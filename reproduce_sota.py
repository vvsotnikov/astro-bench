"""Reproduce Kuznetsov et al. (JINST 2024) SOTA result end-to-end.

Reimplements the kgnn pipeline in plain PyTorch:
- Data: QGS spectra with cuts Ze<30, Ne>4.8, Age∈(0.2,1.48), Nmu>3.6
- Architecture: LeNet (4 conv + 3 linear, 36.6K params)
- Features: channels=[1,2] (e/γ + μ), reco_features=[Ne, Nmu, Age, Ze]
- Split: 70% train / 21% val / 9% test (torch random_split, seed=42)
- Training: Adam lr=3e-4, ReduceLROnPlateau, weighted CE + batch fraction loss
- Augmentation: Rotate90 + Flip
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import sys

DATA_DIR = "data"
DEVICE = "cuda"
BATCH_SIZE = 1024
MAX_EPOCHS = 80
LR = 3e-4
SEED = 42

# Quality cuts matching Kuznetsov et al.
CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}

# Feature normalization constants (from kgnn source)
NORM = {
    'Ne': (5.31, 0.5),    # (mean, std)
    'Nmu': (4.3, 0.42),   # (mean, std)
    'Age': (1.0, 1.0),    # subtract 1 (i.e. center=1, scale=1)
    'Ze': (0.0, 60.0),    # min-max: (Ze - 0) / 60
}

PARTICLE_NAMES = ["proton", "helium", "carbon", "silicon", "iron"]
PARTICLE_IDS = [14, 402, 1206, 2814, 5626]

def p(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class BigQGSDataset(Dataset):
    """Reimplementation of kgnn.data.BigQGSDataset."""

    RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
    TRUE_HEADERS = ['primary_id', 'E', 'Xc', 'Yc', 'Ze', 'Az']

    def __init__(self, mat_file, feat_file, true_file, cut_data=None):
        raw_mat = np.load(mat_file)['matrices']
        raw_feat = np.load(feat_file)['features']
        raw_true = np.load(true_file)['true_features']

        # features columns: [part_type, E, Xc, Yc, core_dist, Ze, Az, Ne, Nmu, Age]
        # reco = features[:, 1:] → [E, Xc, Yc, Core_distance, Ze, Az, Ne, Nmu, Age]
        reco = raw_feat[:, 1:]

        # true: primary_id (from features col 0) + true E, Xc, Yc, Ze, Az
        true = np.concatenate([
            raw_feat[:, [0]],  # particle type from reconstructed
            raw_true[:, [0, 2, 3, 4, 5]]  # E, Xc, Yc, Ze, Az from true
        ], axis=1)

        self.arrays = torch.from_numpy(raw_mat)
        self.reconstructed = torch.from_numpy(reco.astype(np.float32))
        self.true = torch.from_numpy(true.astype(np.float32))

        # Apply cuts
        if cut_data:
            mask = torch.ones(len(self.reconstructed), dtype=torch.bool)
            for feat_name, (lo, hi) in cut_data.items():
                idx = self.RECO_HEADERS.index(feat_name)
                col = self.reconstructed[:, idx]
                mask &= (col > lo) & (col < hi)
            self.arrays = self.arrays[mask]
            self.reconstructed = self.reconstructed[mask]
            self.true = self.true[mask]

        p(f"  Loaded {len(self)} events after cuts")

    def __len__(self):
        return self.reconstructed.shape[0]

    def __getitem__(self, idx):
        reco_dict = {name: self.reconstructed[idx, i] for i, name in enumerate(self.RECO_HEADERS)}
        true_dict = {name: self.true[idx, i] for i, name in enumerate(self.TRUE_HEADERS)}
        return {
            'arrays': self.arrays[idx],
            'reconstructed': reco_dict,
            'true': true_dict,
        }


class TransformDataset(Dataset):
    """Wraps a Subset with transforms (like kgnn DatasetFromSubset)."""
    def __init__(self, subset, augment=False):
        self.subset = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        item = self.subset[idx]
        arrays = item['arrays'].clone()
        if self.augment:
            # Rotate90(1, 0.5) — 90° rotation with 50% prob
            if torch.rand(1) < 0.5:
                arrays = torch.rot90(arrays.float(), 1, [0, 1]).to(arrays.dtype)
            # Rotate90(2, 0.5) — 180° rotation with 50% prob
            if torch.rand(1) < 0.5:
                arrays = torch.rot90(arrays.float(), 2, [0, 1]).to(arrays.dtype)
            # Flip(0, 0.5) — horizontal flip with 50% prob
            if torch.rand(1) < 0.5:
                arrays = torch.flip(arrays.float(), dims=[0]).to(arrays.dtype)
        return {
            'arrays': arrays,
            'reconstructed': item['reconstructed'],
            'true': item['true'],
        }


# ---------------------------------------------------------------------------
# Model: LeNet architecture (exact copy from kgnn)
# ---------------------------------------------------------------------------

class LeNetArchitecture(nn.Module):
    """4 conv layers (all 16 channels) + 3 linear layers. 36.6K params."""

    def __init__(self, n_classes=5, n_reco_features=4, n_channels=2):
        super().__init__()
        ch = 16
        self.conv0 = nn.Conv2d(n_channels, ch, 2)
        self.conv1 = nn.Conv2d(ch, ch, 3)
        self.conv2 = nn.Conv2d(ch, ch, 4)
        self.conv3 = nn.Conv2d(ch, ch, 2)
        self.fc1 = nn.Linear(ch * 3 * 3 + n_reco_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)
        self.fc_on_conv = nn.Linear(ch * 3 * 3, n_classes)

    def forward(self, x, x_reco=None, return_cnn_out=False):
        # x: (B, 16, 16, C) — channels last
        x = x.permute(0, 3, 1, 2).float()
        x = F.max_pool2d(F.relu(self.conv0(x)), 2, stride=1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=1)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x_cnn = x
        if x_reco is not None:
            x = torch.cat((x, x_reco), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if return_cnn_out:
            return x, self.fc_on_conv(x_cnn)
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def extract_batch(batch, channels=[1, 2], reco_features=['Ne', 'Nmu', 'Age', 'Ze']):
    """Extract model inputs from batch dict (matches kgnn _extract_from_batch)."""
    x = batch['arrays'].float()[:, :, :, channels]

    reco_list = []
    for feat in reco_features:
        val = batch['reconstructed'][feat].float().reshape(-1, 1)
        mean, scale = NORM[feat]
        val = (val - mean) / scale
        reco_list.append(val)
    x_reco = torch.cat(reco_list, dim=1) if reco_list else None

    # Labels: primary_id → class index (1-indexed in their data, 0-indexed for us)
    y = batch['true']['primary_id'].long() - 1
    return x, x_reco, y


def compute_loss(model, x, x_reco, y, ce_loss_fn):
    """Compute loss matching kgnn: CE/5 + CNN-only CE/5 + batch fraction MSE."""
    output, output_cnn = model(x, x_reco, return_cnn_out=True)

    # Main CE loss (divided by 5, as in kgnn)
    loss = ce_loss_fn(output, y).mean() / 5

    # CNN-only auxiliary loss
    if output_cnn is not None:
        loss += ce_loss_fn(output_cnn, y).mean() / 5

    # Batch fraction loss
    logits = F.softmax(output, dim=1)
    items, counts = torch.unique(logits.argmax(dim=1), sorted=True, return_counts=True)
    y_pred = torch.zeros(5, device=x.device)
    y_pred[items.long()] = counts.float()
    items_true, counts_true = torch.unique(y, sorted=True, return_counts=True)
    y_true = torch.zeros(5, device=x.device)
    y_true[items_true.long()] = counts_true.float()
    loss += ((y_pred / y_pred.sum() - y_true / y_true.sum()) ** 2).mean()

    return loss, output


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    # --- Load data ---
    p("Loading QGS spectra data with Kuznetsov cuts...")
    dataset = BigQGSDataset(
        f"{DATA_DIR}/qgs_spectra_matrices.npz",
        f"{DATA_DIR}/qgs_spectra_features.npz",
        f"{DATA_DIR}/qgs_spectra_true_features.npz",
        cut_data=CUTS,
    )
    p(f"Total events after cuts: {len(dataset)}")

    # --- Split: 70% train / 21% val / 9% test ---
    # Matches: train_test_split(dataset, test_part=0.3) then split(valid, test_part=0.3)
    n = len(dataset)
    n_valid = int(n * 0.3)
    n_train = n - n_valid
    train_subset, valid_subset = random_split(
        dataset, [n_train, n_valid],
        generator=torch.Generator().manual_seed(42)
    )

    n_test = int(n_valid * 0.3)
    n_val = n_valid - n_test
    val_subset, test_subset = random_split(
        valid_subset, [n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    p(f"Split: train={len(train_subset)}, val={len(val_subset)}, test={len(test_subset)}")

    # Wrap with transforms
    train_data = TransformDataset(train_subset, augment=True)
    val_data = TransformDataset(val_subset, augment=False)
    test_data = TransformDataset(test_subset, augment=False)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    # --- Class weights ---
    # Count classes in full dataset (before split)
    all_labels = dataset.true[:, 0].long() - 1  # primary_id → 0-indexed
    _, counts = torch.unique(all_labels, sorted=True, return_counts=True)
    weights = counts[0].float() / counts.float()
    p(f"Class counts: {counts.tolist()}")
    p(f"Class weights: {weights.tolist()}")

    # --- Model ---
    model = LeNetArchitecture(n_classes=5, n_reco_features=4, n_channels=2).to(DEVICE)
    n_params = sum(pp.numel() for pp in model.parameters())
    p(f"Model params: {n_params:,}")

    ce_loss_fn = nn.CrossEntropyLoss(weight=weights.to(DEVICE), reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=7,
    )

    # --- Training loop ---
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 7  # early stopping

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        train_loss_sum, train_correct, train_total = 0, 0, 0
        for batch in train_loader:
            x, x_reco, y = extract_batch(batch)
            x, x_reco, y = x.to(DEVICE), x_reco.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            loss, output = compute_loss(model, x, x_reco, y, ce_loss_fn)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(y)
            train_correct += (output.argmax(1) == y).sum().item()
            train_total += len(y)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss_sum, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, x_reco, y = extract_batch(batch)
                x, x_reco, y = x.to(DEVICE), x_reco.to(DEVICE), y.to(DEVICE)
                loss, output = compute_loss(model, x, x_reco, y, ce_loss_fn)
                val_loss_sum += loss.item() * len(y)
                val_correct += (output.argmax(1) == y).sum().item()
                val_total += len(y)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]['lr']
        p(f"Ep {epoch+1:3d}/{MAX_EPOCHS}: "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
          f"lr={lr:.2e} [{time.time()-t0:.0f}s]")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "reproduce_sota_best.pt")
            p(f"  >>> New best val_loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                p(f"  Early stopping at epoch {epoch+1}")
                break

    # --- Evaluate on test set ---
    p("\n" + "=" * 60)
    p("Evaluating on test set...")
    model.load_state_dict(torch.load("reproduce_sota_best.pt", weights_only=True))
    model.eval()

    all_preds, all_trues, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x, x_reco, y = extract_batch(batch)
            x, x_reco, y = x.to(DEVICE), x_reco.to(DEVICE), y.to(DEVICE)
            output = model(x, x_reco)
            probs = F.softmax(output, dim=1)
            all_preds.append(output.argmax(1).cpu().numpy())
            all_trues.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    probs = np.concatenate(all_probs)

    acc = accuracy_score(trues, preds)
    p(f"Test accuracy: {acc:.4f} ({acc*100:.2f}%)")
    p(f"\n{classification_report(trues, preds, target_names=PARTICLE_NAMES)}")

    cm = confusion_matrix(trues, preds)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    p("Confusion matrix (row-normalized):")
    abbr = [n[:2] for n in PARTICLE_NAMES]
    p("     " + " ".join(f"{a:>6}" for a in abbr))
    for i in range(5):
        p(f"  {abbr[i]:>2} " + " ".join(f"{cm_norm[i,j]:>6.3f}" for j in range(5)))

    # --- Evaluate with fraction error (matching verify.py methodology) ---
    p("\nComputing fraction error (1001 grid ensembles × 5000 events)...")

    MIXTURE_SIZE = 5000
    MIXTURE_SEED = 2026
    GRID_STEP = 0.1

    def generate_fraction_grid(n_classes=5, step=GRID_STEP):
        n_steps = round(1.0 / step)
        fractions = []
        def _recurse(remaining, depth, current):
            if depth == n_classes - 1:
                current.append(remaining * step)
                fractions.append(current[:])
                current.pop()
                return
            for i in range(remaining + 1):
                current.append(i * step)
                _recurse(remaining - i, depth + 1, current)
                current.pop()
        _recurse(n_steps, 0, [])
        return np.array(fractions)

    rng = np.random.default_rng(MIXTURE_SEED)
    class_indices = {c: np.where(trues == c)[0] for c in range(5)}
    fractions = generate_fraction_grid()
    n_ensembles = len(fractions)

    all_errors = []
    for mix_idx in range(n_ensembles):
        target_fracs = fractions[mix_idx]
        counts_per_class = np.round(target_fracs * MIXTURE_SIZE).astype(int)
        diff = MIXTURE_SIZE - counts_per_class.sum()
        if diff != 0:
            counts_per_class[np.argmax(counts_per_class)] += diff

        sampled_preds = []
        actual_true_fracs = np.zeros(5)
        for c in range(5):
            n_sample = counts_per_class[c]
            if n_sample <= 0:
                continue
            idx = rng.choice(class_indices[c], size=n_sample, replace=True)
            sampled_preds.append(preds[idx])
            actual_true_fracs[c] = n_sample
        actual_true_fracs /= actual_true_fracs.sum()
        all_sampled = np.concatenate(sampled_preds)
        pred_counts = np.bincount(all_sampled, minlength=5)[:5]
        pred_fracs = pred_counts / pred_counts.sum()
        all_errors.append(np.abs(actual_true_fracs - pred_fracs))

    all_errors = np.array(all_errors)
    mean_frac_error = all_errors.mean()
    per_class = [all_errors[:, c].mean() for c in range(5)]

    p(f"\nMean fraction error: {mean_frac_error:.4f}")
    p(f"Per-class: " + " ".join(f"{PARTICLE_NAMES[c][:2]}={per_class[c]:.4f}" for c in range(5)))

    # --- Also evaluate on full valid set (val + test, matching their "valid_data0") ---
    p("\n" + "=" * 60)
    p("Evaluating on full validation set (val + test)...")
    full_valid_data = TransformDataset(valid_subset, augment=False)
    full_valid_loader = DataLoader(full_valid_data, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=4, pin_memory=True, persistent_workers=True)

    all_preds_v, all_trues_v = [], []
    with torch.no_grad():
        for batch in full_valid_loader:
            x, x_reco, y = extract_batch(batch)
            x, x_reco, y = x.to(DEVICE), x_reco.to(DEVICE), y.to(DEVICE)
            output = model(x, x_reco)
            all_preds_v.append(output.argmax(1).cpu().numpy())
            all_trues_v.append(y.cpu().numpy())

    preds_v = np.concatenate(all_preds_v)
    trues_v = np.concatenate(all_trues_v)
    acc_v = accuracy_score(trues_v, preds_v)
    p(f"Full valid accuracy: {acc_v:.4f} ({acc_v*100:.2f}%)")

    # Fraction error on full valid
    class_indices_v = {c: np.where(trues_v == c)[0] for c in range(5)}
    rng2 = np.random.default_rng(MIXTURE_SEED)
    all_errors_v = []
    for mix_idx in range(n_ensembles):
        target_fracs = fractions[mix_idx]
        counts_per_class = np.round(target_fracs * MIXTURE_SIZE).astype(int)
        diff = MIXTURE_SIZE - counts_per_class.sum()
        if diff != 0:
            counts_per_class[np.argmax(counts_per_class)] += diff
        sampled_preds = []
        actual_true_fracs = np.zeros(5)
        for c in range(5):
            n_sample = counts_per_class[c]
            if n_sample <= 0:
                continue
            idx = rng2.choice(class_indices_v[c], size=n_sample, replace=True)
            sampled_preds.append(preds_v[idx])
            actual_true_fracs[c] = n_sample
        actual_true_fracs /= actual_true_fracs.sum()
        all_sampled = np.concatenate(sampled_preds)
        pred_counts = np.bincount(all_sampled, minlength=5)[:5]
        pred_fracs = pred_counts / pred_counts.sum()
        all_errors_v.append(np.abs(actual_true_fracs - pred_fracs))

    all_errors_v = np.array(all_errors_v)
    mean_frac_error_v = all_errors_v.mean()
    per_class_v = [all_errors_v[:, c].mean() for c in range(5)]
    p(f"Full valid fraction error: {mean_frac_error_v:.4f}")
    p(f"Per-class: " + " ".join(f"{PARTICLE_NAMES[c][:2]}={per_class_v[c]:.4f}" for c in range(5)))

    elapsed = time.time() - t0
    p(f"\nTotal time: {elapsed/60:.1f} min")
    p("---")
    p(f"test_accuracy: {acc:.4f}")
    p(f"test_fraction_error: {mean_frac_error:.4f}")
    p(f"valid_accuracy: {acc_v:.4f}")
    p(f"valid_fraction_error: {mean_frac_error_v:.4f}")


if __name__ == "__main__":
    main()
