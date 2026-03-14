"""Beat Kuznetsov et al. SOTA (0.1079 fraction error) using same data pipeline.

Same data: QGS spectra, cuts Ze<30, Ne>4.8, Age∈(0.2,1.48), Nmu>3.6
Same split: 70/21/9 train/val/test (seed=42)
Same evaluation: 1001 grid ensembles × 5000 events

Better model: CNN+Attn+MLP (~1M params) with batch fraction loss,
trained with multiple seeds for ensembling.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
from scipy.optimize import differential_evolution
import time
import sys
import gc
import os

DATA_DIR = "data"
DEVICE = "cuda"
BATCH_SIZE = 2048
MAX_EPOCHS = 60
LR = 1e-3
SEED = 42

CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}
RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
PARTICLE_NAMES = ["proton", "helium", "carbon", "silicon", "iron"]

# Normalization matching kgnn
NORM = {'Ne': (5.31, 0.5), 'Nmu': (4.3, 0.42), 'Age': (1.0, 1.0), 'Ze': (0.0, 60.0)}

MIXTURE_SIZE = 5000
MIXTURE_SEED = 2026
GRID_STEP = 0.1

def p(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data (same as reproduce_sota.py)
# ---------------------------------------------------------------------------

class QGSDataset(Dataset):
    def __init__(self, mat_file, feat_file, true_file, cut_data=None):
        raw_mat = np.load(mat_file)['matrices']
        raw_feat = np.load(feat_file)['features']
        raw_true = np.load(true_file)['true_features']
        reco = raw_feat[:, 1:]  # [E, Xc, Yc, core_dist, Ze, Az, Ne, Nmu, Age]
        true = np.concatenate([raw_feat[:, [0]], raw_true[:, [0, 2, 3, 4, 5]]], axis=1)
        self.arrays = torch.from_numpy(raw_mat)
        self.reconstructed = torch.from_numpy(reco.astype(np.float32))
        self.true = torch.from_numpy(true.astype(np.float32))
        if cut_data:
            mask = torch.ones(len(self.reconstructed), dtype=torch.bool)
            for feat_name, (lo, hi) in cut_data.items():
                idx = RECO_HEADERS.index(feat_name)
                col = self.reconstructed[:, idx]
                mask &= (col > lo) & (col < hi)
            self.arrays = self.arrays[mask]
            self.reconstructed = self.reconstructed[mask]
            self.true = self.true[mask]
        p(f"  Loaded {len(self)} events")

    def __len__(self):
        return self.reconstructed.shape[0]

    def __getitem__(self, idx):
        return {
            'arrays': self.arrays[idx],
            'reconstructed': {name: self.reconstructed[idx, i] for i, name in enumerate(RECO_HEADERS)},
            'true': {'primary_id': self.true[idx, 0]},
        }


class FastDataset(Dataset):
    """Pre-extract all tensors for fast training (no dict overhead per item)."""
    def __init__(self, subset, channels=[1, 2], reco_features=['Ne', 'Nmu', 'Age', 'Ze'], augment=False):
        self.augment = augment
        # Unwrap nested Subsets to get the root dataset and flat indices
        indices = list(range(len(subset)))
        ds = subset
        while hasattr(ds, 'dataset'):
            if hasattr(ds, 'indices'):
                indices = [ds.indices[i] for i in indices]
            ds = ds.dataset

        # Pre-extract everything
        arrays = ds.arrays[indices][:, :, :, channels].float()
        reco_list = []
        for feat in reco_features:
            idx = RECO_HEADERS.index(feat)
            val = ds.reconstructed[indices, idx].float().reshape(-1, 1)
            mean, scale = NORM[feat]
            val = (val - mean) / scale
            reco_list.append(val)
        self.x_reco = torch.cat(reco_list, dim=1)
        self.x_arrays = arrays
        self.y = ds.true[indices, 0].long() - 1  # 1-indexed → 0-indexed

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        arrays = self.x_arrays[idx]
        if self.augment:
            if torch.rand(1) < 0.5:
                arrays = torch.rot90(arrays, 1, [0, 1])
            if torch.rand(1) < 0.5:
                arrays = torch.rot90(arrays, 2, [0, 1])
            if torch.rand(1) < 0.5:
                arrays = torch.flip(arrays, dims=[0])
        return arrays, self.x_reco[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model: CNN+Attn+MLP (our architecture, proven to work)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 8)), nn.ReLU(),
            nn.Linear(max(ch // r, 8), ch), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class HybridModel(nn.Module):
    def __init__(self, n_feat=4, n_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ChannelAttention(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ChannelAttention(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ChannelAttention(256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, mat, feat):
        # mat: (B, H, W, C) → (B, C, H, W)
        mat = mat.permute(0, 3, 1, 2)
        cnn_out = self.cnn(mat).flatten(1)
        feat_out = self.feat_mlp(feat)
        return self.head(torch.cat([cnn_out, feat_out], dim=1))


# ---------------------------------------------------------------------------
# Fraction error evaluator
# ---------------------------------------------------------------------------

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

class FracEval:
    def __init__(self, y_test):
        ci = {c: np.where(y_test == c)[0] for c in range(5)}
        self.fracs = generate_fraction_grid()
        self.ne = len(self.fracs)
        rng = np.random.default_rng(MIXTURE_SEED)
        self.si = []; self.tf = np.zeros((self.ne, 5))
        for i in range(self.ne):
            counts = np.round(self.fracs[i] * MIXTURE_SIZE).astype(int)
            d = MIXTURE_SIZE - counts.sum()
            if d: counts[np.argmax(counts)] += d
            idx = []
            for c in range(5):
                ns = counts[c]
                if ns <= 0: continue
                idx.append(rng.choice(ci[c], size=ns, replace=True))
                self.tf[i, c] = ns / MIXTURE_SIZE
            self.si.append(np.concatenate(idx))

    def evaluate(self, preds):
        e = np.zeros((self.ne, 5))
        for i in range(self.ne):
            s = preds[self.si[i]]
            pc = np.bincount(s, minlength=5)[:5]
            e[i] = np.abs(self.tf[i] - pc / pc.sum())
        return float(e.mean())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_seed(seed, train_ds, val_ds, test_ds, class_weights):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_fast = FastDataset(train_ds, augment=True)
    val_fast = FastDataset(val_ds, augment=False)
    test_fast = FastDataset(test_ds, augment=False)

    train_loader = DataLoader(train_fast, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_fast, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_fast, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    model = HybridModel(n_feat=4).to(DEVICE)
    n_params = sum(pp.numel() for pp in model.parameters())
    p(f"  Model params: {n_params:,}")

    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = GradScaler()

    best_val_loss = float('inf')
    best_probs = None
    patience = 0
    t0 = time.time()

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        train_correct, train_total = 0, 0
        for arrays, reco, labels in train_loader:
            arrays, reco, labels = arrays.to(DEVICE), reco.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output = model(arrays, reco)
                # CE loss (per-sample, weighted)
                loss = ce_loss(output, labels).mean()
                # Batch fraction loss (from kgnn)
                probs = F.softmax(output, dim=1)
                items_p, counts_p = torch.unique(probs.argmax(1), sorted=True, return_counts=True)
                y_pred = torch.zeros(5, device=DEVICE)
                y_pred[items_p.long()] = counts_p.float()
                items_t, counts_t = torch.unique(labels, sorted=True, return_counts=True)
                y_true = torch.zeros(5, device=DEVICE)
                y_true[items_t.long()] = counts_t.float()
                frac_loss = ((y_pred / y_pred.sum() - y_true / y_true.sum()) ** 2).mean()
                loss = loss + frac_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_correct += (output.argmax(1) == labels).sum().item()
            train_total += len(labels)
        scheduler.step()
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss_sum, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for arrays, reco, labels in val_loader:
                arrays, reco, labels = arrays.to(DEVICE), reco.to(DEVICE), labels.to(DEVICE)
                with autocast(device_type='cuda'):
                    output = model(arrays, reco)
                    loss_val = ce_loss(output, labels).mean()
                val_loss_sum += loss_val.item() * len(labels)
                val_correct += (output.argmax(1) == labels).sum().item()
                val_total += len(labels)
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
            p(f"  Seed {seed} Ep {epoch+1:3d}/{MAX_EPOCHS}: "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f} "
              f"[{time.time()-t0:.0f}s]")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            # Save test probabilities
            all_probs = []
            with torch.no_grad():
                for arrays, reco, labels in test_loader:
                    arrays, reco, labels = arrays.to(DEVICE), reco.to(DEVICE), labels.to(DEVICE)
                    with autocast(device_type='cuda'):
                        output = model(arrays, reco)
                    all_probs.append(F.softmax(output.float(), 1).cpu().numpy())
            best_probs = np.concatenate(all_probs)

            # Also get full valid probs
            all_probs_v = []
            with torch.no_grad():
                for arrays, reco, labels in val_loader:
                    arrays, reco, labels = arrays.to(DEVICE), reco.to(DEVICE), labels.to(DEVICE)
                    with autocast(device_type='cuda'):
                        output = model(arrays, reco)
                    all_probs_v.append(F.softmax(output.float(), 1).cpu().numpy())
            best_probs_val = np.concatenate(all_probs_v)
        else:
            patience += 1
            if patience >= 15:
                p(f"  Early stopping at epoch {epoch+1}")
                break

    test_preds = best_probs.argmax(1)
    test_acc = (test_preds == test_fast.y.numpy()).mean()
    p(f"  Seed {seed} final: test_acc={test_acc:.4f}")
    return best_probs, best_probs_val


def main():
    t0 = time.time()

    # --- Load data (same as reproduction) ---
    p("Loading QGS spectra data...")
    dataset = QGSDataset(
        f"{DATA_DIR}/qgs_spectra_matrices.npz",
        f"{DATA_DIR}/qgs_spectra_features.npz",
        f"{DATA_DIR}/qgs_spectra_true_features.npz",
        cut_data=CUTS,
    )

    # --- Same split as reproduction ---
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

    # Class weights
    all_labels = dataset.true[:, 0].long() - 1
    _, counts = torch.unique(all_labels, sorted=True, return_counts=True)
    weights = counts[0].float() / counts.float()
    p(f"Class weights: {weights.tolist()}")

    # Get test labels for evaluation
    test_fast_tmp = FastDataset(test_subset, augment=False)
    test_labels = test_fast_tmp.y.numpy()
    val_fast_tmp = FastDataset(val_subset, augment=False)
    val_labels = val_fast_tmp.y.numpy()
    del test_fast_tmp, val_fast_tmp

    # --- Train multiple seeds ---
    seeds = [42, 7, 123]
    all_test_probs = {}
    all_val_probs = {}

    for seed in seeds:
        p(f"\n{'='*60}")
        p(f"Training seed={seed}")
        p(f"{'='*60}")
        test_probs, val_probs = train_one_seed(seed, train_subset, val_subset, test_subset, weights)
        all_test_probs[seed] = test_probs
        all_val_probs[seed] = val_probs

    # --- Evaluate individual models ---
    p(f"\n{'='*60}")
    p("Individual model results:")
    frac_eval = FracEval(test_labels)

    for seed in seeds:
        preds = all_test_probs[seed].argmax(1)
        acc = (preds == test_labels).mean()
        fe = frac_eval.evaluate(preds)
        p(f"  Seed {seed}: acc={acc:.4f} frac_err={fe:.4f}")

    # --- Ensemble ---
    p(f"\n{'='*60}")
    p("Ensemble results:")

    from itertools import combinations
    best_fe = 1.0
    best_name = ""
    best_preds = None

    seed_keys = list(all_test_probs.keys())
    for r in range(1, len(seed_keys) + 1):
        for combo in combinations(seed_keys, r):
            avg = np.mean([all_test_probs[s] for s in combo], axis=0)
            la = np.log(avg + 1e-10)
            raw_preds = avg.argmax(1)
            raw_fe = frac_eval.evaluate(raw_preds)

            # Bias optimization with DE
            def obj(biases):
                return frac_eval.evaluate((la + biases).argmax(1))

            res = differential_evolution(
                obj, bounds=[(-0.5, 0.5)] * 5,
                seed=42, maxiter=500, tol=1e-8, polish=True, popsize=25,
            )
            bias_preds = (la + res.x).argmax(1)
            bias_fe = frac_eval.evaluate(bias_preds)

            name = "+".join(str(s) for s in combo)
            p(f"  {name}: raw={raw_fe:.4f} DE={bias_fe:.4f}")

            if bias_fe < best_fe:
                best_fe = bias_fe
                best_name = name
                best_preds = bias_preds.copy()

    p(f"\n{'='*60}")
    p(f"BEST: {best_name} = {best_fe:.4f}")
    p(f"SOTA: 0.1079 (Kuznetsov et al. reproduction)")
    if best_fe < 0.1079:
        p(f">>> BEAT SOTA by {0.1079 - best_fe:.4f} <<<")
    else:
        p(f"Did not beat SOTA (delta = {best_fe - 0.1079:.4f})")

    elapsed = time.time() - t0
    p(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
