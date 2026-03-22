"""v1: CNN + engineered physics features for gamma/hadron separation.

Architecture:
- CNN branch: 5-layer conv net on log1p(16x16x2 matrices), processes spatial muon/electron patterns
- Feature branch: MLP on original + engineered features (Ne/Nmu ratio, muon channel stats)
- Combined head: classification

Key physics: gamma rays have near-zero muons → muon channel (ch1) is primary discriminator.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from load_data import load_train, load_test, check_gpu_free
from verify import evaluate


class GammaDataset(Dataset):
    def __init__(self, matrices, features, labels, mat_stats, feat_stats):
        self.matrices = matrices
        self.features = features
        self.labels = labels
        self.mat_mean, self.mat_std = mat_stats  # (2,) each
        self.feat_mean, self.feat_std = feat_stats  # (9,) each

    def __len__(self):
        return len(self.labels)

    def _engineer_features(self, mat_raw, feat_raw):
        """Engineer physics-motivated features from raw data."""
        E, Ze, Az, Ne, Nmu = feat_raw

        # Key physics discriminator: Ne/Nmu in log space
        Ne_Nmu = Ne - Nmu  # log(Ne/Nmu) — gamma showers have high Ne/Nmu

        # Muon channel statistics from spatial data
        muon_ch = mat_raw[:, :, 1]  # (16, 16)
        elec_ch = mat_raw[:, :, 0]  # (16, 16)

        muon_sum = np.log1p(muon_ch.sum())
        muon_nz_frac = (muon_ch > 0).sum() / 256.0  # sparsity

        elec_sum = np.log1p(elec_ch.sum())
        spatial_ratio = elec_sum - muon_sum  # spatial Ne/Nmu proxy

        # Return 9 engineered features
        return np.array([E, Ze, Ne, Nmu, Ne_Nmu, muon_sum, muon_nz_frac, elec_sum, spatial_ratio],
                        dtype=np.float32)

    def __getitem__(self, idx):
        mat_raw = self.matrices[idx].astype(np.float32)  # (16, 16, 2)
        feat_raw = self.features[idx].astype(np.float32)  # (5,)

        # Log1p transform + normalize matrices
        mat_log = np.log1p(mat_raw)  # sparse → log space
        mat_norm = (mat_log - self.mat_mean) / self.mat_std  # (16, 16, 2)
        mat_chw = mat_norm.transpose(2, 0, 1)  # (2, 16, 16)

        # Engineered features
        eng_feat = self._engineer_features(mat_raw, feat_raw)
        feat_norm = (eng_feat - self.feat_mean) / self.feat_std

        label = float(self.labels[idx])

        return (torch.from_numpy(mat_chw.copy()),
                torch.from_numpy(feat_norm),
                torch.tensor(label, dtype=torch.float32))


def compute_matrix_stats(matrices, n_samples=200_000):
    """Compute per-channel mean/std on log1p-transformed data."""
    rng = np.random.default_rng(42)
    n = len(matrices)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    sample = np.array(matrices[idx]).astype(np.float32)  # (n_samples, 16, 16, 2)
    sample_log = np.log1p(sample)
    mean = sample_log.mean(axis=(0, 1, 2))  # (2,)
    std = sample_log.std(axis=(0, 1, 2))    # (2,)
    std[std == 0] = 1.0
    return mean.reshape(1, 1, 2), std.reshape(1, 1, 2)


def compute_feature_stats(matrices, features, n_samples=200_000):
    """Compute engineered feature mean/std."""
    rng = np.random.default_rng(42)
    n = len(features)
    idx = rng.choice(n, size=min(n_samples, n), replace=False)

    eng_feats = []
    for i in idx:
        mat = matrices[i].astype(np.float32)
        feat = features[i].astype(np.float32)
        E, Ze, Az, Ne, Nmu = feat
        Ne_Nmu = Ne - Nmu
        muon_ch = mat[:, :, 1]
        elec_ch = mat[:, :, 0]
        muon_sum = np.log1p(muon_ch.sum())
        muon_nz_frac = (muon_ch > 0).sum() / 256.0
        elec_sum = np.log1p(elec_ch.sum())
        spatial_ratio = elec_sum - muon_sum
        eng_feats.append(np.array([E, Ze, Ne, Nmu, Ne_Nmu, muon_sum, muon_nz_frac, elec_sum, spatial_ratio],
                                   dtype=np.float32))

    eng_feats = np.stack(eng_feats)
    mean = eng_feats.mean(axis=0)
    std = eng_feats.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


class GammaCNN(nn.Module):
    def __init__(self, n_scalar=9):
        super().__init__()

        # CNN branch: processes (2, 16, 16) matrices
        # Focus on learning muon channel patterns
        self.cnn = nn.Sequential(
            # Block 1: (2, 16, 16) -> (32, 16, 16)
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # Block 2: -> (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),  # -> (64, 8, 8)
            # Block 3: -> (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            # Block 4: -> (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.MaxPool2d(2),  # -> (256, 4, 4)
            # Block 5: -> (256, 4, 4)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),  # -> (256, 1, 1)
        )
        self.cnn_dim = 256

        # Feature branch: MLP on engineered scalar features
        self.feat_net = nn.Sequential(
            nn.Linear(n_scalar, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.feat_dim = 128

        # Combined head
        combined_dim = self.cnn_dim + self.feat_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, matrices, scalars):
        # matrices: (B, 2, 16, 16)
        cnn_out = self.cnn(matrices).squeeze(-1).squeeze(-1)  # (B, 256)
        feat_out = self.feat_net(scalars)  # (B, 128)
        combined = torch.cat([cnn_out, feat_out], dim=1)
        return self.head(combined).squeeze(-1)  # (B,) logit for class 1 (hadron)


def main():
    check_gpu_free()
    device = torch.device("cuda:0")
    print(f"Device: {device}")

    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")
    n_gamma_tr = int((np.array(y_train) == 0).sum())
    n_hadron_tr = int((np.array(y_train) == 1).sum())
    print(f"  Train gamma: {n_gamma_tr:,}  hadron: {n_hadron_tr:,}")

    print("Computing normalization statistics...")
    mat_stats = compute_matrix_stats(X_train)
    feat_stats = compute_feature_stats(X_train, f_train)
    print(f"  Matrix stats computed (per-channel log1p)")
    print(f"  Feature stats: mean={feat_stats[0][:5]}")

    # Validation split (stratified by class)
    rng = np.random.default_rng(42)
    y_arr = np.array(y_train)
    gamma_idx = np.where(y_arr == 0)[0]
    hadron_idx = np.where(y_arr == 1)[0]

    rng.shuffle(gamma_idx)
    rng.shuffle(hadron_idx)

    val_frac = 0.1
    n_val_g = int(len(gamma_idx) * val_frac)
    n_val_h = int(len(hadron_idx) * val_frac)

    val_idx = np.concatenate([gamma_idx[:n_val_g], hadron_idx[:n_val_h]])
    train_idx = np.concatenate([gamma_idx[n_val_g:], hadron_idx[n_val_h:]])

    print(f"  Train split: {len(train_idx):,}  Val split: {len(val_idx):,}")

    # Create indexed datasets
    class IndexedDataset(Dataset):
        def __init__(self, base_dataset, indices):
            self.base = base_dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.base[self.indices[i]]

    full_train_ds = GammaDataset(X_train, f_train, y_train, mat_stats, feat_stats)
    full_test_ds = GammaDataset(X_test, f_test, y_test, mat_stats, feat_stats)

    train_ds = IndexedDataset(full_train_ds, train_idx)
    val_ds = IndexedDataset(full_train_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(full_test_ds, batch_size=4096, shuffle=False, num_workers=8, pin_memory=True)

    model = GammaCNN(n_scalar=9).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # BCEWithLogitsLoss with pos_weight for class imbalance
    # Label convention: 0=gamma, 1=hadron
    # pos_weight upweights the positive class (hadron=1) in training
    # But we want to be sensitive to gammas → use inverse pos_weight
    pos_weight = torch.tensor([n_hadron_tr / n_gamma_tr], device=device)
    print(f"  pos_weight: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_loss = float('inf')
    best_epoch = 0
    best_model_path = "model_v1_best.pt"

    print("\nTraining...")
    for epoch in range(50):
        # Train
        model.train()
        total_loss = total = 0
        for X_b, f_b, y_b in train_loader:
            X_b, f_b, y_b = X_b.to(device), f_b.to(device), y_b.to(device)
            logits = model(X_b, f_b)
            loss = criterion(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)
            total += len(y_b)

        scheduler.step()

        # Validate
        model.eval()
        val_loss = val_total = 0
        with torch.no_grad():
            for X_b, f_b, y_b in val_loader:
                X_b, f_b, y_b = X_b.to(device), f_b.to(device), y_b.to(device)
                logits = model(X_b, f_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(y_b)
                val_total += len(y_b)

        val_loss_avg = val_loss / val_total
        print(f"Epoch {epoch+1:2d}/50: train_loss={total_loss/total:.4f} val_loss={val_loss_avg:.4f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)

    print(f"\nBest model at epoch {best_epoch}, val_loss={best_val_loss:.4f}")

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Generate test predictions
    # logit > 0 → model predicts hadron; gamma_score = -logit (higher = more gamma-like)
    all_gamma_scores = []
    with torch.no_grad():
        for X_b, f_b, _ in test_loader:
            X_b, f_b = X_b.to(device), f_b.to(device)
            logits = model(X_b, f_b)
            gamma_scores = -logits.cpu().numpy()  # higher = more gamma-like
            all_gamma_scores.append(gamma_scores)

    gamma_scores = np.concatenate(all_gamma_scores)
    print(f"Test scores: min={gamma_scores.min():.3f} max={gamma_scores.max():.3f} "
          f"mean={gamma_scores.mean():.3f}")

    # Official evaluation
    evaluate(gamma_scores,
             "v1: CNN (5 conv layers) + engineered features (Ne/Nmu ratio, muon channel stats), "
             "log1p transform, 50 epochs cosine LR, BCELoss pos_weight",
             pred_path="predictions_v1.npz")

    np.savez("predictions_v1.npz", gamma_scores=gamma_scores)


if __name__ == "__main__":
    main()
