"""
v2: CNN on 16x16x2 spatial matrices + MLP on engineered features.
Physics: muon density map channel is the KEY discriminant for gammas.
Gamma showers have near-zero muons - the muon channel spatial pattern is crucial.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda:0")
BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def engineer_features(f):
    """Build feature vector from raw 5-feature array [E, Ze, Az, Ne, Nmu]."""
    E = f[:, 0]
    Ze = f[:, 1]
    Az = f[:, 2]
    Ne = f[:, 3]
    Nmu = f[:, 4]

    # Core discriminant: Ne - Nmu (in log space = log ratio)
    ne_nmu_diff = Ne - Nmu

    Ze_norm = Ze / 30.0
    Ne_norm = (Ne - 5.0) / 0.7
    Nmu_norm = (Nmu - 3.5) / 0.7
    E_norm = (E - 16.0) / 1.0

    Az_rad = np.radians(Az)
    Az_cos = np.cos(Az_rad)
    Az_sin = np.sin(Az_rad)

    cos_ze = np.cos(np.radians(Ze))
    ne_e_ratio = Ne - E

    feats = np.stack([
        E_norm, Ze_norm, Az_cos, Az_sin, Ne_norm, Nmu_norm,
        ne_nmu_diff,
        cos_ze,
        ne_e_ratio,
        Ne * Ze_norm,
        Nmu * cos_ze,
    ], axis=1).astype(np.float32)
    return feats


class GammaDataset(Dataset):
    def __init__(self, matrices, features, labels, feat_mean=None, feat_std=None):
        self.matrices = matrices
        self.features = features
        self.labels = labels
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Matrix: (2, 16, 16) - use log1p for sparse data
        mat = self.matrices[idx].astype(np.float32)  # (16, 16, 2)
        mat = np.log1p(mat)
        mat = mat.transpose(2, 0, 1)  # (2, 16, 16)

        feat = self.features[idx].copy()
        if self.feat_mean is not None:
            feat = (feat - self.feat_mean) / self.feat_std

        return (torch.FloatTensor(mat),
                torch.FloatTensor(feat),
                int(self.labels[idx]))


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class GammaCNN(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        # CNN for matrices
        self.cnn = nn.Sequential(
            CNNBlock(2, 32),
            nn.MaxPool2d(2),   # 8x8
            CNNBlock(32, 64),
            nn.MaxPool2d(2),   # 4x4
            CNNBlock(64, 128),
            nn.AdaptiveAvgPool2d(2),  # 2x2
            nn.Flatten(),  # 512
        )
        cnn_out = 128 * 2 * 2  # 512

        # MLP for features
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion
        self.head = nn.Sequential(
            nn.Linear(cnn_out + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, mat, feat):
        cnn_out = self.cnn(mat)
        feat_out = self.feat_mlp(feat)
        x = torch.cat([cnn_out, feat_out], dim=1)
        return self.head(x)


def survival_at_75(scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])[::-1]
    ng = len(sg)
    idx = int(np.ceil(0.75 * ng)) - 1
    thr = sg[min(idx, ng - 1)]
    survival = (scores[is_hadron] >= thr).sum() / is_hadron.sum()
    return survival


def main():
    print("Loading data...")
    f_train_raw = np.load(f"{BASE}/data/gamma_train/features.npy", mmap_mode='r')
    m_train_raw = np.load(f"{BASE}/data/gamma_train/matrices.npy", mmap_mode='r')
    y_train_raw = np.load(f"{BASE}/data/gamma_train/labels_gamma.npy", mmap_mode='r')

    f_test_raw = np.load(f"{BASE}/data/gamma_test/features.npy", mmap_mode='r')
    m_test_raw = np.load(f"{BASE}/data/gamma_test/matrices.npy", mmap_mode='r')
    y_test_raw = np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r')

    print(f"Train: {len(f_train_raw):,}")
    print(f"Test:  {len(f_test_raw):,}")

    # Engineer features
    f_train = engineer_features(np.array(f_train_raw))
    f_test = engineer_features(np.array(f_test_raw))
    y_train = np.array(y_train_raw)
    y_test = np.array(y_test_raw)

    # Feature normalization
    feat_mean = f_train.mean(0)
    feat_std = f_train.std(0)
    feat_std[feat_std < 1e-8] = 1.0

    # Train/val split (90/10)
    n = len(f_train)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    n_val = int(n * 0.1)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    print(f"Train split: {len(tr_idx):,} train, {len(val_idx):,} val")

    # Create datasets
    train_ds = GammaDataset(m_train_raw[tr_idx], f_train[tr_idx], y_train[tr_idx], feat_mean, feat_std)
    val_ds = GammaDataset(m_train_raw[val_idx], f_train[val_idx], y_train[val_idx], feat_mean, feat_std)
    test_ds = GammaDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = GammaCNN(f_train.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Class weights
    n_gamma = (y_train[tr_idx] == 0).sum()
    n_hadron = (y_train[tr_idx] == 1).sum()
    w_gamma = n_hadron / n_gamma
    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
    print(f"Class weights: gamma={w_gamma:.1f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_survival = 1.0
    best_scores_test = None
    n_epochs = 15

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_total = 0
        for mat, feat, y in train_loader:
            mat, feat, y = mat.to(DEVICE), feat.to(DEVICE), y.to(DEVICE)
            logits = model(mat, feat)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            n_total += len(y)
        scheduler.step()

        # Validate
        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for mat, feat, y in val_loader:
                mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                logits = model(mat, feat)
                probs = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
                val_scores.append(probs)
                val_labels.append(y.numpy())
        val_scores = np.concatenate(val_scores)
        val_labels = np.concatenate(val_labels)
        val_survival = survival_at_75(val_scores, val_labels)

        print(f"Epoch {epoch+1:2d}/{n_epochs}: loss={total_loss/n_total:.4f} "
              f"val_survival@75={val_survival:.2e} lr={optimizer.param_groups[0]['lr']:.5f}")

        if val_survival < best_val_survival:
            best_val_survival = val_survival
            # Get test scores
            test_scores = []
            with torch.no_grad():
                for mat, feat, y in test_loader:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    logits = model(mat, feat)
                    probs = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
                    test_scores.append(probs)
            best_scores_test = np.concatenate(test_scores)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v2.pt")
            print(f"  -> New best val survival: {val_survival:.2e}")

    print(f"\nBest val survival@75: {best_val_survival:.2e}")
    np.save(f"{OUT_DIR}/probs_v2.npy", best_scores_test)
    np.savez(f"{OUT_DIR}/predictions_v2.npz", gamma_scores=best_scores_test)
    np.savez(f"{OUT_DIR}/predictions.npz", gamma_scores=best_scores_test)
    print(f"Saved predictions ({len(best_scores_test)} scores)")

    print("\n---")
    print(f"metric: {best_val_survival:.4e}")
    print("description: CNN on 16x16x2 matrices + MLP on engineered features, 15 epochs")


if __name__ == "__main__":
    main()
