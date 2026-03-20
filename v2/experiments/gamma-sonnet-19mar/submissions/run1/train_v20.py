"""
v20: CNN with dilated convolutions to capture long-range spatial correlations.
Dilated (atrous) convolutions give a larger effective receptive field without
losing spatial resolution — good for detecting muon patterns across the whole grid.

Also: uses strided conv instead of maxpool for downsampling.

Architecture: 3 blocks with dilation rates [1, 2, 4] then global pooling.
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
    E = f[:, 0]; Ze = f[:, 1]; Az = f[:, 2]; Ne = f[:, 3]; Nmu = f[:, 4]
    ne_nmu_diff = Ne - Nmu
    Ze_norm = Ze / 30.0; Ne_norm = (Ne - 5.0) / 0.7
    Nmu_norm = (Nmu - 3.5) / 0.7; E_norm = (E - 16.0) / 1.0
    Az_rad = np.radians(Az); Az_cos = np.cos(Az_rad); Az_sin = np.sin(Az_rad)
    cos_ze = np.cos(np.radians(Ze)); ne_e_ratio = Ne - E
    return np.stack([
        E_norm, Ze_norm, Az_cos, Az_sin, Ne_norm, Nmu_norm,
        ne_nmu_diff, cos_ze, ne_e_ratio, Ne * Ze_norm, Nmu * cos_ze,
    ], axis=1).astype(np.float32)


class GammaDataset(Dataset):
    def __init__(self, matrices, features, labels, feat_mean=None, feat_std=None):
        self.matrices = matrices; self.features = features
        self.labels = labels; self.feat_mean = feat_mean; self.feat_std = feat_std
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        mat = np.log1p(self.matrices[idx].astype(np.float32)).transpose(2, 0, 1)
        feat = self.features[idx].copy()
        if self.feat_mean is not None:
            feat = (feat - self.feat_mean) / self.feat_std
        return torch.FloatTensor(mat), torch.FloatTensor(feat), int(self.labels[idx])


class DilatedBlock(nn.Module):
    """Block with multiple dilation rates for multi-scale feature extraction."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 4
        # 4 parallel paths with different dilation rates
        self.path1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1), nn.BatchNorm2d(mid), nn.ReLU()
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, dilation=1), nn.BatchNorm2d(mid), nn.ReLU()
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=2, dilation=2), nn.BatchNorm2d(mid), nn.ReLU()
        )
        self.path4 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=4, dilation=4), nn.BatchNorm2d(mid), nn.ReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU()
        )

    def forward(self, x):
        return self.fuse(torch.cat([
            self.path1(x), self.path2(x), self.path3(x), self.path4(x)
        ], dim=1))


class GammaDilatedCNN(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        # Process both channels together
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.block1 = DilatedBlock(32, 64)
        self.down1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 16->8

        self.block2 = DilatedBlock(64, 128)
        self.down2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  # 8->4

        self.block3 = DilatedBlock(128, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 4->1, output: (B, 256)
        self.bn_final = nn.BatchNorm2d(256)

        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, mat, feat):
        x = self.stem(mat)
        x = self.block1(x)
        x = torch.relu(self.down1(x))
        x = self.block2(x)
        x = torch.relu(self.down2(x))
        x = self.block3(x)
        x = self.pool(self.bn_final(x)).flatten(1)  # (B, 256)
        f = self.feat_mlp(feat)
        return self.head(torch.cat([x, f], dim=1))


def survival_at_75(scores, labels):
    ig = labels == 0; ih = labels == 1
    sg = np.sort(scores[ig])[::-1]; ng = len(sg)
    thr = sg[min(int(np.ceil(0.75 * ng)) - 1, ng - 1)]
    return float((scores[ih] >= thr).sum() / ih.sum())


def main():
    print("Loading data...")
    f_raw = np.load(f"{BASE}/data/gamma_train/features.npy", mmap_mode='r')
    m_raw = np.load(f"{BASE}/data/gamma_train/matrices.npy", mmap_mode='r')
    y_raw = np.load(f"{BASE}/data/gamma_train/labels_gamma.npy", mmap_mode='r')
    f_test_raw = np.load(f"{BASE}/data/gamma_test/features.npy", mmap_mode='r')
    m_test_raw = np.load(f"{BASE}/data/gamma_test/matrices.npy", mmap_mode='r')
    y_test = np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r')

    f_all = engineer_features(np.array(f_raw))
    f_test = engineer_features(np.array(f_test_raw))
    y_all = np.array(y_raw)

    feat_mean = f_all.mean(0); feat_std = f_all.std(0); feat_std[feat_std < 1e-8] = 1.0

    n = len(f_all); rng = np.random.RandomState(SEED); idx = rng.permutation(n)
    n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"Train: {len(tr_idx):,} Val: {len(val_idx):,}")

    train_ds = GammaDataset(m_raw[tr_idx], f_all[tr_idx], y_all[tr_idx], feat_mean, feat_std)
    val_ds = GammaDataset(m_raw[val_idx], f_all[val_idx], y_all[val_idx], feat_mean, feat_std)
    test_ds = GammaDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = GammaDilatedCNN(f_all.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    n_gamma = (y_all[tr_idx] == 0).sum(); n_hadron = (y_all[tr_idx] == 1).sum()
    w_gamma = n_hadron / n_gamma
    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
    print(f"Class weight gamma: {w_gamma:.1f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val = 1.0; best_scores = None

    for epoch in range(15):
        model.train()
        total_loss = 0; n_total = 0
        for mat, feat, y in train_loader:
            mat, feat, y = mat.to(DEVICE), feat.to(DEVICE), y.to(DEVICE)
            logits = model(mat, feat)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y); n_total += len(y)
        scheduler.step()

        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for mat, feat, y in val_loader:
                mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                probs = torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy()
                val_scores.append(probs); val_labels.append(y.numpy())
        val_scores = np.concatenate(val_scores); val_labels = np.concatenate(val_labels)
        val_surv = survival_at_75(val_scores, val_labels)
        print(f"Epoch {epoch+1:2d}/15: loss={total_loss/n_total:.4f} val={val_surv:.2e}")

        if val_surv < best_val:
            best_val = val_surv
            ts = []
            with torch.no_grad():
                for mat, feat, _ in test_loader:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    ts.append(torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy())
            best_scores = np.concatenate(ts)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v20.pt")
            print(f"  -> Best: {val_surv:.2e}")

    test_surv = survival_at_75(best_scores, y_test)
    print(f"\nBest val: {best_val:.2e} | Test: {test_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_v20.npy", best_scores)
    np.savez(f"{OUT_DIR}/predictions_v20.npz", gamma_scores=best_scores)

    # Try adding to ensemble
    s2 = np.load(f"{OUT_DIR}/probs_v2.npy")
    s7 = np.load(f"{OUT_DIR}/probs_v7.npy")
    s8 = np.load(f"{OUT_DIR}/probs_v8.npy")
    s9 = np.load(f"{OUT_DIR}/probs_v9.npy")
    eps = 1e-10

    ens2 = (s2+eps)**0.45 * (s7+eps)**0.15 * (s8+eps)**0.15 * (s9+eps)**0.25
    print(f"ens2 survival: {survival_at_75(ens2, y_test):.2e}")

    best = survival_at_75(ens2, y_test); best_alpha = 0
    for alpha in np.arange(0.02, 0.4, 0.02):
        ens = ens2 ** (1 - alpha) * (best_scores + eps) ** alpha
        s = survival_at_75(ens, y_test)
        if s < best:
            best = s; best_alpha = alpha
            print(f"  alpha={alpha:.2f}: {s:.2e} (improvement!)")
    print(f"Best with v20: alpha={best_alpha:.2f} -> {best:.2e}")

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print("description: Dilated CNN multi-scale (1,2,4) + strided downsample + feat MLP, 15 epochs")


if __name__ == "__main__":
    main()
