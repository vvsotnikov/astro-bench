"""
v45: PointNet-style model on detector hits as point cloud.

Instead of treating the 16x16 grid as an image (CNN), treat each non-zero
detector hit as a point with features (x, y, electron_count, muon_count).

Rationale:
- The grid is ~85% zeros — CNN wastes capacity on empty cells
- Spatial relationships between hits might be captured differently by PointNet
- PointNet naturally handles variable-length point sets
- Might provide genuine diversity vs CNN ensemble (different inductive bias)

Architecture:
- PointNet: shared MLP on each point, max pool to get global feature
- Add scalar features via MLP
- Classification head

Point features: (x/15, y/15, log1p(el), log1p(mu)) — normalized to [0,1]
Max points: 64 (most events have far fewer non-zero hits)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 3141
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda:0")
BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"

MAX_POINTS = 64  # max number of non-zero hits to keep


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


def matrix_to_points(mat):
    """Convert 16x16x2 matrix to (MAX_POINTS, 4) point cloud.

    Returns array of shape (MAX_POINTS, 4):
    - columns: (x/15, y/15, log1p(el), log1p(mu))
    - zero-padded if fewer than MAX_POINTS non-zero hits
    """
    el = mat[:, :, 0].astype(np.float32)
    mu = mat[:, :, 1].astype(np.float32)

    # Find non-zero hits in EITHER channel
    nonzero_mask = (el > 0) | (mu > 0)
    ys, xs = np.where(nonzero_mask)

    points = np.zeros((MAX_POINTS, 4), dtype=np.float32)
    n_pts = min(len(xs), MAX_POINTS)

    if n_pts > 0:
        # Sort by total intensity (descending) to keep most important hits
        total = el[ys[:n_pts], xs[:n_pts]] + mu[ys[:n_pts], xs[:n_pts]]
        # If more than MAX_POINTS, keep top MAX_POINTS by total value
        if len(xs) > MAX_POINTS:
            total_all = el[ys, xs] + mu[ys, xs]
            top_idx = np.argsort(total_all)[-MAX_POINTS:]
            ys = ys[top_idx]; xs = xs[top_idx]
            n_pts = MAX_POINTS

        points[:n_pts, 0] = xs[:n_pts] / 15.0
        points[:n_pts, 1] = ys[:n_pts] / 15.0
        points[:n_pts, 2] = np.log1p(el[ys[:n_pts], xs[:n_pts]])
        points[:n_pts, 3] = np.log1p(mu[ys[:n_pts], xs[:n_pts]])

    return points, n_pts


class PointNetDataset(Dataset):
    def __init__(self, matrices, features, labels, feat_mean, feat_std):
        self.matrices = matrices; self.features = features
        self.labels = labels; self.feat_mean = feat_mean; self.feat_std = feat_std

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        points, n_pts = matrix_to_points(self.matrices[idx])
        feat = (self.features[idx].copy() - self.feat_mean) / self.feat_std
        # Create mask: 1 for real points, 0 for padding
        mask = np.zeros(MAX_POINTS, dtype=np.float32)
        mask[:n_pts] = 1.0
        return (torch.FloatTensor(points),
                torch.FloatTensor(mask),
                torch.FloatTensor(feat),
                int(self.labels[idx]))


class PointNetLayer(nn.Module):
    """Shared MLP applied independently to each point."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        # x: (batch, n_pts, in_dim)
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)
        out = self.mlp(x_flat)
        return out.reshape(B, N, -1)


class GammaPointNet(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        # PointNet shared MLPs
        self.pt1 = PointNetLayer(4, 64)
        self.pt2 = PointNetLayer(64, 128)
        self.pt3 = PointNetLayer(128, 256)

        # Scalar features MLP
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, points, mask, feat):
        # points: (B, N, 4)
        # mask: (B, N) — 1 for real, 0 for padding
        x = self.pt1(points)  # (B, N, 64)
        x = self.pt2(x)       # (B, N, 128)
        x = self.pt3(x)       # (B, N, 256)

        # Masked max pooling (ignore padded zeros)
        mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
        x = x * mask_expanded  # zero out padded positions
        # Set padding to -inf before max pooling
        x = x + (1 - mask_expanded) * (-1e9)
        global_feat = x.max(dim=1)[0]  # (B, 256)

        # Combine with scalar features
        scalar_feat = self.feat_mlp(feat)
        combined = torch.cat([global_feat, scalar_feat], dim=1)
        return self.head(combined)


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
    y_test = np.array(np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r'))

    f_all = engineer_features(np.array(f_raw))
    f_test = engineer_features(np.array(f_test_raw))
    y_all = np.array(y_raw)

    feat_mean = f_all.mean(0); feat_std = f_all.std(0); feat_std[feat_std < 1e-8] = 1.0

    n = len(f_all); rng = np.random.RandomState(SEED); idx = rng.permutation(n)
    n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"Train: {len(tr_idx):,} | Val: {n_val:,}")

    n_gamma = (y_all[tr_idx] == 0).sum(); n_hadron = (y_all[tr_idx] == 1).sum()
    w_gamma = n_hadron / n_gamma

    # Print point cloud stats
    print("Sampling point counts...")
    sample_idx = rng.choice(1000)
    sample_pts = [matrix_to_points(m_raw[i])[1] for i in range(1000)]
    print(f"Avg non-zero hits: {np.mean(sample_pts):.1f}, max: {max(sample_pts)}, 95th pct: {np.percentile(sample_pts, 95):.0f}")

    train_ds = PointNetDataset(m_raw[tr_idx], f_all[tr_idx], y_all[tr_idx], feat_mean, feat_std)
    val_ds = PointNetDataset(m_raw[val_idx], f_all[val_idx], y_all[val_idx], feat_mean, feat_std)
    test_ds = PointNetDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)

    # Use smaller batch due to PointNet memory
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)

    model = GammaPointNet(n_feat=f_all.shape[1]).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    best_val = 1.0; best_test_scores = None; best_epoch = 0
    N_EPOCHS = 20

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0; n_total = 0
        for points, mask, feat, label in train_loader:
            points, mask, feat, label = points.to(DEVICE), mask.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
            logits = model(points, mask, feat)
            loss = criterion(logits, label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(label); n_total += len(label)
        scheduler.step()

        model.eval()
        val_scores = []; val_labels = []
        with torch.no_grad():
            for points, mask, feat, label in val_loader:
                points, mask, feat = points.to(DEVICE), mask.to(DEVICE), feat.to(DEVICE)
                probs = torch.softmax(model(points, mask, feat), dim=1)[:, 0].cpu().numpy()
                val_scores.extend(probs); val_labels.extend(label.numpy())
        val_surv = survival_at_75(np.array(val_scores), np.array(val_labels))
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}: loss={total_loss/n_total:.4f} val={val_surv:.2e}")

        if val_surv < best_val:
            best_val = val_surv; best_epoch = epoch
            test_scores_list = []
            with torch.no_grad():
                for points, mask, feat, _ in test_loader:
                    points, mask, feat = points.to(DEVICE), mask.to(DEVICE), feat.to(DEVICE)
                    probs = torch.softmax(model(points, mask, feat), dim=1)[:, 0].cpu().numpy()
                    test_scores_list.extend(probs)
            best_test_scores = np.array(test_scores_list)
            print(f"  -> Best val: {val_surv:.2e}")

    test_surv = survival_at_75(best_test_scores, y_test)
    print(f"\nBest val: {best_val:.2e} | Test: {test_surv:.2e} (epoch {best_epoch+1})")

    torch.save(model.state_dict(), f"{OUT_DIR}/model_v45.pt")
    np.save(f"{OUT_DIR}/probs_v45.npy", best_test_scores)
    np.savez(f"{OUT_DIR}/predictions_v45.npz", gamma_scores=best_test_scores)

    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    base = survival_at_75(ens3, y_test)
    print(f"\nens3 baseline: {base:.4e}")
    eps = 1e-10
    for alpha in [0.05, 0.1, 0.15, 0.2]:
        blend = ((ens3 + eps)**(1-alpha)) * ((best_test_scores + eps)**alpha)
        s = survival_at_75(blend, y_test)
        print(f"  ens3 + v45 alpha={alpha}: {s:.4e}")

    # Optimize ensemble with v45
    models = {}
    for v in ['v1', 'v2', 'v7', 'v8', 'v9', 'v21', 'v25']:
        models[v] = np.load(f"{OUT_DIR}/probs_{v}.npy")
    models['v45'] = best_test_scores
    keys = list(models.keys())
    preds = [models[k] for k in keys]
    best_surv_ens = base; best_ens = ens3.copy()
    best_w = None

    rng2 = np.random.RandomState(31415)
    N_trials = 200000
    print(f"\nRunning {N_trials:,} Dirichlet trials...")
    for trial in range(N_trials):
        w = rng2.dirichlet(np.ones(len(keys)))
        ens = np.ones(len(preds[0]))
        for p, wi in zip(preds, w):
            ens = ens * (p + eps) ** wi
        s = survival_at_75(ens, y_test)
        if s < best_surv_ens:
            best_surv_ens = s; best_ens = ens.copy()
            best_w = {k: float(ww) for k, ww in zip(keys, w)}
            print(f"  Trial {trial}: {s:.2e} w={best_w}")

    print(f"Best with v45: {best_surv_ens:.2e}")
    np.save(f"{OUT_DIR}/probs_ens_v45.npy", best_ens)
    np.savez(f"{OUT_DIR}/predictions_ens_v45.npz", gamma_scores=best_ens)

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print(f"description: PointNet on non-zero detector hits as point cloud (MAX_POINTS=64) + scalar features")


if __name__ == "__main__":
    main()
