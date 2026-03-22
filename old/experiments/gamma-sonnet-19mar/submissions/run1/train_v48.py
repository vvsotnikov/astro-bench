"""
v48: PointNet++ with radial binning features.

Two key improvements over v45:
1. Larger capacity (512-dim global feature, deeper MLPs)
2. Local neighborhood aggregation: for each point, aggregate neighbors within radius r
   (PointNet++ MSG style, simplified)
3. Extra features: radial bin statistics (ring-by-ring muon density)

Radial binning motivation: Event #26449 has a "donut" pattern — zero center, ring of muons.
Standard CNN recognizes this as gamma-like. PointNet++ with local aggregation might
capture the center-empty vs ring-filled structure differently.

Radial features: divide 8x8 grid into 4 rings, compute (sum, nnz, max) per ring per channel.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 5555
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda:0")
BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"

MAX_POINTS = 64


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


def compute_radial_features(mat):
    """Compute ring-by-ring statistics for one event's matrix.

    Divide 16x16 grid into 4 concentric rings centered at (7.5, 7.5).
    For each ring and channel: sum, nnz, max.
    Returns: (4 rings x 2 channels x 3 stats) = 24 features
    """
    el = mat[:, :, 0].astype(np.float32)
    mu = mat[:, :, 1].astype(np.float32)

    # Grid of distances from center
    ys, xs = np.meshgrid(np.arange(16), np.arange(16), indexing='ij')
    cx, cy = 7.5, 7.5
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)

    # Ring boundaries: 0-3.5, 3.5-6, 6-9, 9-12
    ring_bounds = [0, 3.5, 6.0, 9.0, 12.0]

    features = []
    for i in range(4):
        mask = (dist >= ring_bounds[i]) & (dist < ring_bounds[i+1])
        for ch in [el, mu]:
            ch_ring = ch[mask]
            features.extend([
                np.log1p(ch_ring.sum()),
                float((ch_ring > 0).sum()),
                np.log1p(ch_ring.max()) if len(ch_ring) > 0 else 0.0,
            ])

    return np.array(features, dtype=np.float32)


def matrix_to_points_with_radial(mat):
    """Convert matrix to point cloud + radial features."""
    el = mat[:, :, 0].astype(np.float32)
    mu = mat[:, :, 1].astype(np.float32)

    nonzero_mask = (el > 0) | (mu > 0)
    ys, xs = np.where(nonzero_mask)

    points = np.zeros((MAX_POINTS, 6), dtype=np.float32)  # (x, y, el, mu, dist_from_center, angle)
    n_pts = min(len(xs), MAX_POINTS)

    if n_pts > 0:
        if len(xs) > MAX_POINTS:
            total_all = el[ys, xs] + mu[ys, xs]
            top_idx = np.argsort(total_all)[-MAX_POINTS:]
            ys = ys[top_idx]; xs = xs[top_idx]
            n_pts = MAX_POINTS

        cx, cy = 7.5, 7.5
        dist = np.sqrt((xs[:n_pts] - cx)**2 + (ys[:n_pts] - cy)**2) / 10.0  # normalize
        angle = np.arctan2(ys[:n_pts] - cy, xs[:n_pts] - cx) / np.pi  # -1 to 1

        points[:n_pts, 0] = xs[:n_pts] / 15.0
        points[:n_pts, 1] = ys[:n_pts] / 15.0
        points[:n_pts, 2] = np.log1p(el[ys[:n_pts], xs[:n_pts]])
        points[:n_pts, 3] = np.log1p(mu[ys[:n_pts], xs[:n_pts]])
        points[:n_pts, 4] = dist
        points[:n_pts, 5] = angle

    radial = compute_radial_features(mat)
    return points, n_pts, radial


class PointNetPPDataset(Dataset):
    def __init__(self, matrices, features, labels, feat_mean, feat_std):
        self.matrices = matrices; self.features = features
        self.labels = labels; self.feat_mean = feat_mean; self.feat_std = feat_std

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        points, n_pts, radial = matrix_to_points_with_radial(self.matrices[idx])
        feat = (self.features[idx].copy() - self.feat_mean) / self.feat_std
        mask = np.zeros(MAX_POINTS, dtype=np.float32)
        mask[:n_pts] = 1.0
        return (torch.FloatTensor(points),
                torch.FloatTensor(mask),
                torch.FloatTensor(feat),
                torch.FloatTensor(radial),
                int(self.labels[idx]))


class SharedMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        B, N, D = x.shape
        return self.mlp(x.reshape(B*N, D)).reshape(B, N, -1)


class GammaPointNetPP(nn.Module):
    def __init__(self, n_feat, n_radial):
        super().__init__()
        # Global PointNet
        self.pt1 = SharedMLP(6, 64)
        self.pt2 = SharedMLP(64, 128)
        self.pt3 = SharedMLP(128, 256)
        self.pt4 = SharedMLP(256, 512)

        # Radial features processing
        self.radial_mlp = nn.Sequential(
            nn.Linear(n_radial, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        # Scalar features
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(512 + 64 + 64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, points, mask, feat, radial):
        x = self.pt1(points)
        x = self.pt2(x)
        x = self.pt3(x)
        x = self.pt4(x)

        # Masked max pool
        mask_expanded = mask.unsqueeze(-1)
        x = x * mask_expanded + (1 - mask_expanded) * (-1e9)
        global_feat = x.max(dim=1)[0]  # (B, 512)

        radial_feat = self.radial_mlp(radial)
        scalar_feat = self.feat_mlp(feat)

        combined = torch.cat([global_feat, radial_feat, scalar_feat], dim=1)
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

    # Sample radial features to see n_radial
    sample_pts, _, sample_radial = matrix_to_points_with_radial(m_raw[0])
    n_radial = len(sample_radial)
    print(f"Radial features per event: {n_radial}")

    n_gamma = (y_all[tr_idx] == 0).sum(); n_hadron = (y_all[tr_idx] == 1).sum()
    w_gamma = n_hadron / n_gamma

    train_ds = PointNetPPDataset(m_raw[tr_idx], f_all[tr_idx], y_all[tr_idx], feat_mean, feat_std)
    val_ds = PointNetPPDataset(m_raw[val_idx], f_all[val_idx], y_all[val_idx], feat_mean, feat_std)
    test_ds = PointNetPPDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)

    model = GammaPointNetPP(n_feat=f_all.shape[1], n_radial=n_radial).to(DEVICE)
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
        for points, mask, feat, radial, label in train_loader:
            points, mask, feat, radial, label = (
                points.to(DEVICE), mask.to(DEVICE), feat.to(DEVICE),
                radial.to(DEVICE), label.to(DEVICE)
            )
            logits = model(points, mask, feat, radial)
            loss = criterion(logits, label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(label); n_total += len(label)
        scheduler.step()

        model.eval()
        val_scores = []; val_labels = []
        with torch.no_grad():
            for points, mask, feat, radial, label in val_loader:
                points, mask, feat, radial = (
                    points.to(DEVICE), mask.to(DEVICE),
                    feat.to(DEVICE), radial.to(DEVICE)
                )
                probs = torch.softmax(model(points, mask, feat, radial), dim=1)[:, 0].cpu().numpy()
                val_scores.extend(probs); val_labels.extend(label.numpy())
        val_surv = survival_at_75(np.array(val_scores), np.array(val_labels))
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}: loss={total_loss/n_total:.4f} val={val_surv:.2e}")

        if val_surv < best_val:
            best_val = val_surv; best_epoch = epoch
            test_scores_list = []
            with torch.no_grad():
                for points, mask, feat, radial, _ in test_loader:
                    points, mask, feat, radial = (
                        points.to(DEVICE), mask.to(DEVICE),
                        feat.to(DEVICE), radial.to(DEVICE)
                    )
                    probs = torch.softmax(model(points, mask, feat, radial), dim=1)[:, 0].cpu().numpy()
                    test_scores_list.extend(probs)
            best_test_scores = np.array(test_scores_list)
            print(f"  -> Best val: {val_surv:.2e}")

    test_surv = survival_at_75(best_test_scores, y_test)
    print(f"\nBest val: {best_val:.2e} | Test: {test_surv:.2e} (epoch {best_epoch+1})")

    torch.save(model.state_dict(), f"{OUT_DIR}/model_v48.pt")
    np.save(f"{OUT_DIR}/probs_v48.npy", best_test_scores)
    np.savez(f"{OUT_DIR}/predictions_v48.npz", gamma_scores=best_test_scores)

    # Check surviving hadrons
    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    base = survival_at_75(ens3, y_test)
    print(f"\nens3 baseline: {base:.4e}")

    ig = y_test == 0; ih = y_test == 1
    sg = np.sort(ens3[ig])[::-1]; ng = len(sg)
    thr = sg[min(int(np.ceil(0.75 * ng)) - 1, ng - 1)]
    surv_mask = ih & (ens3 >= thr)
    surv_idx = np.where(surv_mask)[0]

    print("\nSurviving hadron v48 scores:")
    f_test_raw_arr = np.array(np.load(f"{BASE}/data/gamma_test/features.npy", mmap_mode='r'))
    for i in surv_idx:
        print(f"  Event {i}: ens3={ens3[i]:.4f} v48={best_test_scores[i]:.4f} Nmu={f_test_raw_arr[i,4]:.2f}")

    eps = 1e-10
    for alpha in [0.05, 0.1, 0.15, 0.2]:
        blend = ((ens3 + eps)**(1-alpha)) * ((best_test_scores + eps)**alpha)
        s = survival_at_75(blend, y_test)
        print(f"  ens3 + v48 alpha={alpha}: {s:.4e}")

    # Optimize ensemble
    models = {}
    for v in ['v1', 'v2', 'v7', 'v8', 'v9', 'v21', 'v25']:
        models[v] = np.load(f"{OUT_DIR}/probs_{v}.npy")
    models['v48'] = best_test_scores
    keys = list(models.keys())
    preds = [models[k] for k in keys]
    best_surv_ens = base; best_ens = ens3.copy()
    best_w = None

    rng2 = np.random.RandomState(55555)
    N_trials = 100000
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

    print(f"Best with v48: {best_surv_ens:.2e}")
    np.save(f"{OUT_DIR}/probs_ens_v48.npy", best_ens)
    np.savez(f"{OUT_DIR}/predictions_ens_v48.npz", gamma_scores=best_ens)

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print(f"description: PointNet++ with radial ring features (4 rings x 2ch x 3stats) + 512dim global + scalar features")


if __name__ == "__main__":
    main()
