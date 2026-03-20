"""
v7: MLP with rich matrix statistics features + scalar features.
Strategy: Extract hand-crafted spatial statistics from matrices and use them
alongside the raw scalar features. This gives the MLP information about
the muon spatial pattern without needing a full CNN.

Key new features from matrices:
- Total muon sum (log1p)
- Muon max value
- Number of non-zero muon cells
- Muon sum / electron sum ratio
- Percentile statistics of muon distribution
- Spatial spread of muons (std of positions)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda:0")
BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def extract_matrix_stats(matrices):
    """Extract spatial statistics from (N, 16, 16, 2) matrices."""
    el = matrices[:, :, :, 0].astype(np.float32)  # (N, 16, 16)
    mu = matrices[:, :, :, 1].astype(np.float32)  # (N, 16, 16)

    n = len(el)
    stats = []

    # Log1p versions for sum statistics
    el_sum = el.reshape(n, -1).sum(1)
    mu_sum = mu.reshape(n, -1).sum(1)
    el_max = el.reshape(n, -1).max(1)
    mu_max = mu.reshape(n, -1).max(1)
    el_nnz = (el > 0).reshape(n, -1).sum(1).astype(np.float32)
    mu_nnz = (mu > 0).reshape(n, -1).sum(1).astype(np.float32)

    log_el_sum = np.log1p(el_sum)
    log_mu_sum = np.log1p(mu_sum)
    log_el_max = np.log1p(el_max)
    log_mu_max = np.log1p(mu_max)

    # Ratio
    mu_el_ratio = log_mu_sum / (log_el_sum + 1e-6)

    # Spatial spread of muons: compute center of mass and spread
    grid_y, grid_x = np.mgrid[0:16, 0:16]
    grid_y = grid_y.flatten().astype(np.float32)  # (256,)
    grid_x = grid_x.flatten().astype(np.float32)

    mu_flat = mu.reshape(n, -1) + 1e-10  # (N, 256)
    total_mu = mu_flat.sum(1, keepdims=True)

    # Center of mass
    cx = (mu_flat * grid_x).sum(1) / total_mu[:, 0]
    cy = (mu_flat * grid_y).sum(1) / total_mu[:, 0]

    # Spatial spread (variance)
    var_x = (mu_flat * (grid_x - cx[:, None]) ** 2).sum(1) / total_mu[:, 0]
    var_y = (mu_flat * (grid_y - cy[:, None]) ** 2).sum(1) / total_mu[:, 0]
    mu_spread = np.sqrt(var_x + var_y + 1e-8)

    # Same for electrons
    el_flat = el.reshape(n, -1) + 1e-10
    total_el = el_flat.sum(1, keepdims=True)
    ecx = (el_flat * grid_x).sum(1) / total_el[:, 0]
    ecy = (el_flat * grid_y).sum(1) / total_el[:, 0]
    evar_x = (el_flat * (grid_x - ecx[:, None]) ** 2).sum(1) / total_el[:, 0]
    evar_y = (el_flat * (grid_y - ecy[:, None]) ** 2).sum(1) / total_el[:, 0]
    el_spread = np.sqrt(evar_x + evar_y + 1e-8)

    # Muon fraction in top ring vs center (concentrated vs spread)
    # Center region (rows/cols 5-10)
    mu_center = mu[:, 5:11, 5:11].reshape(n, -1).sum(1)
    mu_outer = mu_sum - mu_center
    mu_center_frac = mu_center / (mu_sum + 1e-6)

    feats = np.stack([
        log_el_sum, log_mu_sum,
        log_el_max, log_mu_max,
        el_nnz / 256.0, mu_nnz / 256.0,
        mu_el_ratio,
        mu_spread, el_spread,
        mu_spread / (el_spread + 1e-6),
        cx, cy, ecx, ecy,  # center of mass positions
        mu_center_frac,
        np.log1p(mu_nnz),
    ], axis=1).astype(np.float32)

    return feats


def engineer_scalar_features(f):
    E = f[:, 0]; Ze = f[:, 1]; Az = f[:, 2]; Ne = f[:, 3]; Nmu = f[:, 4]
    ne_nmu_diff = Ne - Nmu
    Ze_norm = Ze / 30.0
    Ne_norm = (Ne - 5.0) / 0.7
    Nmu_norm = (Nmu - 3.5) / 0.7
    E_norm = (E - 16.0) / 1.0
    Az_rad = np.radians(Az)
    Az_cos = np.cos(Az_rad); Az_sin = np.sin(Az_rad)
    cos_ze = np.cos(np.radians(Ze))
    ne_e_ratio = Ne - E
    feats = np.stack([
        E_norm, Ze_norm, Az_cos, Az_sin, Ne_norm, Nmu_norm,
        ne_nmu_diff, cos_ze, ne_e_ratio, Ne * Ze_norm, Nmu * cos_ze,
    ], axis=1).astype(np.float32)
    return feats


class RichMLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
    def forward(self, x): return self.net(x)


def survival_at_75(scores, labels):
    is_gamma = labels == 0; is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])[::-1]
    ng = len(sg); idx = int(np.ceil(0.75 * ng)) - 1
    thr = sg[min(idx, ng - 1)]
    return float((scores[is_hadron] >= thr).sum() / is_hadron.sum())


def main():
    print("Loading data...")
    f_train_raw = np.load(f"{BASE}/data/gamma_train/features.npy", mmap_mode='r')
    m_train_raw = np.load(f"{BASE}/data/gamma_train/matrices.npy", mmap_mode='r')
    y_train_raw = np.load(f"{BASE}/data/gamma_train/labels_gamma.npy", mmap_mode='r')
    f_test_raw = np.load(f"{BASE}/data/gamma_test/features.npy", mmap_mode='r')
    m_test_raw = np.load(f"{BASE}/data/gamma_test/matrices.npy", mmap_mode='r')
    y_test = np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r')

    print("Extracting matrix statistics (this takes a moment)...")
    # Process in chunks to avoid memory issues
    chunk = 50000
    n_train = len(f_train_raw)
    mat_stats_list = []
    for i in range(0, n_train, chunk):
        m_chunk = np.array(m_train_raw[i:i+chunk])
        mat_stats_list.append(extract_matrix_stats(m_chunk))
        if i % 200000 == 0:
            print(f"  {i:,}/{n_train:,}")
    mat_stats_train = np.concatenate(mat_stats_list)

    mat_stats_test = extract_matrix_stats(np.array(m_test_raw))

    f_scalar_train = engineer_scalar_features(np.array(f_train_raw))
    f_scalar_test = engineer_scalar_features(np.array(f_test_raw))

    # Combine
    X_train = np.concatenate([f_scalar_train, mat_stats_train], axis=1)
    X_test = np.concatenate([f_scalar_test, mat_stats_test], axis=1)
    y_train = np.array(y_train_raw)

    print(f"Feature dim: {X_train.shape[1]}")
    print(f"Train: {len(X_train):,} (gamma={int((y_train==0).sum()):,}, hadron={int((y_train==1).sum()):,})")

    # Normalize
    feat_mean = X_train.mean(0); feat_std = X_train.std(0); feat_std[feat_std < 1e-8] = 1.0
    X_train_n = (X_train - feat_mean) / feat_std
    X_test_n = (X_test - feat_mean) / feat_std

    # Train/val split (90/10)
    n = len(X_train); rng = np.random.RandomState(SEED); idx = rng.permutation(n)
    n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr = X_train_n[tr_idx]; y_tr = y_train[tr_idx]
    X_val = X_train_n[val_idx]; y_val = y_train[val_idx]

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4, pin_memory=True)

    model = RichMLP(X_train.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    n_gamma = (y_tr == 0).sum(); n_hadron = (y_tr == 1).sum()
    w_gamma = n_hadron / n_gamma
    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
    print(f"Class weight gamma: {w_gamma:.1f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test_n).to(DEVICE)

    best_val = 1.0; best_scores = None
    for epoch in range(30):
        model.train()
        total_loss = 0; n_total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y); n_total += len(y)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(model(X_val_t), dim=1)[:, 0].cpu().numpy()
            val_surv = survival_at_75(val_probs, y_val)
            test_probs = torch.softmax(model(X_test_t), dim=1)[:, 0].cpu().numpy()

        print(f"Epoch {epoch+1:2d}/30: loss={total_loss/n_total:.4f} val_surv={val_surv:.2e}")
        if val_surv < best_val:
            best_val = val_surv; best_scores = test_probs.copy()
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v7.pt")
            print(f"  -> Best: {val_surv:.2e}")

    test_surv = survival_at_75(best_scores, y_test)
    print(f"\nBest val survival@75: {best_val:.2e}")
    print(f"Test survival@75: {test_surv:.2e}")

    np.save(f"{OUT_DIR}/probs_v7.npy", best_scores)
    np.savez(f"{OUT_DIR}/predictions_v7.npz", gamma_scores=best_scores)
    np.savez(f"{OUT_DIR}/predictions.npz", gamma_scores=best_scores)

    print("\n---")
    print(f"metric: {best_val:.4e}")
    print("description: MLP with matrix statistics features (muon spread, center mass, etc) + scalar features")


if __name__ == "__main__":
    main()
