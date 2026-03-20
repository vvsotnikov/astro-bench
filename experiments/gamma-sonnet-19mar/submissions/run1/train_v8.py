"""
v8: Improved CNN+MLP with:
1. Better feature engineering: add matrix statistics to scalar features
2. More careful training: patience-based early stopping
3. Warm restart cosine schedule
4. Both models (v2 + v8) will be ensembled

Key change: integrate matrix statistics into the feature MLP branch,
giving it explicit access to muon spatial info alongside CNN features.
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


def extract_matrix_stats(matrices):
    """Fast statistics from matrices. matrices: (N, 16, 16, 2)"""
    el = matrices[:, :, :, 0]
    mu = matrices[:, :, :, 1]
    n = len(el)

    el_flat = el.reshape(n, -1).astype(np.float32)
    mu_flat = mu.reshape(n, -1).astype(np.float32)

    log_el_sum = np.log1p(el_flat.sum(1))
    log_mu_sum = np.log1p(mu_flat.sum(1))
    log_el_max = np.log1p(el_flat.max(1))
    log_mu_max = np.log1p(mu_flat.max(1))
    el_nnz = (el_flat > 0).sum(1).astype(np.float32)
    mu_nnz = (mu_flat > 0).sum(1).astype(np.float32)

    # Grid positions for spatial stats
    grid_y = np.repeat(np.arange(16), 16).astype(np.float32)
    grid_x = np.tile(np.arange(16), 16).astype(np.float32)

    mu_w = mu_flat + 1e-8
    total_mu = mu_w.sum(1)
    cx = (mu_w * grid_x).sum(1) / total_mu
    cy = (mu_w * grid_y).sum(1) / total_mu
    var_x = (mu_w * (grid_x - cx[:, None]) ** 2).sum(1) / total_mu
    var_y = (mu_w * (grid_y - cy[:, None]) ** 2).sum(1) / total_mu
    mu_spread = np.sqrt(var_x + var_y + 1e-8)

    return np.stack([
        log_el_sum, log_mu_sum, log_el_max, log_mu_max,
        el_nnz / 256.0, mu_nnz / 256.0,
        log_mu_sum / (log_el_sum + 1e-6),
        mu_spread,
    ], axis=1).astype(np.float32)


def engineer_features(f, m_stats=None):
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

    scalars = np.stack([
        E_norm, Ze_norm, Az_cos, Az_sin, Ne_norm, Nmu_norm,
        ne_nmu_diff, cos_ze, ne_e_ratio, Ne * Ze_norm, Nmu * cos_ze,
    ], axis=1).astype(np.float32)

    if m_stats is not None:
        return np.concatenate([scalars, m_stats], axis=1)
    return scalars


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


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.conv(x)


class GammaCNNv8(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        # Larger CNN than v2
        self.cnn = nn.Sequential(
            CNNBlock(2, 48), nn.MaxPool2d(2),       # 8x8
            CNNBlock(48, 96), nn.MaxPool2d(2),      # 4x4
            CNNBlock(96, 192), nn.AdaptiveAvgPool2d(2), nn.Flatten(),  # 768
        )
        cnn_out = 192 * 4  # 768

        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(cnn_out + 128, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, mat, feat):
        return self.head(torch.cat([self.cnn(mat), self.feat_mlp(feat)], dim=1))


def survival_at_75(scores, labels):
    is_gamma = labels == 0; is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])[::-1]; ng = len(sg)
    thr = sg[min(int(np.ceil(0.75 * ng)) - 1, ng - 1)]
    return float((scores[is_hadron] >= thr).sum() / is_hadron.sum())


def main():
    print("Loading data...")
    f_raw = np.load(f"{BASE}/data/gamma_train/features.npy", mmap_mode='r')
    m_raw = np.load(f"{BASE}/data/gamma_train/matrices.npy", mmap_mode='r')
    y_raw = np.load(f"{BASE}/data/gamma_train/labels_gamma.npy", mmap_mode='r')
    f_test_raw = np.load(f"{BASE}/data/gamma_test/features.npy", mmap_mode='r')
    m_test_raw = np.load(f"{BASE}/data/gamma_test/matrices.npy", mmap_mode='r')
    y_test = np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r')

    print("Extracting matrix stats...")
    m_all = np.array(m_raw)
    mat_stats_train = extract_matrix_stats(m_all)
    mat_stats_test = extract_matrix_stats(np.array(m_test_raw))

    f_all_feats = engineer_features(np.array(f_raw), mat_stats_train)
    f_test_feats = engineer_features(np.array(f_test_raw), mat_stats_test)
    y_all = np.array(y_raw)

    feat_mean = f_all_feats.mean(0); feat_std = f_all_feats.std(0)
    feat_std[feat_std < 1e-8] = 1.0

    n = len(f_all_feats); rng = np.random.RandomState(SEED); idx = rng.permutation(n)
    n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]

    print(f"Train: {len(tr_idx):,} Val: {len(val_idx):,} Features: {f_all_feats.shape[1]}")

    train_ds = GammaDataset(m_all[tr_idx], f_all_feats[tr_idx], y_all[tr_idx], feat_mean, feat_std)
    val_ds = GammaDataset(m_all[val_idx], f_all_feats[val_idx], y_all[val_idx], feat_mean, feat_std)
    test_ds = GammaDataset(np.array(m_test_raw), f_test_feats, y_test, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = GammaCNNv8(f_all_feats.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    n_gamma = (y_all[tr_idx] == 0).sum(); n_hadron = (y_all[tr_idx] == 1).sum()
    w_gamma = n_hadron / n_gamma
    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
    print(f"Class weight gamma: {w_gamma:.1f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val = 1.0; best_scores = None; patience = 0; max_patience = 8

    for epoch in range(40):
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

        print(f"Epoch {epoch+1:2d}/40: loss={total_loss/n_total:.4f} val={val_surv:.2e}")
        if val_surv < best_val:
            best_val = val_surv; patience = 0
            ts = []
            with torch.no_grad():
                for mat, feat, _ in test_loader:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    ts.append(torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy())
            best_scores = np.concatenate(ts)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v8.pt")
            print(f"  -> Best: {val_surv:.2e}")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    test_surv = survival_at_75(best_scores, y_test)
    print(f"\nBest val survival: {best_val:.2e} | Test survival: {test_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_v8.npy", best_scores)
    np.savez(f"{OUT_DIR}/predictions_v8.npz", gamma_scores=best_scores)
    np.savez(f"{OUT_DIR}/predictions.npz", gamma_scores=best_scores)

    print("\n---")
    print(f"metric: {best_val:.4e}")
    print("description: Larger CNN (48-96-192 channels) + matrix stats in features, patience=8")


if __name__ == "__main__":
    main()
