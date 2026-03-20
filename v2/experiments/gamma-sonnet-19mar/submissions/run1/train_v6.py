"""
v6: 3-seed ensemble of v2 architecture WITHOUT quality cuts.
v2 (seed 42) got 4.09e-04. Let's try 3 seeds and geometric mean.
Key lesson from v5: quality cuts on training hurt. Use all data.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda:0")
BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def engineer_features(f):
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


class GammaDataset(Dataset):
    def __init__(self, matrices, features, labels, feat_mean=None, feat_std=None):
        self.matrices = matrices; self.features = features
        self.labels = labels; self.feat_mean = feat_mean; self.feat_std = feat_std
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)
        mat = np.log1p(mat).transpose(2, 0, 1)
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


class GammaCNN(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.cnn = nn.Sequential(
            CNNBlock(2, 32), nn.MaxPool2d(2),
            CNNBlock(32, 64), nn.MaxPool2d(2),
            CNNBlock(64, 128), nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2),
        )
    def forward(self, mat, feat):
        return self.head(torch.cat([self.cnn(mat), self.feat_mlp(feat)], dim=1))


def survival_at_75(scores, labels):
    is_gamma = labels == 0; is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])[::-1]
    ng = len(sg)
    idx = int(np.ceil(0.75 * ng)) - 1
    thr = sg[min(idx, ng - 1)]
    return float((scores[is_hadron] >= thr).sum() / is_hadron.sum())


def train_seed(seed, f_tr, m_tr, y_tr, f_val, m_val, y_val, f_test, m_test, feat_mean, feat_std):
    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = GammaDataset(m_tr, f_tr, y_tr, feat_mean, feat_std)
    val_ds = GammaDataset(m_val, f_val, y_val, feat_mean, feat_std)
    test_ds = GammaDataset(m_test, f_test, np.zeros(len(f_test), dtype=np.int8), feat_mean, feat_std)
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = GammaCNN(f_tr.shape[1]).to(DEVICE)
    n_gamma = (y_tr == 0).sum(); n_hadron = (y_tr == 1).sum()
    w_gamma = n_hadron / n_gamma
    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val = 1.0; best_test_scores = None
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
        print(f"  S{seed} E{epoch+1}/15: loss={total_loss/n_total:.4f} val_surv={val_surv:.2e}")
        if val_surv < best_val:
            best_val = val_surv
            ts = []
            with torch.no_grad():
                for mat, feat, _ in test_loader:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    ts.append(torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy())
            best_test_scores = np.concatenate(ts)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v6_s{seed}.pt")
            print(f"    -> Best: {val_surv:.2e}")
    return best_test_scores


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

    n = len(f_all); rng = np.random.RandomState(42); idx = rng.permutation(n)
    n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]
    f_tr = f_all[tr_idx]; m_tr = m_raw[tr_idx]; y_tr = y_all[tr_idx]
    f_val = f_all[val_idx]; m_val = m_raw[val_idx]; y_val = y_all[val_idx]
    print(f"Train: {len(tr_idx):,} Val: {len(val_idx):,}")

    all_scores = []
    for seed in [42, 123, 777]:
        print(f"\nSeed {seed}...")
        scores = train_seed(seed, f_tr, m_tr, y_tr, f_val, m_val, y_val,
                            f_test, np.array(m_test_raw), feat_mean, feat_std)
        all_scores.append(scores)
        np.save(f"{OUT_DIR}/probs_v6_s{seed}.npy", scores)
        print(f"  Test survival@75: {survival_at_75(scores, y_test):.2e}")

    eps = 1e-10
    ens = np.exp(np.mean([np.log(s + eps) for s in all_scores], axis=0))
    print(f"\nEnsemble: {survival_at_75(ens, y_test):.2e}")

    np.save(f"{OUT_DIR}/probs_v6.npy", ens)
    np.savez(f"{OUT_DIR}/predictions_v6.npz", gamma_scores=ens)
    np.savez(f"{OUT_DIR}/predictions.npz", gamma_scores=ens)
    final_surv = survival_at_75(ens, y_test)

    print("\n---")
    print(f"metric: {final_surv:.4e}")
    print("description: 3-seed ensemble of v2 CNN+MLP, no quality cuts, geometric mean")


if __name__ == "__main__":
    main()
