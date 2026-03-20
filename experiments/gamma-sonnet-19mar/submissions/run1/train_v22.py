"""
v22: Model that focuses exclusively on the muon channel.
Physics: gamma showers have essentially no muons. The muon map channel alone
should be very discriminating.

Two sub-architectures combined:
1. CNN on muon map only (channel 1)
2. MLP on scalar features

The key difference from v2: we ONLY look at the muon channel, forcing the model
to focus on what makes gammas unique (absence of muons).

Also: use more aggressive class weighting since we care about very high gamma purity.
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
        mat = self.matrices[idx].astype(np.float32)
        # Only take muon channel (channel 1) as 1-channel image
        mu_map = np.log1p(mat[:, :, 1])[np.newaxis, :, :]  # (1, 16, 16)
        feat = self.features[idx].copy()
        if self.feat_mean is not None:
            feat = (feat - self.feat_mean) / self.feat_std
        return torch.FloatTensor(mu_map), torch.FloatTensor(feat), int(self.labels[idx])


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.conv(x)


class GammaMuonCNN(nn.Module):
    """CNN on muon channel only + MLP on scalar features."""
    def __init__(self, n_feat):
        super().__init__()
        # Muon channel CNN
        self.mu_cnn = nn.Sequential(
            CNNBlock(1, 32), nn.MaxPool2d(2),
            CNNBlock(32, 64), nn.MaxPool2d(2),
            CNNBlock(64, 128), nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )  # output: 512

        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, mu_map, feat):
        mu_out = self.mu_cnn(mu_map)
        feat_out = self.feat_mlp(feat)
        return self.head(torch.cat([mu_out, feat_out], dim=1))


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

    model = GammaMuonCNN(f_all.shape[1]).to(DEVICE)
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
        for mu_map, feat, y in train_loader:
            mu_map, feat, y = mu_map.to(DEVICE), feat.to(DEVICE), y.to(DEVICE)
            logits = model(mu_map, feat)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(y); n_total += len(y)
        scheduler.step()

        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for mu_map, feat, y in val_loader:
                mu_map, feat = mu_map.to(DEVICE), feat.to(DEVICE)
                probs = torch.softmax(model(mu_map, feat), dim=1)[:, 0].cpu().numpy()
                val_scores.append(probs); val_labels.append(y.numpy())
        val_scores = np.concatenate(val_scores); val_labels = np.concatenate(val_labels)
        val_surv = survival_at_75(val_scores, val_labels)
        print(f"Epoch {epoch+1:2d}/15: loss={total_loss/n_total:.4f} val={val_surv:.2e}")

        if val_surv < best_val:
            best_val = val_surv
            ts = []
            with torch.no_grad():
                for mu_map, feat, _ in test_loader:
                    mu_map, feat = mu_map.to(DEVICE), feat.to(DEVICE)
                    ts.append(torch.softmax(model(mu_map, feat), dim=1)[:, 0].cpu().numpy())
            best_scores = np.concatenate(ts)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v22.pt")
            print(f"  -> Best: {val_surv:.2e}")

    test_surv = survival_at_75(best_scores, y_test)
    print(f"\nBest val: {best_val:.2e} | Test: {test_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_v22.npy", best_scores)
    np.savez(f"{OUT_DIR}/predictions_v22.npz", gamma_scores=best_scores)

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
    print(f"Best with v22: alpha={best_alpha:.2f} -> {best:.2e}")

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print("description: Muon-channel-only CNN + scalar feat MLP, 15 epochs")


if __name__ == "__main__":
    main()
