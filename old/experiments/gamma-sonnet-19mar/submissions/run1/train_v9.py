"""
v9: Cross-channel attention CNN.
Key idea: correlate electron and muon spatial patterns.
In gamma showers, electrons and muons should be spatially decorrelated.
In hadron showers, they come from the same hadronic cascade.

Architecture:
- Process electron and muon channels separately with individual CNNs
- Apply cross-channel attention: muon channel attending to electron channel
- Fuse with scalar features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    Ze_norm = Ze / 30.0
    Ne_norm = (Ne - 5.0) / 0.7
    Nmu_norm = (Nmu - 3.5) / 0.7
    E_norm = (E - 16.0) / 1.0
    Az_rad = np.radians(Az)
    Az_cos = np.cos(Az_rad); Az_sin = np.sin(Az_rad)
    cos_ze = np.cos(np.radians(Ze))
    ne_e_ratio = Ne - E
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


class ChannelCNN(nn.Module):
    """Process a single channel with a CNN backbone."""
    def __init__(self, out_channels=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(32, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
        )
    def forward(self, x): return self.layers(x)


class CrossChannelAttention(nn.Module):
    """Muon channel attends to electron channel positions."""
    def __init__(self, ch):
        super().__init__()
        self.proj_q = nn.Conv2d(ch, ch // 4, 1)  # muon queries
        self.proj_k = nn.Conv2d(ch, ch // 4, 1)  # electron keys
        self.proj_v = nn.Conv2d(ch, ch, 1)        # electron values

    def forward(self, mu_feat, el_feat):
        """mu_feat, el_feat: (B, C, H, W)"""
        B, C, H, W = mu_feat.shape
        q = self.proj_q(mu_feat).flatten(2)   # (B, C//4, H*W)
        k = self.proj_k(el_feat).flatten(2)   # (B, C//4, H*W)
        v = self.proj_v(el_feat).flatten(2)   # (B, C, H*W)

        scale = (C // 4) ** -0.5
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * scale, dim=-1)  # (B, HW, HW)
        out = torch.bmm(v, attn.transpose(1, 2))  # (B, C, HW)
        return out.reshape(B, C, H, W)


class GammaCNNv9(nn.Module):
    def __init__(self, n_feat, ch=64):
        super().__init__()
        self.el_cnn = ChannelCNN(ch)
        self.mu_cnn = ChannelCNN(ch)

        self.cross_attn = CrossChannelAttention(ch)

        # Combine and deeper processing
        self.combine = nn.Sequential(
            nn.Conv2d(ch * 3, 128, 3, padding=1),  # mu_feat + el_feat + cross_attn
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

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
        el = mat[:, 0:1]; mu = mat[:, 1:2]
        el_feat = self.el_cnn(el)
        mu_feat = self.mu_cnn(mu)
        cross = self.cross_attn(mu_feat, el_feat)

        combined = self.combine(torch.cat([el_feat, mu_feat, cross], dim=1))
        feat_out = self.feat_mlp(feat)
        return self.head(torch.cat([combined, feat_out], dim=1))


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

    train_ds = GammaDataset(m_raw[tr_idx], f_all[tr_idx], y_all[tr_idx], feat_mean, feat_std)
    val_ds = GammaDataset(m_raw[val_idx], f_all[val_idx], y_all[val_idx], feat_mean, feat_std)
    test_ds = GammaDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = GammaCNNv9(f_all.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    n_gamma = (y_all[tr_idx] == 0).sum(); n_hadron = (y_all[tr_idx] == 1).sum()
    w_gamma = n_hadron / n_gamma
    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)

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
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v9.pt")
            print(f"  -> Best: {val_surv:.2e}")

    print(f"\nBest val: {best_val:.2e} | Test: {survival_at_75(best_scores, y_test):.2e}")
    np.save(f"{OUT_DIR}/probs_v9.npy", best_scores)
    np.savez(f"{OUT_DIR}/predictions_v9.npz", gamma_scores=best_scores)

    print("\n---")
    print(f"metric: {best_val:.4e}")
    print("description: Cross-channel attention CNN (mu attending to el), 15 epochs")


if __name__ == "__main__":
    main()
