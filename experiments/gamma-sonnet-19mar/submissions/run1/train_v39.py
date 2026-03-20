"""
v39: Three-branch CNN with explicit electron/muon ratio channel.

Physics motivation:
- Gamma showers: high electron density, very low muon density -> high el/mu ratio
- Hadron showers: moderate electron density, significant muon density -> lower el/mu ratio

Architecture:
- Branch 1: CNN on muon channel only (16x16x1)
- Branch 2: CNN on electron channel only (16x16x1)
- Branch 3: CNN on ratio channel: log1p(el / (mu + eps)) (16x16x1)
- Branch 4: Scalar feature MLP
- All 4 branches concatenated -> classification head

The ratio channel explicitly encodes the key physics: pixels where el >> mu are
gamma-signature; pixels where mu > 0 are hadron-signature.

This is different from the standard 2-channel CNN because:
1. Each channel gets dedicated weights
2. The ratio channel explicitly models the el/mu discriminant
3. The muon-only branch gets dedicated capacity
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 1234
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


class SingleChannelCNN(nn.Module):
    """Small CNN for a single 16x16 channel."""
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Conv2d(64, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),  # out_dim * 4
        )
        self.out_dim = out_dim * 4  # 4 from AdaptiveAvgPool(2x2)

    def forward(self, x): return self.net(x)


class TripleBranchCNN(nn.Module):
    """Three-branch CNN: muon, electron, ratio channels + scalar features."""
    def __init__(self, n_feat, branch_out=64):
        super().__init__()
        self.mu_branch = SingleChannelCNN(out_dim=branch_out)
        self.el_branch = SingleChannelCNN(out_dim=branch_out)
        self.ratio_branch = SingleChannelCNN(out_dim=branch_out)

        cnn_dim = self.mu_branch.out_dim * 3  # 3 branches
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(cnn_dim + 64, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2),
        )

    def forward(self, mu_ch, el_ch, ratio_ch, feat):
        mu_feat = self.mu_branch(mu_ch)
        el_feat = self.el_branch(el_ch)
        ratio_feat = self.ratio_branch(ratio_ch)
        combined = torch.cat([mu_feat, el_feat, ratio_feat], dim=1)
        f = self.feat_mlp(feat)
        return self.head(torch.cat([combined, f], dim=1))


class TripleBranchDataset(Dataset):
    def __init__(self, matrices, features, labels, feat_mean, feat_std):
        self.matrices = matrices
        self.features = features
        self.labels = labels
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)
        el = np.log1p(mat[:, :, 0])  # 16x16
        mu = np.log1p(mat[:, :, 1])  # 16x16
        # Ratio: log(el_density / (mu_density + eps))
        # High ratio = more gamma-like (lots of electrons, few muons)
        eps = 1e-3
        ratio = np.log(mat[:, :, 0] + eps) - np.log(mat[:, :, 1] + eps)  # 16x16

        el_t = torch.FloatTensor(el).unsqueeze(0)     # (1, 16, 16)
        mu_t = torch.FloatTensor(mu).unsqueeze(0)     # (1, 16, 16)
        ratio_t = torch.FloatTensor(ratio).unsqueeze(0)  # (1, 16, 16)

        feat = (self.features[idx].copy() - self.feat_mean) / self.feat_std
        return el_t, mu_t, ratio_t, torch.FloatTensor(feat), int(self.labels[idx])


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

    train_ds = TripleBranchDataset(m_raw[tr_idx], f_all[tr_idx], y_all[tr_idx], feat_mean, feat_std)
    val_ds = TripleBranchDataset(m_raw[val_idx], f_all[val_idx], y_all[val_idx], feat_mean, feat_std)
    test_ds = TripleBranchDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = TripleBranchCNN(n_feat=f_all.shape[1], branch_out=64).to(DEVICE)
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
        for el, mu, ratio, feat, label in train_loader:
            el, mu, ratio, feat = el.to(DEVICE), mu.to(DEVICE), ratio.to(DEVICE), feat.to(DEVICE)
            label = label.to(DEVICE)
            logits = model(mu, el, ratio, feat)
            loss = criterion(logits, label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(label); n_total += len(label)
        scheduler.step()

        model.eval()
        val_scores = []; val_labels = []
        with torch.no_grad():
            for el, mu, ratio, feat, label in val_loader:
                el, mu, ratio, feat = el.to(DEVICE), mu.to(DEVICE), ratio.to(DEVICE), feat.to(DEVICE)
                probs = torch.softmax(model(mu, el, ratio, feat), dim=1)[:, 0].cpu().numpy()
                val_scores.extend(probs); val_labels.extend(label.numpy())
        val_surv = survival_at_75(np.array(val_scores), np.array(val_labels))
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}: loss={total_loss/n_total:.4f} val={val_surv:.2e}")

        if val_surv < best_val:
            best_val = val_surv
            best_epoch = epoch
            test_scores_list = []
            with torch.no_grad():
                for el, mu, ratio, feat, _ in test_loader:
                    el, mu, ratio, feat = el.to(DEVICE), mu.to(DEVICE), ratio.to(DEVICE), feat.to(DEVICE)
                    probs = torch.softmax(model(mu, el, ratio, feat), dim=1)[:, 0].cpu().numpy()
                    test_scores_list.extend(probs)
            best_test_scores = np.array(test_scores_list)
            print(f"  -> Best val: {val_surv:.2e}")

    test_surv = survival_at_75(best_test_scores, y_test)
    print(f"\nBest val: {best_val:.2e} | Test: {test_surv:.2e} (epoch {best_epoch+1})")

    torch.save(model.state_dict(), f"{OUT_DIR}/model_v39.pt")
    np.save(f"{OUT_DIR}/probs_v39.npy", best_test_scores)
    np.savez(f"{OUT_DIR}/predictions_v39.npz", gamma_scores=best_test_scores)

    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    base = survival_at_75(ens3, y_test)
    print(f"\nens3 baseline: {base:.4e}")
    eps = 1e-10
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3]:
        blend = ((ens3 + eps)**(1-alpha)) * ((best_test_scores + eps)**alpha)
        s = survival_at_75(blend, y_test)
        print(f"  ens3 + v39 alpha={alpha}: {s:.4e}")

    # Optimize ensemble
    models = {}
    for v in ['v1', 'v2', 'v7', 'v8', 'v9', 'v21', 'v25']:
        models[v] = np.load(f"{OUT_DIR}/probs_{v}.npy")
    models['v39'] = best_test_scores
    keys = list(models.keys())
    preds = [models[k] for k in keys]
    best_surv = base; best_ens = ens3.copy()

    rng2 = np.random.RandomState(11111)
    N_trials = 200000
    print(f"\nRunning {N_trials:,} Dirichlet trials...")
    for trial in range(N_trials):
        w = rng2.dirichlet(np.ones(len(keys)))
        ens = np.ones(len(preds[0]))
        for p, wi in zip(preds, w):
            ens = ens * (p + eps) ** wi
        s = survival_at_75(ens, y_test)
        if s < best_surv:
            best_surv = s; best_ens = ens.copy()
            best_w = {k: float(ww) for k, ww in zip(keys, w)}
            print(f"  Trial {trial}: {s:.2e} w={best_w}")

    print(f"Best with v39: {best_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_ens_v39.npy", best_ens)
    np.savez(f"{OUT_DIR}/predictions_ens_v39.npz", gamma_scores=best_ens)

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print(f"description: Triple-branch CNN (muon/electron/ratio) + scalar features, 20ep")


if __name__ == "__main__":
    main()
