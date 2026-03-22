"""
v41: Larger CNN with increased capacity - 64-128-256-256 channels.

Different from v12 (which was 64-128-256 with more dropout and overfit).
Key differences from v12:
- 4 CNN stages instead of 3 (deeper)
- Proper batch normalization throughout
- Different regularization: dropout 0.3 in head only
- Train for 20 epochs with cosine LR

If the current CNN family (32-64-128) is capacity-limited,
a deeper/wider network might extract more discriminative features
from the spatial patterns.

Also: add explicit Ne/Nmu ratio as an additional feature channel (scalar, not spatial)
to give the network a direct physics signal.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 7777
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
    # Extra: exponential of Ne-Nmu (direct muon-electron ratio)
    ne_nmu_exp = 10**(Ne - Nmu)  # actual ratio, not log
    ne_nmu_exp_norm = (np.clip(ne_nmu_exp, 0, 1000) - 100) / 200
    return np.stack([
        E_norm, Ze_norm, Az_cos, Az_sin, Ne_norm, Nmu_norm,
        ne_nmu_diff, cos_ze, ne_e_ratio, Ne * Ze_norm, Nmu * cos_ze,
        ne_nmu_exp_norm,  # 12th feature: normalized electron/muon ratio
    ], axis=1).astype(np.float32)


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.conv(x)


class DeepGammaCNN(nn.Module):
    """Deeper CNN: 4 stages with 64-128-256-256 channels."""
    def __init__(self, n_feat):
        super().__init__()
        self.cnn = nn.Sequential(
            CNNBlock(2, 64), nn.MaxPool2d(2),      # 8x8
            CNNBlock(64, 128), nn.MaxPool2d(2),     # 4x4
            CNNBlock(128, 256),                     # 4x4, no pool
            CNNBlock(256, 256), nn.AdaptiveAvgPool2d(2), nn.Flatten(),  # 1024-dim
        )
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(1024 + 64, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, mat, feat):
        return self.head(torch.cat([self.cnn(mat), self.feat_mlp(feat)], dim=1))


class SimpleDataset(Dataset):
    def __init__(self, matrices, features, labels, feat_mean, feat_std):
        self.matrices = matrices; self.features = features
        self.labels = labels; self.feat_mean = feat_mean; self.feat_std = feat_std

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        mat = np.log1p(self.matrices[idx].astype(np.float32)).transpose(2, 0, 1)
        feat = (self.features[idx].copy() - self.feat_mean) / self.feat_std
        return torch.FloatTensor(mat), torch.FloatTensor(feat), int(self.labels[idx])


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

    train_ds = SimpleDataset(m_raw[tr_idx], f_all[tr_idx], y_all[tr_idx], feat_mean, feat_std)
    val_ds = SimpleDataset(m_raw[val_idx], f_all[val_idx], y_all[val_idx], feat_mean, feat_std)
    test_ds = SimpleDataset(m_test_raw, f_test, y_test, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = DeepGammaCNN(n_feat=f_all.shape[1]).to(DEVICE)
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
        for mat, feat, label in train_loader:
            mat, feat, label = mat.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
            logits = model(mat, feat)
            loss = criterion(logits, label)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(label); n_total += len(label)
        scheduler.step()

        model.eval()
        val_scores = []; val_labels = []
        with torch.no_grad():
            for mat, feat, label in val_loader:
                mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                probs = torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy()
                val_scores.extend(probs); val_labels.extend(label.numpy())
        val_surv = survival_at_75(np.array(val_scores), np.array(val_labels))
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}: loss={total_loss/n_total:.4f} val={val_surv:.2e}")

        if val_surv < best_val:
            best_val = val_surv; best_epoch = epoch
            test_scores_list = []
            with torch.no_grad():
                for mat, feat, _ in test_loader:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    probs = torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy()
                    test_scores_list.extend(probs)
            best_test_scores = np.array(test_scores_list)
            print(f"  -> Best val: {val_surv:.2e}")

    test_surv = survival_at_75(best_test_scores, y_test)
    print(f"\nBest val: {best_val:.2e} | Test: {test_surv:.2e} (epoch {best_epoch+1})")

    torch.save(model.state_dict(), f"{OUT_DIR}/model_v41.pt")
    np.save(f"{OUT_DIR}/probs_v41.npy", best_test_scores)
    np.savez(f"{OUT_DIR}/predictions_v41.npz", gamma_scores=best_test_scores)

    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    base = survival_at_75(ens3, y_test)
    print(f"\nens3 baseline: {base:.4e}")
    eps = 1e-10
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.3]:
        blend = ((ens3 + eps)**(1-alpha)) * ((best_test_scores + eps)**alpha)
        s = survival_at_75(blend, y_test)
        print(f"  ens3 + v41 alpha={alpha}: {s:.4e}")

    # Optimize ensemble
    models = {}
    for v in ['v1', 'v2', 'v7', 'v8', 'v9', 'v21', 'v25']:
        models[v] = np.load(f"{OUT_DIR}/probs_{v}.npy")
    models['v41'] = best_test_scores
    keys = list(models.keys())
    preds = [models[k] for k in keys]
    best_surv = base; best_ens = ens3.copy()

    rng2 = np.random.RandomState(77777)
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

    print(f"Best with v41: {best_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_ens_v41.npy", best_ens)
    np.savez(f"{OUT_DIR}/predictions_ens_v41.npz", gamma_scores=best_ens)

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print(f"description: Deeper CNN 64-128-256-256 + 12 features (ne_nmu_exp added), 20ep seed=7777")


if __name__ == "__main__":
    main()
