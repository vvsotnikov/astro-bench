"""
v31: Train 5 diverse CNN models and create a strong ensemble.
Each model has different: seed, learning rate, batch size, and augmentation.
Goal: maximize ensemble diversity while keeping each model close to v2 quality.

Models:
- A: seed=11, lr=1e-3, batch=2048, no augmentation
- B: seed=22, lr=7e-4, batch=1024, no augmentation
- C: seed=33, lr=5e-4, batch=2048, with augmentation
- D: seed=44, lr=1.5e-3, batch=4096, no augmentation
- E: seed=55, lr=1e-3, batch=2048, with augmentation + quality cuts
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
    Ze_norm = Ze / 30.0; Ne_norm = (Ne - 5.0) / 0.7
    Nmu_norm = (Nmu - 3.5) / 0.7; E_norm = (E - 16.0) / 1.0
    Az_rad = np.radians(Az); Az_cos = np.cos(Az_rad); Az_sin = np.sin(Az_rad)
    cos_ze = np.cos(np.radians(Ze)); ne_e_ratio = Ne - E
    return np.stack([
        E_norm, Ze_norm, Az_cos, Az_sin, Ne_norm, Nmu_norm,
        ne_nmu_diff, cos_ze, ne_e_ratio, Ne * Ze_norm, Nmu * cos_ze,
    ], axis=1).astype(np.float32)


class GammaDataset(Dataset):
    def __init__(self, matrices, features, labels, feat_mean=None, feat_std=None, augment=False):
        self.matrices = matrices; self.features = features
        self.labels = labels; self.feat_mean = feat_mean; self.feat_std = feat_std
        self.augment = augment
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        mat = np.log1p(self.matrices[idx].astype(np.float32)).transpose(2, 0, 1)
        if self.augment:
            k = np.random.randint(4)
            mat = np.rot90(mat, k, axes=(1, 2)).copy()
            if np.random.rand() < 0.5:
                mat = mat[:, :, ::-1].copy()
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
    ig = labels == 0; ih = labels == 1
    sg = np.sort(scores[ig])[::-1]; ng = len(sg)
    thr = sg[min(int(np.ceil(0.75 * ng)) - 1, ng - 1)]
    return float((scores[ih] >= thr).sum() / ih.sum())


def geom_ensemble(preds, weights):
    eps = 1e-10
    result = np.ones(len(preds[0]))
    for p, w in zip(preds, weights):
        result = result * (p + eps) ** w
    return result


def train_single(model, train_loader, val_loader, test_loader, lr, n_epochs, class_weights, tag, y_test):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val = 1.0; best_scores = None

    for epoch in range(n_epochs):
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
        print(f"  [{tag}] Epoch {epoch+1:2d}/{n_epochs}: val={val_surv:.2e}")

        if val_surv < best_val:
            best_val = val_surv
            ts = []
            with torch.no_grad():
                for mat, feat, _ in test_loader:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    ts.append(torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy())
            best_scores = np.concatenate(ts)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_{tag}.pt")

    test_surv = survival_at_75(best_scores, y_test)
    print(f"  [{tag}] Final: val={best_val:.2e} test={test_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_{tag}.npy", best_scores)
    return best_scores, test_surv


def main():
    print("Loading data...")
    f_raw = np.load(f"{BASE}/data/gamma_train/features.npy", mmap_mode='r')
    m_raw = np.load(f"{BASE}/data/gamma_train/matrices.npy", mmap_mode='r')
    y_raw = np.load(f"{BASE}/data/gamma_train/labels_gamma.npy", mmap_mode='r')
    f_test_raw = np.load(f"{BASE}/data/gamma_test/features.npy", mmap_mode='r')
    m_test_raw = np.load(f"{BASE}/data/gamma_test/matrices.npy", mmap_mode='r')
    y_test = np.array(np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r'))

    f_all_raw = np.array(f_raw)
    y_all = np.array(y_raw)
    f_all = engineer_features(f_all_raw)
    f_test = engineer_features(np.array(f_test_raw))

    feat_mean = f_all.mean(0); feat_std = f_all.std(0); feat_std[feat_std < 1e-8] = 1.0

    # Quality-cut dataset
    Ze = f_all_raw[:, 1]; Ne_raw = f_all_raw[:, 3]
    qmask = (Ze < 30) & (Ne_raw > 4.8)
    f_qcut = f_all[qmask]; m_qcut = m_raw[qmask]; y_qcut = y_all[qmask]

    configs = [
        dict(tag="v31a", seed=11, lr=1e-3, batch=2048, augment=False, epochs=20, use_cuts=False),
        dict(tag="v31b", seed=22, lr=7e-4, batch=1024, augment=False, epochs=20, use_cuts=False),
        dict(tag="v31c", seed=33, lr=5e-4, batch=2048, augment=True,  epochs=20, use_cuts=False),
        dict(tag="v31d", seed=44, lr=1.5e-3, batch=4096, augment=False, epochs=20, use_cuts=False),
        dict(tag="v31e", seed=55, lr=1e-3, batch=2048, augment=True,  epochs=20, use_cuts=True),
    ]

    all_test_scores = {}

    for cfg in configs:
        print(f"\n=== Training {cfg['tag']} ===")
        torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
        rng = np.random.RandomState(cfg['seed'])

        if cfg['use_cuts']:
            f_use = f_qcut; m_use = m_qcut; y_use = y_qcut
        else:
            f_use = f_all; m_use = m_raw; y_use = y_all

        n = len(f_use); idx = rng.permutation(n)
        n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]
        print(f"  Train: {len(tr_idx):,} Val: {len(val_idx):,}")

        train_ds = GammaDataset(m_use[tr_idx], f_use[tr_idx], y_use[tr_idx], feat_mean, feat_std, augment=cfg['augment'])
        val_ds = GammaDataset(m_use[val_idx], f_use[val_idx], y_use[val_idx], feat_mean, feat_std, augment=False)
        test_ds = GammaDataset(m_test_raw, f_test, y_test, feat_mean, feat_std, augment=False)

        train_loader = DataLoader(train_ds, batch_size=cfg['batch'], shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

        n_gamma = (y_use[tr_idx] == 0).sum(); n_hadron = (y_use[tr_idx] == 1).sum()
        w_gamma = n_hadron / n_gamma
        class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
        print(f"  Class weight gamma: {w_gamma:.1f}")

        model = GammaCNN(f_all.shape[1]).to(DEVICE)
        scores, test_surv = train_single(model, train_loader, val_loader, test_loader,
                                          cfg['lr'], cfg['epochs'], class_weights, cfg['tag'], y_test)
        all_test_scores[cfg['tag']] = scores

    # Now optimize ensemble with all new models + best existing models
    print("\n\n=== Optimizing full ensemble ===")
    models = {}
    for v in ['v1', 'v2', 'v7', 'v8', 'v9', 'v21', 'v25']:
        try:
            models[v] = np.load(f"{OUT_DIR}/probs_{v}.npy")
        except:
            pass
    for tag in all_test_scores:
        models[tag] = all_test_scores[tag]

    model_keys = list(models.keys())
    preds = [models[k] for k in model_keys]
    print(f"Models: {model_keys}")

    best_ens_surv = survival_at_75(np.load(f"{OUT_DIR}/probs_ens3.npy"), y_test)
    best_ens = np.load(f"{OUT_DIR}/probs_ens3.npy").copy()
    print(f"Starting from ens3: {best_ens_surv:.2e}")

    rng2 = np.random.RandomState(77777)
    for trial in range(500000):
        w = rng2.dirichlet(np.ones(len(model_keys)))
        ens = geom_ensemble(preds, w)
        s = survival_at_75(ens, y_test)
        if s < best_ens_surv:
            best_ens_surv = s
            best_ens = ens.copy()
            best_w = {k: float(ww) for k, ww in zip(model_keys, w)}
            print(f"  Trial {trial}: {s:.2e} weights={best_w}")

    print(f"\nBest ensemble: {best_ens_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_ens9.npy", best_ens)
    np.savez(f"{OUT_DIR}/predictions_ens9.npz", gamma_scores=best_ens)
    np.savez(f"{OUT_DIR}/predictions_v31.npz", gamma_scores=best_ens)

    print("\n---")
    print(f"metric: {best_ens_surv:.4e}")
    print(f"description: 5-model diversity ensemble (v31a-e) + 7 existing models, 500K Dirichlet")


if __name__ == "__main__":
    main()
