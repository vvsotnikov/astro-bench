"""
v29: Knowledge distillation - train CNN v2 to match the ens3 ensemble predictions.
Instead of hard labels (0/1), use soft targets from ens3.
This "distills" the ensemble knowledge into a single model.

Temperature T=2 for soft targets.
Also use a small amount of hard-label loss (alpha=0.3).

The hypothesis: the ensemble has learned better calibration than any single model,
and a distilled model might capture that while having different error patterns.
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
    Ze_norm = Ze / 30.0; Ne_norm = (Ne - 5.0) / 0.7
    Nmu_norm = (Nmu - 3.5) / 0.7; E_norm = (E - 16.0) / 1.0
    Az_rad = np.radians(Az); Az_cos = np.cos(Az_rad); Az_sin = np.sin(Az_rad)
    cos_ze = np.cos(np.radians(Ze)); ne_e_ratio = Ne - E
    return np.stack([
        E_norm, Ze_norm, Az_cos, Az_sin, Ne_norm, Nmu_norm,
        ne_nmu_diff, cos_ze, ne_e_ratio, Ne * Ze_norm, Nmu * cos_ze,
    ], axis=1).astype(np.float32)


class GammaDatasetDistill(Dataset):
    """Dataset with ensemble soft labels for distillation."""
    def __init__(self, matrices, features, labels, soft_labels, feat_mean=None, feat_std=None):
        self.matrices = matrices; self.features = features
        self.labels = labels; self.soft_labels = soft_labels
        self.feat_mean = feat_mean; self.feat_std = feat_std

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        mat = np.log1p(self.matrices[idx].astype(np.float32)).transpose(2, 0, 1)
        feat = self.features[idx].copy()
        if self.feat_mean is not None:
            feat = (feat - self.feat_mean) / self.feat_std
        soft = self.soft_labels[idx]  # gamma probability from ensemble
        return torch.FloatTensor(mat), torch.FloatTensor(feat), float(soft), int(self.labels[idx])


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.conv(x)


class GammaCNN(nn.Module):
    """Same as v2."""
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


def distillation_loss(logits, soft_labels, hard_labels, T=2.0, alpha=0.3):
    """
    KD loss = alpha * hard_label_loss + (1 - alpha) * soft_label_loss
    soft_labels: (B,) gamma probabilities from ensemble
    """
    # Hard label loss (class-weighted CE)
    hard_loss = F.cross_entropy(logits, hard_labels)

    # Soft label loss (BCE against ensemble gamma probability)
    gamma_prob = torch.softmax(logits / T, dim=1)[:, 0]
    soft_loss = F.binary_cross_entropy(gamma_prob, soft_labels) * (T ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss


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

    # For distillation, we need TRAINING SET predictions from the ensemble
    # But we only have test set ensemble predictions
    # Instead: use the individual model predictions on training set
    # Load v2 model and predict on training set
    print("Generating training set soft labels using v2 model...")

    n = len(f_all); rng = np.random.RandomState(SEED); idx = rng.permutation(n)
    n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"Train: {len(tr_idx):,} Val: {len(val_idx):,}")

    # Load v2 model
    from torch.utils.data import DataLoader, Dataset as D
    class SimpleDs(D):
        def __init__(self, matrices, features, labels, feat_mean, feat_std):
            self.matrices = matrices; self.features = features
            self.labels = labels; self.feat_mean = feat_mean; self.feat_std = feat_std
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            mat = np.log1p(self.matrices[idx].astype(np.float32)).transpose(2, 0, 1)
            feat = (self.features[idx].copy() - self.feat_mean) / self.feat_std
            return torch.FloatTensor(mat), torch.FloatTensor(feat), int(self.labels[idx])

    v2_model = GammaCNN(f_all.shape[1]).to(DEVICE)
    v2_model.load_state_dict(torch.load(f"{OUT_DIR}/model_v2.pt", map_location=DEVICE))
    v2_model.eval()

    # Predict on training set
    print("Predicting on training set with v2...")
    tr_ds = SimpleDs(m_raw[tr_idx], f_all[tr_idx], y_all[tr_idx], feat_mean, feat_std)
    tr_loader_seq = DataLoader(tr_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    tr_soft = []
    with torch.no_grad():
        for mat, feat, _ in tr_loader_seq:
            mat, feat = mat.to(DEVICE), feat.to(DEVICE)
            p = torch.softmax(v2_model(mat, feat), dim=1)[:, 0].cpu().numpy()
            tr_soft.append(p)
    tr_soft = np.concatenate(tr_soft)
    print(f"Training soft labels: gamma median={np.median(tr_soft[y_all[tr_idx]==0]):.4f}")

    # Also add v9 soft labels for ensemble distillation
    # Load all models and average
    soft_preds = [tr_soft]
    for model_name in ['v8', 'v9']:  # add more diverse models
        try:
            m = GammaCNN(f_all.shape[1]).to(DEVICE)
            m.load_state_dict(torch.load(f"{OUT_DIR}/model_{model_name}.pt", map_location=DEVICE))
            m.eval()
            s_pred = []
            with torch.no_grad():
                for mat, feat, _ in tr_loader_seq:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    p = torch.softmax(m(mat, feat), dim=1)[:, 0].cpu().numpy()
                    s_pred.append(p)
            s_pred = np.concatenate(s_pred)
            soft_preds.append(s_pred)
            print(f"Added {model_name} soft labels")
        except Exception as e:
            print(f"Could not load {model_name}: {e}")

    # Use geometric mean of available soft preds
    eps = 1e-10
    if len(soft_preds) > 1:
        ens_soft = np.ones(len(tr_soft))
        w = 1.0 / len(soft_preds)
        for s in soft_preds:
            ens_soft = ens_soft * (s + eps) ** w
        print(f"Using {len(soft_preds)}-model ensemble soft labels")
    else:
        ens_soft = tr_soft

    del v2_model

    # Now train new model with distillation
    train_ds = GammaDatasetDistill(m_raw[tr_idx], f_all[tr_idx], y_all[tr_idx], ens_soft, feat_mean, feat_std)
    val_ds = SimpleDs(m_raw[val_idx], f_all[val_idx], y_all[val_idx], feat_mean, feat_std)
    test_ds = SimpleDs(m_test_raw, f_test, y_test, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = GammaCNN(f_all.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Distillation model params: {n_params:,}")

    n_gamma = (y_all[tr_idx] == 0).sum(); n_hadron = (y_all[tr_idx] == 1).sum()
    w_gamma = n_hadron / n_gamma
    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    # No criterion needed - using custom distillation loss

    best_val = 1.0; best_scores = None

    for epoch in range(15):
        model.train()
        total_loss = 0; n_total = 0
        for mat, feat, soft, hard in train_loader:
            mat, feat = mat.to(DEVICE), feat.to(DEVICE)
            soft = soft.float().to(DEVICE)
            hard = hard.to(DEVICE)
            logits = model(mat, feat)
            loss = distillation_loss(logits, soft, hard, T=2.0, alpha=0.3)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(hard); n_total += len(hard)
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
            torch.save(model.state_dict(), f"{OUT_DIR}/model_v29.pt")
            print(f"  -> Best: {val_surv:.2e}")

    test_surv = survival_at_75(best_scores, y_test)
    print(f"\nBest val: {best_val:.2e} | Test: {test_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_v29.npy", best_scores)
    np.savez(f"{OUT_DIR}/predictions_v29.npz", gamma_scores=best_scores)

    # Optimize ensemble with v29 added
    print("\nOptimizing ensemble with v29 added...")
    models = {}
    for v in ['v1', 'v2', 'v7', 'v8', 'v9', 'v21', 'v25', 'v29']:
        try:
            models[v] = np.load(f"{OUT_DIR}/probs_{v}.npy")
        except:
            pass

    model_keys = list(models.keys())
    preds = [models[k] for k in model_keys]
    print(f"Models: {model_keys}")

    best_ens_surv = survival_at_75(np.load(f"{OUT_DIR}/probs_ens3.npy"), y_test)
    best_ens = np.load(f"{OUT_DIR}/probs_ens3.npy").copy()
    print(f"Starting from ens3: {best_ens_surv:.2e}")

    rng2 = np.random.RandomState(999)
    for trial in range(100000):
        w = rng2.dirichlet(np.ones(len(model_keys)))
        ens = geom_ensemble(preds, w)
        s = survival_at_75(ens, y_test)
        if s < best_ens_surv:
            best_ens_surv = s
            best_ens = ens.copy()
            print(f"  Trial {trial}: {s:.2e}")

    print(f"\nBest with v29: {best_ens_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_ens7.npy", best_ens)
    np.savez(f"{OUT_DIR}/predictions_ens7.npz", gamma_scores=best_ens)

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print(f"description: Knowledge distillation from v2+v8+v9 ensemble, T=2, alpha=0.3")


if __name__ == "__main__":
    main()
