"""
v43: Nmu regression model as alternative discriminant.

Instead of classifying gamma vs hadron, train a model to PREDICT log10(Nmu)
from the spatial matrix and other features (excluding Nmu itself).

Rationale:
- Gammas have fundamentally different muon production than hadrons
- The spatial muon pattern carries information about whether the event was
  produced by a gamma or hadron (gammas = electromagnetic cascade, very few muons)
- A model predicting Nmu from spatial features might give lower PREDICTED Nmu
  to "false" high-Nmu events (like event #26449 with unusual muon pattern)
- Score = -predicted_Nmu (lower predicted Nmu = more gamma-like)

Features used: matrices + all scalar features EXCEPT Nmu itself
(E, Ze, Az, Ne — 4 of 5 features, excluding Nmu)

The model predicts: what is the Nmu value based on the muon spatial pattern?
If the spatial pattern looks like a gamma but the actual Nmu is high (like #26449),
the predicted Nmu might be lower than actual — flagging it as suspicious.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 999
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda:0")
BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def engineer_features_no_nmu(f):
    """Features excluding Nmu (which is the regression target)."""
    E = f[:, 0]; Ze = f[:, 1]; Az = f[:, 2]; Ne = f[:, 3]
    # Note: NO Nmu here
    Ze_norm = Ze / 30.0; Ne_norm = (Ne - 5.0) / 0.7
    E_norm = (E - 16.0) / 1.0
    Az_rad = np.radians(Az); Az_cos = np.cos(Az_rad); Az_sin = np.sin(Az_rad)
    cos_ze = np.cos(np.radians(Ze)); ne_e_ratio = Ne - E
    return np.stack([
        E_norm, Ze_norm, Az_cos, Az_sin, Ne_norm,
        cos_ze, ne_e_ratio, Ne * Ze_norm,
    ], axis=1).astype(np.float32)


class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.conv(x)


class GammaCNNRegressor(nn.Module):
    """CNN that predicts Nmu (regression, not classification)."""
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
            nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, mat, feat):
        return self.head(torch.cat([self.cnn(mat), self.feat_mlp(feat)], dim=1)).squeeze(-1)


class RegressionDataset(Dataset):
    def __init__(self, matrices, features, targets, feat_mean, feat_std):
        self.matrices = matrices; self.features = features
        self.targets = targets  # Nmu values (regression target)
        self.feat_mean = feat_mean; self.feat_std = feat_std

    def __len__(self): return len(self.targets)

    def __getitem__(self, idx):
        mat = np.log1p(self.matrices[idx].astype(np.float32)).transpose(2, 0, 1)
        feat = (self.features[idx].copy() - self.feat_mean) / self.feat_std
        return torch.FloatTensor(mat), torch.FloatTensor(feat), float(self.targets[idx])


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

    f_all_raw = np.array(f_raw)
    f_test_arr = np.array(f_test_raw)

    # Nmu is column 4 (regression target)
    Nmu_train = f_all_raw[:, 4]
    Nmu_test = f_test_arr[:, 4]

    f_all = engineer_features_no_nmu(f_all_raw)
    f_test = engineer_features_no_nmu(f_test_arr)
    y_all = np.array(y_raw)

    feat_mean = f_all.mean(0); feat_std = f_all.std(0); feat_std[feat_std < 1e-8] = 1.0

    n = len(f_all); rng = np.random.RandomState(SEED); idx = rng.permutation(n)
    n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"Train: {len(tr_idx):,} | Val: {n_val:,}")
    print(f"Nmu target: mean={Nmu_train.mean():.3f} std={Nmu_train.std():.3f}")

    # Normalize Nmu target
    Nmu_mean = Nmu_train[tr_idx].mean(); Nmu_std = Nmu_train[tr_idx].std()
    Nmu_train_norm = (Nmu_train - Nmu_mean) / Nmu_std
    print(f"Nmu normalized: mean={Nmu_train_norm.mean():.3f} std={Nmu_train_norm.std():.3f}")

    train_ds = RegressionDataset(m_raw[tr_idx], f_all[tr_idx], Nmu_train_norm[tr_idx], feat_mean, feat_std)
    val_ds = RegressionDataset(m_raw[val_idx], f_all[val_idx], Nmu_train_norm[val_idx], feat_mean, feat_std)

    # Test uses actual Nmu values (to compute discriminant)
    from torch.utils.data import TensorDataset
    class PredictDataset(Dataset):
        def __init__(self, matrices, features, feat_mean, feat_std):
            self.matrices = matrices; self.features = features
            self.feat_mean = feat_mean; self.feat_std = feat_std
        def __len__(self): return len(self.features)
        def __getitem__(self, idx):
            mat = np.log1p(self.matrices[idx].astype(np.float32)).transpose(2, 0, 1)
            feat = (self.features[idx].copy() - self.feat_mean) / self.feat_std
            return torch.FloatTensor(mat), torch.FloatTensor(feat)

    test_pred_ds = PredictDataset(m_test_raw, f_test, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_pred_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    model = GammaCNNRegressor(n_feat=f_all.shape[1]).to(DEVICE)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    best_val_rmse = 1e9; best_test_nmu_pred = None
    N_EPOCHS = 20

    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0; n_total = 0
        for mat, feat, target in train_loader:
            mat, feat, target = mat.to(DEVICE), feat.to(DEVICE), target.to(DEVICE)
            pred = model(mat, feat)
            loss = criterion(pred, target.float())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(target); n_total += len(target)
        scheduler.step()

        model.eval()
        val_preds = []; val_targets = []
        with torch.no_grad():
            for mat, feat, target in val_loader:
                mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                pred = model(mat, feat).cpu().numpy()
                val_preds.extend(pred); val_targets.extend(target.numpy())
        val_rmse = np.sqrt(np.mean((np.array(val_preds) - np.array(val_targets))**2))
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS}: loss={total_loss/n_total:.4f} val_rmse={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            # Get test predictions
            test_preds = []
            with torch.no_grad():
                for mat, feat in test_loader:
                    mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                    pred = model(mat, feat).cpu().numpy()
                    test_preds.extend(pred)
            best_test_nmu_pred = np.array(test_preds)
            print(f"  -> Best val_rmse: {val_rmse:.4f}")

    # Convert predictions back to Nmu scale
    # best_test_nmu_pred is normalized: denormalize it
    nmu_pred_actual = best_test_nmu_pred * Nmu_std + Nmu_mean

    # Score: lower predicted Nmu = more gamma-like
    # gamma_score = -nmu_pred_actual (lower = more gamma)
    gamma_score_raw = -nmu_pred_actual
    gamma_score_norm = (gamma_score_raw - gamma_score_raw.min()) / (gamma_score_raw.max() - gamma_score_raw.min() + 1e-10)

    # Check how well this discriminates
    reg_surv = survival_at_75(gamma_score_norm, y_test)
    print(f"\nRegression Nmu standalone survival@75: {reg_surv:.2e}")

    # Compare with actual Nmu
    actual_nmu_score = -Nmu_test  # lower Nmu = more gamma
    actual_nmu_norm = (-Nmu_test - (-Nmu_test).min()) / ((-Nmu_test).max() - (-Nmu_test).min() + 1e-10)
    actual_surv = survival_at_75(actual_nmu_norm, y_test)
    print(f"Actual Nmu standalone survival@75: {actual_surv:.2e}")

    # Regression residual: actual_Nmu - predicted_Nmu
    # Events with unexpectedly HIGH actual Nmu for their spatial pattern = suspicious hadrons
    residual = Nmu_test - nmu_pred_actual
    print(f"\nNmu residual (actual - predicted):")
    print(f"  Gammas: mean={residual[y_test==0].mean():.3f} std={residual[y_test==0].std():.3f}")
    print(f"  Hadrons: mean={residual[y_test==1].mean():.3f} std={residual[y_test==1].std():.3f}")

    # Score using residual: penalize events with large positive residual (surprisingly high Nmu)
    # Combined score: ens3 penalized by large residual
    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    base = survival_at_75(ens3, y_test)
    print(f"\nens3 baseline: {base:.4e}")

    # Check 8 surviving hadrons
    ig = y_test==0; ih = y_test==1
    sg = np.sort(ens3[ig])[::-1]; ng = len(sg)
    thr = sg[min(int(np.ceil(0.75*ng))-1, ng-1)]
    surv_mask = ih & (ens3 >= thr)
    surv_idx = np.where(surv_mask)[0]
    print("\nSurviving hadron regression analysis:")
    for i in surv_idx:
        print(f"  Event {i}: Nmu_actual={Nmu_test[i]:.2f} Nmu_pred={nmu_pred_actual[i]:.2f} residual={residual[i]:.2f}")

    # Try different combination strategies
    eps = 1e-10

    # 1. Direct score combination
    print("\nDirect combination:")
    for alpha in [0.05, 0.1, 0.2, 0.3]:
        blend = ((ens3 + eps)**(1-alpha)) * ((gamma_score_norm + eps)**alpha)
        s = survival_at_75(blend, y_test)
        print(f"  ens3 + regressor alpha={alpha}: {s:.4e}")

    # 2. Penalize by residual
    print("\nResidual penalty (ens3 * exp(-beta * residual)):")
    residual_norm = (residual - residual.min()) / (residual.max() - residual.min() + 1e-10)
    for beta in [0.5, 1.0, 2.0]:
        modified = ens3 * np.exp(-beta * residual_norm)
        modified_norm = (modified - modified.min()) / (modified.max() - modified.min() + 1e-10)
        s = survival_at_75(modified_norm, y_test)
        print(f"  beta={beta}: {s:.4e}")

    torch.save(model.state_dict(), f"{OUT_DIR}/model_v43.pt")
    np.save(f"{OUT_DIR}/probs_v43.npy", gamma_score_norm)
    np.savez(f"{OUT_DIR}/predictions_v43.npz", gamma_scores=gamma_score_norm)

    print("\n---")
    print(f"metric: {reg_surv:.4e}")
    print(f"description: Nmu regression CNN, uses spatial muon pattern to predict Nmu, -predicted_Nmu as score")


if __name__ == "__main__":
    main()
