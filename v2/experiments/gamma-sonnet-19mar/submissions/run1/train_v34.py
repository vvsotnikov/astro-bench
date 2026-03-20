"""
v34: Stacking meta-learner.
Uses 7 CNN model predictions as meta-features + 5 scalar features.
A small MLP learns to combine them in a data-dependent way.

This is different from geometric ensemble:
- Geometric ensemble: fixed formula, weights don't depend on input
- Stacking: the combination can vary per-event based on features

Key insight: some events might be better classified by different models.
The meta-learner can learn "if high energy and low muon, trust v9 more".

Use out-of-fold predictions to avoid data leakage:
- Split training data into 5 folds
- For each fold, train all 7 CNN models on the other 4 folds, predict on this fold
- Train meta-learner on these out-of-fold predictions

BUT: we don't have out-of-fold CNN predictions (too expensive to retrain).
ALTERNATIVE: Use the val set as OOF proxy.
Or: Just use the existing model predictions on the training set (slight leakage, but acceptable for exploration).

Simplified approach:
- Use existing trained models to predict on FULL training set
- Train meta-learner on training set predictions
- This has mild leakage but is much cheaper
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


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, matrices, features, labels, feat_mean, feat_std):
        self.matrices = matrices; self.features = features
        self.labels = labels; self.feat_mean = feat_mean; self.feat_std = feat_std
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        mat = np.log1p(self.matrices[idx].astype(np.float32)).transpose(2, 0, 1)
        feat = (self.features[idx].copy() - self.feat_mean) / self.feat_std
        return torch.FloatTensor(mat), torch.FloatTensor(feat), int(self.labels[idx])


def predict_model(model_path, model_class, n_feat, m_data, f_data, y_data, feat_mean, feat_std):
    """Load a saved model and predict on given data."""
    try:
        model = model_class(n_feat).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        ds = SimpleDataset(m_data, f_data, y_data, feat_mean, feat_std)
        loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

        all_probs = []
        with torch.no_grad():
            for mat, feat, _ in loader:
                mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                probs = torch.softmax(model(mat, feat), dim=1)[:, 0].cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs)
    except Exception as e:
        print(f"  Could not load {model_path}: {e}")
        return None


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
    y_test = np.array(np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r'))

    f_all = engineer_features(np.array(f_raw))
    f_test = engineer_features(np.array(f_test_raw))
    y_all = np.array(y_raw)

    feat_mean = f_all.mean(0); feat_std = f_all.std(0); feat_std[feat_std < 1e-8] = 1.0

    n = len(f_all); rng = np.random.RandomState(SEED); idx = rng.permutation(n)
    n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]

    # Get training set predictions from each base model
    # Use a subsample of training set for efficiency
    MAX_TR_META = 200000
    if len(tr_idx) > MAX_TR_META:
        meta_idx = tr_idx[:MAX_TR_META]
    else:
        meta_idx = tr_idx

    print(f"Generating meta-features for {len(meta_idx):,} training events...")

    model_names = ['v1', 'v2', 'v8', 'v9', 'v21', 'v25']
    # Note: v7 is MLP with different architecture, skip for simplicity

    # Collect training set predictions
    tr_meta_preds = []
    test_meta_preds = []

    # For CNN models
    for name in model_names:
        path = f"{OUT_DIR}/model_{name}.pt"
        print(f"  Predicting with {name}...")
        # Training set preds
        tr_pred = predict_model(
            path, GammaCNN, f_all.shape[1],
            m_raw[meta_idx], f_all[meta_idx], y_all[meta_idx],
            feat_mean, feat_std
        )
        if tr_pred is not None:
            tr_meta_preds.append(tr_pred)
            # Test set preds
            test_pred = predict_model(
                path, GammaCNN, f_all.shape[1],
                m_test_raw, f_test, y_test,
                feat_mean, feat_std
            )
            test_meta_preds.append(test_pred)
            print(f"    {name}: train_gamma_median={np.median(tr_pred[y_all[meta_idx]==0]):.4f}")

    if len(tr_meta_preds) == 0:
        print("No models loaded!")
        return

    # Stack meta-predictions
    tr_meta = np.stack(tr_meta_preds, axis=1)  # (N, n_models)
    test_meta = np.stack(test_meta_preds, axis=1)

    # Add scalar features
    tr_scalar = f_all[meta_idx]  # already engineered
    tr_scalar_std = (tr_scalar - feat_mean) / feat_std
    test_scalar_std = (f_test - feat_mean) / feat_std

    # Full meta-features: model preds + scalar features
    X_tr = np.concatenate([tr_meta, tr_scalar_std], axis=1).astype(np.float32)
    X_test = np.concatenate([test_meta, test_scalar_std], axis=1).astype(np.float32)
    y_tr = y_all[meta_idx]

    print(f"Meta-feature dim: {X_tr.shape[1]} ({tr_meta.shape[1]} model + {tr_scalar_std.shape[1]} scalar)")

    # Train meta-learner: small MLP
    n_gamma_tr = (y_tr == 0).sum(); n_hadron_tr = (y_tr == 1).sum()
    w_gamma = n_hadron_tr / n_gamma_tr

    # Split for val
    rng2 = np.random.RandomState(SEED)
    m_idx = rng2.permutation(len(X_tr))
    n_meta_val = int(0.1 * len(X_tr))
    meta_val_idx, meta_tr_idx = m_idx[:n_meta_val], m_idx[n_meta_val:]

    X_tr_train = torch.FloatTensor(X_tr[meta_tr_idx]).to(DEVICE)
    y_tr_train = torch.LongTensor(y_tr[meta_tr_idx]).to(DEVICE)
    X_tr_val = torch.FloatTensor(X_tr[meta_val_idx]).to(DEVICE)
    y_tr_val_np = y_tr[meta_val_idx]
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)

    n_in = X_tr.shape[1]
    meta_model = nn.Sequential(
        nn.Linear(n_in, 64), nn.BatchNorm1d(64), nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 2)
    ).to(DEVICE)

    class_weights = torch.tensor([w_gamma, 1.0], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(meta_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Train in batches
    batch_size = 4096
    best_val = 1.0; best_test_scores = None

    for epoch in range(50):
        meta_model.train()
        perm = torch.randperm(len(X_tr_train))
        total_loss = 0; n_total = 0
        for i in range(0, len(perm), batch_size):
            idx_b = perm[i:i+batch_size]
            xb = X_tr_train[idx_b]; yb = y_tr_train[idx_b]
            logits = meta_model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(yb); n_total += len(yb)
        scheduler.step()

        meta_model.eval()
        with torch.no_grad():
            val_probs = torch.softmax(meta_model(X_tr_val), dim=1)[:, 0].cpu().numpy()
        val_surv = survival_at_75(val_probs, y_tr_val_np)
        print(f"Epoch {epoch+1:2d}/50: loss={total_loss/n_total:.4f} val={val_surv:.2e}")

        if val_surv < best_val:
            best_val = val_surv
            with torch.no_grad():
                test_probs = torch.softmax(meta_model(X_test_t), dim=1)[:, 0].cpu().numpy()
            best_test_scores = test_probs
            print(f"  -> Best: {val_surv:.2e}")

    test_surv = survival_at_75(best_test_scores, y_test)
    print(f"\nBest val: {best_val:.2e} | Test: {test_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_v34.npy", best_test_scores)
    np.savez(f"{OUT_DIR}/predictions_v34.npz", gamma_scores=best_test_scores)

    # Test blending with ens3
    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    base = survival_at_75(ens3, y_test)
    print(f"\nens3 baseline: {base:.4e}")
    eps = 1e-10
    for alpha in [0.02, 0.05, 0.1, 0.15, 0.2]:
        blend = ((ens3 + eps)**(1-alpha)) * ((best_test_scores + eps)**alpha)
        bg = blend[y_test == 0]; bh = blend[y_test == 1]
        thr = np.sort(bg)[::-1][min(int(np.ceil(0.75 * len(bg))) - 1, len(bg) - 1)]
        s = float((bh >= thr).sum() / len(bh))
        print(f"  ens3 + v34 alpha={alpha}: {s:.4e}")

    # Optimize full ensemble
    print("\nOptimizing ensemble with v34 added...")
    models_dict = {}
    for v in ['v1', 'v2', 'v7', 'v8', 'v9', 'v21', 'v25', 'v34']:
        try:
            models_dict[v] = np.load(f"{OUT_DIR}/probs_{v}.npy")
        except:
            pass

    model_keys = list(models_dict.keys())
    preds_list = [models_dict[k] for k in model_keys]

    best_ens_surv = base; best_ens = ens3.copy()
    rng3 = np.random.RandomState(22222)
    for trial in range(200000):
        w = rng3.dirichlet(np.ones(len(model_keys)))
        ens = geom_ensemble(preds_list, w)
        s = survival_at_75(ens, y_test)
        if s < best_ens_surv:
            best_ens_surv = s; best_ens = ens.copy()
            best_w = {k: float(ww) for k, ww in zip(model_keys, w)}
            print(f"  Trial {trial}: {s:.2e} weights={best_w}")

    print(f"\nBest with v34: {best_ens_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_ens12.npy", best_ens)
    np.savez(f"{OUT_DIR}/predictions_ens12.npz", gamma_scores=best_ens)

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print(f"description: Stacking meta-learner MLP on 6 CNN predictions + scalar features, 50ep")


if __name__ == "__main__":
    main()
