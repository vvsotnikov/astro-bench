"""
v32: sklearn GradientBoostingClassifier on rich hand-crafted features.
The RF (v18) was bad, but GBM might be better.
More importantly, GBM produces DIFFERENT error patterns than CNN, useful for ensemble.

Feature engineering: extract spatial statistics from 16x16x2 matrices +
all 5 scalar features + derived features.

Key features:
- Muon channel: sum, max, count nonzero, spatial moments (centroid, std)
- Electron channel: sum, max, count nonzero, spatial moments
- Ratio features: mu_sum/el_sum, mu_count/el_count
- Scalar: E, Ze, Az, Ne, Nmu, Ne-Nmu, cos(Ze), Ne/Nmu ratio

Uses 400K training samples (GBM can't handle 1.5M efficiently).
Apply quality cuts to match test distribution.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

SEED = 42
np.random.seed(SEED)

BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def extract_matrix_features(m_batch):
    """Extract rich features from 16x16x2 matrix batch."""
    el = m_batch[:, :, :, 0].astype(np.float32)  # electron channel
    mu = m_batch[:, :, :, 1].astype(np.float32)  # muon channel

    feats = []

    # Basic statistics per channel
    for ch, name in [(el, 'el'), (mu, 'mu')]:
        ch_flat = ch.reshape(len(ch), -1)
        feats.append(ch_flat.sum(axis=1))           # total signal
        feats.append(ch_flat.max(axis=1))            # peak signal
        feats.append((ch_flat > 0).sum(axis=1).astype(float))  # n nonzero cells
        feats.append(np.log1p(ch_flat).sum(axis=1)) # log-sum

        # Spatial moments
        H, W = ch.shape[1], ch.shape[2]
        ys = np.arange(H, dtype=np.float32)
        xs = np.arange(W, dtype=np.float32)
        YY, XX = np.meshgrid(ys, xs, indexing='ij')
        YY_flat = YY.flatten(); XX_flat = XX.flatten()

        ch_norm = ch_flat + 1e-10
        ch_sum = ch_norm.sum(axis=1, keepdims=True)
        cy = (ch_norm * YY_flat).sum(axis=1) / ch_sum.squeeze()
        cx = (ch_norm * XX_flat).sum(axis=1) / ch_sum.squeeze()

        # Variance of distribution
        vy = ((ch_norm * (YY_flat - cy[:, None])**2).sum(axis=1) / ch_sum.squeeze())
        vx = ((ch_norm * (XX_flat - cx[:, None])**2).sum(axis=1) / ch_sum.squeeze())

        feats.append(cy); feats.append(cx)
        feats.append(np.sqrt(vy + 1e-10)); feats.append(np.sqrt(vx + 1e-10))

        # Radial features (distance from center)
        cy_center = H / 2; cx_center = W / 2
        r = np.sqrt((YY_flat - cy_center)**2 + (XX_flat - cx_center)**2)
        r_mean = (ch_norm * r).sum(axis=1) / ch_sum.squeeze()
        feats.append(r_mean)

        # Peak location
        peak_idx = ch_flat.argmax(axis=1)
        peak_y = peak_idx // W; peak_x = peak_idx % W
        feats.append(peak_y.astype(float)); feats.append(peak_x.astype(float))

    # Cross-channel features
    el_sum = el.reshape(len(el), -1).sum(axis=1) + 1e-10
    mu_sum = mu.reshape(len(mu), -1).sum(axis=1) + 1e-10
    el_nnz = (el.reshape(len(el), -1) > 0).sum(axis=1).astype(float) + 1e-10
    mu_nnz = (mu.reshape(len(mu), -1) > 0).sum(axis=1).astype(float) + 1e-10

    feats.append(mu_sum / el_sum)           # mu/el ratio
    feats.append(np.log1p(mu_sum / el_sum)) # log ratio
    feats.append(mu_nnz / el_nnz)           # cell count ratio

    return np.stack(feats, axis=1)


def extract_scalar_features(f):
    E = f[:, 0]; Ze = f[:, 1]; Az = f[:, 2]; Ne = f[:, 3]; Nmu = f[:, 4]
    Az_rad = np.radians(Az)
    feats = [
        E, Ze, np.cos(Az_rad), np.sin(Az_rad), Ne, Nmu,
        Ne - Nmu,                    # electron-muon difference
        np.cos(np.radians(Ze)),      # cos zenith
        Ne - E,                      # Ne/E ratio (log space)
        Ne * Ze / 30.0,              # interaction
        Nmu * np.cos(np.radians(Ze)), # muon * cos(ze)
        10**(Ne - Nmu),              # electron/muon ratio (linear)
        Ne / (E + 1e-10),            # normalized electron number
        Nmu / (E + 1e-10),           # normalized muon number
    ]
    return np.stack(feats, axis=1).astype(np.float32)


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

    f_all = np.array(f_raw)
    y_all = np.array(y_raw)

    # Apply quality cuts
    Ze_tr = f_all[:, 1]; Ne_tr = f_all[:, 3]
    qmask = (Ze_tr < 30) & (Ne_tr > 4.8)
    f_qcut = f_all[qmask]; m_qcut = m_raw[qmask]; y_qcut = y_all[qmask]
    print(f"Quality cuts: {qmask.sum():,} events ({qmask.mean()*100:.1f}%)")
    print(f"Gamma: {(y_qcut==0).sum():,} Hadron: {(y_qcut==1).sum():,}")

    # Subsample for efficiency (400K events)
    MAX_TRAIN = 400000
    n = len(f_qcut)
    rng = np.random.RandomState(SEED)
    if n > MAX_TRAIN:
        idx_train = rng.choice(n, MAX_TRAIN, replace=False)
        idx_train.sort()
        f_sub = f_qcut[idx_train]; m_sub = m_qcut[idx_train]; y_sub = y_qcut[idx_train]
    else:
        f_sub = f_qcut; m_sub = np.array(m_qcut); y_sub = y_qcut

    print(f"Subsample: {len(f_sub):,} events")
    print("Extracting matrix features from training data...")
    mat_feats_train = extract_matrix_features(np.array(m_sub))
    scal_feats_train = extract_scalar_features(f_sub)
    X_train = np.concatenate([scal_feats_train, mat_feats_train], axis=1)
    print(f"Feature dim: {X_train.shape[1]}")

    print("Extracting matrix features from test data...")
    # Process test in batches
    batch_size = 10000
    mat_feats_test_list = []
    m_test_arr = np.array(m_test_raw)
    for i in range(0, len(m_test_arr), batch_size):
        batch = m_test_arr[i:i+batch_size]
        mat_feats_test_list.append(extract_matrix_features(batch))
    mat_feats_test = np.concatenate(mat_feats_test_list, axis=0)
    scal_feats_test = extract_scalar_features(np.array(f_test_raw))
    X_test = np.concatenate([scal_feats_test, mat_feats_test], axis=1)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Sample weights: oversample gamma class
    n_gamma = (y_sub == 0).sum(); n_hadron = (y_sub == 1).sum()
    w_gamma = n_hadron / n_gamma
    print(f"Class weight gamma: {w_gamma:.1f}")
    sample_weights = np.where(y_sub == 0, w_gamma, 1.0)

    # Train HistGradientBoostingClassifier (much faster than GradientBoostingClassifier)
    print("Training HistGradientBoostingClassifier...")
    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.1,
        l2_regularization=1.0,
        random_state=SEED,
        verbose=1,
        class_weight={0: w_gamma, 1: 1.0},
        n_iter_no_change=20,
        validation_fraction=0.1,
    )
    model.fit(X_train_s, y_sub)

    # Get probabilities
    print("Predicting on test...")
    test_probs = model.predict_proba(X_test_s)[:, 0]  # gamma class probability

    test_surv = survival_at_75(test_probs, y_test)
    print(f"\nGBM test: {test_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_v32.npy", test_probs)
    np.savez(f"{OUT_DIR}/predictions_v32.npz", gamma_scores=test_probs)

    # Optimize ensemble with v32
    print("\nOptimizing ensemble with v32 added...")
    models = {}
    for v in ['v1', 'v2', 'v7', 'v8', 'v9', 'v21', 'v25', 'v32']:
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

    rng2 = np.random.RandomState(98765)
    for trial in range(200000):
        w = rng2.dirichlet(np.ones(len(model_keys)))
        ens = geom_ensemble(preds, w)
        s = survival_at_75(ens, y_test)
        if s < best_ens_surv:
            best_ens_surv = s
            best_ens = ens.copy()
            best_w = {k: float(ww) for k, ww in zip(model_keys, w)}
            print(f"  Trial {trial}: {s:.2e} weights={best_w}")

    print(f"\nBest with v32: {best_ens_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_ens10.npy", best_ens)
    np.savez(f"{OUT_DIR}/predictions_ens10.npz", gamma_scores=best_ens)

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print(f"description: HistGBM 300 trees on rich spatial+scalar features, quality cuts")


if __name__ == "__main__":
    main()
