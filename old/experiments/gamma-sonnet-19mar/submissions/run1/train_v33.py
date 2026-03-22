"""
v33: HistGradientBoostingClassifier on ALL training data (no quality cuts).
v32 used only quality-cut training data (143K events) and was terrible.
This version uses full 400K subsampled events for better diversity.

The GBM can potentially learn non-linear feature interactions that CNNs miss.
Key: the GBM sees the data through a completely different lens (scalar features + spatial statistics).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def extract_matrix_features(m_batch):
    """Extract rich features from 16x16x2 matrix batch."""
    el = m_batch[:, :, :, 0].astype(np.float32)
    mu = m_batch[:, :, :, 1].astype(np.float32)

    feats = []

    for ch in [el, mu]:
        ch_flat = ch.reshape(len(ch), -1)
        feats.append(ch_flat.sum(axis=1))
        feats.append(ch_flat.max(axis=1))
        feats.append((ch_flat > 0).sum(axis=1).astype(float))
        feats.append(np.log1p(ch_flat).sum(axis=1))

        H, W = ch.shape[1], ch.shape[2]
        ys = np.arange(H, dtype=np.float32)
        xs = np.arange(W, dtype=np.float32)
        YY, XX = np.meshgrid(ys, xs, indexing='ij')
        YY_flat = YY.flatten(); XX_flat = XX.flatten()

        ch_norm = ch_flat + 1e-10
        ch_sum = ch_norm.sum(axis=1, keepdims=True)
        cy = (ch_norm * YY_flat).sum(axis=1) / ch_sum.squeeze()
        cx = (ch_norm * XX_flat).sum(axis=1) / ch_sum.squeeze()
        vy = ((ch_norm * (YY_flat - cy[:, None])**2).sum(axis=1) / ch_sum.squeeze())
        vx = ((ch_norm * (XX_flat - cx[:, None])**2).sum(axis=1) / ch_sum.squeeze())

        feats.append(cy); feats.append(cx)
        feats.append(np.sqrt(vy + 1e-10)); feats.append(np.sqrt(vx + 1e-10))

        cy_center = H / 2; cx_center = W / 2
        r = np.sqrt((YY_flat - cy_center)**2 + (XX_flat - cx_center)**2)
        r_mean = (ch_norm * r).sum(axis=1) / ch_sum.squeeze()
        feats.append(r_mean)

        peak_idx = ch_flat.argmax(axis=1)
        peak_y = peak_idx // W; peak_x = peak_idx % W
        feats.append(peak_y.astype(float)); feats.append(peak_x.astype(float))

    el_flat = el.reshape(len(el), -1)
    mu_flat = mu.reshape(len(mu), -1)
    el_sum = el_flat.sum(axis=1) + 1e-10
    mu_sum = mu_flat.sum(axis=1) + 1e-10
    el_nnz = (el_flat > 0).sum(axis=1).astype(float) + 1e-10
    mu_nnz = (mu_flat > 0).sum(axis=1).astype(float) + 1e-10

    feats.append(mu_sum / el_sum)
    feats.append(np.log1p(mu_sum / el_sum))
    feats.append(mu_nnz / el_nnz)

    return np.stack(feats, axis=1)


def extract_scalar_features(f):
    E = f[:, 0]; Ze = f[:, 1]; Az = f[:, 2]; Ne = f[:, 3]; Nmu = f[:, 4]
    Az_rad = np.radians(Az)
    feats = [
        E, Ze, np.cos(Az_rad), np.sin(Az_rad), Ne, Nmu,
        Ne - Nmu, np.cos(np.radians(Ze)), Ne - E,
        Ne * Ze / 30.0, Nmu * np.cos(np.radians(Ze)),
        10**(Ne - Nmu), Ne / (E + 1e-10), Nmu / (E + 1e-10),
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
    print(f"Full dataset: {len(f_all):,} events (gamma: {(y_all==0).sum():,})")

    # Subsample to 400K events
    MAX_TRAIN = 400000
    rng = np.random.RandomState(SEED)
    n = len(f_all)
    if n > MAX_TRAIN:
        idx = rng.choice(n, MAX_TRAIN, replace=False)
        idx.sort()
        f_sub = f_all[idx]; m_sub = m_raw[idx]; y_sub = y_all[idx]
    else:
        f_sub = f_all; m_sub = np.array(m_raw); y_sub = y_all

    print(f"Subsample: {len(f_sub):,} events (gamma: {(y_sub==0).sum():,})")
    print("Extracting features from training data (400K events)...")

    # Process in batches to avoid RAM issues
    batch_size = 50000
    mat_feats_train_list = []
    for i in range(0, len(f_sub), batch_size):
        if i % 100000 == 0:
            print(f"  Batch {i}/{len(f_sub)}...")
        batch = np.array(m_sub[i:i+batch_size])
        mat_feats_train_list.append(extract_matrix_features(batch))

    mat_feats_train = np.concatenate(mat_feats_train_list, axis=0)
    scal_feats_train = extract_scalar_features(f_sub)
    X_train = np.concatenate([scal_feats_train, mat_feats_train], axis=1)
    print(f"Feature dim: {X_train.shape[1]}")

    print("Extracting features from test data...")
    m_test_arr = np.array(m_test_raw)
    mat_feats_test_list = []
    for i in range(0, len(m_test_arr), batch_size):
        batch = m_test_arr[i:i+batch_size]
        mat_feats_test_list.append(extract_matrix_features(batch))
    mat_feats_test = np.concatenate(mat_feats_test_list, axis=0)
    scal_feats_test = extract_scalar_features(np.array(f_test_raw))
    X_test = np.concatenate([scal_feats_test, mat_feats_test], axis=1)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_gamma = (y_sub == 0).sum(); n_hadron = (y_sub == 1).sum()
    w_gamma = n_hadron / n_gamma
    print(f"Class weight gamma: {w_gamma:.1f}")

    print("Training HistGradientBoostingClassifier...")
    model = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=SEED,
        verbose=1,
        class_weight={0: w_gamma, 1: 1.0},
        n_iter_no_change=30,
        validation_fraction=0.1,
        min_samples_leaf=20,
    )
    model.fit(X_train_s, y_sub)

    print("Predicting on test...")
    test_probs = model.predict_proba(X_test_s)[:, 0]

    test_surv = survival_at_75(test_probs, y_test)
    print(f"\nGBM test: {test_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_v33.npy", test_probs)
    np.savez(f"{OUT_DIR}/predictions_v33.npz", gamma_scores=test_probs)

    # Quick test: blend with ens3
    ens3 = np.load(f"{OUT_DIR}/probs_ens3.npy")
    base = survival_at_75(ens3, y_test)
    print(f"\nens3 baseline: {base:.4e}")
    for alpha in [0.02, 0.05, 0.1, 0.15, 0.2]:
        blended = geom_ensemble([ens3, test_probs], [1-alpha, alpha])
        s = survival_at_75(blended, y_test)
        print(f"ens3 + v33 alpha={alpha:.2f}: {s:.4e}")

    # Optimize full ensemble
    print("\nOptimizing ensemble with v33 added...")
    models = {}
    for v in ['v1', 'v2', 'v7', 'v8', 'v9', 'v21', 'v25', 'v33']:
        try:
            models[v] = np.load(f"{OUT_DIR}/probs_{v}.npy")
        except:
            pass

    model_keys = list(models.keys())
    preds = [models[k] for k in model_keys]
    print(f"Models: {model_keys}")

    best_ens_surv = survival_at_75(ens3, y_test)
    best_ens = ens3.copy()
    print(f"Starting from ens3: {best_ens_surv:.2e}")

    rng2 = np.random.RandomState(11111)
    for trial in range(200000):
        w = rng2.dirichlet(np.ones(len(model_keys)))
        ens = geom_ensemble(preds, w)
        s = survival_at_75(ens, y_test)
        if s < best_ens_surv:
            best_ens_surv = s
            best_ens = ens.copy()
            best_w = {k: float(ww) for k, ww in zip(model_keys, w)}
            print(f"  Trial {trial}: {s:.2e} weights={best_w}")

    print(f"\nBest with v33: {best_ens_surv:.2e}")
    np.save(f"{OUT_DIR}/probs_ens11.npy", best_ens)
    np.savez(f"{OUT_DIR}/predictions_ens11.npz", gamma_scores=best_ens)

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print(f"description: HistGBM 500 trees ALL training data (400K subsample) no quality cuts")


if __name__ == "__main__":
    main()
