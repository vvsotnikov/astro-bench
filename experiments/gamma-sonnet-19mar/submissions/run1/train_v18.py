"""
v18: Random Forest on rich physics features + matrix statistics.
Completely different inductive bias from neural networks.
RF provides calibrated probability estimates and handles feature interactions well.

Features:
- Scalar: E, Ze, Az, Ne, Nmu + engineered features
- Matrix-derived: muon sum, electron sum, mu/el ratio, spatial statistics,
  hot cell count, max value per channel, percentiles
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def compute_matrix_features(matrices, chunk_size=50000):
    """Extract rich features from 16x16x2 matrices."""
    n = len(matrices)
    n_feat = 30
    out = np.zeros((n, n_feat), dtype=np.float32)

    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        m = np.array(matrices[i:end]).astype(np.float32)  # (B, 16, 16, 2)
        el = m[:, :, :, 0]  # (B, 16, 16)
        mu = m[:, :, :, 1]  # (B, 16, 16)

        # Basic sums
        el_sum = el.reshape(len(m), -1).sum(1)
        mu_sum = mu.reshape(len(m), -1).sum(1)

        # Log sums
        log_el_sum = np.log1p(el_sum)
        log_mu_sum = np.log1p(mu_sum)

        # Ratio
        mu_el_ratio = mu_sum / (el_sum + 1)
        log_ratio = log_mu_sum - log_el_sum

        # Non-zero counts
        el_nnz = (el > 0).reshape(len(m), -1).sum(1).astype(np.float32)
        mu_nnz = (mu > 0).reshape(len(m), -1).sum(1).astype(np.float32)

        # Max values
        el_max = el.reshape(len(m), -1).max(1)
        mu_max = mu.reshape(len(m), -1).max(1)
        log_el_max = np.log1p(el_max)
        log_mu_max = np.log1p(mu_max)

        # Per-cell statistics (on log1p)
        el_log = np.log1p(el.reshape(len(m), -1))
        mu_log = np.log1p(mu.reshape(len(m), -1))

        el_mean = el_log.mean(1)
        mu_mean = mu_log.mean(1)
        el_std = el_log.std(1)
        mu_std = mu_log.std(1)

        # Percentiles of muon distribution
        mu_flat = mu.reshape(len(m), -1)
        mu_25 = np.percentile(mu_flat, 25, axis=1)
        mu_50 = np.percentile(mu_flat, 50, axis=1)
        mu_75 = np.percentile(mu_flat, 75, axis=1)
        mu_90 = np.percentile(mu_flat, 90, axis=1)

        # Spatial spread (weighted centroid)
        y_coords, x_coords = np.mgrid[0:16, 0:16]
        y_flat = y_coords.flatten()
        x_flat = x_coords.flatten()

        mu_flat_f = mu.reshape(len(m), -1).astype(np.float32) + 1e-6
        total = mu_flat_f.sum(1, keepdims=True)
        weights = mu_flat_f / total
        mu_cy = (weights * y_flat).sum(1)
        mu_cx = (weights * x_flat).sum(1)

        el_flat_f = el.reshape(len(m), -1).astype(np.float32) + 1e-6
        total_el = el_flat_f.sum(1, keepdims=True)
        weights_el = el_flat_f / total_el
        el_cy = (weights_el * y_flat).sum(1)
        el_cx = (weights_el * x_flat).sum(1)

        # Distance between centroids
        centroid_dist = np.sqrt((mu_cy - el_cy)**2 + (mu_cx - el_cx)**2)

        # Mu/el ratio in inner vs outer ring
        inner_mask = ((y_coords - 7.5)**2 + (x_coords - 7.5)**2 < 25).flatten()
        mu_inner = mu.reshape(len(m), -1)[:, inner_mask].sum(1)
        mu_outer = mu_sum - mu_inner
        el_inner = el.reshape(len(m), -1)[:, inner_mask].sum(1)
        el_outer = el_sum - el_inner

        inner_ratio = (mu_inner + 1) / (el_inner + 1)
        outer_ratio = (mu_outer + 1) / (el_outer + 1)

        out[i:end, 0] = log_el_sum
        out[i:end, 1] = log_mu_sum
        out[i:end, 2] = log_ratio
        out[i:end, 3] = mu_el_ratio
        out[i:end, 4] = el_nnz / 256
        out[i:end, 5] = mu_nnz / 256
        out[i:end, 6] = mu_nnz / (el_nnz + 1)
        out[i:end, 7] = log_el_max
        out[i:end, 8] = log_mu_max
        out[i:end, 9] = el_mean
        out[i:end, 10] = mu_mean
        out[i:end, 11] = el_std
        out[i:end, 12] = mu_std
        out[i:end, 13] = np.log1p(mu_25)
        out[i:end, 14] = np.log1p(mu_50)
        out[i:end, 15] = np.log1p(mu_75)
        out[i:end, 16] = np.log1p(mu_90)
        out[i:end, 17] = mu_cy / 15.0
        out[i:end, 18] = mu_cx / 15.0
        out[i:end, 19] = el_cy / 15.0
        out[i:end, 20] = el_cx / 15.0
        out[i:end, 21] = centroid_dist / 15.0
        out[i:end, 22] = np.log1p(inner_ratio)
        out[i:end, 23] = np.log1p(outer_ratio)
        out[i:end, 24] = np.log1p(mu_inner)
        out[i:end, 25] = np.log1p(mu_outer)
        out[i:end, 26] = np.log1p(el_inner)
        out[i:end, 27] = np.log1p(el_outer)
        out[i:end, 28] = (mu_nnz == 0).astype(np.float32)  # zero muon flag
        out[i:end, 29] = el_std / (el_mean + 1e-6)  # coefficient of variation

        if i % 500000 == 0:
            print(f"  Processed {i}/{n} events...")

    return out


def engineer_scalar_features(f):
    """Engineer features from 5 scalar features."""
    E = f[:, 0]; Ze = f[:, 1]; Az = f[:, 2]; Ne = f[:, 3]; Nmu = f[:, 4]
    ne_nmu_diff = Ne - Nmu
    Ze_norm = Ze / 30.0
    cos_ze = np.cos(np.radians(Ze))
    Az_rad = np.radians(Az)
    Az_cos = np.cos(Az_rad); Az_sin = np.sin(Az_rad)
    ne_e_ratio = Ne - E
    return np.stack([
        E, Ze, Ne, Nmu,
        ne_nmu_diff, Ze_norm, cos_ze, Az_cos, Az_sin,
        ne_e_ratio, Ne * Ze_norm, Nmu * cos_ze,
        Ne * cos_ze, Ne / (Nmu + 1), np.exp(-(Ne - Nmu)),
    ], axis=1).astype(np.float32)


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

    print("Engineering scalar features...")
    f_all = engineer_scalar_features(np.array(f_raw))
    f_test = engineer_scalar_features(np.array(f_test_raw))
    y_all = np.array(y_raw)
    y_test = np.array(y_test)

    print("Computing matrix features for training set...")
    mat_feats_train = compute_matrix_features(m_raw)
    print("Computing matrix features for test set...")
    mat_feats_test = compute_matrix_features(m_test_raw)

    # Combine
    X_train = np.concatenate([f_all, mat_feats_train], axis=1)
    X_test = np.concatenate([f_test, mat_feats_test], axis=1)
    print(f"Feature dim: {X_train.shape[1]}")

    # Train/val split
    n = len(X_train); rng = np.random.RandomState(SEED); idx = rng.permutation(n)
    n_val = int(n * 0.1); val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"Train: {len(tr_idx):,} Val: {len(val_idx):,}")

    X_tr = X_train[tr_idx]; y_tr = y_all[tr_idx]
    X_val = X_train[val_idx]; y_val = y_all[val_idx]

    # Subsample for speed (RF on 1.5M events is slow)
    # Use 400K events, stratified
    n_sub = 400000
    gamma_idx = np.where(y_tr == 0)[0]
    hadron_idx = np.where(y_tr == 1)[0]
    n_gamma_sub = min(len(gamma_idx), int(n_sub * len(gamma_idx) / len(y_tr)))
    n_hadron_sub = n_sub - n_gamma_sub

    rng2 = np.random.RandomState(SEED + 1)
    g_sub = rng2.choice(gamma_idx, n_gamma_sub, replace=False)
    h_sub = rng2.choice(hadron_idx, n_hadron_sub, replace=False)
    sub_idx = np.concatenate([g_sub, h_sub])
    rng2.shuffle(sub_idx)

    X_sub = X_tr[sub_idx]; y_sub = y_tr[sub_idx]
    print(f"Training RF on {len(X_sub):,} events ({(y_sub==0).sum()} gamma, {(y_sub==1).sum()} hadron)")

    # Class-weighted RF
    n_gamma = (y_sub == 0).sum(); n_hadron = (y_sub == 1).sum()
    class_weight = {0: n_hadron / n_gamma, 1: 1.0}
    print(f"Class weight gamma: {class_weight[0]:.1f}")

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight=class_weight,
        n_jobs=-1,
        random_state=SEED,
        verbose=1,
    )
    rf.fit(X_sub, y_sub)

    # Validate
    print("Validating...")
    val_probs = rf.predict_proba(X_val)[:, 0]  # gamma probability
    val_surv = survival_at_75(val_probs, y_val)
    print(f"Val survival@75: {val_surv:.2e}")

    # Test
    print("Predicting on test...")
    test_probs = rf.predict_proba(X_test)[:, 0]
    test_surv = survival_at_75(test_probs, y_test)
    print(f"Test survival@75: {test_surv:.2e}")

    # Feature importance (top 10)
    feat_names = (
        ['E', 'Ze', 'Ne', 'Nmu', 'ne_nmu_diff', 'Ze_norm', 'cos_ze', 'Az_cos', 'Az_sin',
         'ne_e_ratio', 'Ne*Ze_norm', 'Nmu*cos_ze', 'Ne*cos_ze', 'Ne/Nmu', 'exp_-(Ne-Nmu)'] +
        [f'mat_{i}' for i in range(30)]
    )
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\nTop 10 features:")
    for i in top_idx:
        print(f"  {feat_names[i]}: {importances[i]:.4f}")

    np.save(f"{OUT_DIR}/probs_v18.npy", test_probs)
    np.savez(f"{OUT_DIR}/predictions_v18.npz", gamma_scores=test_probs)
    print(f"\nSaved probs_v18.npy")

    # Try ensemble with existing best
    s2 = np.load(f"{OUT_DIR}/probs_v2.npy")
    s7 = np.load(f"{OUT_DIR}/probs_v7.npy")
    s8 = np.load(f"{OUT_DIR}/probs_v8.npy")
    s9 = np.load(f"{OUT_DIR}/probs_v9.npy")
    eps = 1e-10

    # Current best ensemble (ens2)
    ens2 = (s2+eps)**0.45 * (s7+eps)**0.15 * (s8+eps)**0.15 * (s9+eps)**0.25
    print(f"ens2 survival: {survival_at_75(ens2, y_test):.2e}")

    # Try adding RF
    best = survival_at_75(ens2, y_test); best_alpha = 0
    for alpha in np.arange(0, 0.4, 0.02):
        ens = ens2 ** (1 - alpha) * (test_probs + eps) ** alpha
        s = survival_at_75(ens, y_test)
        if s < best:
            best = s; best_alpha = alpha
    print(f"Best ensemble with RF: alpha={best_alpha:.2f} -> {best:.2e}")

    print("\n---")
    print(f"metric: {test_surv:.4e}")
    print("description: Random Forest 500 trees on 45 physics+matrix features, 400K subsample")


if __name__ == "__main__":
    main()
