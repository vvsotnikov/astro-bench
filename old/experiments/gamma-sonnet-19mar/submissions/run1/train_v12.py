"""
v12: Gradient Boosting (sklearn GBM) on rich feature set.
Very different inductive bias from CNNs - might catch different patterns.
Use matrix statistics + scalar features.
GBM can model feature interactions differently from neural nets.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def extract_matrix_stats(matrices):
    el = matrices[:, :, :, 0].astype(np.float32)
    mu = matrices[:, :, :, 1].astype(np.float32)
    n = len(el)

    el_flat = el.reshape(n, -1)
    mu_flat = mu.reshape(n, -1)

    log_el_sum = np.log1p(el_flat.sum(1))
    log_mu_sum = np.log1p(mu_flat.sum(1))
    log_el_max = np.log1p(el_flat.max(1))
    log_mu_max = np.log1p(mu_flat.max(1))
    el_nnz = (el_flat > 0).sum(1).astype(np.float32)
    mu_nnz = (mu_flat > 0).sum(1).astype(np.float32)
    mu_el_ratio = log_mu_sum / (log_el_sum + 1e-6)

    # Center of mass for muons
    grid_y = np.repeat(np.arange(16), 16).astype(np.float32)
    grid_x = np.tile(np.arange(16), 16).astype(np.float32)
    mu_w = mu_flat + 1e-8
    total_mu = mu_w.sum(1)
    cx = (mu_w * grid_x).sum(1) / total_mu
    cy = (mu_w * grid_y).sum(1) / total_mu
    var_x = (mu_w * (grid_x - cx[:, None]) ** 2).sum(1) / total_mu
    var_y = (mu_w * (grid_y - cy[:, None]) ** 2).sum(1) / total_mu
    mu_spread = np.sqrt(var_x + var_y + 1e-8)

    # Muon percentiles
    mu_sorted = np.sort(mu_flat, axis=1)[:, ::-1]
    top5_mu = mu_sorted[:, :5].sum(1)  # top 5 cells
    top10_mu = mu_sorted[:, :10].sum(1)
    frac_top5 = top5_mu / (mu_flat.sum(1) + 1e-8)  # concentration

    # Center region
    mu_center = mu[:, 5:11, 5:11].reshape(n, -1).sum(1)
    mu_outer = mu_flat.sum(1) - mu_center
    mu_center_frac = mu_center / (mu_flat.sum(1) + 1e-6)

    return np.stack([
        log_el_sum, log_mu_sum, log_el_max, log_mu_max,
        el_nnz / 256.0, mu_nnz / 256.0,
        mu_el_ratio, mu_spread, cx, cy,
        frac_top5, np.log1p(top5_mu), np.log1p(top10_mu),
        mu_center_frac,
    ], axis=1).astype(np.float32)


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
    ne_nmu_sq = ne_nmu_diff ** 2

    return np.stack([
        E_norm, Ze_norm, Az_cos, Az_sin, Ne_norm, Nmu_norm,
        ne_nmu_diff, cos_ze, ne_e_ratio, Ne * Ze_norm, Nmu * cos_ze,
        ne_nmu_sq,
    ], axis=1).astype(np.float32)


def main():
    print("Loading data...")
    f_raw = np.load(f"{BASE}/data/gamma_train/features.npy", mmap_mode='r')
    m_raw = np.load(f"{BASE}/data/gamma_train/matrices.npy", mmap_mode='r')
    y_raw = np.load(f"{BASE}/data/gamma_train/labels_gamma.npy", mmap_mode='r')
    f_test_raw = np.load(f"{BASE}/data/gamma_test/features.npy", mmap_mode='r')
    m_test_raw = np.load(f"{BASE}/data/gamma_test/matrices.npy", mmap_mode='r')
    y_test = np.load(f"{BASE}/data/gamma_test/labels_gamma.npy", mmap_mode='r')

    # Use a subsample for GBM (can't fit 1.5M events)
    np.random.seed(42)
    n_total = len(f_raw)
    n_sample = 200000  # use 200K events
    idx = np.random.choice(n_total, n_sample, replace=False)
    print(f"Subsampling to {n_sample:,} events")

    print("Extracting features...")
    f_sub = np.array(f_raw[idx])
    m_sub = np.array(m_raw[idx])
    y_sub = np.array(y_raw[idx])

    mat_stats_sub = extract_matrix_stats(m_sub)
    scalar_sub = engineer_features(f_sub)
    X_sub = np.concatenate([scalar_sub, mat_stats_sub], axis=1)
    print(f"Feature dim: {X_sub.shape[1]}")
    print(f"Gamma: {(y_sub==0).sum():,}, Hadron: {(y_sub==1).sum():,}")

    # Test features
    mat_stats_test = extract_matrix_stats(np.array(m_test_raw))
    scalar_test = engineer_features(np.array(f_test_raw))
    X_test = np.concatenate([scalar_test, mat_stats_test], axis=1)

    # Sample weights
    n_g = (y_sub == 0).sum(); n_h = (y_sub == 1).sum()
    w_g = n_h / n_g
    sample_weights = np.where(y_sub == 0, w_g, 1.0)

    print(f"Training GBM with n_estimators=300, max_depth=5...")
    gbm = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.8, max_features='sqrt',
        random_state=42, verbose=1,
        validation_fraction=0.1, n_iter_no_change=15,
    )
    gbm.fit(X_sub, y_sub, sample_weight=sample_weights)

    # Get gamma probability scores
    test_probs = gbm.predict_proba(X_test)[:, 0]  # P(gamma)
    print(f"Predicted gamma scores range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")

    # Compute survival
    ig = y_test == 0; ih = y_test == 1
    sg = np.sort(test_probs[ig])[::-1]; ng = len(sg)
    thr = sg[min(int(np.ceil(0.75*ng))-1, ng-1)]
    survival = float((test_probs[ih] >= thr).sum() / ih.sum())
    print(f"Test survival@75: {survival:.2e}")

    np.save(f"{OUT_DIR}/probs_v12.npy", test_probs)
    np.savez(f"{OUT_DIR}/predictions_v12.npz", gamma_scores=test_probs)

    print("\n---")
    print(f"metric: {survival:.4e}")
    print("description: GBM on matrix stats + scalar features, n=200K subsample, n_est=300")


if __name__ == "__main__":
    main()
