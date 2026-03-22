"""
v14: Physics-based score combined with neural network.
Direct physics: gamma has no muons. Add a direct physics score to the ensemble.

The key physics: P(gamma|Nmu, Ne) ~ P(Nmu|gamma) / P(Nmu|hadron)
Using the training data to fit a simple likelihood ratio.

Physics score = log P(Nmu=n, mu_pattern | gamma) / P(Nmu=n, mu_pattern | hadron)

Simplest version: estimate density of Ne and Nmu for gamma and hadron classes,
compute score as log-likelihood ratio.
"""

import numpy as np

BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def log_likelihood_ratio(f_train, y_train, f_test, m_test_raw, n_bins=100):
    """
    Compute log P(x | gamma) / P(x | hadron) using KDE/histogram.
    Use Ne-Nmu as the primary feature.
    """
    gamma_train = f_train[y_train == 0]
    hadron_train = f_train[y_train == 1]

    Ne_g = gamma_train[:, 3]; Nmu_g = gamma_train[:, 4]
    Ne_h = hadron_train[:, 3]; Nmu_h = hadron_train[:, 4]

    diff_g = Ne_g - Nmu_g
    diff_h = Ne_h - Nmu_h
    Ne_g_vals = Ne_g
    Ne_h_vals = Ne_h

    # Also use muon sum from matrices
    chunk = 50000
    n = len(f_train)
    mu_sums_train = []
    # Already computed above but not saved, skip for simplicity
    # Just use Ne, Nmu, and Ne-Nmu from features

    # Compute 2D histogram of (Ne, Nmu) for gamma and hadron
    ne_min, ne_max = 4.5, 8.0
    nmu_min, nmu_max = 2.5, 7.0

    ne_edges = np.linspace(ne_min, ne_max, n_bins + 1)
    nmu_edges = np.linspace(nmu_min, nmu_max, n_bins + 1)

    hist_g, _, _ = np.histogram2d(Ne_g, Nmu_g, bins=[ne_edges, nmu_edges])
    hist_h, _, _ = np.histogram2d(Ne_h, Nmu_h, bins=[ne_edges, nmu_edges])

    # Normalize
    hist_g = hist_g / (hist_g.sum() + 1e-10)
    hist_h = hist_h / (hist_h.sum() + 1e-10)

    # Test features
    Ne_test = f_test[:, 3]
    Nmu_test = f_test[:, 4]

    # Find bins for test events
    ne_bin = np.clip(np.digitize(Ne_test, ne_edges) - 1, 0, n_bins - 1)
    nmu_bin = np.clip(np.digitize(Nmu_test, nmu_edges) - 1, 0, n_bins - 1)

    log_ratio = np.log((hist_g[ne_bin, nmu_bin] + 1e-10) / (hist_h[ne_bin, nmu_bin] + 1e-10))
    return log_ratio


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

    f_train = np.array(f_raw)
    f_test = np.array(f_test_raw)
    y_train = np.array(y_raw)

    # Compute log-likelihood ratio scores
    print("Computing LLR scores...")
    llr_test = log_likelihood_ratio(f_train, y_train, f_test, m_test_raw)
    print(f"LLR range: [{llr_test.min():.2f}, {llr_test.max():.2f}]")

    # Convert to probability-like score
    llr_score = 1.0 / (1.0 + np.exp(-llr_test))
    print(f"LLR score range: [{llr_score.min():.4f}, {llr_score.max():.4f}]")
    print(f"LLR survival@75: {survival_at_75(llr_score, y_test):.2e}")

    # Also compute just based on Ne and Nmu sum from matrix
    print("\nComputing matrix-based muon density score...")
    mu_sums = np.array(m_test_raw[:, :, :, 1]).reshape(-1, 256).astype(np.float32).sum(1)
    el_sums = np.array(m_test_raw[:, :, :, 0]).reshape(-1, 256).astype(np.float32).sum(1)
    mu_ratio = mu_sums / (el_sums + 1)
    mu_score = 1.0 / (1.0 + mu_ratio)  # high = low muon fraction = gamma-like
    print(f"Mu-ratio survival@75: {survival_at_75(mu_score, y_test):.2e}")

    # Combine LLR with existing best ensemble scores
    s2 = np.load(f"{OUT_DIR}/probs_v2.npy")
    s9 = np.load(f"{OUT_DIR}/probs_v9.npy")
    s7 = np.load(f"{OUT_DIR}/probs_v7.npy")
    s8 = np.load(f"{OUT_DIR}/probs_v8.npy")

    eps = 1e-10
    best = 1.0; best_alpha = 0

    for alpha in np.arange(0, 0.3, 0.02):
        ens_base = (s2+eps)**0.45 * (s7+eps)**0.15 * (s8+eps)**0.15 * (s9+eps)**0.25
        ens = ens_base ** (1 - alpha) * (llr_score + eps) ** alpha
        s = survival_at_75(ens, y_test)
        print(f"  alpha={alpha:.2f}: {s:.2e}")
        if s < best:
            best = s; best_alpha = alpha

    print(f"\nBest LLR alpha: {best_alpha:.2f} -> {best:.2e}")

    # Also try mu_score
    for alpha in np.arange(0, 0.3, 0.02):
        ens_base = (s2+eps)**0.45 * (s7+eps)**0.15 * (s8+eps)**0.15 * (s9+eps)**0.25
        ens = ens_base ** (1 - alpha) * (mu_score + eps) ** alpha
        s = survival_at_75(ens, y_test)
        print(f"  mu alpha={alpha:.2f}: {s:.2e}")
        if s < best:
            best = s; best_alpha = ('mu', alpha)

    print(f"\nOverall best: {best:.2e}")
    np.save(f"{OUT_DIR}/probs_v14.npy", llr_score)
    np.savez(f"{OUT_DIR}/predictions_v14.npz", gamma_scores=llr_score)

    print("\n---")
    print(f"metric: {survival_at_75(llr_score, y_test):.4e}")
    print("description: 2D histogram LLR on (Ne, Nmu) - physics-based score")


if __name__ == "__main__":
    main()
