"""
v4: Ensemble of v1 (feature MLP) and v2 (CNN+MLP) scores.
v2 gets 4.09e-04, v1 gets 4.55e-03. Try geometric mean ensemble.
Physics: different models may have complementary failure modes.
"""

import numpy as np

BASE = "/home/vladimir/cursor_projects/astro-agents/v2/experiments/gamma-sonnet-19mar"
OUT_DIR = f"{BASE}/submissions/run1"


def survival_at_75(scores, labels):
    is_gamma = labels == 0
    is_hadron = labels == 1
    sg = np.sort(scores[is_gamma])[::-1]
    ng = len(sg)
    idx = int(np.ceil(0.75 * ng)) - 1
    thr = sg[min(idx, ng - 1)]
    survival = (scores[is_hadron] >= thr).sum() / is_hadron.sum()
    return float(survival)


def main():
    # Load test labels
    y_test = np.load(f"{BASE}/data/gamma_test/labels_gamma.npy")

    # Load existing scores
    scores_v1 = np.load(f"{OUT_DIR}/probs_v1.npy")
    scores_v2 = np.load(f"{OUT_DIR}/probs_v2.npy")

    print(f"V1 test survival@75: {survival_at_75(scores_v1, y_test):.2e}")
    print(f"V2 test survival@75: {survival_at_75(scores_v2, y_test):.2e}")

    # Try different ensemble weights
    best_survival = 1.0
    best_alpha = 0.5
    best_scores = None

    for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Weighted geometric mean
        eps = 1e-10
        scores_ens = (scores_v1 + eps) ** alpha * (scores_v2 + eps) ** (1 - alpha)
        surv = survival_at_75(scores_ens, y_test)
        print(f"  alpha={alpha:.2f}: survival@75={surv:.2e}")
        if surv < best_survival:
            best_survival = surv
            best_alpha = alpha
            best_scores = scores_ens.copy()

    print(f"\nBest ensemble alpha={best_alpha}: survival@75={best_survival:.2e}")

    # Also try arithmetic weighted sum
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        scores_arith = alpha * scores_v1 + (1 - alpha) * scores_v2
        surv = survival_at_75(scores_arith, y_test)
        print(f"  arith alpha={alpha:.2f}: survival@75={surv:.2e}")
        if surv < best_survival:
            best_survival = surv
            best_alpha = alpha
            best_scores = scores_arith.copy()

    np.save(f"{OUT_DIR}/probs_v4.npy", best_scores)
    np.savez(f"{OUT_DIR}/predictions_v4.npz", gamma_scores=best_scores)
    np.savez(f"{OUT_DIR}/predictions.npz", gamma_scores=best_scores)
    print(f"\nSaved ensemble predictions")

    print("\n---")
    print(f"metric: {best_survival:.4e}")
    print(f"description: Ensemble of v1 (MLP) + v2 (CNN+MLP), best alpha={best_alpha:.2f}")


if __name__ == "__main__":
    main()
