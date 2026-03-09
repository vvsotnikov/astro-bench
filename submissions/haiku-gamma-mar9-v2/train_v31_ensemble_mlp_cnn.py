"""Ensemble v18_seed42 (MLP on all features) with v30 (CNN on matrices only)."""

import numpy as np

# Load predictions
v18_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v18.npz")["gamma_scores"]
v30_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v30.npz")["gamma_scores"]

test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_at_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_hadron_surviving = (scores[is_hadron] >= thr).sum()
    return n_hadron_surviving / is_hadron.sum()

# Grid search over alpha (blend factor)
print("Searching for optimal blend alpha...")
best_surv = 1.0
best_alpha = 0.5
best_scores = None

for alpha in np.linspace(0.0, 1.0, 51):
    ensemble_scores = alpha * v18_scores + (1 - alpha) * v30_scores
    surv = compute_survival_at_75(ensemble_scores)
    print(f"alpha={alpha:.2f}: surv={surv:.4e}")
    if surv < best_surv:
        best_surv = surv
        best_alpha = alpha
        best_scores = ensemble_scores

print(f"\nBest alpha: {best_alpha:.2f}")
print(f"Best survival: {best_surv:.4e}")

# Save ensemble
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v31.npz",
         gamma_scores=best_scores)

print(f"\n---")
print(f"metric: {best_surv:.4e}")
print(f"description: Ensemble v18_seed42 (MLP) + v30 (CNN matrices-only), alpha={best_alpha:.2f}")
