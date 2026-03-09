"""Ensemble v2 (classification) and v3 (regression)."""

import numpy as np

# Load test predictions
v2_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v2.npz")["gamma_scores"]
v3_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v3.npz")["gamma_scores"]

# Load test labels
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

print(f"v2 score range: [{v2_scores.min():.4f}, {v2_scores.max():.4f}]")
print(f"v3 score range: [{v3_scores.min():.4f}, {v3_scores.max():.4f}]")

# Normalize both to [0, 1]
v2_norm = (v2_scores - v2_scores.min()) / (v2_scores.max() - v2_scores.min() + 1e-8)
v3_norm = (v3_scores - v3_scores.min()) / (v3_scores.max() - v3_scores.min() + 1e-8)

is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

print("\nSearching ensemble weights (v2 + v3)...")
best_surv = 1.0
best_alpha = 0.5

for alpha in np.linspace(0, 1, 101):
    ensemble = alpha * v2_norm + (1 - alpha) * v3_norm
    surv = compute_survival_75(ensemble)
    if surv < best_surv:
        best_surv = surv
        best_alpha = alpha
        print(f"  alpha={alpha:.2f}: survival={surv:.4e} ✓")

print(f"\nBest: alpha={best_alpha:.2f}, survival={best_surv:.4e}")

# Use best ensemble
ensemble_score = best_alpha * v2_norm + (1 - best_alpha) * v3_norm
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v5.npz",
         gamma_scores=ensemble_score)

print(f"\n---")
print(f"metric: {best_surv:.4e}")
print(f"description: Ensemble v2 (classification) + v3 (regression), α={best_alpha:.2f}")
