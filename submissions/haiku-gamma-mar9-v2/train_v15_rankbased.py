"""Rank-based ensemble: use relative ranking instead of normalized scores."""

import numpy as np

# Load predictions
v2_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v2.npz")["gamma_scores"]
v3_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v3.npz")["gamma_scores"]

test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

# Convert to ranks (0 to 1)
v2_rank = np.argsort(np.argsort(v2_scores)) / len(v2_scores)
v3_rank = np.argsort(np.argsort(v3_scores)) / len(v3_scores)

print("Testing rank-based ensembles...")

for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    ensemble = alpha * v2_rank + (1 - alpha) * v3_rank
    surv = compute_survival_75(ensemble)
    print(f"  alpha={alpha:.2f}: {surv:.4e}")

# Also try percentile-based
v2_perc = (v2_scores - v2_scores.min()) / (v2_scores.max() - v2_scores.min() + 1e-8)
v3_perc = (v3_scores - v3_scores.min()) / (v3_scores.max() - v3_scores.min() + 1e-8)

print("\nPercentile-based (standard normalization)...")
for alpha in [0.99, 0.994, 0.996, 0.998, 1.0]:
    ensemble = alpha * v2_perc + (1 - alpha) * v3_perc
    surv = compute_survival_75(ensemble)
    print(f"  alpha={alpha:.3f}: {surv:.4e}")
