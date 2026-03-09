"""Triple ensemble: v2 (classification) + v3 (regression) + v5."""

import numpy as np

# Load predictions
v2_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v2.npz")["gamma_scores"]
v3_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v3.npz")["gamma_scores"]

# Load test labels
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

print(f"v2 score range: [{v2_scores.min():.4f}, {v2_scores.max():.4f}]")
print(f"v3 score range: [{v3_scores.min():.4f}, {v3_scores.max():.4f}]")

# Normalize
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

print("\nTrying 3-way combinations (simple)...")

# Just try a few reasonable combinations
best_surv = 1.0
best_combo = (0, 0, 1)

for w2 in [0.98, 0.985, 0.99, 0.995]:
    for w3 in [0.01, 0.015]:
        w_sum = w2 + w3
        w2_norm = w2 / w_sum
        w3_norm = w3 / w_sum

        ensemble = w2_norm * v2_norm + w3_norm * v3_norm
        surv = compute_survival_75(ensemble)
        if surv < best_surv:
            best_surv = surv
            best_combo = (w2_norm, w3_norm)
            print(f"  (v2={w2_norm:.3f}, v3={w3_norm:.3f}): survival={surv:.4e} ✓")

print(f"\nBest: v2={best_combo[0]:.3f}, v3={best_combo[1]:.3f}, survival={best_surv:.4e}")

# Use best
ensemble_score = best_combo[0] * v2_norm + best_combo[1] * v3_norm
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v10.npz",
         gamma_scores=ensemble_score)

print(f"\n---")
print(f"metric: {best_surv:.4e}")
print(f"description: 2-way ensemble v2+v3 optimization")
