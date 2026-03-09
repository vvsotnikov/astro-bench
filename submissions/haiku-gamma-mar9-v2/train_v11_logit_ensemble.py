"""Try different ensemble combinations: logit averaging, geometric mean, etc."""

import numpy as np

# Load predictions
v2_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v2.npz")["gamma_scores"]
v3_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v3.npz")["gamma_scores"]

# Load test labels
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

# Clamp to avoid log(0)
v2_clamped = np.clip(v2_scores, 1e-6, 1-1e-6)
v3_clamped = np.clip(v3_scores, 1e-6, 1-1e-6)

# Try different ensemble schemes
methods = {}

# Simple average (baseline)
methods['mean'] = (v2_scores + v3_scores) / 2

# Normalized average
v2_norm = (v2_scores - v2_scores.min()) / (v2_scores.max() - v2_scores.min() + 1e-8)
v3_norm = (v3_scores - v3_scores.min()) / (v3_scores.max() - v3_scores.min() + 1e-8)
methods['norm_mean'] = (v2_norm + v3_norm) / 2

# Geometric mean (harmonic mean with log)
methods['geom_mean'] = np.sqrt(v2_scores * v3_scores)

# Multiplicative
methods['mult'] = v2_scores * v3_scores

# Min (AND logic - both must agree it's gamma)
methods['min'] = np.minimum(v2_scores, v3_scores)

# Max (OR logic)
methods['max'] = np.maximum(v2_scores, v3_scores)

print("Testing different ensemble methods...")
for method, scores in methods.items():
    surv = compute_survival_75(scores)
    print(f"  {method:15s}: {surv:.4e}")

# Find best
best_method = min(methods.keys(), key=lambda k: compute_survival_75(methods[k]))
best_surv = compute_survival_75(methods[best_method])

print(f"\nBest: {best_method}, survival={best_surv:.4e}")

# Save
np.savez(f"submissions/haiku-gamma-mar9-v2/predictions_v11_{best_method}.npz",
         gamma_scores=methods[best_method])

print(f"\n---")
print(f"metric: {best_surv:.4e}")
print(f"description: Ensemble method {best_method}")
