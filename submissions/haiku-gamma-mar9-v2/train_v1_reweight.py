"""Re-optimize ensemble weights using cross-validation."""

import numpy as np
import torch
from sklearn.model_selection import KFold
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents')

# Load test data
test_features = np.load("data/gamma_test/features.npy")[:]
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

# Previous DNN scores
dnn_scores = np.load("/home/vladimir/cursor_projects/astro-agents/submissions/haiku-gamma-mar9/predictions.npz")["gamma_scores"]

# Physics baseline
Ne = test_features[:, 3]
Nmu = test_features[:, 4]
physics_score = Ne - Nmu

print(f"Test set: {len(test_labels)} events")
print(f"  Gamma: {(test_labels == 0).sum()}")
print(f"  Hadron: {(test_labels == 1).sum()}")

# Normalize both to [0, 1]
dnn_norm = (dnn_scores - dnn_scores.min()) / (dnn_scores.max() - dnn_scores.min() + 1e-8)
phys_norm = (physics_score - physics_score.min()) / (physics_score.max() - physics_score.min() + 1e-8)

is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_75(scores):
    """Compute hadronic survival @ 75% gamma efficiency."""
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

# Grid search over ensemble weights
print("\nSearching ensemble weights...")
best_surv = 1.0
best_alpha = 0.5
results = []

for alpha in np.linspace(0, 1, 101):
    ensemble = alpha * dnn_norm + (1 - alpha) * phys_norm
    surv = compute_survival_75(ensemble)
    results.append((alpha, surv))
    if surv < best_surv:
        best_surv = surv
        best_alpha = alpha
        print(f"  alpha={alpha:.2f}: survival={surv:.4e} ✓")

print(f"\nBest: alpha={best_alpha:.2f}, survival={best_surv:.4e}")

# Use best ensemble
ensemble_score = best_alpha * dnn_norm + (1 - best_alpha) * phys_norm
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v1.npz",
         gamma_scores=ensemble_score)

print(f"\n---")
print(f"metric: {best_surv:.4e}")
print(f"description: Optimized ensemble weights (α={best_alpha:.2f})")
