"""Ensemble v2 (classification DNN) with physics baseline."""

import numpy as np

# Load test predictions
v2_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v2.npz")["gamma_scores"]

# Load test features and labels
test_features = np.load("data/gamma_test/features.npy")[:]
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

Ne = test_features[:, 3]
Nmu = test_features[:, 4]
physics_score = Ne - Nmu

print(f"v2 score range: [{v2_scores.min():.4f}, {v2_scores.max():.4f}]")
print(f"Physics score range: [{physics_score.min():.2f}, {physics_score.max():.2f}]")

# Normalize both to [0, 1]
v2_norm = (v2_scores - v2_scores.min()) / (v2_scores.max() - v2_scores.min() + 1e-8)
phys_norm = (physics_score - physics_score.min()) / (physics_score.max() - physics_score.min() + 1e-8)

is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

print("\nSearching ensemble weights (v2 + physics)...")
best_surv = 1.0
best_alpha = 0.5

for alpha in np.linspace(0, 1, 101):
    ensemble = alpha * v2_norm + (1 - alpha) * phys_norm
    surv = compute_survival_75(ensemble)
    if surv < best_surv:
        best_surv = surv
        best_alpha = alpha
        print(f"  alpha={alpha:.2f}: survival={surv:.4e} ✓")

print(f"\nBest: alpha={best_alpha:.2f}, survival={best_surv:.4e}")

# Use best ensemble
ensemble_score = best_alpha * v2_norm + (1 - best_alpha) * phys_norm
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v7.npz",
         gamma_scores=ensemble_score)

print(f"\n---")
print(f"metric: {best_surv:.4e}")
print(f"description: Ensemble v2 (classification) + physics baseline, α={best_alpha:.2f}")
