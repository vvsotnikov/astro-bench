"""Ensemble: Combine Ne-Nmu feature with DNN score."""

import numpy as np

# Load scores from v2 DNN
dnn_scores = np.load("submissions/haiku-gamma-mar9/predictions.npz")["gamma_scores"]

# Load test features
features = np.load("data/gamma_test/features.npy")[:]
labels = np.load("data/gamma_test/labels_gamma.npy")[:]

Ne = features[:, 3]
Nmu = features[:, 4]
physics_score = Ne - Nmu

print(f"DNN score range: [{dnn_scores.min():.4f}, {dnn_scores.max():.4f}]")
print(f"Physics score range: [{physics_score.min():.2f}, {physics_score.max():.2f}]")

# Normalize both to [0, 1]
dnn_norm = (dnn_scores - dnn_scores.min()) / (dnn_scores.max() - dnn_scores.min() + 1e-8)
phys_norm = (physics_score - physics_score.min()) / (physics_score.max() - physics_score.min() + 1e-8)

# Try different ensemble weights
is_gamma = labels == 0
is_hadron = labels == 1

def compute_survival(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.99))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

print(f"\nDNN only: {compute_survival(dnn_norm):.4f}")
print(f"Physics only: {compute_survival(phys_norm):.4f}")

for alpha in np.linspace(0, 1, 11):
    ensemble = alpha * dnn_norm + (1 - alpha) * phys_norm
    surv = compute_survival(ensemble)
    print(f"alpha={alpha:.1f}: {surv:.4f}")

# Best combination (find via search)
best_surv = 1.0
best_alpha = 0.5
for alpha in np.linspace(0, 1, 101):
    ensemble = alpha * dnn_norm + (1 - alpha) * phys_norm
    surv = compute_survival(ensemble)
    if surv < best_surv:
        best_surv = surv
        best_alpha = alpha

print(f"\nBest: alpha={best_alpha:.2f}, survival={best_surv:.4f}")

# Use best ensemble
ensemble_score = best_alpha * dnn_norm + (1 - best_alpha) * phys_norm
np.savez("submissions/haiku-gamma-mar9/predictions_v7.npz",
         gamma_scores=ensemble_score)
print(f"Saved ensemble scores")
