"""Weighted ensemble of v9 (CNN) + v20 (ViT) — different inductive biases."""

import numpy as np

print("Loading predictions from v9 and v20...")

# v9: Attention CNN + features @ 3.50e-04
v9_npz = np.load("submissions/haiku-gamma-mar9-v3/predictions_v9.npz")
v9_scores = v9_npz["gamma_scores"]

# v20: Vision Transformer @ 6.72e-04
v20_npz = np.load("submissions/haiku-gamma-mar9-v3/predictions_v20.npz")
v20_scores = v20_npz["gamma_scores"]

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

# Test different weights
print("\nOptimizing weights for v9 + v20 ensemble...")
best_survival = 1.0
best_weights = (1.0, 0.0)

for w9 in np.linspace(0.0, 1.0, 11):
    w20 = 1.0 - w9
    ensemble = w9 * v9_scores + w20 * v20_scores
    surv = compute_survival_75(ensemble)

    if surv < best_survival:
        best_survival = surv
        best_weights = (w9, w20)

w9, w20 = best_weights
ensemble_scores = w9 * v9_scores + w20 * v20_scores
final_surv = compute_survival_75(ensemble_scores)

print(f"\nBest weights: v9={w9:.2f}, v20={w20:.2f}")
print(f"v9 alone: {compute_survival_75(v9_scores):.4e}")
print(f"v20 alone: {compute_survival_75(v20_scores):.4e}")
print(f"Ensemble: {final_surv:.4e}")

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v51.npz", gamma_scores=ensemble_scores)

print(f"\n---")
print(f"metric: {final_surv:.4e}")
print(f"description: Weighted ensemble of v9 CNN ({w9:.2f}) + v20 ViT ({w20:.2f})")
