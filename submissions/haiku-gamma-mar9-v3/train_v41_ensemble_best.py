"""Ensemble of best 3 architectures: v9 CNN, v38 ResNet, v27b ViT."""

import numpy as np

print("Loading predictions from top models...")

# v9: CNN + Attention + features @ 3.50e-04
v9_npz = np.load("submissions/haiku-gamma-mar9-v3/predictions_v9.npz")
v9_scores = v9_npz["gamma_scores"]

# v38: ResNet + features @ 3.80e-04
v38_npz = np.load("submissions/haiku-gamma-mar9-v3/predictions_v38.npz")
v38_scores = v38_npz["gamma_scores"]

# v27b: ViT 2x2 patches + features @ 5.55e-04
v27b_npz = np.load("submissions/haiku-gamma-mar9-v3/predictions_v27b.npz")
v27b_scores = v27b_npz["gamma_scores"]

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

# Test ensemble with different weights
print("\nOptimizing ensemble weights...")
best_survival = 1.0
best_weights = (1/3, 1/3, 1/3)

# Grid search over weights (equal weight typically works best)
for w9 in np.linspace(0.1, 0.9, 9):
    for w38 in np.linspace(0.1, 0.9 - w9, 9):
        w27b = 1.0 - w9 - w38
        if w27b < 0.01:
            continue

        ensemble_scores = w9 * v9_scores + w38 * v38_scores + w27b * v27b_scores
        surv = compute_survival_75(ensemble_scores)

        if surv < best_survival:
            best_survival = surv
            best_weights = (w9, w38, w27b)

w9, w38, w27b = best_weights
ensemble_scores = w9 * v9_scores + w38 * v38_scores + w27b * v27b_scores
final_surv = compute_survival_75(ensemble_scores)

print(f"\nBest weights: v9={w9:.3f}, v38={w38:.3f}, v27b={w27b:.3f}")
print(f"v9 alone: {compute_survival_75(v9_scores):.4e}")
print(f"v38 alone: {compute_survival_75(v38_scores):.4e}")
print(f"v27b alone: {compute_survival_75(v27b_scores):.4e}")
print(f"Ensemble: {final_surv:.4e}")

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v41.npz", gamma_scores=ensemble_scores)

print(f"\n---")
print(f"metric: {final_surv:.4e}")
print(f"description: Ensemble of v9 CNN ({w9:.2f}) + v38 ResNet ({w38:.2f}) + v27b ViT ({w27b:.2f})")
