#!/usr/bin/env python3
"""Ensemble of v3, v4, v6 with weighted averaging.

v3: CNN + MLP fusion, 5.54e-04
v4: ResNet (no proper skip connections), 5.54e-04
v6: CNN with channel attention + MLP fusion, 3.50e-04 [BEST]

Weighted: 0.1 * v3 + 0.1 * v4 + 0.8 * v6
Favor v6 since it's the best.
"""

import numpy as np
import os

os.makedirs("submissions/run1", exist_ok=True)

print("Loading probability predictions...")

probs_v3 = np.load("submissions/run1/probs_v3.npy")
probs_v4 = np.load("submissions/run1/probs_v4.npy")
probs_v6 = np.load("submissions/run1/probs_v6.npy")

print(f"v3 shape: {probs_v3.shape}, mean: {probs_v3.mean():.4f}")
print(f"v4 shape: {probs_v4.shape}, mean: {probs_v4.mean():.4f}")
print(f"v6 shape: {probs_v6.shape}, mean: {probs_v6.mean():.4f}")

# Weighted ensemble: favor v6
ensemble_probs = 0.1 * probs_v3 + 0.1 * probs_v4 + 0.8 * probs_v6

print(f"\nEnsemble shape: {ensemble_probs.shape}")
print(f"Ensemble min: {ensemble_probs.min():.6f}, max: {ensemble_probs.max():.6f}")
print(f"Ensemble mean: {ensemble_probs.mean():.6f}, std: {ensemble_probs.std():.6f}")

# Save as predictions.npz
np.savez("submissions/run1/predictions_v31.npz", gamma_scores=ensemble_probs)

print("Saved ensemble predictions to submissions/run1/predictions_v31.npz")
