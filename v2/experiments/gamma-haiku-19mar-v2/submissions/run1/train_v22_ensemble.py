#!/usr/bin/env python3
"""Ensemble of v6 with three random seeds (42, 43, 44).

Averages gamma_scores from:
- v6: seed=42 (original)
- v20: seed=43 (different split)
- v21: seed=44 (another different split)

Takes the mean probability across the three runs.
"""

import numpy as np
import os

os.makedirs("submissions/run1", exist_ok=True)

print("Loading probability predictions from three seeds...")

# Load the three seed variants
probs_v6 = np.load("submissions/run1/probs_v6.npy")    # seed=42
probs_v20 = np.load("submissions/run1/probs_v20.npy")  # seed=43
probs_v21 = np.load("submissions/run1/probs_v21.npy")  # seed=44

print(f"v6 shape: {probs_v6.shape}")
print(f"v20 shape: {probs_v20.shape}")
print(f"v21 shape: {probs_v21.shape}")

# Average the gamma scores
ensemble_probs = (probs_v6 + probs_v20 + probs_v21) / 3.0

print(f"Ensemble shape: {ensemble_probs.shape}")
print(f"Ensemble min: {ensemble_probs.min():.6f}, max: {ensemble_probs.max():.6f}")
print(f"Ensemble mean: {ensemble_probs.mean():.6f}, std: {ensemble_probs.std():.6f}")

# Save as predictions.npz
np.savez("submissions/run1/predictions_v22.npz", gamma_scores=ensemble_probs)

print("Saved ensemble predictions to submissions/run1/predictions_v22.npz")
