#!/usr/bin/env python3
"""Gamma/hadron v8: Simple ensemble of best models.

Average the gamma scores from v3, v4, v6 (all had best single model results).
"""

import numpy as np
import os

# Load predictions from best models
print("Loading predictions from v3, v4, v6...")
probs_v3 = np.load("submissions/run1/probs_v3.npy")
probs_v4 = np.load("submissions/run1/probs_v4.npy")
probs_v6 = np.load("submissions/run1/probs_v6.npy")

print(f"  v3: {probs_v3.shape}")
print(f"  v4: {probs_v4.shape}")
print(f"  v6: {probs_v6.shape}")

# Average scores
gamma_scores = (probs_v3 + probs_v4 + probs_v6) / 3.0

# Save predictions
os.makedirs("submissions/run1", exist_ok=True)
np.savez("submissions/run1/predictions_v8.npz", gamma_scores=gamma_scores)
np.save("submissions/run1/probs_v8.npy", gamma_scores)

# Compute survival @ 75% gamma efficiency
labels = np.load("data/gamma_test/labels_gamma.npy")
is_gamma = labels == 0
is_hadron = labels == 1
sg = np.sort(gamma_scores[is_gamma])
ng = len(sg)
thr_75 = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
n_hadron_surviving = (gamma_scores[is_hadron] >= thr_75).sum()
survival_75 = n_hadron_surviving / is_hadron.sum() if is_hadron.sum() > 0 else 1.0

print(f"\nEnsemble survival rate @ 75% gamma efficiency: {survival_75:.2e}")
print(f"Saved to submissions/run1/predictions_v8.npz")
