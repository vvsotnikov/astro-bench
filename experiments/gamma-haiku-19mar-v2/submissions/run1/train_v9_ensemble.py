#!/usr/bin/env python3
"""Gamma/hadron v9: Weighted ensemble giving more weight to v6.

v6 is the best single model, so give it more weight.
"""

import numpy as np
import os

# Load predictions from best models
print("Loading predictions from v3, v4, v6...")
probs_v3 = np.load("submissions/run1/probs_v3.npy")
probs_v4 = np.load("submissions/run1/probs_v4.npy")
probs_v6 = np.load("submissions/run1/probs_v6.npy")

# Weighted average: v6 gets higher weight
gamma_scores = (0.2 * probs_v3 + 0.2 * probs_v4 + 0.6 * probs_v6)

# Save predictions
os.makedirs("submissions/run1", exist_ok=True)
np.savez("submissions/run1/predictions_v9.npz", gamma_scores=gamma_scores)
np.save("submissions/run1/probs_v9.npy", gamma_scores)

# Compute survival @ 75% gamma efficiency
labels = np.load("data/gamma_test/labels_gamma.npy")
is_gamma = labels == 0
is_hadron = labels == 1
sg = np.sort(gamma_scores[is_gamma])
ng = len(sg)
thr_75 = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
n_hadron_surviving = (gamma_scores[is_hadron] >= thr_75).sum()
survival_75 = n_hadron_surviving / is_hadron.sum() if is_hadron.sum() > 0 else 1.0

print(f"\nWeighted ensemble survival rate @ 75% gamma efficiency: {survival_75:.2e}")
print(f"Saved to submissions/run1/predictions_v9.npz")
