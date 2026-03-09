"""Simple physics-based threshold on Ne - Nmu.

From explore_gamma.py:
- Gamma (Ne-Nmu): median 2.65
- Hadron (Ne-Nmu): median 0.96

This is nearly perfectly separated. A simple threshold should work very well.
"""

import numpy as np

# Load test
features = np.load("data/gamma_test/features.npy", mmap_mode="r")[:]
labels = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")[:]

# Extract features
Ne = features[:, 3]
Nmu = features[:, 4]
Ne_minus_Nmu = Ne - Nmu

is_gamma = labels == 0
is_hadron = labels == 1

print(f"Test: {is_gamma.sum()} gamma, {is_hadron.sum()} hadron\n")

# Find 99% gamma efficiency threshold
gamma_scores = Ne_minus_Nmu.copy()
sg = np.sort(gamma_scores[is_gamma])
ng = len(sg)
thr_99 = sg[max(0, int(np.floor(ng * (1 - 0.99))))]

print(f"99% gamma efficiency threshold (Ne-Nmu >= {thr_99:.2f})")
n_gamma_kept = (gamma_scores[is_gamma] >= thr_99).sum()
n_hadron_kept = (gamma_scores[is_hadron] >= thr_99).sum()
print(f"  Gamma kept: {n_gamma_kept}/{is_gamma.sum()} = {n_gamma_kept/is_gamma.sum():.2%}")
print(f"  Hadron kept: {n_hadron_kept}/{is_hadron.sum()} = {n_hadron_kept/is_hadron.sum():.2%}")
print(f"  Survival: {n_hadron_kept/is_hadron.sum():.4f}\n")

# Try different features
features_list = [
    ("Ne", Ne),
    ("Nmu", Nmu),
    ("Ne - Nmu", Ne_minus_Nmu),
    ("Ne / Nmu", Ne / (Nmu + 0.01)),
    ("E (energy)", features[:, 0]),
]

print("=== Testing different features ===")
for fname, scores in features_list:
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.99))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    surv = n_surv / is_hadron.sum()
    print(f"{fname:20s}: threshold={thr:7.2f}, survival={surv:.4f}")

# Use Ne - Nmu as the score
print(f"\nUsing Ne - Nmu as gamma_scores...")
np.savez("submissions/haiku-gamma-mar9/predictions_v6.npz",
         gamma_scores=Ne_minus_Nmu)
print(f"Saved ({len(Ne_minus_Nmu)} scores)")
