#!/usr/bin/env python3
"""Gamma/hadron classifier: Simple Nmu threshold baseline.

Physics-driven approach: gammas have median Nmu ≈ 3.07, hadrons ≈ 3.54.
Try a simple threshold and compare to neural network.
"""

import numpy as np

# Load test data
f_test = np.load('data/gamma_test/features.npy', mmap_mode='r')
y_test = np.load('data/gamma_test/labels_gamma.npy', mmap_mode='r')

print("=" * 80)
print("SIMPLE NMU THRESHOLD BASELINE")
print("=" * 80)

# Nmu is at index 4 (0-indexed: E, Ze, Az, Ne, Nmu)
Nmu_test = f_test[:, 4]

print(f"\nNmu statistics:")
gamma_mask = y_test == 0
hadron_mask = y_test == 1

print(f"  Gamma  Nmu: median={np.median(Nmu_test[gamma_mask]):.3f}, "
      f"mean={Nmu_test[gamma_mask].mean():.3f}, "
      f"std={Nmu_test[gamma_mask].std():.3f}")
print(f"  Hadron Nmu: median={np.median(Nmu_test[hadron_mask]):.3f}, "
      f"mean={Nmu_test[hadron_mask].mean():.3f}, "
      f"std={Nmu_test[hadron_mask].std():.3f}")

# Try different thresholds and compute survival@99%
print(f"\nTesting thresholds (lower Nmu -> more likely gamma):")
print(f"{'Threshold':>12} {'GammaKeep':>12} {'HadronSurv':>12} {'Notes':>20}")
print("-" * 60)

best_survival = 1.0
best_threshold = None
best_scores = None

# Create scores: 1 - Nmu/7 (higher scores = lower Nmu = more like gamma)
# Normalize to roughly [0, 1]
scores = 1.0 - (Nmu_test / 7.0)
scores = np.clip(scores, 0, 1)

ng = gamma_mask.sum()
target_gamma_keep = int(np.ceil(0.99 * ng))

# Test different score thresholds
for percentile in np.arange(1, 100, 1):
    threshold = np.percentile(Nmu_test[gamma_mask], percentile)

    # Count gammas with Nmu <= threshold
    gammas_kept = (Nmu_test[gamma_mask] <= threshold).sum()

    if gammas_kept >= target_gamma_keep:
        # This threshold achieves at least 99% gamma efficiency
        # Compute hadron survival
        hadrons_kept = (Nmu_test[hadron_mask] <= threshold).sum()
        survival = hadrons_kept / hadron_mask.sum()

        status = ""
        if survival < best_survival:
            best_survival = survival
            best_threshold = threshold
            best_scores = scores.copy()
            status = "← BEST"

        if percentile % 10 == 1 or survival < 0.1:
            print(f"  {threshold:10.3f}  {gammas_kept:10d}/{ng}  {survival:12.2e} {status:>20}")

print(f"\n{'='*80}")
print(f"Best threshold: {best_threshold:.3f} (Nmu)")
print(f"Best survival @ 99% gamma eff: {best_survival:.2e}")

# Save predictions
np.savez(
    "submissions/haiku-gamma-mar8/predictions_nmu_threshold.npz",
    gamma_scores=best_scores,
)
print(f"Saved predictions")

# Compare to MLP baseline
mlp_preds = np.load('submissions/haiku-gamma-mar8/predictions.npz')
mlp_scores = mlp_preds['gamma_scores']

print(f"\n{'='*80}")
print(f"COMPARISON: NMU THRESHOLD vs MLP")
print(f"  NMU threshold survival: {best_survival:.2e}")

# Compute MLP survival at 99%
is_gamma = y_test == 0
is_hadron = y_test == 1
sg = np.sort(mlp_scores[is_gamma])
ng = len(sg)
thr_99_mlp = sg[max(0, int(np.floor(ng * (1 - 0.99))))]
n_hadron_surviving_mlp = (mlp_scores[is_hadron] >= thr_99_mlp).sum()
survival_mlp = n_hadron_surviving_mlp / is_hadron.sum()

print(f"  MLP survival:           {survival_mlp:.2e}")
print(f"  Ratio (NMU / MLP):      {best_survival / survival_mlp:.3f}x")

if best_survival < survival_mlp:
    print(f"  -> NMU THRESHOLD IS BETTER!")
else:
    print(f"  -> MLP is still better")
