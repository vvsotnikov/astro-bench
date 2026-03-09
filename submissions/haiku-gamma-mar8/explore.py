#!/usr/bin/env python3
"""Explore gamma/hadron data and understand the physics."""
import numpy as np

# Load training data
X_train = np.load('/home/vladimir/cursor_projects/astro-agents/data/gamma_train/matrices.npy', mmap_mode='r')
f_train = np.load('/home/vladimir/cursor_projects/astro-agents/data/gamma_train/features.npy', mmap_mode='r')
y_train = np.load('/home/vladimir/cursor_projects/astro-agents/data/gamma_train/labels_gamma.npy', mmap_mode='r')

# Load test data
X_test = np.load('/home/vladimir/cursor_projects/astro-agents/data/gamma_test/matrices.npy', mmap_mode='r')
f_test = np.load('/home/vladimir/cursor_projects/astro-agents/data/gamma_test/features.npy', mmap_mode='r')
y_test = np.load('/home/vladimir/cursor_projects/astro-agents/data/gamma_test/labels_gamma.npy', mmap_mode='r')

print("=" * 80)
print("GAMMA/HADRON DATA EXPLORATION")
print("=" * 80)

print(f"\nTraining set:")
print(f"  X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
print(f"  f_train shape: {f_train.shape}, dtype: {f_train.dtype}")
print(f"  y_train shape: {y_train.shape}, dtype: {y_train.dtype}")

print(f"\nTest set:")
print(f"  X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
print(f"  f_test shape: {f_test.shape}, dtype: {f_test.dtype}")
print(f"  y_test shape: {y_test.shape}, dtype: {y_test.dtype}")

# Class distribution
gamma_count_train = (y_train == 0).sum()
hadron_count_train = (y_train == 1).sum()
print(f"\nTraining class distribution:")
print(f"  Gamma (0):  {gamma_count_train:,} ({100*gamma_count_train/len(y_train):.2f}%)")
print(f"  Hadron (1): {hadron_count_train:,} ({100*hadron_count_train/len(y_train):.2f}%)")
print(f"  Total:      {len(y_train):,}")

gamma_count_test = (y_test == 0).sum()
hadron_count_test = (y_test == 1).sum()
print(f"\nTest class distribution:")
print(f"  Gamma (0):  {gamma_count_test:,} ({100*gamma_count_test/len(y_test):.2f}%)")
print(f"  Hadron (1): {hadron_count_test:,} ({100*hadron_count_test/len(y_test):.2f}%)")
print(f"  Total:      {len(y_test):,}")

# Feature columns: E, Ze, Az, Ne, Nmu
print(f"\nFeature descriptions:")
print(f"  [0] E:   log10(energy/eV)")
print(f"  [1] Ze:  zenith angle (degrees)")
print(f"  [2] Az:  azimuth angle (degrees)")
print(f"  [3] Ne:  log10(electron number)")
print(f"  [4] Nmu: log10(muon number)")

# Feature statistics
print(f"\nTraining feature statistics:")
for i in range(5):
    print(f"  f_train[:, {i}] - mean: {f_train[:, i].mean():.3f}, std: {f_train[:, i].std():.3f}, min: {f_train[:, i].min():.3f}, max: {f_train[:, i].max():.3f}")

# Key physics discriminant: Ne/Nmu ratio
Ne_train = 10.0 ** f_train[:, 3]  # electron number
Nmu_train = 10.0 ** f_train[:, 4]  # muon number
ratio_train = Ne_train / (Nmu_train + 1e-6)

Ne_test = 10.0 ** f_test[:, 3]
Nmu_test = 10.0 ** f_test[:, 4]
ratio_test = Ne_test / (Nmu_test + 1e-6)

print(f"\nKey physics discriminant: Ne/Nmu ratio")
print(f"Training:")
print(f"  Gamma mean Ne/Nmu:  {ratio_train[y_train==0].mean():.2f}")
print(f"  Hadron mean Ne/Nmu: {ratio_train[y_train==1].mean():.2f}")
print(f"Test:")
print(f"  Gamma mean Ne/Nmu:  {ratio_test[y_test==0].mean():.2f}")
print(f"  Hadron mean Ne/Nmu: {ratio_test[y_test==1].mean():.2f}")

# Muon statistics (key for gamma separation)
print(f"\nMuon statistics (key discriminant for gamma rays):")
print(f"Training:")
print(f"  Gamma mean log10(Nmu):  {f_train[y_train==0, 4].mean():.3f}, std: {f_train[y_train==0, 4].std():.3f}")
print(f"  Hadron mean log10(Nmu): {f_train[y_train==1, 4].mean():.3f}, std: {f_train[y_train==1, 4].std():.3f}")
print(f"Test:")
print(f"  Gamma mean log10(Nmu):  {f_test[y_test==0, 4].mean():.3f}, std: {f_test[y_test==0, 4].std():.3f}")
print(f"  Hadron mean log10(Nmu): {f_test[y_test==1, 4].mean():.3f}, std: {f_test[y_test==1, 4].std():.3f}")

# Matrix sparsity
print(f"\nMatrix sparsity:")
sparsity_train = (X_train[:] == 0).sum() / X_train.size
print(f"  Training: {100*sparsity_train:.2f}% zeros")
sparsity_test = (X_test[:] == 0).sum() / X_test.size
print(f"  Test: {100*sparsity_test:.2f}% zeros")

# Check for NaN/Inf
print(f"\nData integrity:")
print(f"  X_train NaN: {np.isnan(X_train[:]).sum()}, Inf: {np.isinf(X_train[:]).sum()}")
print(f"  f_train NaN: {np.isnan(f_train[:]).sum()}, Inf: {np.isinf(f_train[:]).sum()}")
print(f"  X_test NaN: {np.isnan(X_test[:]).sum()}, Inf: {np.isinf(X_test[:]).sum()}")
print(f"  f_test NaN: {np.isnan(f_test[:]).sum()}, Inf: {np.isinf(f_test[:]).sum()}")

print("\n" + "=" * 80)
