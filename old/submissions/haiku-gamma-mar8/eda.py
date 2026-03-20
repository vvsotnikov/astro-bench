#!/usr/bin/env python3
"""Comprehensive EDA for gamma/hadron classification."""

import numpy as np
from sklearn.preprocessing import StandardScaler
import json

# Load all data
X_train_mat = np.load('data/gamma_train/matrices.npy', mmap_mode='r')
f_train = np.load('data/gamma_train/features.npy', mmap_mode='r')
y_train = np.load('data/gamma_train/labels_gamma.npy', mmap_mode='r')

X_test_mat = np.load('data/gamma_test/matrices.npy', mmap_mode='r')
f_test = np.load('data/gamma_test/features.npy', mmap_mode='r')
y_test = np.load('data/gamma_test/labels_gamma.npy', mmap_mode='r')

# Load best model predictions
best_preds = np.load('submissions/haiku-gamma-mar8/predictions.npz')
gamma_scores = best_preds['gamma_scores']

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - GAMMA/HADRON CLASSIFICATION")
print("=" * 80)

# 1. CLASS DISTRIBUTION
print("\n1. CLASS DISTRIBUTION")
print(f"   Training: {(y_train==0).sum():,} gamma ({100*(y_train==0).sum()/len(y_train):.2f}%), "
      f"{(y_train==1).sum():,} hadron ({100*(y_train==1).sum()/len(y_train):.2f}%)")
print(f"   Test:     {(y_test==0).sum():,} gamma ({100*(y_test==0).sum()/len(y_test):.2f}%), "
      f"{(y_test==1).sum():,} hadron ({100*(y_test==1).sum()/len(y_test):.2f}%)")

# 2. FEATURE STATISTICS BY CLASS
print("\n2. FEATURE STATISTICS (Training Set)")
print("   Feature distributions:")
feature_names = ['E (log10 eV)', 'Ze (deg)', 'Az (deg)', 'Ne (log10)', 'Nmu (log10)']
for i, fname in enumerate(feature_names):
    gamma_mean = f_train[y_train==0, i].mean()
    hadron_mean = f_train[y_train==1, i].mean()
    gamma_std = f_train[y_train==0, i].std()
    hadron_std = f_train[y_train==1, i].std()

    # Effect size (Cohen's d)
    cohens_d = (gamma_mean - hadron_mean) / np.sqrt((gamma_std**2 + hadron_std**2) / 2)

    print(f"   {fname:20s}: gamma={gamma_mean:7.3f}±{gamma_std:.3f}, "
          f"hadron={hadron_mean:7.3f}±{hadron_std:.3f}, d={cohens_d:6.3f}")

# 3. KEY PHYSICS DISCRIMINANTS
print("\n3. KEY PHYSICS DISCRIMINANTS (Training Set)")

# Ne/Nmu ratio
Ne_train = 10.0 ** f_train[:, 3]
Nmu_train = 10.0 ** f_train[:, 4]
ratio_train = Ne_train / (Nmu_train + 1e-6)

gamma_ratio = ratio_train[y_train==0]
hadron_ratio = ratio_train[y_train==1]

print(f"   Ne/Nmu ratio:")
print(f"     Gamma:  median={np.median(gamma_ratio):8.1f}, mean={gamma_ratio.mean():8.1f}")
print(f"     Hadron: median={np.median(hadron_ratio):8.1f}, mean={hadron_ratio.mean():8.1f}")

# Muon fraction in detector
print(f"\n   Muon density in detector (channel 1):")
muon_ch = X_train_mat[:, :, :, 1]
muon_sum_train = muon_ch.reshape(len(X_train_mat), -1).sum(axis=1)
electron_ch = X_train_mat[:, :, :, 0]
electron_sum_train = electron_ch.reshape(len(X_train_mat), -1).sum(axis=1)

gamma_muon = muon_sum_train[y_train==0]
hadron_muon = muon_sum_train[y_train==1]
gamma_electron = electron_sum_train[y_train==0]
hadron_electron = electron_sum_train[y_train==1]

print(f"     Gamma muon:  median={np.median(gamma_muon):8.1f}, mean={gamma_muon.mean():8.1f}")
print(f"     Hadron muon: median={np.median(hadron_muon):8.1f}, mean={hadron_muon.mean():8.1f}")
print(f"     Gamma electron:  median={np.median(gamma_electron):8.1f}, mean={gamma_electron.mean():8.1f}")
print(f"     Hadron electron: median={np.median(hadron_electron):8.1f}, mean={hadron_electron.mean():8.1f}")

# 4. MATRIX STATISTICS
print("\n4. MATRIX STATISTICS")
print(f"   Matrix shape: {X_train_mat.shape}")
print(f"   Train sparsity (zeros): {100*(X_train_mat[:]==0).sum()/X_train_mat.size:.1f}%")
print(f"   Test sparsity (zeros):  {100*(X_test_mat[:]==0).sum()/X_test_mat.size:.1f}%")

# Spatial concentration
gamma_mat = X_train_mat[y_train==0]
hadron_mat = X_train_mat[y_train==1]

gamma_max_per_sample = gamma_mat.reshape(len(gamma_mat), -1).max(axis=1)
hadron_max_per_sample = hadron_mat.reshape(len(hadron_mat), -1).max(axis=1)

gamma_std_per_sample = gamma_mat.reshape(len(gamma_mat), -1).std(axis=1)
hadron_std_per_sample = hadron_mat.reshape(len(hadron_mat), -1).std(axis=1)

print(f"   Max value per sample:")
print(f"     Gamma:  median={np.median(gamma_max_per_sample):8.1f}, mean={gamma_max_per_sample.mean():8.1f}")
print(f"     Hadron: median={np.median(hadron_max_per_sample):8.1f}, mean={hadron_max_per_sample.mean():8.1f}")

print(f"   Std dev per sample (spatial spread):")
print(f"     Gamma:  median={np.median(gamma_std_per_sample):8.3f}, mean={gamma_std_per_sample.mean():8.3f}")
print(f"     Hadron: median={np.median(hadron_std_per_sample):8.3f}, mean={hadron_std_per_sample.mean():8.3f}")

# 5. MODEL PERFORMANCE ANALYSIS
print("\n5. MODEL PERFORMANCE ON TEST SET")
is_gamma_test = y_test == 0
is_hadron_test = y_test == 1

print(f"   Score distribution:")
print(f"     Gamma  - min: {gamma_scores[is_gamma_test].min():.6f}, "
      f"median: {np.median(gamma_scores[is_gamma_test]):.6f}, "
      f"max: {gamma_scores[is_gamma_test].max():.6f}")
print(f"     Hadron - min: {gamma_scores[is_hadron_test].min():.6f}, "
      f"median: {np.median(gamma_scores[is_hadron_test]):.6f}, "
      f"max: {gamma_scores[is_hadron_test].max():.6f}")

# Separation metric: ROC AUC
from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(y_test, -gamma_scores)  # negate because gamma=0, hadron=1, and we want high scores for gamma
print(f"   ROC AUC: {auc:.4f}")

# 6. FAILURE CASES
print("\n6. FAILURE CASES ANALYSIS")

# Gammas misclassified (low score)
low_gamma_thresh = np.percentile(gamma_scores[is_gamma_test], 5)
low_gamma_idx = np.where((y_test == 0) & (gamma_scores < low_gamma_thresh))[0]

print(f"   Gammas with low scores (bottom 5%, < {low_gamma_thresh:.4f}):")
print(f"     Count: {len(low_gamma_idx)}")
if len(low_gamma_idx) > 0:
    print(f"     Energy (E): median={np.median(f_test[low_gamma_idx, 0]):.3f}, "
          f"mean={f_test[low_gamma_idx, 0].mean():.3f}")
    print(f"     Zenith (Ze): median={np.median(f_test[low_gamma_idx, 1]):.1f}, "
          f"mean={f_test[low_gamma_idx, 1].mean():.1f}")
    print(f"     Ne: median={np.median(f_test[low_gamma_idx, 3]):.3f}, "
          f"mean={f_test[low_gamma_idx, 3].mean():.3f}")
    print(f"     Nmu: median={np.median(f_test[low_gamma_idx, 4]):.3f}, "
          f"mean={f_test[low_gamma_idx, 4].mean():.3f}")

# Hadrons misclassified (high score)
high_hadron_thresh = np.percentile(gamma_scores[is_hadron_test], 95)
high_hadron_idx = np.where((y_test == 1) & (gamma_scores > high_hadron_thresh))[0]

print(f"\n   Hadrons with high scores (top 5%, > {high_hadron_thresh:.4f}):")
print(f"     Count: {len(high_hadron_idx)}")
if len(high_hadron_idx) > 0:
    print(f"     Energy (E): median={np.median(f_test[high_hadron_idx, 0]):.3f}, "
          f"mean={f_test[high_hadron_idx, 0].mean():.3f}")
    print(f"     Zenith (Ze): median={np.median(f_test[high_hadron_idx, 1]):.1f}, "
          f"mean={f_test[high_hadron_idx, 1].mean():.1f}")
    print(f"     Ne: median={np.median(f_test[high_hadron_idx, 3]):.3f}, "
          f"mean={f_test[high_hadron_idx, 3].mean():.3f}")
    print(f"     Nmu: median={np.median(f_test[high_hadron_idx, 4]):.3f}, "
          f"mean={f_test[high_hadron_idx, 4].mean():.3f}")

# 7. ENERGY-DEPENDENT ANALYSIS
print("\n7. ENERGY-DEPENDENT SEPARATION QUALITY")
energy_bins = [(14.0, 14.5), (14.5, 15.0), (15.0, 15.5), (15.5, 16.0),
               (16.0, 16.5), (16.5, 17.0), (17.0, 18.0)]

for emin, emax in energy_bins:
    mask = (f_test[:, 0] >= emin) & (f_test[:, 0] < emax)
    n_gamma_bin = (y_test[mask] == 0).sum()
    n_hadron_bin = (y_test[mask] == 1).sum()

    if n_gamma_bin > 0 and n_hadron_bin > 0:
        auc_bin = roc_auc_score(y_test[mask], -gamma_scores[mask])
        gamma_median_score = np.median(gamma_scores[mask & (y_test==0)])
        hadron_median_score = np.median(gamma_scores[mask & (y_test==1)])

        print(f"   E=[{emin:.1f}, {emax:.1f}): AUC={auc_bin:.3f}, "
              f"γ_median={gamma_median_score:.4f}, h_median={hadron_median_score:.4f}, "
              f"n_γ={n_gamma_bin}, n_h={n_hadron_bin}")

# 8. ZENITH-DEPENDENT ANALYSIS
print("\n8. ZENITH-DEPENDENT SEPARATION QUALITY")
zenith_bins = [(0, 10), (10, 20), (20, 30)]

for zmin, zmax in zenith_bins:
    mask = (f_test[:, 1] >= zmin) & (f_test[:, 1] < zmax)
    n_gamma_bin = (y_test[mask] == 0).sum()
    n_hadron_bin = (y_test[mask] == 1).sum()

    if n_gamma_bin > 0 and n_hadron_bin > 0:
        auc_bin = roc_auc_score(y_test[mask], -gamma_scores[mask])
        gamma_median_score = np.median(gamma_scores[mask & (y_test==0)])
        hadron_median_score = np.median(gamma_scores[mask & (y_test==1)])

        print(f"   Ze=[{zmin:2d}, {zmax:2d}): AUC={auc_bin:.3f}, "
              f"γ_median={gamma_median_score:.4f}, h_median={hadron_median_score:.4f}, "
              f"n_γ={n_gamma_bin}, n_h={n_hadron_bin}")

# 9. FEATURE IMPORTANCE PROXY
print("\n9. FEATURE IMPORTANCE (via separability)")
print("   Cohen's d values (larger = better separation):")

# Compute d values for test set features
for i, fname in enumerate(feature_names):
    gamma_vals = f_test[y_test==0, i]
    hadron_vals = f_test[y_test==1, i]
    d = (gamma_vals.mean() - hadron_vals.mean()) / np.sqrt((gamma_vals.std()**2 + hadron_vals.std()**2) / 2)
    print(f"   {fname:20s}: d={d:6.3f}")

print("\n" + "=" * 80)
