"""Comprehensive EDA and Data Analysis for KASCADE Gamma/Hadron Task.

Explores:
1. Class imbalance and label distributions
2. Feature correlations and separability
3. Matrix sparsity and structure
4. Distribution shifts (train vs test)
5. Physics regimes (E, Ze, Ne bins)
6. Error patterns in v41 predictions
"""

import numpy as np
from scipy import stats

# Load data
print("Loading data...")
train_matrices = np.load("data/gamma_train/matrices.npy", mmap_mode="r")
train_features = np.load("data/gamma_train/features.npy")[:]
train_labels = np.load("data/gamma_train/labels_gamma.npy")[:]

test_matrices = np.load("data/gamma_test/matrices.npy", mmap_mode="r")
test_features = np.load("data/gamma_test/features.npy")[:]
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

print(f"Train: {len(train_labels)} samples")
print(f"Test: {len(test_labels)} samples")

# ============================================================================
# 1. CLASS BALANCE AND LABEL DISTRIBUTIONS
# ============================================================================

print("\n" + "="*70)
print("1. CLASS BALANCE")
print("="*70)

train_gamma_ct = np.sum(train_labels == 0)
train_hadron_ct = np.sum(train_labels == 1)
test_gamma_ct = np.sum(test_labels == 0)
test_hadron_ct = np.sum(test_labels == 1)

print(f"\nTRAIN SET:")
print(f"  Gamma:  {train_gamma_ct:>8} ({100*train_gamma_ct/len(train_labels):.1f}%)")
print(f"  Hadron: {train_hadron_ct:>8} ({100*train_hadron_ct/len(train_labels):.1f}%)")
print(f"  Ratio (H/G): {train_hadron_ct/train_gamma_ct:.1f}:1")

print(f"\nTEST SET:")
print(f"  Gamma:  {test_gamma_ct:>8} ({100*test_gamma_ct/len(test_labels):.1f}%)")
print(f"  Hadron: {test_hadron_ct:>8} ({100*test_hadron_ct/len(test_labels):.1f}%)")
print(f"  Ratio (H/G): {test_hadron_ct/test_gamma_ct:.1f}:1")

print(f"\nIMBALANCE RATIO:")
print(f"  Train: {train_hadron_ct/train_gamma_ct:.1f}:1")
print(f"  Test:  {test_hadron_ct/test_gamma_ct:.1f}:1")
ratio_diff = abs((train_hadron_ct/train_gamma_ct) - (test_hadron_ct/test_gamma_ct))
print(f"  Status: {'MATCHED ✓' if ratio_diff < 0.1 else 'MISMATCH ✗'}")

# ============================================================================
# 2. FEATURE STATISTICS AND SEPARABILITY
# ============================================================================

print("\n" + "="*70)
print("2. FEATURE STATISTICS & SEPARABILITY")
print("="*70)

feat_names = ["E", "Ze", "Az", "Ne", "Nmu"]

print("\nFEATURE RANGES:")
print(f"{'Feature':<8} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
print("-" * 56)

for i, name in enumerate(feat_names):
    col = train_features[:, i]
    print(f"{name:<8} {col.min():<12.3f} {col.max():<12.3f} {col.mean():<12.3f} {col.std():<12.3f}")

print("\nFEATURE SEPARATION (Mann-Whitney U test, p-value):")
print(f"{'Feature':<8} {'Gamma Mean':<12} {'Hadron Mean':<12} {'p-value':<12} {'Significant'}")
print("-" * 60)

for i, name in enumerate(feat_names):
    gamma_vals = train_features[train_labels == 0, i]
    hadron_vals = train_features[train_labels == 1, i]

    stat, pval = stats.mannwhitneyu(gamma_vals, hadron_vals)

    sig = 'YES ✓' if pval < 0.05 else 'NO ✗'
    print(f"{name:<8} {gamma_vals.mean():<12.3f} {hadron_vals.mean():<12.3f} {pval:<12.2e} {sig}")

print("\nKEY FINDING: Ne-Nmu RATIO (engineered feature):")
ne_minus_nmu = train_features[:, 3] - train_features[:, 4]
gamma_ratio = ne_minus_nmu[train_labels == 0]
hadron_ratio = ne_minus_nmu[train_labels == 1]

print(f"  Gamma (Ne-Nmu):  mean={gamma_ratio.mean():.2f}, std={gamma_ratio.std():.2f}")
print(f"  Hadron (Ne-Nmu): mean={hadron_ratio.mean():.2f}, std={hadron_ratio.std():.2f}")
d_prime = abs(gamma_ratio.mean() - hadron_ratio.mean()) / np.sqrt((gamma_ratio.std()**2 + hadron_ratio.std()**2)/2)
print(f"  Separation (d'): {d_prime:.2f}")

# ============================================================================
# 3. MATRIX SPARSITY AND STRUCTURE
# ============================================================================

print("\n" + "="*70)
print("3. MATRIX SPARSITY & STRUCTURE")
print("="*70)

# Sample matrices to check sparsity
sample_indices = np.random.choice(len(train_matrices), 1000, replace=False)
sparsity_pct = []
for idx in sample_indices:
    mat = train_matrices[idx]
    nonzero = np.count_nonzero(mat)
    total = mat.shape[0] * mat.shape[1] * mat.shape[2]
    sparsity_pct.append(100 * nonzero / total)

print(f"\nMatrix Sparsity (16×16×2 = 512 elements):")
print(f"  Mean non-zero: {np.mean(sparsity_pct):.1f}%")
print(f"  Median non-zero: {np.median(sparsity_pct):.1f}%")
print(f"  Min non-zero: {np.min(sparsity_pct):.1f}%")
print(f"  Max non-zero: {np.max(sparsity_pct):.1f}%")

# Check per-class sparsity
gamma_mask = train_labels[sample_indices] == 0
hadron_mask = train_labels[sample_indices] == 1

gamma_sparsity = []
for idx in sample_indices[gamma_mask]:
    nonzero = np.count_nonzero(train_matrices[idx])
    gamma_sparsity.append(100 * nonzero / 512)

hadron_sparsity = []
for idx in sample_indices[hadron_mask]:
    nonzero = np.count_nonzero(train_matrices[idx])
    hadron_sparsity.append(100 * nonzero / 512)

if gamma_sparsity and hadron_sparsity:
    print(f"\nSparsity by Class:")
    print(f"  Gamma:  mean={np.mean(gamma_sparsity):.1f}%, std={np.std(gamma_sparsity):.1f}%")
    print(f"  Hadron: mean={np.mean(hadron_sparsity):.1f}%, std={np.std(hadron_sparsity):.1f}%")

# ============================================================================
# 4. DISTRIBUTION SHIFTS (Train vs Test)
# ============================================================================

print("\n" + "="*70)
print("4. DISTRIBUTION SHIFTS (Train vs Test)")
print("="*70)

print(f"\nFeature Statistics Comparison:")
print(f"{'Feature':<8} {'Train Mean':<12} {'Test Mean':<12} {'Diff %':<12} {'Significant'}")
print("-" * 60)

for i, name in enumerate(feat_names):
    train_col = train_features[:, i]
    test_col = test_features[:, i]

    pct_diff = 100 * abs(train_col.mean() - test_col.mean()) / train_col.mean()
    stat, pval = stats.ttest_ind(train_col, test_col)

    sig = 'YES ✗' if pval < 0.05 else 'NO ✓'
    print(f"{name:<8} {train_col.mean():<12.3f} {test_col.mean():<12.3f} {pct_diff:<12.1f}% {sig}")

# Check quality cuts effect on test
print(f"\nQuality Cuts on Test Set (Ze<30, Ne>4.8):")
test_ze = test_features[:, 1]
test_ne = test_features[:, 3]

quality_mask = (test_ze < 30) & (test_ne > 4.8)
n_with_cuts = np.sum(quality_mask)
n_total = len(test_features)

print(f"  Events passing cuts: {n_with_cuts}/{n_total} ({100*n_with_cuts/n_total:.1f}%)")
print(f"  Events failing cuts: {n_total - n_with_cuts}/{n_total} ({100*(n_total-n_with_cuts)/n_total:.1f}%)")

# Check if test truly has these cuts
print(f"\nTest Set Quality Bounds:")
print(f"  Ze range: [{test_ze.min():.1f}, {test_ze.max():.1f}]")
print(f"  Ne range: [{test_ne.min():.1f}, {test_ne.max():.1f}]")
print(f"  Status: {'Cuts applied ✓' if test_ze.max() < 30 and test_ne.min() > 4.8 else 'Full range'}")

# ============================================================================
# 5. PHYSICS REGIMES AND PERFORMANCE VARIATIONS
# ============================================================================

print("\n" + "="*70)
print("5. PHYSICS REGIMES & PERFORMANCE VARIATIONS")
print("="*70)

# Load v41 predictions if available
try:
    v41_data = np.load("submissions/haiku-gamma-mar9-v3/predictions_v41.npz")
    v41_scores = v41_data["gamma_scores"]
    has_predictions = True
    print("\nLoaded v41 predictions")
except:
    has_predictions = False
    print("\nv41 predictions not found")

if has_predictions:
    print(f"\nv41 Score Statistics:")
    print(f"  Mean: {v41_scores.mean():.4f}")
    print(f"  Std: {v41_scores.std():.4f}")
    print(f"  Min: {v41_scores.min():.4f}")
    print(f"  Max: {v41_scores.max():.4f}")

    # Performance by energy
    print(f"\nPerformance by Energy Bins:")
    print(f"{'E Range':<15} {'n_gamma':<8} {'n_hadron':<10} {'gamma_acc':<12} {'hadron_reject':<12}")
    print("-" * 60)

    e_bins = np.percentile(test_features[:, 0], [0, 33, 67, 100])

    for j in range(len(e_bins)-1):
        mask = (test_features[:, 0] >= e_bins[j]) & (test_features[:, 0] < e_bins[j+1])

        bin_labels = test_labels[mask]
        bin_scores = v41_scores[mask]

        n_gamma = np.sum(bin_labels == 0)
        n_hadron = np.sum(bin_labels == 1)

        if n_gamma > 0:
            gamma_scores = bin_scores[bin_labels == 0]
            thr = np.sort(gamma_scores)[max(0, int(np.floor(len(gamma_scores) * 0.25)))]
            gamma_acc = np.sum(bin_scores[bin_labels == 0] >= thr) / n_gamma
            hadron_reject = 1 - np.sum(bin_scores[bin_labels == 1] >= thr) / n_hadron if n_hadron > 0 else 0
        else:
            gamma_acc = 0
            hadron_reject = 0

        print(f"[{e_bins[j]:>6.1f}, {e_bins[j+1]:<6.1f}) {n_gamma:<8} {n_hadron:<10} {gamma_acc:<12.1%} {hadron_reject:<12.1%}")

print("\n" + "="*70)
print("EDA COMPLETE")
print("="*70)
