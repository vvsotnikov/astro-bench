import numpy as np

# Load test data to understand the distribution
labels = np.load('data/gamma_test/labels_gamma.npy', mmap_mode='r')[:]
features = np.load('data/gamma_test/features.npy', mmap_mode='r')[:]

# Features: E, Ze, Az, Ne, Nmu
E = features[:, 0]
Ze = features[:, 1]
Az = features[:, 2]
Ne = features[:, 3]
Nmu = features[:, 4]

is_gamma = labels == 0
is_hadron = labels == 1

print(f"Test set: {len(labels)} total ({is_gamma.sum()} gamma, {is_hadron.sum()} hadron)")
print(f"Imbalance: gamma/{is_hadron.sum()} = {is_gamma.sum()/is_hadron.sum():.3f}")

print("\n=== Nmu distribution (strongest discriminant) ===")
print(f"Gamma Nmu:  median={np.median(Nmu[is_gamma]):.2f}, mean={Nmu[is_gamma].mean():.2f}, min={Nmu[is_gamma].min():.2f}, max={Nmu[is_gamma].max():.2f}")
print(f"Hadron Nmu: median={np.median(Nmu[is_hadron]):.2f}, mean={Nmu[is_hadron].mean():.2f}, min={Nmu[is_hadron].min():.2f}, max={Nmu[is_hadron].max():.2f}")

print("\n=== Ne distribution ===")
print(f"Gamma Ne:  median={np.median(Ne[is_gamma]):.2f}, mean={Ne[is_gamma].mean():.2f}")
print(f"Hadron Ne: median={np.median(Ne[is_hadron]):.2f}, mean={Ne[is_hadron].mean():.2f}")

print("\n=== Energy distribution ===")
print(f"Gamma E:  median={np.median(E[is_gamma]):.2f}, mean={E[is_gamma].mean():.2f}")
print(f"Hadron E: median={np.median(E[is_hadron]):.2f}, mean={E[is_hadron].mean():.2f}")

# Test a simple Nmu threshold
print("\n=== Simple Nmu threshold test ===")
sg = np.sort(Nmu[is_gamma])
ng = len(sg)
thr_99_nmu = sg[max(0, int(np.floor(ng * (1 - 0.99))))]
print(f"99% gamma efficiency threshold (Nmu): {thr_99_nmu:.2f}")

# Count how many hadrons survive this threshold (interpreted as "low Nmu = gamma-like")
# Higher threshold = more lenient on gammas, more hadrons pass through
# We want to count hadrons with Nmu <= threshold (low muon count = gamma-like)
n_hadron_surviving = (Nmu[is_hadron] <= thr_99_nmu).sum()
survival_99 = n_hadron_surviving / is_hadron.sum()
print(f"Hadronic survival @ 99% gamma eff (Nmu <= threshold): {survival_99:.3f}")

# Try inverted: low Nmu is gamma
print("\nAlternative: count gammas with Nmu >= some value to find 99% efficiency")
# Find threshold where 99% of gammas pass
sg_sorted = np.sort(Nmu[is_gamma])
thr_99_nmu_v2 = sg_sorted[int(np.floor(ng * (1 - 0.99)))]
n_hadron_surviving_v2 = (Nmu[is_hadron] >= thr_99_nmu_v2).sum()
survival_99_v2 = n_hadron_surviving_v2 / is_hadron.sum()
print(f"Threshold (Nmu >= {thr_99_nmu_v2:.2f}): {survival_99_v2:.3f}")

print("\n=== Explore Ne/Nmu ratio ===")
ratio = Ne - Nmu
print(f"Gamma (Ne-Nmu): median={np.median(ratio[is_gamma]):.2f}, mean={ratio[is_gamma].mean():.2f}")
print(f"Hadron (Ne-Nmu): median={np.median(ratio[is_hadron]):.2f}, mean={ratio[is_hadron].mean():.2f}")
