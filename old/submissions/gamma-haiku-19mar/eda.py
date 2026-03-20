"""Quick EDA: understand feature distributions and separability."""

import numpy as np

def main():
    # Load data
    X_train = np.load("data/gamma_train/features.npy", mmap_mode="r").astype(np.float32)
    y_train = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")

    is_gamma = y_train == 0
    is_hadron = y_train == 1

    feature_names = ["E", "Ze", "Az", "Ne", "Nmu"]

    print("Feature distributions (mean ± std):\n")
    for i, name in enumerate(feature_names):
        gamma_mean = X_train[is_gamma, i].mean()
        gamma_std = X_train[is_gamma, i].std()
        hadron_mean = X_train[is_hadron, i].mean()
        hadron_std = X_train[is_hadron, i].std()
        print(f"{name:4s}: gamma {gamma_mean:.3f}±{gamma_std:.3f}, hadron {hadron_mean:.3f}±{hadron_std:.3f}, diff={gamma_mean-hadron_mean:.3f}")

    # Check muon channel properties (should be strongest signal)
    print("\nMuon-specific analysis:")
    print(f"  Gamma median Nmu: {np.median(X_train[is_gamma, 4]):.3f}")
    print(f"  Hadron median Nmu: {np.median(X_train[is_hadron, 4]):.3f}")
    print(f"  Gamma 10th percentile Nmu: {np.percentile(X_train[is_gamma, 4], 10):.3f}")
    print(f"  Hadron 90th percentile Nmu: {np.percentile(X_train[is_hadron, 4], 90):.3f}")

    # Check Ne/Nmu ratio
    ne_nmu_gamma = X_train[is_gamma, 3] - X_train[is_gamma, 4]  # log ratio
    ne_nmu_hadron = X_train[is_hadron, 3] - X_train[is_hadron, 4]
    print(f"\nNe/Nmu ratio (log10 Ne - log10 Nmu):")
    print(f"  Gamma: {ne_nmu_gamma.mean():.3f} ± {ne_nmu_gamma.std():.3f}")
    print(f"  Hadron: {ne_nmu_hadron.mean():.3f} ± {ne_nmu_hadron.std():.3f}")
    print(f"  Difference: {ne_nmu_gamma.mean() - ne_nmu_hadron.mean():.3f}")

if __name__ == "__main__":
    main()
