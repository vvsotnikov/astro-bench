"""EDA: Explore data distributions, class separability, and find opportunities."""
import numpy as np
from collections import Counter

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
PARTICLE_NAMES = ["proton", "helium", "carbon", "silicon", "iron"]

def p(msg):
    print(msg, flush=True)

def main():
    # Load train and test
    p("Loading data...")
    y_train = np.load(f"{DATA_DIR}/composition_train/labels_composition.npy", mmap_mode='r')
    f_train = np.load(f"{DATA_DIR}/composition_train/features.npy", mmap_mode='r')
    y_test = np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r')
    f_test = np.load(f"{DATA_DIR}/composition_test/features.npy", mmap_mode='r')

    n_train, n_test = len(y_train), len(y_test)
    p(f"Train: {n_train}, Test: {n_test}")

    # Class balance
    p("\n=== Class balance ===")
    for split, y in [("Train", y_train), ("Test", y_test)]:
        counts = Counter(np.array(y))
        p(f"{split}:")
        for c in range(5):
            p(f"  {PARTICLE_NAMES[c]}: {counts[c]} ({counts[c]/len(y)*100:.1f}%)")

    # Feature stats
    p("\n=== Feature distributions (Train) ===")
    feat_names = ["E", "Ze", "Az", "Ne", "Nmu"]
    # Sample to avoid loading all 5.5M
    n_sample = min(500000, n_train)
    idx = np.random.default_rng(42).choice(n_train, n_sample, replace=False)
    idx.sort()
    f_sample = np.array(f_train[idx], dtype=np.float32)
    y_sample = np.array(y_train[idx], dtype=np.int8)

    for i, name in enumerate(feat_names):
        vals = f_sample[:, i]
        p(f"\n  {name}: mean={vals.mean():.3f} std={vals.std():.3f} min={vals.min():.3f} max={vals.max():.3f}")
        for c in range(5):
            cv = vals[y_sample == c]
            p(f"    {PARTICLE_NAMES[c]}: mean={cv.mean():.3f} std={cv.std():.3f}")

    p("\n=== Feature distributions (Test) ===")
    f_test_np = np.array(f_test, dtype=np.float32)
    y_test_np = np.array(y_test, dtype=np.int8)
    for i, name in enumerate(feat_names):
        vals = f_test_np[:, i]
        p(f"\n  {name}: mean={vals.mean():.3f} std={vals.std():.3f} min={vals.min():.3f} max={vals.max():.3f}")

    # Quality cuts on train
    p("\n=== Quality cuts analysis ===")
    E_tr = f_sample[:, 0]
    Ze_tr = f_sample[:, 1]
    Ne_tr = f_sample[:, 3]
    Nmu_tr = f_sample[:, 4]

    cuts = Ze_tr < 30
    p(f"Ze < 30: {cuts.sum()}/{n_sample} ({cuts.mean()*100:.1f}%)")
    cuts2 = cuts & (Ne_tr > 4.8)
    p(f"+ Ne > 4.8: {cuts2.sum()}/{n_sample} ({cuts2.mean()*100:.1f}%)")

    # Distribution of Ne-Nmu (key discriminant) per class
    p("\n=== Ne-Nmu ratio per class ===")
    ratio = f_sample[:, 3] - f_sample[:, 4]  # log10(Ne) - log10(Nmu) = log10(Ne/Nmu)
    for c in range(5):
        r = ratio[y_sample == c]
        p(f"  {PARTICLE_NAMES[c]}: mean={r.mean():.3f} std={r.std():.3f}")

    # Same for test
    ratio_test = f_test_np[:, 3] - f_test_np[:, 4]
    p("\n=== Ne-Nmu ratio per class (Test) ===")
    for c in range(5):
        r = ratio_test[y_test_np == c]
        p(f"  {PARTICLE_NAMES[c]}: mean={r.mean():.3f} std={r.std():.3f}")

    # Energy-dependent accuracy from v8 predictions
    p("\n=== Matrix sparsity analysis ===")
    # Load a small batch of matrices
    matrices = np.load(f"{DATA_DIR}/composition_train/matrices.npy", mmap_mode='r')
    m_batch = np.array(matrices[:10000], dtype=np.float32)
    ch0 = m_batch[:, :, :, 0]  # electron
    ch1 = m_batch[:, :, :, 1]  # muon
    p(f"  Channel 0 (e/γ): {(ch0 == 0).mean()*100:.1f}% zeros, median nonzero={np.median(ch0[ch0 > 0]):.1f}")
    p(f"  Channel 1 (μ): {(ch1 == 0).mean()*100:.1f}% zeros, median nonzero={np.median(ch1[ch1 > 0]):.1f}")

    # Per-class matrix statistics
    y_batch = np.array(y_train[:10000], dtype=np.int8)
    for c in range(5):
        mask = y_batch == c
        if mask.sum() < 10:
            continue
        m_c = m_batch[mask]
        ch0_c = m_c[:, :, :, 0]
        ch1_c = m_c[:, :, :, 1]
        p(f"  {PARTICLE_NAMES[c]}: e_sum={ch0_c.sum(axis=(1,2)).mean():.0f} μ_sum={ch1_c.sum(axis=(1,2)).mean():.0f} "
          f"e_nnz={np.mean(ch0_c > 0)*100:.1f}% μ_nnz={np.mean(ch1_c > 0)*100:.1f}%")

    # Train/test energy distribution comparison
    p("\n=== Train/Test energy distribution ===")
    E_test = f_test_np[:, 0]
    energy_bins = [(14, 14.5), (14.5, 15), (15, 15.5), (15.5, 16), (16, 16.5), (16.5, 17), (17, 18)]
    p(f"{'Bin':<12} {'Train%':>8} {'Test%':>8}")
    for lo, hi in energy_bins:
        tr_frac = ((E_tr >= lo) & (E_tr < hi)).mean() * 100
        te_frac = ((E_test >= lo) & (E_test < hi)).mean() * 100
        p(f"[{lo}-{hi}){'':<3} {tr_frac:>8.1f} {te_frac:>8.1f}")

    # Check if train has Age feature (it doesn't - only 5 features)
    p(f"\nTrain features shape: {f_train.shape}")
    p(f"Test features shape: {f_test.shape}")

    # Confusion analysis: which classes overlap most?
    p("\n=== Class overlap analysis (Mahalanobis-like) ===")
    # Use Ne-Nmu + E as 2D discriminant space
    for c1 in range(5):
        for c2 in range(c1+1, 5):
            m1 = y_sample == c1
            m2 = y_sample == c2
            f1 = np.column_stack([ratio[m1], f_sample[m1, 0]])
            f2 = np.column_stack([ratio[m2], f_sample[m2, 0]])
            # Mean distance
            d_ratio = abs(f1[:, 0].mean() - f2[:, 0].mean()) / (0.5*(f1[:, 0].std() + f2[:, 0].std()))
            d_e = abs(f1[:, 1].mean() - f2[:, 1].mean()) / (0.5*(f1[:, 1].std() + f2[:, 1].std()))
            p(f"  {PARTICLE_NAMES[c1]}-{PARTICLE_NAMES[c2]}: d_ratio={d_ratio:.2f} d_E={d_e:.2f}")


if __name__ == "__main__":
    main()
