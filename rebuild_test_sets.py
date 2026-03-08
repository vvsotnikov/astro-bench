"""One-off script: rebuild test sets with quality cuts applied.

Quality cuts (from KASCADE collaboration, applied to both tasks):
    Ze < 30 degrees (zenith angle)
    Ne > 4.8 (log10 electron number)
    0.2 < Age < 1.48 (shower age)

These use raw 10-column features (indices: Ze=5, Ne=7, Age=9).
Applied to test sets only — train sets keep all events.
"""

import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

FEATURE_INDICES = [1, 5, 6, 7, 8]  # E, Ze, Az, Ne, Nmu
PARTICLE_ID_MAP = {14: 0, 402: 1, 1206: 2, 2814: 3, 5626: 4}

COMPOSITION_SOURCES = ["qgs_spectra", "epos_spectra"]
GAMMA_SOURCES = ["qgs-4_gm_pr", "LHC_gm_pr", "sibyll-23c_gm_pr"]


def apply_quality_cuts(features_10col: np.ndarray) -> np.ndarray:
    """Return boolean mask for events passing quality cuts."""
    return (
        (features_10col[:, 5] < 30)
        & (features_10col[:, 7] > 4.8)
        & (features_10col[:, 9] > 0.2)
        & (features_10col[:, 9] < 1.48)
    )


def rebuild_test(task: str, sources: list[str], seed: int = 2026):
    """Rebuild test set with quality cuts from raw data (vectorized)."""
    print(f"\n=== Rebuilding {task}_test ===")

    # Load all raw data
    raw_features = []
    raw_matrices = []
    raw_true = []
    offsets = [0]
    for src in sources:
        m = np.load(DATA_DIR / f"{src}_matrices.npz")["matrices"]
        f = np.load(DATA_DIR / f"{src}_features.npz")["features"]
        t = np.load(DATA_DIR / f"{src}_true_features.npz")["true_features"]
        raw_matrices.append(m)
        raw_features.append(f)
        raw_true.append(t)
        offsets.append(offsets[-1] + len(m))
        print(f"  {src}: {len(m):,} events")

    total = offsets[-1]

    # Create the same split as before (seed=2026, 80/20)
    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    test_size = int(0.2 * total)
    test_indices = np.sort(indices[:test_size])

    print(f"  Test split: {len(test_indices):,} events (before cuts)")

    # Vectorized: process per source dataset
    all_matrices = []
    all_features = []
    all_labels = []

    for ds_idx in range(len(sources)):
        lo, hi = offsets[ds_idx], offsets[ds_idx + 1]
        # Which test indices fall in this source?
        mask = (test_indices >= lo) & (test_indices < hi)
        local_indices = test_indices[mask] - lo

        if len(local_indices) == 0:
            continue

        print(f"  Processing {sources[ds_idx]}: {len(local_indices):,} test events")

        # Load features for these indices and apply quality cuts
        feats = raw_features[ds_idx][local_indices]
        cut_mask = apply_quality_cuts(feats)
        local_indices = local_indices[cut_mask]
        print(f"    After cuts: {len(local_indices):,} ({cut_mask.sum() / len(cut_mask) * 100:.1f}%)")

        if len(local_indices) == 0:
            continue

        # Extract data
        mats = raw_matrices[ds_idx][local_indices, :, :, 1:].astype(np.float16)
        feat_out = raw_features[ds_idx][local_indices][:, FEATURE_INDICES].astype(np.float16)

        raw_ids = raw_true[ds_idx][local_indices, 1]
        if task == "composition":
            labels = np.array([PARTICLE_ID_MAP.get(int(pid), -1) for pid in raw_ids], dtype=np.int8)
        else:
            labels = np.where(raw_ids == 1, 0, 1).astype(np.int8)

        all_matrices.append(mats)
        all_features.append(feat_out)
        all_labels.append(labels)

    # Concatenate
    matrices_out = np.concatenate(all_matrices)
    features_out = np.concatenate(all_features)
    labels_out = np.concatenate(all_labels)
    n = len(labels_out)

    print(f"  Total after cuts: {n:,}")

    # Save
    out_dir = DATA_DIR / "upload" / f"{task}_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "matrices.npy", matrices_out)
    np.save(out_dir / "features.npy", features_out)
    label_name = "labels_composition" if task == "composition" else "labels_gamma"
    np.save(out_dir / f"{label_name}.npy", labels_out)

    # Print stats
    if task == "composition":
        names = ["proton", "helium", "carbon", "silicon", "iron"]
        for i, name in enumerate(names):
            count = (labels_out == i).sum()
            print(f"  {name}: {count:,} ({count / n * 100:.1f}%)")
    else:
        n_gamma = (labels_out == 0).sum()
        print(f"  gamma: {n_gamma:,}, hadron: {n - n_gamma:,}")

    print(f"  Saved to {out_dir}")


if __name__ == "__main__":
    rebuild_test("composition", COMPOSITION_SOURCES)
    rebuild_test("gamma", GAMMA_SOURCES)
    print("\nDone. Upload with:")
    print("  aws s3 sync data/upload/ s3://kascade-sim-data/v2/")
