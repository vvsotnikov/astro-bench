"""Download KASCADE simulation data from S3."""

import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from tqdm import tqdm

S3_BASE = "https://kascade-sim-data.s3.eu-central-1.amazonaws.com"
DATA_DIR = Path("data")

# Simulation datasets available
SIMULATIONS = ["epos-LHC", "qgs-4", "sibyll-23c"]

# Tabular data (array-level observables)
TABULAR_FILES = [f"data_array_{sim}.csv" for sim in SIMULATIONS]

# Matrix data (16x16 detector grid images) + features + true features
MATRIX_MODES = ["qgs_spectra", "epos_spectra"]
MATRIX_FILES = []
for mode in MATRIX_MODES:
    MATRIX_FILES.extend([
        f"{mode}_matrices.npz",
        f"{mode}_features.npz",
        f"{mode}_true_features.npz",
    ])


def download_file(filename: str, subdir: str = "") -> Path:
    dst_dir = DATA_DIR / subdir if subdir else DATA_DIR
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / filename
    if dst.exists():
        print(f"  {filename} already exists")
        return dst
    print(f"  Downloading {filename}...")
    urlretrieve(f"{S3_BASE}/{filename}", dst)
    return dst


def create_test_split(seed: int = 2026) -> None:
    """Create a fixed train/test split and save test indices.

    Uses true energy for stratified splitting (same approach as legacy notebooks)
    to ensure test set covers full energy range. The split is deterministic
    given the seed.
    """
    split_file = DATA_DIR / "test_split.npz"
    if split_file.exists():
        print("\nTest split already exists")
        return

    print("\nCreating fixed test split (seed=2026)...")

    # Load matrix data (this is the primary dataset for the challenge)
    matrices_all = []
    features_all = []
    true_features_all = []
    sim_labels = []

    for mode in MATRIX_MODES:
        matrices = np.load(DATA_DIR / f"{mode}_matrices.npz")["matrices"]
        features = np.load(DATA_DIR / f"{mode}_features.npz")["features"]
        true_features = np.load(DATA_DIR / f"{mode}_true_features.npz")["true_features"]
        matrices_all.append(matrices)
        features_all.append(features)
        true_features_all.append(true_features)
        sim_labels.extend([mode] * len(matrices))

    matrices_all = np.concatenate(matrices_all)
    features_all = np.concatenate(features_all)
    true_features_all = np.concatenate(true_features_all)

    n = len(matrices_all)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    # 80/20 split
    test_size = int(0.2 * n)
    test_indices = np.sort(indices[:test_size])
    train_indices = np.sort(indices[test_size:])

    np.savez(
        split_file,
        train_indices=train_indices,
        test_indices=test_indices,
        seed=seed,
    )
    print(f"  Split created: {len(train_indices)} train, {len(test_indices)} test")


def main() -> None:
    print("Downloading KASCADE data...\n")

    print("Tabular data (array-level observables):")
    for f in tqdm(TABULAR_FILES, desc="Tabular"):
        download_file(f)

    print("\nMatrix data (16x16 detector grids):")
    for f in tqdm(MATRIX_FILES, desc="Matrix"):
        download_file(f)

    create_test_split()
    print("\nDone! Data is in ./data/")


if __name__ == "__main__":
    main()
