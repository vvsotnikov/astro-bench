"""Download and prepare KASCADE mass composition data.

Downloads QGSJet-II.04 simulation from S3, applies quality cuts matching
the published pipeline (Kuznetsov et al., JINST 2024), and splits 70/30.

Output: data/composition_train/ and data/composition_test/
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from tqdm import tqdm

S3_BASE = "https://kascade-sim-data.s3.eu-central-1.amazonaws.com"
DATA_DIR = Path("data")

PARTICLE_NAMES = ["proton", "helium", "carbon", "silicon", "iron"]
RECO_HEADERS = ['particle_type', 'E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']

# Quality cuts matching published pipeline
CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, float('inf')), 'Nmu': (3.6, float('inf'))}
SPLIT_SEED = 42
TRAIN_FRACTION = 0.7


def download_file(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    from urllib.request import urlopen
    with urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
    with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {dst.name}") as pbar:
        def hook(count, block_size, total_size):
            pbar.update(block_size)
        urlretrieve(url, dst, reporthook=hook)


def _init_results_tsv():
    """Ensure results.tsv exists with baseline as attempt 0."""
    tsv = Path("results.tsv")
    if not tsv.exists():
        with open(tsv, "w") as f:
            f.write("attempt\tfrac_error\taccuracy\tpredictions\tdescription\n")
            f.write("0\t0.1073\t0.504\tbaseline/predictions.npz\tLeNet baseline (published, Kuznetsov et al. JINST 2024)\n")
        print("  Initialized results.tsv with baseline (attempt 0)")
    else:
        with open(tsv) as f:
            content = f.read()
        if "0\t0.1073" not in content:
            # Prepend baseline if missing
            lines = content.strip().split("\n")
            header = lines[0]
            rest = lines[1:]
            with open(tsv, "w") as f:
                f.write(header + "\n")
                f.write("0\t0.1073\t0.504\tbaseline/predictions.npz\tLeNet baseline (published, Kuznetsov et al. JINST 2024)\n")
                for line in rest:
                    f.write(line + "\n")


def main() -> None:
    train_dir = DATA_DIR / "composition_train"
    test_dir = DATA_DIR / "composition_test"

    if all((d / f).exists() for d in [train_dir, test_dir]
           for f in ["matrices.npy", "features.npy", "labels_composition.npy"]):
        print("Data already prepared.")
        return

    # Download raw QGS files
    for suffix in ["matrices", "features"]:
        path = DATA_DIR / f"qgs_spectra_{suffix}.npz"
        if not path.exists():
            print(f"Downloading {suffix}...")
            download_file(f"{S3_BASE}/qgs_spectra_{suffix}.npz", path)

    # Load
    print("Loading raw data...")
    raw_mat = np.load(DATA_DIR / "qgs_spectra_matrices.npz")['matrices']
    raw_feat = np.load(DATA_DIR / "qgs_spectra_features.npz")['features']
    n_total = len(raw_feat)
    print(f"  Raw: {n_total:,} events")

    # Apply cuts (float32 to match published kgnn pipeline)
    reco = raw_feat[:, 1:].astype(np.float32)
    labels_raw = raw_feat[:, 0].astype(int)

    mask = np.ones(n_total, dtype=bool)
    for feat_name, (lo, hi) in CUTS.items():
        idx = RECO_HEADERS.index(feat_name) - 1
        mask &= (reco[:, idx] > lo) & (reco[:, idx] < hi)
    print(f"  After cuts: {mask.sum():,} events")

    # Extract channels [1,2] and features [E, Ze, Az, Ne, Nmu, Age]
    matrices = raw_mat[mask][:, :, :, [1, 2]]
    feat_indices = [0, 4, 5, 6, 7, 8]  # E, Ze, Az, Ne, Nmu, Age in reco
    features = reco[mask][:, feat_indices]
    labels = (labels_raw[mask] - 1).astype(np.int8)

    for i, name in enumerate(PARTICLE_NAMES):
        count = int((labels == i).sum())
        print(f"    {name}: {count:,} ({count / len(labels) * 100:.1f}%)")

    # Split 70/30
    n = len(labels)
    n_train = n - int(n * (1 - TRAIN_FRACTION))
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(SPLIT_SEED)).numpy()

    for split_name, idx in [("train", indices[:n_train]), ("test", indices[n_train:])]:
        out_dir = DATA_DIR / f"composition_{split_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "matrices.npy", matrices[idx].astype(np.float32))
        np.save(out_dir / "features.npy", features[idx].astype(np.float32))
        np.save(out_dir / "labels_composition.npy", labels[idx])
        print(f"  {split_name}: {len(idx):,} events → {out_dir}/")

    # Always ensure results.tsv has baseline (attempt 0)
    _init_results_tsv()

    print("\nDone! Load data with:")
    print("  np.load('data/composition_train/matrices.npy', mmap_mode='r')")


if __name__ == "__main__":
    main()
