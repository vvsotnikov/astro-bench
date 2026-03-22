"""Download and prepare KASCADE gamma/hadron separation data.

Combines QGSJet-II + EPOS-LHC + SIBYLL gamma+proton simulations.
Split: 80/20 (seed=2026). Quality cuts applied to test set only.

Output: data/gamma_train/ and data/gamma_test/
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from tqdm import tqdm

S3_BASE = "https://kascade-sim-data.s3.eu-central-1.amazonaws.com"
DATA_DIR = Path("data")

SOURCES = ["qgs-4_gm_pr", "LHC_gm_pr", "sibyll-23c_gm_pr"]
RECO_HEADERS = ['particle_type', 'E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']

# Quality cuts for test set (no Nmu cut — gammas have near-zero muons)
TEST_CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, float('inf'))}
SPLIT_SEED = 2026
TRAIN_FRACTION = 0.8


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


def main() -> None:
    train_dir = DATA_DIR / "gamma_train"
    test_dir = DATA_DIR / "gamma_test"

    if all((d / f).exists() for d in [train_dir, test_dir]
           for f in ["matrices.npy", "features.npy", "labels_gamma.npy"]):
        print("Data already prepared.")
        return

    # Download raw files
    for mode in SOURCES:
        for suffix in ["matrices", "features"]:
            path = DATA_DIR / f"{mode}_{suffix}.npz"
            if not path.exists():
                print(f"Downloading {mode}_{suffix}...")
                download_file(f"{S3_BASE}/{mode}_{suffix}.npz", path)

    # Load and combine
    print("Loading raw gamma data...")
    all_mat, all_feat = [], []
    for mode in SOURCES:
        all_mat.append(np.load(DATA_DIR / f"{mode}_matrices.npz")['matrices'])
        all_feat.append(np.load(DATA_DIR / f"{mode}_features.npz")['features'])
    raw_mat = np.concatenate(all_mat)
    raw_feat = np.concatenate(all_feat)
    n_total = len(raw_feat)
    print(f"  Combined: {n_total:,} events")

    labels = raw_feat[:, 0].astype(np.int8)  # 0=gamma, 1=hadron
    print(f"  gamma={int((labels == 0).sum()):,}, hadron={int((labels == 1).sum()):,}")

    # Features: [E, Ze, Az, Ne, Nmu] — 5 features (no Age)
    features = raw_feat[:, [1, 5, 6, 7, 8]].astype(np.float32)
    matrices = raw_mat[:, :, :, [1, 2]]

    # Split 80/20
    n_train = n_total - int(n_total * (1 - TRAIN_FRACTION))
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(SPLIT_SEED)).numpy()
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # Apply cuts to test only
    reco_test = raw_feat[test_idx, 1:].astype(np.float32)
    test_mask = np.ones(len(test_idx), dtype=bool)
    for feat_name, (lo, hi) in TEST_CUTS.items():
        i = RECO_HEADERS.index(feat_name) - 1
        test_mask &= (reco_test[:, i] > lo) & (reco_test[:, i] < hi)
    test_idx = test_idx[test_mask]

    print(f"  train={len(train_idx):,} (no cuts), test={len(test_idx):,} (after cuts)")
    print(f"    test gamma={int((labels[test_idx] == 0).sum()):,}, hadron={int((labels[test_idx] == 1).sum()):,}")

    # Save
    for split_name, idx in [("train", train_idx), ("test", test_idx)]:
        out_dir = DATA_DIR / f"gamma_{split_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "matrices.npy", matrices[idx].astype(np.float32))
        np.save(out_dir / "features.npy", features[idx].astype(np.float32))
        np.save(out_dir / "labels_gamma.npy", labels[idx])
        print(f"  {split_name}: {len(idx):,} events → {out_dir}/")

    # Initialize results.tsv with baseline
    tsv = Path("results.tsv")
    if not tsv.exists():
        with open(tsv, "w") as f:
            f.write("attempt\tsurvival_75\tpredictions\tdescription\n")
            f.write("0\t1.0e-02\tbaseline/predictions.npz\tRF baseline (published, Kostunin et al. ICRC 2021)\n")
        print("  Initialized results.tsv with baseline (attempt 0)")

    print("\nDone!")


if __name__ == "__main__":
    main()
