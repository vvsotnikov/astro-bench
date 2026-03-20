"""Download KASCADE simulation data from S3 and prepare train/test splits.

Task 1 — Mass composition (5-class):
    Downloads raw QGSJet-II.04 simulation data and processes it to match
    the published pipeline (Kuznetsov et al., JINST 2024):
    - Quality cuts: Ze<30, Ne>4.8, Age∈(0.2,1.48), Nmu>3.6
    - Split: 70% train / 30% test (torch random_split, seed=42)
    - Output: data/composition_train/ and data/composition_test/

Task 2 — Gamma/hadron separation (binary):
    Downloads pre-processed data (QGSJet-II + EPOS-LHC + SIBYLL):
    - Output: data/gamma_train/ and data/gamma_test/

All output files are .npy format (float16/int8), memory-mappable:
    matrices.npy            (N, 16, 16, 2) float16
    features.npy            (N, 6) float16   — composition
    features.npy            (N, 5) float16   — gamma
    labels_composition.npy  (N,) int8        — Task 1 only (0-4)
    labels_gamma.npy        (N,) int8        — Task 2 only (0=gamma, 1=hadron)
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from tqdm import tqdm

S3_BASE = "https://kascade-sim-data.s3.eu-central-1.amazonaws.com"
DATA_DIR = Path("data")

COMPOSITION_FEATURE_NAMES = ["E", "Ze", "Az", "Ne", "Nmu", "Age"]
GAMMA_FEATURE_NAMES = ["E", "Ze", "Az", "Ne", "Nmu"]
PARTICLE_NAMES = ["proton", "helium", "carbon", "silicon", "iron"]

# Raw data column layout (features.npz):
# [0]=particle_type, [1]=E, [2]=Xc, [3]=Yc, [4]=Core_dist, [5]=Ze, [6]=Az, [7]=Ne, [8]=Nmu, [9]=Age
RECO_HEADERS = ['particle_type', 'E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']

# Quality cuts matching Kuznetsov et al. (JINST 2024)
COMPOSITION_CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, float('inf')), 'Nmu': (3.6, float('inf'))}

# Split parameters matching published pipeline
SPLIT_SEED = 42
TRAIN_FRACTION = 0.7  # 70% train, 30% test


def download_file(url: str, dst: Path) -> None:
    """Download a file with progress bar."""
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


def prepare_composition() -> None:
    """Download raw QGS data and prepare composition train/test splits.

    Matches the published pipeline (Kuznetsov et al., JINST 2024):
    1. Load qgs_spectra_{matrices,features}.npz
    2. Apply quality cuts: Ze<30, Ne>4.8, Age∈(0.2,1.48), Nmu>3.6
    3. Extract matrix channels [1,2] (e/γ + μ) and features [E, Ze, Az, Ne, Nmu, Age]
    4. Split 70/30 using torch.random_split(generator=Generator().manual_seed(42))
    5. Save as .npy files
    """
    train_dir = DATA_DIR / "composition_train"
    test_dir = DATA_DIR / "composition_test"

    # Check if already prepared
    if all((d / f).exists() for d in [train_dir, test_dir]
           for f in ["matrices.npy", "features.npy", "labels_composition.npy"]):
        print("  composition — already prepared")
        return

    # Download raw files
    raw_files = {
        "matrices": DATA_DIR / "qgs_spectra_matrices.npz",
        "features": DATA_DIR / "qgs_spectra_features.npz",
    }
    for name, path in raw_files.items():
        if not path.exists():
            url = f"{S3_BASE}/qgs_spectra_{name}.npz"
            print(f"  Downloading {name}...")
            download_file(url, path)

    # Load raw data
    print("  Loading raw QGS spectra data...")
    raw_mat = np.load(raw_files["matrices"])['matrices']   # (N, 16, 16, 3)
    raw_feat = np.load(raw_files["features"])['features']  # (N, 10)
    n_total = len(raw_feat)
    print(f"  Raw: {n_total:,} events")

    # Parse columns — convert to float32 before cuts to match published kgnn pipeline
    reco = raw_feat[:, 1:].astype(np.float32)  # skip particle_type → [E, Xc, Yc, Core_dist, Ze, Az, Ne, Nmu, Age]
    labels_raw = raw_feat[:, 0].astype(int)  # 1-5

    # Apply quality cuts
    mask = np.ones(n_total, dtype=bool)
    for feat_name, (lo, hi) in COMPOSITION_CUTS.items():
        idx = RECO_HEADERS.index(feat_name) - 1  # -1 because reco skips column 0
        mask &= (reco[:, idx] > lo) & (reco[:, idx] < hi)

    print(f"  After cuts: {mask.sum():,} events (from {n_total:,})")

    # Extract arrays
    matrices = raw_mat[mask][:, :, :, [1, 2]]  # channels [1,2]: e/γ + μ
    # Features: [E, Ze, Az, Ne, Nmu, Age] — indices [0, 4, 5, 6, 7, 8] of reco
    feat_indices = [
        RECO_HEADERS.index('E') - 1,
        RECO_HEADERS.index('Ze') - 1,
        RECO_HEADERS.index('Az') - 1,
        RECO_HEADERS.index('Ne') - 1,
        RECO_HEADERS.index('Nmu') - 1,
        RECO_HEADERS.index('Age') - 1,
    ]
    features = reco[mask][:, feat_indices]
    labels = (labels_raw[mask] - 1).astype(np.int8)  # 1-5 → 0-4

    # Verify class distribution
    for i, name in enumerate(PARTICLE_NAMES):
        count = int((labels == i).sum())
        print(f"    {name}: {count:,} ({count / len(labels) * 100:.1f}%)")

    # Split: 70% train / 30% test (matching published pipeline)
    n = len(labels)
    n_train = n - int(n * (1 - TRAIN_FRACTION))
    n_test = n - n_train
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(SPLIT_SEED)).numpy()
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    print(f"  Split: train={n_train:,}, test={n_test:,}")

    # Save
    for split_name, idx in [("train", train_idx), ("test", test_idx)]:
        out_dir = DATA_DIR / f"composition_{split_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "matrices.npy", matrices[idx].astype(np.float32))
        np.save(out_dir / "features.npy", features[idx].astype(np.float32))
        np.save(out_dir / "labels_composition.npy", labels[idx])
        print(f"  Saved {split_name}: {len(idx):,} events → {out_dir}/")


GAMMA_SOURCES = ["qgs-4_gm_pr", "LHC_gm_pr", "sibyll-23c_gm_pr"]
GAMMA_CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, float('inf'))}
# No Nmu cut for gamma — gamma showers have near-zero muons
GAMMA_SPLIT_SEED = 2026
GAMMA_TRAIN_FRACTION = 0.8


def prepare_gamma() -> None:
    """Prepare gamma/hadron data from raw files.

    Combines QGSJet-II + EPOS-LHC + SIBYLL gamma+proton simulations.
    Split: 80/20 (seed=2026), cuts applied to test set only.
    """
    train_dir = DATA_DIR / "gamma_train"
    test_dir = DATA_DIR / "gamma_test"

    if all((d / f).exists() for d in [train_dir, test_dir]
           for f in ["matrices.npy", "features.npy", "labels_gamma.npy"]):
        print("  gamma — already prepared")
        return

    # Download raw files if needed
    for mode in GAMMA_SOURCES:
        for suffix in ["matrices", "features"]:
            path = DATA_DIR / f"{mode}_{suffix}.npz"
            if not path.exists():
                url = f"{S3_BASE}/{mode}_{suffix}.npz"
                print(f"  Downloading {mode}_{suffix}...")
                download_file(url, path)

    # Load and combine
    print("  Loading raw gamma data...")
    all_mat, all_feat = [], []
    for mode in GAMMA_SOURCES:
        m = np.load(DATA_DIR / f"{mode}_matrices.npz")['matrices']
        f = np.load(DATA_DIR / f"{mode}_features.npz")['features']
        all_mat.append(m)
        all_feat.append(f)
    raw_mat = np.concatenate(all_mat)
    raw_feat = np.concatenate(all_feat)
    n_total = len(raw_feat)
    print(f"  Combined: {n_total:,} events")

    # Labels: particle type 0=gamma, 1=hadron (already 0/1 in the data)
    labels = raw_feat[:, 0].astype(np.int8)
    print(f"  gamma={int((labels==0).sum()):,}, hadron={int((labels==1).sum()):,}")

    # Extract features: [E, Ze, Az, Ne, Nmu] — 5 features (no Age for gamma)
    # Raw layout: [part_type, E, Xc, Yc, core_dist, Ze, Az, Ne, Nmu, Age]
    feat_indices = [1, 5, 6, 7, 8]  # E, Ze, Az, Ne, Nmu
    features = raw_feat[:, feat_indices]

    # Extract matrix channels [1, 2]
    matrices = raw_mat[:, :, :, [1, 2]]

    # Split 80/20
    n_train = n_total - int(n_total * (1 - GAMMA_TRAIN_FRACTION))
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(GAMMA_SPLIT_SEED)).numpy()
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # Apply cuts to test set only
    reco_test = raw_feat[test_idx, 1:].astype(np.float32)
    test_mask = np.ones(len(test_idx), dtype=bool)
    for feat_name, (lo, hi) in GAMMA_CUTS.items():
        i = RECO_HEADERS.index(feat_name) - 1
        test_mask &= (reco_test[:, i] > lo) & (reco_test[:, i] < hi)
    test_idx_cut = test_idx[test_mask]

    print(f"  Split: train={len(train_idx):,} (no cuts), test={len(test_idx_cut):,} (after cuts)")
    print(f"    test gamma={int((labels[test_idx_cut]==0).sum()):,}, hadron={int((labels[test_idx_cut]==1).sum()):,}")

    # Save train (no cuts)
    train_dir.mkdir(parents=True, exist_ok=True)
    np.save(train_dir / "matrices.npy", matrices[train_idx].astype(np.float32))
    np.save(train_dir / "features.npy", features[train_idx].astype(np.float32))
    np.save(train_dir / "labels_gamma.npy", labels[train_idx])
    print(f"  Saved train: {len(train_idx):,} events → {train_dir}/")

    # Save test (with cuts)
    test_dir.mkdir(parents=True, exist_ok=True)
    np.save(test_dir / "matrices.npy", matrices[test_idx_cut].astype(np.float32))
    np.save(test_dir / "features.npy", features[test_idx_cut].astype(np.float32))
    np.save(test_dir / "labels_gamma.npy", labels[test_idx_cut])
    print(f"  Saved test: {len(test_idx_cut):,} events → {test_dir}/")


def print_stats(task: str) -> None:
    """Print dataset statistics."""
    for split in ["train", "test"]:
        d = DATA_DIR / f"{task}_{split}"
        if not d.exists():
            continue
        matrices = np.load(d / "matrices.npy", mmap_mode="r")
        features = np.load(d / "features.npy", mmap_mode="r")
        n = len(matrices)

        if task == "composition":
            labels = np.load(d / "labels_composition.npy", mmap_mode="r")
            print(f"  {task}_{split}: {n:,} events, {features.shape[1]} features")
            for i, name in enumerate(PARTICLE_NAMES):
                count = int((labels == i).sum())
                print(f"    {name}: {count:,} ({count / n * 100:.1f}%)")
        else:
            labels = np.load(d / "labels_gamma.npy", mmap_mode="r")
            n_gamma = int((labels == 0).sum())
            n_hadron = int((labels == 1).sum())
            print(f"  {task}_{split}: {n:,} events — {n_gamma:,} gamma, {n_hadron:,} hadron")


def main() -> None:
    tasks = sys.argv[1:] if len(sys.argv) > 1 else ["composition", "gamma"]
    valid_tasks = ["composition", "gamma"]

    for task in tasks:
        if task not in valid_tasks:
            print(f"Unknown task: {task}. Choose from: {valid_tasks}")
            sys.exit(1)

    for task in tasks:
        print(f"\n=== Preparing: {task} ===")
        if task == "composition":
            prepare_composition()
        elif task == "gamma":
            prepare_gamma()

    print("\n=== Dataset Summary ===")
    for task in tasks:
        print_stats(task)

    print("\nDone! Files are memory-mappable:")
    print("  np.load('data/composition_train/matrices.npy', mmap_mode='r')")


if __name__ == "__main__":
    main()
