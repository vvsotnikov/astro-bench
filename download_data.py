"""Download pre-processed KASCADE simulation data from S3.

Two datasets, one per task:

Task 1 — Mass composition (5-class):
    data/composition_train/   and   data/composition_test/
    Sources: QGSJet-II + EPOS-LHC (~7M hadron events, 5 classes)

Task 2 — Gamma/hadron separation (binary):
    data/gamma_train/   and   data/gamma_test/
    Sources: QGSJet-II + EPOS-LHC + SIBYLL gamma+proton (~1.9M events)

All files are .npy format (float16 matrices/features, int8 labels), memory-mappable:
    matrices.npy          (N, 16, 16, 2) float16
    features.npy          (N, 5) float16
    labels_composition.npy (N,) int8  — Task 1 only (0-4)
    labels_gamma.npy      (N,) int8  — Task 2 only (0=gamma, 1=hadron)
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from tqdm import tqdm

S3_BASE = "https://kascade-sim-data.s3.eu-central-1.amazonaws.com/v2"
DATA_DIR = Path("data")

FEATURE_NAMES = ["E", "Ze", "Az", "Ne", "Nmu"]
PARTICLE_NAMES = ["proton", "helium", "carbon", "silicon", "iron"]

# What to download per task
DATASETS = {
    "composition": {
        "splits": ["train", "test"],
        "files": ["matrices.npy", "features.npy", "labels_composition.npy"],
    },
    "gamma": {
        "splits": ["train", "test"],
        "files": ["matrices.npy", "features.npy", "labels_gamma.npy"],
    },
}


def download_file(url: str, dst: Path) -> None:
    """Download a file with progress bar."""
    if dst.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Get file size for progress bar
    from urllib.request import urlopen

    with urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))

    with tqdm(total=total, unit="B", unit_scale=True, desc=f"  {dst.name}") as pbar:

        def hook(count, block_size, total_size):
            pbar.update(block_size)

        urlretrieve(url, dst, reporthook=hook)


def download_task(task: str) -> None:
    """Download all files for a task."""
    cfg = DATASETS[task]
    for split in cfg["splits"]:
        out_dir = DATA_DIR / f"{task}_{split}"
        for filename in cfg["files"]:
            dst = out_dir / filename
            if dst.exists():
                print(f"  {task}_{split}/{filename} — already exists")
                continue
            url = f"{S3_BASE}/{task}_{split}/{filename}"
            download_file(url, dst)


def print_stats(task: str) -> None:
    """Print dataset statistics."""
    for split in ["train", "test"]:
        d = DATA_DIR / f"{task}_{split}"
        if not d.exists():
            continue
        matrices = np.load(d / "matrices.npy", mmap_mode="r")
        n = len(matrices)

        if task == "composition":
            labels = np.load(d / "labels_composition.npy", mmap_mode="r")
            print(f"  {task}_{split}: {n:,} events")
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

    for task in tasks:
        if task not in DATASETS:
            print(f"Unknown task: {task}. Choose from: {list(DATASETS.keys())}")
            sys.exit(1)

    for task in tasks:
        print(f"\n=== Downloading: {task} ===")
        download_task(task)

    print("\n=== Dataset Summary ===")
    for task in tasks:
        print_stats(task)

    print("\nDone! Files are memory-mappable:")
    print("  np.load('data/composition_train/matrices.npy', mmap_mode='r')")


if __name__ == "__main__":
    main()
