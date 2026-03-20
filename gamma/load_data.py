"""Load KASCADE gamma/hadron data. Import this from any script, regardless of cwd.

Usage:
    from load_data import load_train, load_test

    X_train, f_train, y_train = load_train()  # matrices, features, labels
    X_test, f_test, y_test = load_test()
"""

from pathlib import Path
import numpy as np

_DATA_DIR = Path(__file__).parent / "data"


def load_train():
    """Load training data (1.5M events, no quality cuts)."""
    d = _DATA_DIR / "gamma_train"
    return (
        np.load(d / "matrices.npy", mmap_mode="r"),
        np.load(d / "features.npy", mmap_mode="r"),
        np.load(d / "labels_gamma.npy", mmap_mode="r"),
    )


def load_test():
    """Load test data (~36K events, quality cuts applied)."""
    d = _DATA_DIR / "gamma_test"
    return (
        np.load(d / "matrices.npy", mmap_mode="r"),
        np.load(d / "features.npy", mmap_mode="r"),
        np.load(d / "labels_gamma.npy", mmap_mode="r"),
    )
