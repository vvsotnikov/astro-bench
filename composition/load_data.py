"""Load KASCADE mass composition data. Import this from any script, regardless of cwd.

Usage:
    from load_data import load_train, load_test

    X_train, f_train, y_train = load_train()  # matrices, features, labels
    X_test, f_test, y_test = load_test()
"""

from pathlib import Path
import numpy as np

_DATA_DIR = Path(__file__).parent / "data"


def load_train():
    """Load training data (268K events, quality cuts pre-applied)."""
    d = _DATA_DIR / "composition_train"
    return (
        np.load(d / "matrices.npy", mmap_mode="r"),
        np.load(d / "features.npy", mmap_mode="r"),
        np.load(d / "labels_composition.npy", mmap_mode="r"),
    )


def load_test():
    """Load test data (115K events, quality cuts pre-applied)."""
    d = _DATA_DIR / "composition_test"
    return (
        np.load(d / "matrices.npy", mmap_mode="r"),
        np.load(d / "features.npy", mmap_mode="r"),
        np.load(d / "labels_composition.npy", mmap_mode="r"),
    )
