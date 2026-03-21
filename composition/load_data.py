"""Load KASCADE mass composition data. Import this from any script, regardless of cwd.

Usage:
    from load_data import load_train, load_test

    X_train, f_train, y_train = load_train()  # matrices, features, labels
    X_test, f_test, y_test = load_test()
"""

from pathlib import Path
import numpy as np

_DATA_DIR = Path(__file__).parent / "data"


def check_gpu_free():
    """Check that no other training is using the GPU. Call before training.

    Raises RuntimeError if GPU memory usage exceeds 100MB (another job is running).
    Training runs can take up to 24 hours — this is normal. Do NOT kill long-running jobs.
    Wait for them to finish, then start your next experiment.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            used_mb = int(result.stdout.strip().split('\n')[0])
            if used_mb > 100:
                raise RuntimeError(
                    f"GPU has {used_mb}MB in use — another training job is running. "
                    f"Do NOT start a new job. Wait for it to finish first. "
                    f"Training runs can take up to 24 hours — this is normal."
                )
    except FileNotFoundError:
        pass  # nvidia-smi not available, skip check
    except (ValueError, IndexError):
        pass  # parse error, skip check


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
