"""Evaluate composition predictions and log the attempt.

CLI usage:
    python verify.py predictions.npz "CNN+Attention with SAM optimizer, seed=42"

Python usage:
    from verify import evaluate
    metric = evaluate(predictions, "v3: CNN + attention, seed=42")

Each call counts as one attempt (max 50). Results are auto-logged to results.tsv.
Key metric: mean fraction error (lower is better).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

_ROOT = Path(__file__).parent
_DATA_DIR = _ROOT / "data"
_TSV = _ROOT / "results.tsv"
MAX_ATTEMPTS = 50

PARTICLE_NAMES = ["proton", "helium", "carbon", "silicon", "iron"]
MIXTURE_SIZE = 5000
MIXTURE_SEED = 2026
GRID_STEP = 0.1


def _load_test_data():
    test_dir = _DATA_DIR / "composition_test"
    if not test_dir.exists():
        raise FileNotFoundError(f"{test_dir} not found. Run download_data.py first.")
    labels = np.array(np.load(test_dir / "labels_composition.npy", mmap_mode="r"))
    features = np.array(np.load(test_dir / "features.npy", mmap_mode="r"), dtype=np.float32)
    return labels, features


def _generate_fraction_grid(n_classes=5, step=GRID_STEP):
    n_steps = round(1.0 / step)
    fractions = []

    def _recurse(remaining, depth, current):
        if depth == n_classes - 1:
            current.append(remaining * step)
            fractions.append(current[:])
            current.pop()
            return
        for i in range(remaining + 1):
            current.append(i * step)
            _recurse(remaining - i, depth + 1, current)
            current.pop()

    _recurse(n_steps, 0, [])
    return np.array(fractions)


def _compute_fraction_error(truth, pred):
    n_classes = 5
    classes = np.arange(n_classes)
    class_indices = {c: np.where(truth == c)[0] for c in classes}

    if any(len(class_indices[c]) == 0 for c in classes):
        return None

    fractions = _generate_fraction_grid(n_classes, GRID_STEP)
    rng = np.random.default_rng(MIXTURE_SEED)
    all_errors = []

    for mix_idx in range(len(fractions)):
        counts = np.round(fractions[mix_idx] * MIXTURE_SIZE).astype(int)
        diff = MIXTURE_SIZE - counts.sum()
        if diff != 0:
            counts[np.argmax(counts)] += diff

        sampled_preds = []
        actual_true_fracs = np.zeros(n_classes)
        for c in classes:
            if counts[c] <= 0:
                continue
            idx = rng.choice(class_indices[c], size=counts[c], replace=True)
            sampled_preds.append(pred[idx])
            actual_true_fracs[c] = counts[c]

        actual_true_fracs /= actual_true_fracs.sum()
        all_preds = np.concatenate(sampled_preds)
        pred_counts = np.bincount(all_preds, minlength=n_classes)[:n_classes]
        pred_fracs = pred_counts / pred_counts.sum()
        all_errors.append(np.abs(actual_true_fracs - pred_fracs))

    all_errors = np.array(all_errors)

    return {
        "mean": float(all_errors.mean()),
        "per_class_mean": [float(all_errors[:, c].mean()) for c in classes],
        "per_class_max": [float(all_errors[:, c].max()) for c in classes],
        "per_class_p99": [float(np.percentile(all_errors[:, c], 99)) for c in classes],
        "n_ensembles": len(fractions),
    }


def _ensure_baseline():
    if not _TSV.exists():
        with open(_TSV, "w") as f:
            f.write("attempt\tfrac_error\taccuracy\tpredictions\tdescription\n")
            f.write("0\t0.1073\t0.504\tbaseline/predictions.npz\tLeNet baseline (published, Kuznetsov et al. JINST 2024)\n")


def _get_attempt_number():
    _ensure_baseline()
    with open(_TSV) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("attempt")]
    non_baseline = [l for l in lines if not l.startswith("0\t")]
    return len(non_baseline) + 1


def _log_result(attempt, frac_error, accuracy, description, pred_path):
    _ensure_baseline()
    with open(_TSV, "a") as f:
        f.write(f"{attempt}\t{frac_error:.4f}\t{accuracy:.4f}\t{pred_path}\t{description}\n")


def _get_best():
    _ensure_baseline()
    best = float("inf")
    with open(_TSV) as f:
        for line in f:
            if line.startswith("attempt"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    best = min(best, float(parts[1]))
                except ValueError:
                    pass
    return best


def evaluate(predictions, description, pred_path="(from python)"):
    """Evaluate composition predictions and log the attempt. Returns the key metric.

    Args:
        predictions: int array (N,), class labels 0-4
        description: what you tried (logged to results.tsv)
        pred_path: optional path string for logging

    Returns:
        float: mean fraction error (lower is better)

    Raises:
        RuntimeError: if max attempts exceeded
    """
    predictions = np.asarray(predictions, dtype=int)
    labels, features = _load_test_data()

    if len(predictions) != len(labels):
        raise ValueError(f"predictions has {len(predictions)} elements, expected {len(labels)}")

    invalid = (predictions < 0) | (predictions > 4)
    if invalid.any():
        raise ValueError(f"{invalid.sum()} predictions outside valid range [0, 4]")

    attempt = _get_attempt_number()
    if attempt > MAX_ATTEMPTS:
        raise RuntimeError(f"Maximum {MAX_ATTEMPTS} attempts reached. Check results.tsv for your best result.")

    accuracy = float(accuracy_score(labels, predictions))
    results = _compute_fraction_error(labels, predictions)
    if results is None:
        raise ValueError("Test set missing one or more classes")

    fe = results["mean"]

    # Log
    _log_result(attempt, fe, accuracy, description, str(pred_path))

    # Print summary
    abbrev = [n[:2] for n in PARTICLE_NAMES]
    print(f"\n{'=' * 60}")
    print(f"  ATTEMPT: {attempt}/{MAX_ATTEMPTS}")
    print(f"  DESCRIPTION: {description}")
    print(f"  KEY METRIC (mean fraction error): {fe:.4f}")
    print(f"  ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"    ({results['n_ensembles']} grid ensembles × {MIXTURE_SIZE} events)")
    print(f"{'=' * 60}")

    print(f"\n  Per-class fraction error:")
    print(f"  {'':>8}" + "".join(f"{a:>8}" for a in abbrev) + f"{'Mean':>8}")
    print(f"  {'MeanAE':>8}" + "".join(f"{v:>8.4f}" for v in results["per_class_mean"]) + f"{fe:>8.4f}")
    print(f"  {'MaxAE':>8}" + "".join(f"{v:>8.4f}" for v in results["per_class_max"]) + f"{np.mean(results['per_class_max']):>8.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    print(f"\n  Confusion matrix (row-normalized):")
    print(f"  {'':>4}" + "".join(f"{a:>6}" for a in abbrev))
    for i in range(5):
        print(f"  {abbrev[i]:>4}" + "".join(f"{cm_norm[i, j]:>6.2f}" for j in range(5)))

    best = _get_best()
    if fe <= best:
        print(f"\n  *** NEW BEST: {fe:.4f} (previous best: {best:.4f}) ***")
    else:
        print(f"\n  Current best: {best:.4f} (this attempt: {fe:.4f})")

    print(f"\n  Attempts used: {attempt}/{MAX_ATTEMPTS}")
    return fe


# --- CLI entry point ---

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate composition predictions (counts as one attempt)")
    parser.add_argument("predictions", type=Path, help="Path to predictions.npz")
    parser.add_argument("description", type=str,
                        help="What you tried (required). E.g. 'CNN+Attention with SAM, seed=42'")
    args = parser.parse_args()

    if not args.predictions.exists():
        print(f"Error: {args.predictions} not found")
        sys.exit(1)

    pred_data = np.load(args.predictions)
    if "predictions" not in pred_data:
        print("Error: predictions.npz must contain a 'predictions' array (int, classes 0-4)")
        sys.exit(1)

    predictions = pred_data["predictions"]
    evaluate(predictions, args.description, pred_path=str(args.predictions))


if __name__ == "__main__":
    main()
