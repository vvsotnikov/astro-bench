"""Verify a submission's predictions against the held-out test set.

Usage:
    python verify.py submissions/my_submission/predictions.npz

The predictions file must contain:
    - "predictions": array of integer class labels (0-4) for each test sample
      0=proton, 1=helium, 2=carbon, 3=silicon, 4=iron

Output: accuracy, per-class metrics, confusion matrix, energy-binned performance.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = Path("data")
PARTICLE_NAMES = ["proton", "helium", "carbon", "silicon", "iron"]

# Particle ID mapping from raw data to class index
# gamma=1->excluded, H=14->0, He=402->1, C=1206->2, Si=2814->3, Fe=5626->4
PARTICLE_ID_MAP = {14: 0, 402: 1, 1206: 2, 2814: 3, 5626: 4}


def load_test_truth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load test set ground truth.

    Returns:
        classes: integer class labels (0-4) for test samples
        energies: true energies (log10(E/eV)) for test samples
        test_indices: indices into the full concatenated dataset
    """
    split = np.load(DATA_DIR / "test_split.npz")
    test_indices = split["test_indices"]

    # Load true features from both simulation sets
    true_features_list = []
    for mode in ["qgs_spectra", "epos_spectra"]:
        tf = np.load(DATA_DIR / f"{mode}_true_features.npz")["true_features"]
        true_features_list.append(tf)
    true_features = np.concatenate(true_features_list)

    # Extract test set
    test_tf = true_features[test_indices]
    energies = test_tf[:, 0]  # log10(E/eV)
    raw_particle_ids = test_tf[:, 1]

    # Map particle IDs to class indices
    classes = np.full(len(raw_particle_ids), -1, dtype=int)
    for raw_id, class_idx in PARTICLE_ID_MAP.items():
        classes[raw_particle_ids == raw_id] = class_idx

    # Sanity check: no unmapped particles
    unmapped = classes == -1
    if unmapped.any():
        # Gamma rays (id=1) or other particles — exclude from evaluation
        valid = ~unmapped
        return classes[valid], energies[valid], test_indices[valid]

    return classes, energies, test_indices


def evaluate(predictions: np.ndarray, truth: np.ndarray, energies: np.ndarray) -> dict:
    """Compute all evaluation metrics."""
    results = {}

    # Overall accuracy
    results["accuracy"] = float(accuracy_score(truth, predictions))

    # Per-class report
    report = classification_report(
        truth, predictions, target_names=PARTICLE_NAMES, output_dict=True, zero_division=0
    )
    results["per_class"] = {
        name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": report[name]["support"],
        }
        for name in PARTICLE_NAMES
        if name in report
    }
    results["macro_f1"] = report["macro avg"]["f1-score"]
    results["weighted_f1"] = report["weighted avg"]["f1-score"]

    # Confusion matrix
    results["confusion_matrix"] = confusion_matrix(truth, predictions).tolist()

    # Energy-binned accuracy
    energy_bins = [
        (14.0, 15.0, "14.0-15.0"),
        (15.0, 15.5, "15.0-15.5"),
        (15.5, 16.0, "15.5-16.0"),
        (16.0, 16.5, "16.0-16.5"),
        (16.5, 17.0, "16.5-17.0"),
        (17.0, 18.0, "17.0-18.0"),
    ]
    results["energy_binned"] = {}
    for lo, hi, label in energy_bins:
        mask = (energies >= lo) & (energies < hi)
        if mask.sum() > 0:
            results["energy_binned"][label] = {
                "accuracy": float(accuracy_score(truth[mask], predictions[mask])),
                "count": int(mask.sum()),
            }

    return results


def print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  OVERALL ACCURACY: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  MACRO F1:         {results['macro_f1']:.4f}")
    print(f"  WEIGHTED F1:      {results['weighted_f1']:.4f}")
    print(f"{'='*60}")

    print(f"\n{'Per-class metrics':^60}")
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    for name in PARTICLE_NAMES:
        if name in results["per_class"]:
            m = results["per_class"][name]
            print(f"{name:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")

    print(f"\n{'Confusion matrix':^60}")
    cm = np.array(results["confusion_matrix"])
    header = "True\\Pred  " + "  ".join(f"{n[:4]:>6}" for n in PARTICLE_NAMES)
    print(header)
    for i, name in enumerate(PARTICLE_NAMES):
        row = "  ".join(f"{cm[i, j]:>6}" for j in range(len(PARTICLE_NAMES)))
        print(f"{name[:4]:<10} {row}")

    print(f"\n{'Energy-binned accuracy':^60}")
    print(f"{'Bin':<15} {'Accuracy':>10} {'Count':>10}")
    print("-" * 40)
    for label, m in results["energy_binned"].items():
        print(f"{label:<15} {m['accuracy']:>10.4f} {m['count']:>10}")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <predictions.npz>")
        print("  The file must contain a 'predictions' array of int class labels (0-4).")
        sys.exit(1)

    pred_path = Path(sys.argv[1])
    if not pred_path.exists():
        print(f"Error: {pred_path} not found")
        sys.exit(1)

    # Load predictions
    pred_data = np.load(pred_path)
    if "predictions" not in pred_data:
        print("Error: predictions.npz must contain a 'predictions' array")
        sys.exit(1)
    predictions = pred_data["predictions"].astype(int)

    # Load ground truth
    truth, energies, test_indices = load_test_truth()

    # Validate shape
    if len(predictions) != len(truth):
        print(f"Error: predictions has {len(predictions)} elements, expected {len(truth)}")
        sys.exit(1)

    # Validate range
    invalid = (predictions < 0) | (predictions > 4)
    if invalid.any():
        print(f"Error: {invalid.sum()} predictions outside valid range [0, 4]")
        sys.exit(1)

    # Evaluate
    results = evaluate(predictions, truth, energies)
    print_results(results)

    # Save results alongside predictions
    import json
    results_path = pred_path.parent / "metrics.json"
    # Convert numpy types for JSON serialization
    results_json = json.loads(json.dumps(results, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else x))
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
