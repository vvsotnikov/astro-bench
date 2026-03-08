"""Verify a submission's predictions against the held-out test set.

Usage:
    python verify.py submissions/my_submission/predictions.npz
    python verify.py --task gamma submissions/my_submission/predictions.npz

Task 1 (composition, default):
    predictions.npz must contain "predictions": int array of class labels (0-4)
    0=proton, 1=helium, 2=carbon, 3=silicon, 4=iron
    Key metric: overall accuracy.

Task 2 (gamma):
    predictions.npz must contain "gamma_scores": float array of gamma probabilities
    Higher score = more likely to be gamma.
    Key metric: hadronic survival rate at 99% gamma efficiency (lower is better).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = Path("data")
PARTICLE_NAMES = ["proton", "helium", "carbon", "silicon", "iron"]
PARTICLE_ID_MAP = {14: 0, 402: 1, 1206: 2, 2814: 3, 5626: 4}


def load_test_truth(task: str):
    """Load test set ground truth for the given task.

    Returns:
        labels: int8 array — composition (0-4) or gamma (0=gamma, 1=hadron)
        energies: float array — reconstructed E (log10(E/eV)), feature index 0
        n_test: number of test samples
    """
    test_dir = DATA_DIR / f"{task}_test"

    if not test_dir.exists():
        print(f"Error: {test_dir} not found. Run download_data.py first.")
        sys.exit(1)

    if task == "composition":
        labels = np.load(test_dir / "labels_composition.npy", mmap_mode="r")
    else:
        labels = np.load(test_dir / "labels_gamma.npy", mmap_mode="r")

    features = np.load(test_dir / "features.npy", mmap_mode="r")
    energies = np.array(features[:, 0])  # copy from mmap
    return np.array(labels), energies, len(labels)


# ---------------------------------------------------------------------------
# Task 1: Mass composition (5-class)
# ---------------------------------------------------------------------------


def evaluate_composition(predictions, labels_composition, energies):
    """Evaluate 5-class mass composition predictions."""
    truth = labels_composition

    results = {}
    results["accuracy"] = float(accuracy_score(truth, predictions))

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
    results["confusion_matrix"] = confusion_matrix(truth, predictions).tolist()

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


def print_composition_results(results):
    """Pretty-print composition evaluation results."""
    print(f"\n{'='*60}")
    print(f"  TASK: Mass Composition (5-class)")
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
            print(
                f"{name:<10} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1']:>10.4f} {m['support']:>10}"
            )

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


# ---------------------------------------------------------------------------
# Task 2: Gamma/hadron separation (binary)
# ---------------------------------------------------------------------------


def evaluate_gamma(gamma_scores, labels_gamma, energies):
    """Evaluate gamma/hadron separation.

    Key metric: hadronic survival rate at 99% gamma efficiency.
    This is the fraction of hadrons that survive the gamma selection cut
    while retaining 99% of true gammas.
    """
    # Labels: 0=gamma, 1=hadron
    is_gamma = labels_gamma == 0
    is_hadron = labels_gamma == 1
    n_gamma = is_gamma.sum()
    n_hadron = is_hadron.sum()

    if n_gamma == 0 or n_hadron == 0:
        print("Error: test set must contain both gamma and hadron events")
        sys.exit(1)

    results = {}
    results["n_gamma"] = int(n_gamma)
    results["n_hadron"] = int(n_hadron)

    # Sort scores descending (higher = more gamma-like)
    gamma_scores_gamma = np.sort(gamma_scores[is_gamma])[::-1]
    gamma_scores_hadron = gamma_scores[is_hadron]

    # Find threshold at target gamma efficiencies
    target_efficiencies = [0.50, 0.90, 0.95, 0.99]
    results["survival_rates"] = {}

    for eff in target_efficiencies:
        # Threshold: score above which we keep eff fraction of gammas
        idx = int(np.ceil(eff * n_gamma)) - 1
        threshold = gamma_scores_gamma[min(idx, len(gamma_scores_gamma) - 1)]

        # Hadronic survival: fraction of hadrons above threshold
        n_hadron_surviving = (gamma_scores_hadron >= threshold).sum()
        survival_rate = float(n_hadron_surviving / n_hadron)

        results["survival_rates"][f"{eff:.0%}"] = {
            "threshold": float(threshold),
            "gamma_efficiency": float((gamma_scores[is_gamma] >= threshold).sum() / n_gamma),
            "hadron_survival": survival_rate,
            "hadron_surviving": int(n_hadron_surviving),
        }

    # Key metric: survival rate at 99% gamma efficiency
    results["key_metric"] = results["survival_rates"]["99%"]["hadron_survival"]

    # Binary classification metrics at 50% threshold
    binary_preds = (gamma_scores >= 0.5).astype(int)  # 1=gamma, 0=hadron
    truth_binary = is_gamma.astype(int)  # 1=gamma, 0=hadron
    results["accuracy_at_0.5"] = float(accuracy_score(truth_binary, binary_preds))

    # Energy-binned survival rate at 99% gamma efficiency
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
        if mask.sum() == 0:
            continue
        g_mask = mask & is_gamma
        h_mask = mask & is_hadron
        ng = g_mask.sum()
        nh = h_mask.sum()
        if ng == 0 or nh == 0:
            continue
        # 99% gamma efficiency in this bin
        gs = np.sort(gamma_scores[g_mask])[::-1]
        idx = int(np.ceil(0.99 * ng)) - 1
        thr = gs[min(idx, len(gs) - 1)]
        n_surv = (gamma_scores[h_mask] >= thr).sum()
        results["energy_binned"][label] = {
            "hadron_survival": float(n_surv / nh),
            "n_gamma": int(ng),
            "n_hadron": int(nh),
        }

    return results


def print_gamma_results(results):
    """Pretty-print gamma/hadron evaluation results."""
    print(f"\n{'='*60}")
    print(f"  TASK: Gamma/Hadron Separation")
    key = results["key_metric"]
    print(f"  KEY METRIC (hadron survival @ 99% gamma eff): {key:.2e}")
    print(f"  Events: {results['n_gamma']} gamma, {results['n_hadron']} hadron")
    print(f"{'='*60}")

    print(f"\n{'Survival rates at different gamma efficiencies':^60}")
    print(f"{'Gamma eff':<12} {'Hadron survival':>16} {'Hadrons left':>14} {'Threshold':>10}")
    print("-" * 56)
    for label, m in results["survival_rates"].items():
        print(
            f"{label:<12} {m['hadron_survival']:>16.2e} "
            f"{m['hadron_surviving']:>14} {m['threshold']:>10.4f}"
        )

    if results["energy_binned"]:
        print(f"\n{'Energy-binned hadron survival @ 99% gamma eff':^60}")
        print(f"{'Bin':<15} {'Survival':>12} {'Gamma':>8} {'Hadron':>8}")
        print("-" * 48)
        for label, m in results["energy_binned"].items():
            print(
                f"{label:<15} {m['hadron_survival']:>12.2e} "
                f"{m['n_gamma']:>8} {m['n_hadron']:>8}"
            )

    print(f"\n  Published baseline: 1e-6 to 3e-5 (Petrov et al., Chinese Physics C 2023)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Verify predictions against test set")
    parser.add_argument("predictions", type=Path, help="Path to predictions.npz")
    parser.add_argument(
        "--task",
        choices=["composition", "gamma"],
        default="composition",
        help="Which task to evaluate (default: composition)",
    )
    args = parser.parse_args()

    if not args.predictions.exists():
        print(f"Error: {args.predictions} not found")
        sys.exit(1)

    pred_data = np.load(args.predictions)
    labels, energies, n_test = load_test_truth(args.task)

    if args.task == "composition":
        if "predictions" not in pred_data:
            print("Error: predictions.npz must contain a 'predictions' array")
            sys.exit(1)
        predictions = pred_data["predictions"].astype(int)
        if len(predictions) != n_test:
            print(f"Error: predictions has {len(predictions)} elements, expected {n_test}")
            sys.exit(1)
        invalid = (predictions < 0) | (predictions > 4)
        if invalid.any():
            print(f"Error: {invalid.sum()} predictions outside valid range [0, 4]")
            sys.exit(1)
        results = evaluate_composition(predictions, labels, energies)
        print_composition_results(results)

    elif args.task == "gamma":
        if "gamma_scores" not in pred_data:
            print("Error: predictions.npz must contain a 'gamma_scores' float array")
            print("  Higher values = more likely to be gamma")
            sys.exit(1)
        gamma_scores = pred_data["gamma_scores"].astype(float)
        if len(gamma_scores) != n_test:
            print(f"Error: gamma_scores has {len(gamma_scores)} elements, expected {n_test}")
            sys.exit(1)
        results = evaluate_gamma(gamma_scores, labels, energies)
        print_gamma_results(results)

    # Save results
    results_path = args.predictions.parent / f"metrics_{args.task}.json"
    results_json = json.loads(
        json.dumps(
            results,
            default=lambda x: int(x)
            if isinstance(x, np.integer)
            else float(x)
            if isinstance(x, np.floating)
            else x,
        )
    )
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
