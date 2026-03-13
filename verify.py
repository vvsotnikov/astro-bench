"""Verify a submission's predictions against the held-out test set.

Usage:
    python verify.py submissions/my_submission/predictions.npz
    python verify.py --task gamma submissions/my_submission/predictions.npz

Task 1 (composition, default):
    predictions.npz must contain "predictions": int array of class labels (0-4)
    0=proton, 1=helium, 2=carbon, 3=silicon, 4=iron
    Key metric: mean fraction error — how well the classifier recovers
    particle fractions across energy bins and random mixture compositions.
    Lower is better (0 = perfect fraction recovery).

Task 2 (gamma):
    predictions.npz must contain "gamma_scores": float array of gamma probabilities
    Higher score = more likely to be gamma.
    Key metric: hadronic survival rate at 75% gamma efficiency (lower is better).
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


ENERGY_BINS = [
    (14.0, 15.0, "14.0-15.0"),
    (15.0, 15.5, "15.0-15.5"),
    (15.5, 16.0, "15.5-16.0"),
    (16.0, 16.5, "16.0-16.5"),
    (16.5, 17.0, "16.5-17.0"),
    (17.0, 18.0, "17.0-18.0"),
]


def load_test_truth(task: str):
    """Load test set ground truth for the given task.

    Returns:
        labels: int8 array — composition (0-4) or gamma (0=gamma, 1=hadron)
        features: float array — (N, 5) reconstructed features [E, Ze, Az, Ne, Nmu]
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
    return np.array(labels), np.array(features, dtype=np.float32), len(labels)


# ---------------------------------------------------------------------------
# Task 1: Mass composition (5-class)
# ---------------------------------------------------------------------------

# Mixture evaluation parameters
N_MIXTURES = 1000       # number of random mixtures per energy bin
MIXTURE_SIZE = 5000     # events per mixture
MIXTURE_SEED = 2026     # fixed seed for reproducibility


def _fraction_error_for_bin(truth_bin, pred_bin, rng):
    """Compute mean absolute fraction error for one energy bin.

    Samples N_MIXTURES random mixtures from the events in this bin,
    each with random target class fractions drawn from Dirichlet(1,...,1).
    For each mixture, samples MIXTURE_SIZE events with those fractions,
    then compares true fractions vs predicted fractions.

    Returns:
        mean_frac_error: float — mean |true_frac - pred_frac| averaged
                         over classes and mixtures
        per_class_errors: list[float] — mean |true_frac - pred_frac| per class
        details: dict — detailed results for analysis
    """
    n_classes = 5
    classes = np.arange(n_classes)

    # Group event indices by true class
    class_indices = {c: np.where(truth_bin == c)[0] for c in classes}
    class_counts = {c: len(class_indices[c]) for c in classes}

    # Skip if any class is missing (can't form mixtures)
    if any(class_counts[c] == 0 for c in classes):
        return None, None, None

    # Sample random mixture fractions from Dirichlet(1,1,1,1,1) = uniform on simplex
    fractions = rng.dirichlet(np.ones(n_classes), size=N_MIXTURES)

    all_errors = []  # (N_MIXTURES, n_classes)

    for mix_idx in range(N_MIXTURES):
        target_fracs = fractions[mix_idx]
        counts_per_class = np.round(target_fracs * MIXTURE_SIZE).astype(int)
        # Adjust rounding to hit exactly MIXTURE_SIZE
        diff = MIXTURE_SIZE - counts_per_class.sum()
        if diff != 0:
            # Add/subtract from the largest class
            counts_per_class[np.argmax(counts_per_class)] += diff

        # Sample events for this mixture
        sampled_preds = []
        actual_true_fracs = np.zeros(n_classes)
        for c in classes:
            n_sample = counts_per_class[c]
            if n_sample <= 0:
                continue
            # Sample with replacement from this class's events
            idx = rng.choice(class_indices[c], size=n_sample, replace=True)
            sampled_preds.append(pred_bin[idx])
            actual_true_fracs[c] = n_sample

        actual_true_fracs /= actual_true_fracs.sum()
        all_preds = np.concatenate(sampled_preds)

        # Predicted fractions
        pred_counts = np.bincount(all_preds, minlength=n_classes)[:n_classes]
        pred_fracs = pred_counts / pred_counts.sum()

        # Absolute fraction error per class
        errors = np.abs(actual_true_fracs - pred_fracs)
        all_errors.append(errors)

    all_errors = np.array(all_errors)  # (N_MIXTURES, n_classes)
    mean_frac_error = float(all_errors.mean())
    per_class_errors = [float(all_errors[:, c].mean()) for c in classes]

    details = {
        "mean_fraction_error": mean_frac_error,
        "per_class_fraction_error": per_class_errors,
        "n_mixtures": N_MIXTURES,
        "mixture_size": MIXTURE_SIZE,
    }
    return mean_frac_error, per_class_errors, details


def evaluate_composition(predictions, labels_composition, features):
    """Evaluate 5-class mass composition predictions.

    Key metric: mean fraction error — averaged across energy bins and
    random mixture compositions. This measures how well the classifier
    can recover the true particle fractions in a realistic scenario
    where the composition is unknown.
    """
    truth = labels_composition
    energies = features[:, 0]
    rng = np.random.default_rng(MIXTURE_SEED)

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

    # Energy-binned accuracy, confusion matrices, and fraction errors
    results["energy_binned"] = {}
    bin_fraction_errors = []
    for lo, hi, label in ENERGY_BINS:
        mask = (energies >= lo) & (energies < hi)
        if mask.sum() > 0:
            cm = confusion_matrix(truth[mask], predictions[mask], labels=list(range(5)))
            bin_result = {
                "accuracy": float(accuracy_score(truth[mask], predictions[mask])),
                "count": int(mask.sum()),
                "confusion_matrix": cm.tolist(),
            }
            # Fraction error for this energy bin
            mfe, pce, details = _fraction_error_for_bin(
                truth[mask], predictions[mask], rng
            )
            if mfe is not None:
                bin_result["fraction_error"] = details
                bin_fraction_errors.append(mfe)
            results["energy_binned"][label] = bin_result

    # Key metric: mean fraction error across energy bins
    if bin_fraction_errors:
        results["mean_fraction_error"] = float(np.mean(bin_fraction_errors))
    else:
        results["mean_fraction_error"] = 1.0

    return results


def print_composition_results(results):
    """Pretty-print composition evaluation results."""
    mfe = results["mean_fraction_error"]
    print(f"\n{'='*60}")
    print(f"  TASK: Mass Composition (5-class)")
    print(f"  KEY METRIC (mean fraction error): {mfe:.4f}")
    print(f"  ACCURACY: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
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

    print(f"\n{'Energy-binned accuracy & fraction error':^60}")
    print(f"{'Bin':<15} {'Accuracy':>10} {'Frac Err':>10} {'Count':>10}")
    print("-" * 50)
    for label, m in results["energy_binned"].items():
        fe = m.get("fraction_error", {}).get("mean_fraction_error", None)
        fe_str = f"{fe:>10.4f}" if fe is not None else f"{'—':>10}"
        print(f"{label:<15} {m['accuracy']:>10.4f} {fe_str} {m['count']:>10}")

    # Per-class fraction errors per energy bin
    print(f"\n{'Per-class fraction error by energy bin':^60}")
    abbrev = [n[:2] for n in PARTICLE_NAMES]
    print(f"{'Bin':<15}" + "".join(f"{a:>8}" for a in abbrev))
    print("-" * (15 + 8 * 5))
    for label, m in results["energy_binned"].items():
        fe = m.get("fraction_error")
        if fe:
            pce = fe["per_class_fraction_error"]
            vals = "".join(f"{v:>8.4f}" for v in pce)
        else:
            vals = "".join(f"{'—':>8}" for _ in range(5))
        print(f"{label:<15}{vals}")

    # Normalized confusion matrices per energy bin
    print(f"\n{'Energy-binned confusion matrices (row-normalized)':^60}")
    for label, m in results["energy_binned"].items():
        cm = np.array(m["confusion_matrix"])
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums
        print(f"\n  {label} (n={m['count']})")
        print(f"  {'':>4}" + "".join(f"{a:>6}" for a in abbrev))
        for i, name in enumerate(abbrev):
            row = "".join(f"{cm_norm[i, j]:>6.2f}" for j in range(5))
            print(f"  {name:>4}{row}")


# ---------------------------------------------------------------------------
# Task 2: Gamma/hadron separation (binary)
# ---------------------------------------------------------------------------


def _survival_at_efficiency(scores_gamma, scores_hadron, efficiency):
    """Compute hadronic survival rate at a given gamma efficiency."""
    ng = len(scores_gamma)
    nh = len(scores_hadron)
    if ng == 0 or nh == 0:
        return None
    sorted_g = np.sort(scores_gamma)[::-1]
    idx = int(np.ceil(efficiency * ng)) - 1
    thr = sorted_g[min(idx, ng - 1)]
    n_surv = (scores_hadron >= thr).sum()
    return {
        "threshold": float(thr),
        "gamma_efficiency": float((scores_gamma >= thr).sum() / ng),
        "hadron_survival": float(n_surv / nh),
        "hadron_surviving": int(n_surv),
    }


def evaluate_gamma(gamma_scores, labels_gamma, features):
    """Evaluate gamma/hadron separation.

    Key metric: hadronic survival rate at 75% gamma efficiency.
    This is the fraction of hadrons that survive the gamma selection cut
    while retaining 75% of true gammas.
    """
    energies = features[:, 0]
    zeniths = features[:, 1]

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

    # Survival rates at target gamma efficiencies
    target_efficiencies = [0.50, 0.75, 0.90, 0.95, 0.99]
    results["survival_rates"] = {}
    for eff in target_efficiencies:
        sr = _survival_at_efficiency(gamma_scores[is_gamma], gamma_scores[is_hadron], eff)
        results["survival_rates"][f"{eff:.0%}"] = sr

    results["key_metric"] = results["survival_rates"]["75%"]["hadron_survival"]

    # Binary classification metrics at 50% threshold
    binary_preds = (gamma_scores >= 0.5).astype(int)
    truth_binary = is_gamma.astype(int)
    results["accuracy_at_0.5"] = float(accuracy_score(truth_binary, binary_preds))

    # Energy-binned survival rate at 99% gamma efficiency
    results["energy_binned"] = {}
    for lo, hi, label in ENERGY_BINS:
        mask = (energies >= lo) & (energies < hi)
        g_mask = mask & is_gamma
        h_mask = mask & is_hadron
        sr = _survival_at_efficiency(gamma_scores[g_mask], gamma_scores[h_mask], 0.75)
        if sr is not None:
            sr["n_gamma"] = int(g_mask.sum())
            sr["n_hadron"] = int(h_mask.sum())
            results["energy_binned"][label] = sr

    # Zenith-angle-binned survival rate at 99% gamma efficiency
    zenith_bins = [(0, 10, "0-10"), (10, 20, "10-20"), (20, 30, "20-30")]
    results["zenith_binned"] = {}
    for lo, hi, label in zenith_bins:
        mask = (zeniths >= lo) & (zeniths < hi)
        g_mask = mask & is_gamma
        h_mask = mask & is_hadron
        sr = _survival_at_efficiency(gamma_scores[g_mask], gamma_scores[h_mask], 0.75)
        if sr is not None:
            sr["n_gamma"] = int(g_mask.sum())
            sr["n_hadron"] = int(h_mask.sum())
            results["zenith_binned"][label] = sr

    # 2D grid: energy × zenith (matches published analysis format)
    # Uses 8 energy bins from 14-18 and 3 zenith bands
    grid_energy_bins = [(14 + i * 0.5, 14 + (i + 1) * 0.5) for i in range(8)]
    results["energy_zenith_grid"] = {}
    for ze_lo, ze_hi, ze_label in zenith_bins:
        ze_mask = (zeniths >= ze_lo) & (zeniths < ze_hi)
        grid_row = {}
        for e_lo, e_hi in grid_energy_bins:
            e_label = f"{e_lo:.1f}-{e_hi:.1f}"
            mask = ze_mask & (energies >= e_lo) & (energies < e_hi)
            g_mask = mask & is_gamma
            h_mask = mask & is_hadron
            ng, nh = int(g_mask.sum()), int(h_mask.sum())
            sr = _survival_at_efficiency(gamma_scores[g_mask], gamma_scores[h_mask], 0.75)
            if sr is not None:
                sr["n_gamma"] = ng
                sr["n_hadron"] = nh
                grid_row[e_label] = sr
        if grid_row:
            results["energy_zenith_grid"][ze_label] = grid_row

    return results


def print_gamma_results(results):
    """Pretty-print gamma/hadron evaluation results."""
    print(f"\n{'='*60}")
    print(f"  TASK: Gamma/Hadron Separation")
    key = results["key_metric"]
    print(f"  KEY METRIC (hadron survival @ 75% gamma eff): {key:.2e}")
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
        print(f"\n{'Energy-binned hadron survival @ 75% gamma eff':^60}")
        print(f"{'Bin':<15} {'Survival':>12} {'Gamma':>8} {'Hadron':>8}")
        print("-" * 48)
        for label, m in results["energy_binned"].items():
            print(
                f"{label:<15} {m['hadron_survival']:>12.2e} "
                f"{m['n_gamma']:>8} {m['n_hadron']:>8}"
            )

    if results.get("zenith_binned"):
        print(f"\n{'Zenith-binned hadron survival @ 75% gamma eff':^60}")
        print(f"{'Ze (deg)':<15} {'Survival':>12} {'Gamma':>8} {'Hadron':>8}")
        print("-" * 48)
        for label, m in results["zenith_binned"].items():
            print(
                f"{label:<15} {m['hadron_survival']:>12.2e} "
                f"{m['n_gamma']:>8} {m['n_hadron']:>8}"
            )

    if results.get("energy_zenith_grid"):
        print(f"\n{'Energy x Zenith grid — hadron survival @ 75% gamma eff':^72}")
        # Collect all energy bin labels
        all_e_labels = []
        for ze_label, row in results["energy_zenith_grid"].items():
            for e_label in row:
                if e_label not in all_e_labels:
                    all_e_labels.append(e_label)
        header = f"{'Ze (deg)':<10}" + "".join(f"{e:>10}" for e in all_e_labels)
        print(header)
        print("-" * len(header))
        for ze_label, row in results["energy_zenith_grid"].items():
            vals = []
            for e_label in all_e_labels:
                if e_label in row:
                    sr = row[e_label]["hadron_survival"]
                    ng = row[e_label]["n_gamma"]
                    if ng < 5:
                        vals.append(f"{'—':>10}")
                    else:
                        vals.append(f"{sr:>10.2e}")
                else:
                    vals.append(f"{'—':>10}")
            print(f"{ze_label:<10}" + "".join(vals))

    print(f"\n  Published baseline: suppression 1e2-1e3 at ~30-70% gamma eff (Kostunin et al., ICRC 2021)")


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
    labels, features, n_test = load_test_truth(args.task)

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
        results = evaluate_composition(predictions, labels, features)
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
        results = evaluate_gamma(gamma_scores, labels, features)
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
