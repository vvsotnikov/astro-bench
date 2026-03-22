"""Evaluate gamma/hadron predictions and log the attempt.

Usage:
    python verify.py predictions.npz "MLP with engineered features, 3 layers"

Each call counts as one attempt (max 50). Results are auto-logged to results.tsv.

Key metric: hadronic survival rate at 75% gamma efficiency (lower is better).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

DATA_DIR = Path("data")
MAX_ATTEMPTS = 50


def load_test_data():
    test_dir = DATA_DIR / "gamma_test"
    if not test_dir.exists():
        print("Error: data/gamma_test/ not found. Run download_data.py first.")
        sys.exit(1)
    labels = np.array(np.load(test_dir / "labels_gamma.npy", mmap_mode="r"))
    features = np.array(np.load(test_dir / "features.npy", mmap_mode="r"), dtype=np.float32)
    return labels, features


def survival_at_efficiency(scores_gamma, scores_hadron, efficiency):
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


def get_attempt_number():
    tsv = Path("results.tsv")
    if not tsv.exists():
        return 1
    with open(tsv) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("attempt")]
    non_baseline = [l for l in lines if not l.startswith("0\t")]
    return len(non_baseline) + 1


def log_result(attempt, survival, description, pred_path):
    tsv = Path("results.tsv")
    if not tsv.exists():
        with open(tsv, "w") as f:
            f.write("attempt\tsurvival_75\tpredictions\tdescription\n")
    with open(tsv, "a") as f:
        f.write(f"{attempt}\t{survival:.4e}\t{pred_path}\t{description}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate gamma/hadron predictions (counts as one attempt)")
    parser.add_argument("predictions", type=Path, help="Path to predictions.npz")
    parser.add_argument("description", type=str,
                        help="What you tried (required). E.g. 'CNN + attention, 3 seeds ensemble'")
    args = parser.parse_args()

    if not args.predictions.exists():
        print(f"Error: {args.predictions} not found")
        sys.exit(1)

    attempt = get_attempt_number()
    if attempt > MAX_ATTEMPTS:
        print(f"Error: Maximum {MAX_ATTEMPTS} attempts reached.")
        sys.exit(1)

    pred_data = np.load(args.predictions)
    if "gamma_scores" not in pred_data:
        print("Error: predictions.npz must contain 'gamma_scores' (float array, higher = more gamma-like)")
        sys.exit(1)

    gamma_scores = pred_data["gamma_scores"].astype(float)
    labels, features = load_test_data()

    if len(gamma_scores) != len(labels):
        print(f"Error: gamma_scores has {len(gamma_scores)} elements, expected {len(labels)}")
        sys.exit(1)

    is_gamma = labels == 0
    is_hadron = labels == 1

    # Survival at multiple efficiencies
    target_effs = [0.50, 0.75, 0.90, 0.95]
    print(f"\n{'=' * 60}")
    print(f"  ATTEMPT: {attempt}/{MAX_ATTEMPTS}")
    print(f"  DESCRIPTION: {args.description}")

    key_result = survival_at_efficiency(gamma_scores[is_gamma], gamma_scores[is_hadron], 0.75)
    key_metric = key_result["hadron_survival"]
    print(f"  KEY METRIC (hadron survival @ 75% γ eff): {key_metric:.2e}")
    print(f"  Events: {int(is_gamma.sum())} gamma, {int(is_hadron.sum())} hadron")
    print(f"{'=' * 60}")

    print(f"\n  {'Gamma eff':<12} {'Survival':>16} {'Hadrons left':>14} {'Threshold':>10}")
    print(f"  {'-' * 56}")
    for eff in target_effs:
        sr = survival_at_efficiency(gamma_scores[is_gamma], gamma_scores[is_hadron], eff)
        print(f"  {eff:.0%}{'':>8} {sr['hadron_survival']:>16.2e} "
              f"{sr['hadron_surviving']:>14} {sr['threshold']:>10.4f}")

    # Log
    log_result(attempt, key_metric, args.description, str(args.predictions))

    # Best so far
    tsv = Path("results.tsv")
    if tsv.exists():
        best = float("inf")
        with open(tsv) as f:
            for line in f:
                if line.startswith("attempt"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        best = min(best, float(parts[1]))
                    except ValueError:
                        pass
        if key_metric <= best:
            print(f"\n  *** NEW BEST: {key_metric:.2e} ***")
        else:
            print(f"\n  Current best: {best:.2e} (this attempt: {key_metric:.2e})")

    print(f"\n  Attempts used: {attempt}/{MAX_ATTEMPTS}")


if __name__ == "__main__":
    main()
