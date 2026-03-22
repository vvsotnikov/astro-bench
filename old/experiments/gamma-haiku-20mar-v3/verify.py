"""Evaluate gamma/hadron predictions and log the attempt.

CLI usage:
    python verify.py predictions.npz "MLP with engineered features, 3 layers"

Python usage:
    from verify import evaluate
    metric = evaluate(gamma_scores, "v3: CNN + attention, seed=42")

Each call counts as one attempt (max 50). Results are auto-logged to results.tsv.
Key metric: hadronic survival rate at 75% gamma efficiency (lower is better).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent
_DATA_DIR = _ROOT / "data"
_TSV = _ROOT / "results.tsv"
MAX_ATTEMPTS = 50


def _load_test_labels():
    test_dir = _DATA_DIR / "gamma_test"
    if not test_dir.exists():
        raise FileNotFoundError(f"{test_dir} not found. Run download_data.py first.")
    return np.array(np.load(test_dir / "labels_gamma.npy", mmap_mode="r"))


def _survival_at_efficiency(scores_gamma, scores_hadron, efficiency):
    ng, nh = len(scores_gamma), len(scores_hadron)
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


def _ensure_baseline():
    if not _TSV.exists():
        with open(_TSV, "w") as f:
            f.write("attempt\tsurvival_75\tpredictions\tdescription\n")
            f.write("0\t1.0e-02\tbaseline/predictions.npz\tRF baseline (published, Kostunin et al. ICRC 2021)\n")


def _get_attempt_number():
    _ensure_baseline()
    with open(_TSV) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("attempt")]
    non_baseline = [l for l in lines if not l.startswith("0\t")]
    return len(non_baseline) + 1


def _log_result(attempt, survival, description, pred_path):
    _ensure_baseline()
    with open(_TSV, "a") as f:
        f.write(f"{attempt}\t{survival:.4e}\t{pred_path}\t{description}\n")


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


def evaluate(gamma_scores, description, pred_path="(from python)"):
    """Evaluate gamma scores and log the attempt. Returns the key metric.

    Args:
        gamma_scores: float array (N,), higher = more gamma-like
        description: what you tried (logged to results.tsv)
        pred_path: optional path string for logging

    Returns:
        float: hadronic survival rate at 75% gamma efficiency (lower is better)

    Raises:
        RuntimeError: if max attempts exceeded
    """
    gamma_scores = np.asarray(gamma_scores, dtype=float)
    labels = _load_test_labels()

    if len(gamma_scores) != len(labels):
        raise ValueError(f"gamma_scores has {len(gamma_scores)} elements, expected {len(labels)}")

    attempt = _get_attempt_number()
    if attempt > MAX_ATTEMPTS:
        raise RuntimeError(f"Maximum {MAX_ATTEMPTS} attempts reached. Check results.tsv for your best result.")

    is_gamma = labels == 0
    is_hadron = labels == 1

    key_result = _survival_at_efficiency(gamma_scores[is_gamma], gamma_scores[is_hadron], 0.75)
    key_metric = key_result["hadron_survival"]

    # Log
    _log_result(attempt, key_metric, description, str(pred_path))

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  ATTEMPT: {attempt}/{MAX_ATTEMPTS}")
    print(f"  DESCRIPTION: {description}")
    print(f"  KEY METRIC (hadron survival @ 75% γ eff): {key_metric:.2e}")
    print(f"  Events: {int(is_gamma.sum())} gamma, {int(is_hadron.sum())} hadron")
    print(f"{'=' * 60}")

    target_effs = [0.50, 0.75, 0.90, 0.95]
    print(f"\n  {'Gamma eff':<12} {'Survival':>16} {'Hadrons left':>14} {'Threshold':>10}")
    print(f"  {'-' * 56}")
    for eff in target_effs:
        sr = _survival_at_efficiency(gamma_scores[is_gamma], gamma_scores[is_hadron], eff)
        print(f"  {eff:.0%}{'':>8} {sr['hadron_survival']:>16.2e} "
              f"{sr['hadron_surviving']:>14} {sr['threshold']:>10.4f}")

    best = _get_best()
    if key_metric <= best:
        print(f"\n  *** NEW BEST: {key_metric:.2e} ***")
    else:
        print(f"\n  Current best: {best:.2e} (this attempt: {key_metric:.2e})")

    print(f"\n  Attempts used: {attempt}/{MAX_ATTEMPTS}")
    return key_metric


# --- CLI entry point ---

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

    pred_data = np.load(args.predictions)
    if "gamma_scores" not in pred_data:
        print("Error: predictions.npz must contain 'gamma_scores' (float array, higher = more gamma-like)")
        sys.exit(1)

    gamma_scores = pred_data["gamma_scores"]
    evaluate(gamma_scores, args.description, pred_path=str(args.predictions))


if __name__ == "__main__":
    main()
