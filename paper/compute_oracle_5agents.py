"""Compute 3-agent and 5-agent oracle to support the §4.2.2 reviewer question.

Reports:
  - Per-agent Best@50 fraction error
  - 3-agent oracle (Opus/GPT/Sonnet) — current paper baseline
  - 5-agent oracle (+ Kimi, Qwen) — reviewer-requested comparison
  - Independence-null oracle for both
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "composition"))
from verify import _compute_fraction_error, _load_test_data  # noqa: E402

ROOT = Path("/home/vladimir/cursor_projects/astro-bench-experiments")

BEST = {
    "Opus":   "composition-opus-2apr/predictions_v34.npz",
    "GPT":    "composition-gpt-2apr/predictions_v47.npz",
    "Sonnet": "composition-sonnet-27mar/predictions_v44.npz",
    "Kimi":   "composition-kimi-5apr/predictions_v10.npz",
    "Qwen":   "composition-qwen-6apr/predictions_v10.npz",
}


def oracle_pred(labels, preds_list):
    """For each event, if ANY agent predicts true class, oracle picks that class."""
    n = len(labels)
    out = preds_list[0].copy()
    for i in range(n):
        if labels[i] in [p[i] for p in preds_list]:
            out[i] = labels[i]
    return out


def independence_null(preds_list, labels, n_classes=5):
    """P(at least one correct | class c) = 1 - prod_i (1 - p_i(c)) summed over prior."""
    confs = []
    for p in preds_list:
        # p_i(c) = fraction of events with true class c that agent got right
        p_c = np.zeros(n_classes)
        for c in range(n_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                p_c[c] = (p[mask] == c).mean()
        confs.append(p_c)
    priors = np.bincount(labels, minlength=n_classes).astype(float)
    priors /= priors.sum()
    # Per-class P(at least one correct)
    at_least_one = np.ones(n_classes)
    for p_c in confs:
        at_least_one *= (1 - p_c)
    at_least_one = 1 - at_least_one
    return float((priors * at_least_one).sum())


def main():
    labels, _ = _load_test_data()
    print(f"Test set: {len(labels)} events\n")

    preds = {}
    for name, rel in BEST.items():
        arr = np.load(ROOT / rel)["predictions"].astype(int)
        preds[name] = arr
        fe = _compute_fraction_error(labels, arr)["mean"]
        acc = (arr == labels).mean()
        print(f"  {name:>7s}: FE={fe:.5f}  acc={acc:.4f}")

    print("\n=== 3-agent oracle (Opus, GPT, Sonnet) ===")
    preds_3 = [preds[n] for n in ["Opus", "GPT", "Sonnet"]]
    o3 = oracle_pred(labels, preds_3)
    fe_3 = _compute_fraction_error(labels, o3)["mean"]
    acc_3 = (o3 == labels).mean()
    ind_3 = independence_null(preds_3, labels)
    print(f"  Oracle FE:            {fe_3:.5f}")
    print(f"  Oracle accuracy:      {acc_3:.4f}  ({100*acc_3:.2f}%)")
    print(f"  Independence-null acc: {ind_3:.4f}  ({100*ind_3:.2f}%)")

    print("\n=== 5-agent oracle (+ Kimi, Qwen) ===")
    preds_5 = [preds[n] for n in ["Opus", "GPT", "Sonnet", "Kimi", "Qwen"]]
    o5 = oracle_pred(labels, preds_5)
    fe_5 = _compute_fraction_error(labels, o5)["mean"]
    acc_5 = (o5 == labels).mean()
    ind_5 = independence_null(preds_5, labels)
    print(f"  Oracle FE:            {fe_5:.5f}")
    print(f"  Oracle accuracy:      {acc_5:.4f}  ({100*acc_5:.2f}%)")
    print(f"  Independence-null acc: {ind_5:.4f}  ({100*ind_5:.2f}%)")

    print(f"\nDelta (5-agent vs 3-agent): FE {fe_3 - fe_5:+.5f}, acc {acc_5 - acc_3:+.4f}")


if __name__ == "__main__":
    main()
