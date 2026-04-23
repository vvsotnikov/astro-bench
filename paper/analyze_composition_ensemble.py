"""Investigate why majority voting on composition yields exactly 0.1051 = Opus.

Loads the best predictions from each of Opus/GPT/Sonnet composition runs and:
  - Reports per-agent fraction error (recomputed via verify.py's evaluator)
  - Computes majority-vote ensemble
  - Measures Opus dominance: how often Opus is in the winning plurality on
    disagreements, and how often the ensemble output equals Opus's output
  - Reports the fraction error at higher precision (beyond the 4dp rounding)
"""

from pathlib import Path
import sys

import numpy as np

# Use verify.py's _compute_fraction_error directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "composition"))
from verify import _compute_fraction_error, _load_test_data  # noqa: E402


BEST = {
    # Opus v34 is the final 5-model ensemble at 0.1051 (same as v22a/b/c, which no
    # longer have saved .npz files). This is also the model used for the real-data
    # application in Section 6.1.
    "Opus":   "composition-opus-2apr/predictions_v34.npz",
    "GPT":    "composition-gpt-2apr/predictions_v47.npz",
    "Sonnet": "composition-sonnet-27mar/predictions_v44.npz",
}
ROOT = Path("/home/vladimir/cursor_projects/astro-bench-experiments")


def load_predictions():
    preds = {}
    for name, rel in BEST.items():
        arr = np.load(ROOT / rel)["predictions"].astype(int)
        preds[name] = arr
        print(f"  {name:>7s}: {rel}  shape={arr.shape}")
    return preds


def majority_vote(stacked):
    """Per-row majority vote over shape (n_agents, n_events).

    Tie-breaking for 3-way disagreements (all three pick different classes):
    use row 0 (Opus) as the tie-breaker. We measure how often this happens.
    """
    n_agents, n_events = stacked.shape
    out = np.empty(n_events, dtype=int)
    tiebreak_mask = np.zeros(n_events, dtype=bool)
    for i in range(n_events):
        vals, counts = np.unique(stacked[:, i], return_counts=True)
        if counts.max() >= 2:
            out[i] = int(vals[counts.argmax()])
        else:
            # 3-way disagreement, all three classes different
            out[i] = int(stacked[0, i])  # Opus breaks ties
            tiebreak_mask[i] = True
    return out, tiebreak_mask


def main():
    labels, _ = _load_test_data()
    print(f"Test set: {len(labels)} events")
    preds = load_predictions()

    names = ["Opus", "GPT", "Sonnet"]
    stacked = np.stack([preds[n] for n in names])  # (3, N)
    n_events = stacked.shape[1]

    # Per-agent fraction error (full precision)
    print("\n=== Per-agent mean fraction error (full precision) ===")
    individual_fe = {}
    for n in names:
        res = _compute_fraction_error(labels, preds[n])
        fe = res["mean"]
        individual_fe[n] = fe
        print(f"  {n:>7s}: {fe:.8f}  (rounded: {fe:.4f})")

    # Majority-vote ensemble
    mv, tiebreak_mask = majority_vote(stacked)
    res_mv = _compute_fraction_error(labels, mv)
    fe_mv = res_mv["mean"]
    print(f"\n  Majority vote: {fe_mv:.8f}  (rounded: {fe_mv:.4f})")
    print(f"  Opus vs MV: {'EXACTLY EQUAL' if fe_mv == individual_fe['Opus'] else 'DIFFERENT'}")
    print(f"  |Opus - MV| = {abs(fe_mv - individual_fe['Opus']):.3e}")

    # Agreement statistics
    print("\n=== Agreement structure ===")
    all_agree = (stacked[0] == stacked[1]) & (stacked[1] == stacked[2])
    n_all_agree = int(all_agree.sum())
    print(f"  All 3 agree:            {n_all_agree:>7d}  ({100*n_all_agree/n_events:.2f}%)")
    n_disagree = n_events - n_all_agree
    print(f"  Any disagreement:       {n_disagree:>7d}  ({100*n_disagree/n_events:.2f}%)")
    n_tiebreak = int(tiebreak_mask.sum())
    print(f"    3-way disagreement:   {n_tiebreak:>7d}  ({100*n_tiebreak/n_events:.2f}%)")
    print(f"    2-vs-1 disagreement:  {n_disagree - n_tiebreak:>7d}  ({100*(n_disagree-n_tiebreak)/n_events:.2f}%)")

    # Opus dominance in majority vote
    print("\n=== Opus dominance in majority vote ===")
    ensemble_equals_opus = (mv == stacked[0])
    print(f"  MV == Opus (on all events):              {ensemble_equals_opus.sum():>7d} / {n_events}  ({100*ensemble_equals_opus.mean():.2f}%)")

    # On disagreement events, how often is Opus in the winning majority?
    disagree_mask = ~all_agree
    opus_in_majority = (mv == stacked[0]) & disagree_mask
    print(f"  On disagreement events, Opus in winning side: "
          f"{opus_in_majority.sum():>7d} / {n_disagree}  ({100*opus_in_majority.sum()/n_disagree:.2f}%)")

    gpt_in_majority = (mv == stacked[1]) & disagree_mask
    sonnet_in_majority = (mv == stacked[2]) & disagree_mask
    print(f"  On disagreement events, GPT in winning side:  "
          f"{gpt_in_majority.sum():>7d} / {n_disagree}  ({100*gpt_in_majority.sum()/n_disagree:.2f}%)")
    print(f"  On disagreement events, Sonnet in winning:    "
          f"{sonnet_in_majority.sum():>7d} / {n_disagree}  ({100*sonnet_in_majority.sum()/n_disagree:.2f}%)")

    # Also: what if we tie-break differently (GPT or Sonnet)?
    print("\n=== Sensitivity to tie-breaker on 3-way disagreements ===")
    for tb_idx, tb_name in enumerate(names):
        mv_alt = mv.copy()
        mv_alt[tiebreak_mask] = stacked[tb_idx, tiebreak_mask]
        res_alt = _compute_fraction_error(labels, mv_alt)
        print(f"  Tie-broken by {tb_name:>7s}: FE = {res_alt['mean']:.8f}  (rounded: {res_alt['mean']:.4f})")

    # Irreducible-error-under-agreement analysis
    # --- observational ceiling evidence ---
    print("\n=== Agreement-outcome decomposition (observational ceiling evidence) ===")
    all_agree_correct = all_agree & (stacked[0] == labels)
    all_agree_wrong = all_agree & (stacked[0] != labels)
    n_ac = int(all_agree_correct.sum())
    n_aw = int(all_agree_wrong.sum())
    print(f"  All 3 agents agree AND correct:   {n_ac:>7d}  ({100*n_ac/n_events:5.2f}%)")
    print(f"  All 3 agents agree AND WRONG:     {n_aw:>7d}  ({100*n_aw/n_events:5.2f}%)  <-- lower bound on irreducible error")
    # Among disagreement events: how often does majority get it right, or none get it right?
    disagree_mask_local = ~all_agree
    # How often is truth = at least one of the 3 agents' predictions?
    any_correct = (stacked == labels[None, :]).any(axis=0)
    none_correct = disagree_mask_local & ~any_correct
    print(f"  Any disagreement, none correct:   {int(none_correct.sum()):>7d}  "
          f"({100*int(none_correct.sum())/n_events:5.2f}%)  <-- irreducible within this agent pool")
    mv_correct = (mv == labels)
    mv_correct_disagree = disagree_mask_local & mv_correct
    print(f"  Disagreement, majority correct:   {int(mv_correct_disagree.sum()):>7d}  "
          f"({100*int(mv_correct_disagree.sum())/n_events:5.2f}%)")
    mv_wrong_disagree = disagree_mask_local & ~mv_correct & any_correct
    print(f"  Disagreement, majority wrong but some agent right: {int(mv_wrong_disagree.sum()):>7d}  "
          f"({100*int(mv_wrong_disagree.sum())/n_events:5.2f}%)  <-- recoverable by oracle selection")

    total_irreducible = n_aw + int(none_correct.sum())
    print(f"\n  Total 'no agent gets it right' (agreement-wrong + disagreement-all-wrong): "
          f"{total_irreducible:>7d}  ({100*total_irreducible/n_events:5.2f}%)")
    oracle_accuracy = 1 - total_irreducible / n_events
    print(f"  Oracle (pick best agent per event) upper bound on accuracy: {oracle_accuracy:.4f}")

    # Oracle predictions and their fraction error
    oracle_pred = stacked[0].copy()  # start with Opus
    # For each event, if any agent matches truth, use that; otherwise keep Opus (lose anyway)
    for i in range(n_events):
        for j in range(3):
            if stacked[j, i] == labels[i]:
                oracle_pred[i] = labels[i]
                break
    oracle_res = _compute_fraction_error(labels, oracle_pred.astype(int))
    print(f"  Oracle fraction error:    {oracle_res['mean']:.6f}  (rounded: {oracle_res['mean']:.4f})")
    print(f"  vs Opus alone:            {individual_fe['Opus']:.6f}  (rounded: {individual_fe['Opus']:.4f})")
    print(f"  Headroom in fraction error: {individual_fe['Opus'] - oracle_res['mean']:.6f}")
    print(f"  Headroom in accuracy:       {oracle_accuracy - 0.5145:.4f}")

    # Per-class all-wrong rate (where all 3 agents agree on wrong class)
    print("\n  Agreement-wrong rate by TRUE class:")
    print(f"  {'class':>10s}  {'N_true':>7s}  {'N_agree_wrong':>14s}  {'rate':>7s}")
    for c, name_cls in enumerate(['p', 'He', 'C', 'Si', 'Fe']):
        mask_c = labels == c
        n_true = int(mask_c.sum())
        n_wrong = int((all_agree_wrong & mask_c).sum())
        print(f"  {name_cls:>10s}  {n_true:>7d}  {n_wrong:>14d}  {n_wrong/max(1,n_true):>7.4f}")

    print()
    print("=== Per-class prediction counts (summary) ===")
    print(f"  {'Agent':>10s}  " + "  ".join(f"{c:>7s}" for c in ['p', 'He', 'C', 'Si', 'Fe']))
    for n in names:
        counts = np.bincount(preds[n], minlength=5)
        print(f"  {n:>10s}  " + "  ".join(f"{c:>7d}" for c in counts))
    counts_mv = np.bincount(mv, minlength=5)
    print(f"  {'MajVote':>10s}  " + "  ".join(f"{c:>7d}" for c in counts_mv))
    print(f"  {'Truth':>10s}  " + "  ".join(f"{c:>7d}" for c in np.bincount(labels, minlength=5)))


if __name__ == "__main__":
    main()
