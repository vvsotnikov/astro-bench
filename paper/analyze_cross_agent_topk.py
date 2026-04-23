"""Cross-agent top-K overlap analysis for the real-data gamma search.

Loads per-event scores from Opus (gamma_real_full_opus.npz) and Sonnet
(gamma_real_full_sonnet.npz), event-aligned by (run, idx_in_run). For each
cut (E threshold) and each K, computes:

  * Opus top-K and Sonnet top-K by score
  * Overlap |Opus top-K  INTERSECT  Sonnet top-K|
  * Random-overlap expectation under independent selection (K^2 / N)
  * Enrichment factor (observed / expected)

This measures whether high-scoring events are consistent across two
independently-built architectures --- a cross-validation signal that's
absent from a single-classifier candidate list.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OPUS_NPZ = HERE / "gamma_real_full_opus.npz"
SONN_NPZ = HERE / "gamma_real_full_sonnet.npz"


def load_and_align():
    op = np.load(OPUS_NPZ, allow_pickle=True)
    so = np.load(SONN_NPZ, allow_pickle=True)

    # Event-alignment check: both datasets should have the same events in the
    # same order (same cuts, same per-run scanning order).
    n_op = len(op["real_score"])
    n_so = len(so["real_score"])
    print(f"Opus:   {n_op:,} events")
    print(f"Sonnet: {n_so:,} events")
    if n_op != n_so:
        print(f"WARNING: event counts differ ({n_op} vs {n_so}).")

    # Verify alignment by (run, idx_in_run).
    op_key = np.array([f"{r}:{i}" for r, i in zip(op["real_run"], op["real_idx_in_run"])])
    so_key = np.array([f"{r}:{i}" for r, i in zip(so["real_run"], so["real_idx_in_run"])])

    if n_op == n_so and (op_key == so_key).all():
        print("Event alignment: identical order (no remapping needed).")
        return op, so, None

    # Fall back to intersection by key
    print("Event alignment: re-aligning by (run, idx_in_run)...")
    op_idx_map = {k: i for i, k in enumerate(op_key)}
    so_idx_map = {k: i for i, k in enumerate(so_key)}
    common_keys = sorted(set(op_idx_map.keys()) & set(so_idx_map.keys()))
    print(f"  {len(common_keys):,} events in common")
    op_idx = np.array([op_idx_map[k] for k in common_keys])
    so_idx = np.array([so_idx_map[k] for k in common_keys])
    return op, so, (op_idx, so_idx)


def topk_overlap(op_scores, so_scores, mask, k_values, name=""):
    idx = np.where(mask)[0]
    n_pool = len(idx)
    if n_pool == 0:
        print(f"  {name}: no events in pool")
        return

    op_sub = op_scores[idx]
    so_sub = so_scores[idx]

    # Sort DESC to get top-K by score
    op_order = np.argsort(op_sub)[::-1]
    so_order = np.argsort(so_sub)[::-1]

    print(f"\n=== {name}: {n_pool:,} events ===")
    print(f"  {'K':>6s} {'|Op top-K|':>10s} {'|So top-K|':>10s} "
          f"{'overlap':>8s} {'expected':>9s} {'enrichment':>11s}")
    for k in k_values:
        if k > n_pool:
            continue
        op_top = set(op_order[:k].tolist())
        so_top = set(so_order[:k].tolist())
        inter = len(op_top & so_top)
        expected = k * k / n_pool  # under independent uniform selection
        enrich = inter / expected if expected > 0 else float("inf")
        print(f"  {k:>6d} {k:>10d} {k:>10d} "
              f"{inter:>8d} {expected:>9.2f} {enrich:>11.1f}x")


def main():
    op, so, align = load_and_align()

    if align is None:
        op_scores = op["real_score"]
        so_scores = so["real_score"]
        E = op["real_E"]
        run = op["real_run"]
        idx_in_run = op["real_idx_in_run"]
    else:
        op_idx, so_idx = align
        op_scores = op["real_score"][op_idx]
        so_scores = so["real_score"][so_idx]
        E = op["real_E"][op_idx]
        run = op["real_run"][op_idx]
        idx_in_run = op["real_idx_in_run"][op_idx]

    # Score distributions
    print(f"\nOpus   score stats: min={op_scores.min():.3f}, "
          f"med={np.median(op_scores):.3f}, max={op_scores.max():.3f}")
    print(f"Sonnet score stats: min={so_scores.min():.3f}, "
          f"med={np.median(so_scores):.3f}, max={so_scores.max():.3f}")

    # Pearson correlation overall and at the tail
    corr_all = float(np.corrcoef(op_scores, so_scores)[0, 1])
    print(f"\nPearson correlation (all events): {corr_all:.4f}")

    # Top-K overlap across different energy cuts
    k_values = [5, 10, 20, 50, 100, 200, 500, 1000]

    topk_overlap(op_scores, so_scores,
                 mask=np.ones_like(E, dtype=bool),
                 k_values=k_values, name="All events (E >= 14.5)")

    topk_overlap(op_scores, so_scores,
                 mask=(E >= 15.0),
                 k_values=k_values, name="E >= 15.0")

    topk_overlap(op_scores, so_scores,
                 mask=(E >= 15.5),
                 k_values=k_values, name="E >= 15.5")

    topk_overlap(op_scores, so_scores,
                 mask=(E >= 16.0),
                 k_values=k_values, name="E >= 16.0 (PeV regime)")

    topk_overlap(op_scores, so_scores,
                 mask=(E >= 16.5),
                 k_values=k_values, name="E >= 16.5")

    # Named-event check: verify Runs 1971 and 4281 (the two paper-cited events)
    # appear in both top-K lists at E >= 16
    print("\n=== Named events at E >= 16 ===")
    mask_he = E >= 16.0
    he_idx = np.where(mask_he)[0]
    op_he = op_scores[he_idx]
    so_he = so_scores[he_idx]
    run_he = run[he_idx]
    op_order = np.argsort(op_he)[::-1]
    so_order = np.argsort(so_he)[::-1]

    for target_run in ["run1971", "run4281"]:
        op_rank = None
        so_rank = None
        for rank_op, p in enumerate(op_order):
            if run_he[p] == target_run:
                op_rank = rank_op
                break
        for rank_so, p in enumerate(so_order):
            if run_he[p] == target_run:
                so_rank = rank_so
                break
        print(f"  {target_run}: Opus rank {op_rank}, Sonnet rank {so_rank} "
              f"(out of {len(op_he)} events at E>=16)")


if __name__ == "__main__":
    main()
