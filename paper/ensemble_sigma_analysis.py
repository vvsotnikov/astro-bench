"""Compute σ_ensemble for the three top composition agents from per-seed retrains.

Each agent's final ensemble (Opus v34, GPT v47, Sonnet v44) was retrained from scratch
under 3 different outer seeds. This script:
  1. Loads per-seed ensemble probabilities (115064, 5).
  2. Computes mean fraction error per seed (verifier-consistent).
  3. Reports per-agent ensemble σ across the 3 seeds, and pairwise z values
     against measured σ_ensemble (instead of single-model σ).
  4. Writes ensemble_sigma_results.json.

Sonnet logits are stored as sums of softmax over 5 internal models × 8 TTA = 40 views;
they get normalized to probabilities here.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "composition"))
from verify import _compute_fraction_error, _load_test_data  # noqa: E402

ROOT = Path("/home/vladimir/cursor_projects/astro-bench-experiments")

AGENT_PATHS = {
    "Opus 4.6 v34": [
        ROOT / "composition-opus-2apr" / f"opus_v34_probs_s{s}.npy" for s in [1, 7, 100]
    ],
    "GPT-5.4 v47": [
        ROOT / "composition-gpt-2apr" / f"gpt_v47_probs_s{s}.npy" for s in [1, 7, 100]
    ],
    "Sonnet 4.6 v44": [
        ROOT / "composition-sonnet-27mar" / f"sonnet_v44_logits_s{s}.npy" for s in [1, 2, 3]
    ],
}

# v44 saves SUMS of softmax across 5 internal models * 8 TTA = 40 views; normalize.
NEEDS_NORMALIZATION = {"Sonnet 4.6 v44"}


def load_probs(path: Path, normalize: bool) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if normalize:
        arr = arr / arr.sum(axis=1, keepdims=True)
    return arr


def main():
    labels, _ = _load_test_data()
    print(f"Test set: {len(labels)} events")

    results = {}
    for agent, paths in AGENT_PATHS.items():
        ready = [p for p in paths if p.exists()]
        if not ready:
            print(f"\n{agent}: no probs files yet")
            results[agent] = None
            continue
        if len(ready) < len(paths):
            print(f"\n{agent}: {len(ready)}/{len(paths)} files present, partial analysis")
        print(f"\n=== {agent} ===")
        normalize = agent in NEEDS_NORMALIZATION
        fes = []
        for path in ready:
            probs = load_probs(path, normalize)
            preds = probs.argmax(axis=1).astype(int)
            res = _compute_fraction_error(labels, preds)
            fe = res["mean"]
            acc = float((preds == labels).mean())
            print(f"  {path.name:>40s}  fe={fe:.5f}  acc={acc:.4f}")
            fes.append(fe)
        fes_arr = np.array(fes)
        mean = float(fes_arr.mean())
        std = float(fes_arr.std(ddof=1)) if len(fes_arr) >= 2 else None
        print(f"  ----")
        print(f"  mean = {mean:.5f}")
        if std is not None:
            print(f"  std  = {std:.5f}  (n={len(fes_arr)} seeds, ddof=1)")
        results[agent] = {
            "n_seeds": len(fes_arr),
            "fes": [float(x) for x in fes_arr],
            "mean": mean,
            "std": std,
        }

    # Pairwise z analysis using measured ensemble σ where available
    print("\n=== Pairwise z against measured ensemble σ ===")
    summary = {"per_agent": results, "pairwise": {}}
    keys = list(results.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ai, aj = keys[i], keys[j]
            ri, rj = results[ai], results[aj]
            if ri is None or rj is None or ri["std"] is None or rj["std"] is None:
                continue
            delta = ri["mean"] - rj["mean"]
            sigma_pair = np.hypot(ri["std"], rj["std"])  # σ of (xi - xj) under independent ensembles
            z = delta / sigma_pair
            print(
                f"  {ai} ({ri['mean']:.5f}±{ri['std']:.5f}) vs "
                f"{aj} ({rj['mean']:.5f}±{rj['std']:.5f}): "
                f"Δ={delta:+.5f}  σ_pair={sigma_pair:.5f}  z={z:+.2f}"
            )
            summary["pairwise"][f"{ai} vs {aj}"] = {
                "delta": float(delta),
                "sigma_pair": float(sigma_pair),
                "z": float(z),
            }

    out_path = Path(__file__).resolve().parent / "ensemble_sigma_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
