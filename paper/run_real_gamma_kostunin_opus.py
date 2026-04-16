"""Kostunin's walk-up-the-ladder gamma candidate search with Opus 4.6 best model.

Methodology (matching run_real_gamma_search.py used with Sonnet v4):
  1. Fix the classifier threshold at 75% gamma efficiency
     (calibrated on the simulation test set).
  2. Walk up the energy ladder, checking for each cut E > E0:
         N_exp_bg(E > E0) = N_real(E > E0) * p_survival
  3. Any event passing the threshold in a regime where N_exp_bg < 1
     is a genuine gamma candidate.

Uses the 3-seed DualBranchCNN ensemble (Opus v5b, geometric mean).
"""

import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from run_real_gamma_skymap_opus import (  # reuse architecture, features, inference
    CUTS,
    DATA_DIR,
    DEVICE,
    DualBranchCNN,
    EvalDataset,
    OPUS_DIR,
    REAL_TO_SIM,
    compute_normalization_stats,
    engineer_features,
    load_models,
    score_with_ensemble,
)


def calibrate_threshold(models, mean, std):
    """Find gamma score threshold at 75% gamma efficiency on sim test set.
    Returns (threshold, p_survival) where p_survival is the hadronic survival rate.
    """
    print("Calibrating threshold on simulation test set...", flush=True)
    X_test = np.load(DATA_DIR / "gamma_test" / "matrices.npy", mmap_mode="r")
    f_test = np.load(DATA_DIR / "gamma_test" / "features.npy", mmap_mode="r")
    y_test = np.load(DATA_DIR / "gamma_test" / "labels_gamma.npy", mmap_mode="r")

    F_test = engineer_features(np.array(X_test), np.array(f_test, dtype=np.float32))
    F_test_norm = (F_test - mean) / std

    scores = score_with_ensemble(models, np.array(X_test), F_test_norm)
    y_arr = np.array(y_test)

    # sigmoid(logits) = P(gamma). Keep top 75% of gammas.
    gamma_scores = scores[y_arr == 0]
    threshold = float(np.percentile(gamma_scores, 25))
    hadron_scores = scores[y_arr == 1]
    n_hadron_pass = int((hadron_scores >= threshold).sum())
    n_hadron = int((y_arr == 1).sum())
    p_survival = n_hadron_pass / n_hadron
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Hadrons passing: {n_hadron_pass}/{n_hadron} = {p_survival:.4e}")
    return threshold, p_survival


def main():
    t0 = time.time()

    mean, std, n_features = compute_normalization_stats()
    models = load_models(mean, std, n_features)
    threshold, p_survival = calibrate_threshold(models, mean, std)

    # Stream over real data runs, scoring events and tracking energies.
    real_dir = DATA_DIR / "real_kascade"
    run_files = sorted(glob(str(real_dir / "*_matrices.npz")))
    print(f"\nProcessing {len(run_files)} runs (quality cuts only, no energy cut)...",
          flush=True)

    all_energies = []          # log10(E/eV) for every event passing quality cuts
    passing_energies = []      # log10(E/eV) for events passing gamma threshold
    passing_info = []          # detailed per-candidate info

    n_total = 0
    n_passing = 0

    for ri, mpath in enumerate(run_files):
        run_name = Path(mpath).name.replace("_matrices.npz", "")
        fpath = real_dir / f"{run_name}_features.npz"
        efpath = real_dir / f"{run_name}_extra_features.npz"

        feat = np.load(fpath)["features"]
        if feat.ndim != 2 or len(feat) == 0:
            continue

        # Quality cuts only (no energy cut — we want the full distribution)
        ze = feat[:, 3]
        ne = feat[:, 5]
        age = feat[:, 7]
        mask = (
            (ze < CUTS["Ze_max"])
            & (ne > CUTS["Ne_min"])
            & (age > CUTS["Age_min"])
            & (age < CUTS["Age_max"])
        )
        if mask.sum() == 0:
            continue

        mat = np.load(mpath)["matrices"][mask][:, :, :, 1:3]
        feat_cut = feat[mask]
        feat_sim = feat_cut[:, REAL_TO_SIM].astype(np.float32)
        energies = feat_cut[:, 0].astype(np.float32)

        F_real = engineer_features(mat, feat_sim)
        F_real_norm = (F_real - mean) / std

        scores = score_with_ensemble(models, mat, F_real_norm, batch_size=4096)

        n_total += len(energies)
        all_energies.append(energies)

        passing = scores >= threshold
        n_pass = int(passing.sum())
        n_passing += n_pass

        if n_pass > 0:
            passing_energies.append(energies[passing])
            ef = np.load(efpath, allow_pickle=True)["extra_features"][mask][passing]
            for idx_local in np.where(passing)[0]:
                passing_info.append({
                    "E": float(feat_cut[idx_local, 0]),
                    "Ze": float(feat_cut[idx_local, 3]),
                    "Az": float(feat_cut[idx_local, 4]),
                    "Ne": float(feat_cut[idx_local, 5]),
                    "Nmu": float(feat_cut[idx_local, 6]),
                    "score": float(scores[idx_local]),
                    "run": run_name,
                })

        if (ri + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(
                f"  {ri + 1}/{len(run_files)} runs, "
                f"{n_total:,} events, {n_passing:,} passing ({elapsed:.0f}s)",
                flush=True,
            )

    all_energies = np.concatenate(all_energies) if all_energies else np.array([])
    passing_energies = (
        np.concatenate(passing_energies) if passing_energies else np.array([])
    )

    print(f"\nTotal: {n_total:,} events, {n_passing:,} passing ({time.time() - t0:.0f}s)")
    print(f"Overall expected background = N_total * p_survival = "
          f"{n_total * p_survival:.0f}")

    # Energy-binned analysis (Kostunin's method)
    print(f"\n{'=' * 75}")
    print("Energy-binned analysis")
    print(f"{'=' * 75}")
    print(
        f"{'E_min':>7s} {'E_max':>7s} {'N_events':>12s} "
        f"{'N_pass':>8s} {'N_exp_bg':>12s} {'Significant?':>14s}"
    )
    print("-" * 75)
    e_bins = np.arange(14.0, 18.5, 0.5)
    for i in range(len(e_bins) - 1):
        e_lo, e_hi = e_bins[i], e_bins[i + 1]
        n_bin = int(((all_energies >= e_lo) & (all_energies < e_hi)).sum())
        n_pass_bin = int(((passing_energies >= e_lo) & (passing_energies < e_hi)).sum())
        n_exp_bg = n_bin * p_survival
        sig = (
            "*** YES ***"
            if (n_pass_bin > 0 and n_exp_bg < 1)
            else "candidate"
            if (n_pass_bin > 0 and n_exp_bg < 3)
            else ""
        )
        print(
            f"{e_lo:7.1f} {e_hi:7.1f} {n_bin:12,} {n_pass_bin:8,} "
            f"{n_exp_bg:12.2f} {sig:>14s}"
        )

    print(f"\n{'=' * 75}")
    print("Cumulative from high energy (Kostunin's walk-up-the-ladder)")
    print(f"{'=' * 75}")
    print(
        f"{'E_cut':>7s} {'N_events':>12s} {'N_pass':>8s} "
        f"{'N_exp_bg':>12s} {'Significant?':>14s}"
    )
    print("-" * 60)
    for e_cut in [14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5]:
        n_bin = int((all_energies >= e_cut).sum())
        n_pass_bin = int((passing_energies >= e_cut).sum())
        n_exp_bg = n_bin * p_survival
        sig = (
            "*** YES ***"
            if (n_pass_bin > 0 and n_exp_bg < 1)
            else "candidate"
            if (n_pass_bin > 0 and n_exp_bg < 3)
            else ""
        )
        print(
            f"{e_cut:7.1f} {n_bin:12,} {n_pass_bin:8,} "
            f"{n_exp_bg:12.2f} {sig:>14s}"
        )

    # Report the highest-energy passing events (if any)
    if passing_info:
        passing_info.sort(key=lambda d: d["E"], reverse=True)
        print(f"\n{'=' * 75}")
        print("Top 20 highest-energy gamma candidates (Opus v5b, 3-seed geo ensemble)")
        print(f"{'=' * 75}")
        print(f"{'E':>7s} {'Ze':>6s} {'Az':>7s} {'Ne':>6s} {'Nmu':>6s} "
              f"{'Ne-Nmu':>7s} {'score':>7s}  run")
        print("-" * 75)
        for cand in passing_info[:20]:
            print(
                f"{cand['E']:7.3f} {cand['Ze']:6.1f} {cand['Az']:7.1f} "
                f"{cand['Ne']:6.3f} {cand['Nmu']:6.3f} "
                f"{cand['Ne'] - cand['Nmu']:7.3f} {cand['score']:7.4f}  {cand['run']}"
            )

    out_npz = Path(__file__).resolve().parent / "gamma_kostunin_opus.npz"
    np.savez(
        out_npz,
        all_energies=all_energies,
        passing_energies=passing_energies,
        threshold=threshold,
        p_survival=p_survival,
    )
    print(f"\nSaved {out_npz}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
