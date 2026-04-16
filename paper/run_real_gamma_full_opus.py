"""Score ALL real KASCADE events with the Opus v5b 3-seed ensemble.

Unlike the earlier run_real_gamma_skymap_opus.py and run_real_gamma_kostunin_opus.py,
this script saves ALL scores (not just passing events), plus per-event metadata
needed for downstream analysis: energy, zenith, azimuth, Ne, Nmu, timestamp.

This enables proper analysis:
  * Score distribution per energy bin (real vs sim gamma vs sim hadron)
  * Top-N highest-scoring real events per energy bin regardless of threshold
  * Per-bin threshold calibration with known gamma efficiency and hadron p_survival
  * Identification of genuinely gamma-like candidates using per-bin criteria
"""

import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from run_real_gamma_skymap_opus import (
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


def score_sim_test(models, mean, std):
    """Score the simulation test set and return scores + labels + energies."""
    print("Scoring simulation test set for per-bin calibration...", flush=True)
    X_test = np.load(DATA_DIR / "gamma_test" / "matrices.npy", mmap_mode="r")
    f_test = np.load(DATA_DIR / "gamma_test" / "features.npy", mmap_mode="r")
    y_test = np.load(DATA_DIR / "gamma_test" / "labels_gamma.npy", mmap_mode="r")

    F_test = engineer_features(np.array(X_test), np.array(f_test, dtype=np.float32))
    F_test_norm = (F_test - mean) / std
    scores = score_with_ensemble(models, np.array(X_test), F_test_norm)
    y = np.array(y_test)
    energies = np.array(f_test)[:, 0]
    return scores, y, energies


def score_real_data(models, mean, std):
    """Stream over real data, score everything, return arrays of all events."""
    real_dir = DATA_DIR / "real_kascade"
    run_files = sorted(glob(str(real_dir / "*_matrices.npz")))
    print(f"\nScoring {len(run_files)} runs (quality cuts only)...", flush=True)

    all_scores = []
    all_E = []
    all_Ze = []
    all_Az = []
    all_Ne = []
    all_Nmu = []
    all_run = []
    all_idx_in_run = []
    all_timestamp = []

    t0 = time.time()
    n_total = 0

    for ri, mpath in enumerate(run_files):
        run_name = Path(mpath).name.replace("_matrices.npz", "")
        fpath = real_dir / f"{run_name}_features.npz"
        efpath = real_dir / f"{run_name}_extra_features.npz"

        feat = np.load(fpath)["features"]
        if feat.ndim != 2 or len(feat) == 0:
            continue

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
        idx_in_run = np.where(mask)[0]
        feat_sim = feat_cut[:, REAL_TO_SIM].astype(np.float32)

        F_real = engineer_features(mat, feat_sim)
        F_real_norm = (F_real - mean) / std

        scores = score_with_ensemble(models, mat, F_real_norm, batch_size=4096)

        # Pull timestamps for all events
        ef = np.load(efpath, allow_pickle=True)["extra_features"][mask]
        timestamps = np.array([
            str(ef[i][8]) if ef[i][8] is not None else "" for i in range(len(ef))
        ])

        all_scores.append(scores.astype(np.float32))
        all_E.append(feat_cut[:, 0].astype(np.float32))
        all_Ze.append(feat_cut[:, 3].astype(np.float32))
        all_Az.append(feat_cut[:, 4].astype(np.float32))
        all_Ne.append(feat_cut[:, 5].astype(np.float32))
        all_Nmu.append(feat_cut[:, 6].astype(np.float32))
        all_run.append(np.full(len(scores), run_name))
        all_idx_in_run.append(idx_in_run.astype(np.int32))
        all_timestamp.append(timestamps)

        n_total += len(scores)
        if (ri + 1) % 200 == 0:
            print(
                f"  {ri + 1}/{len(run_files)} runs, {n_total:,} events "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    print(f"\nDone scoring: {n_total:,} events in {time.time() - t0:.0f}s")

    return {
        "score": np.concatenate(all_scores),
        "E": np.concatenate(all_E),
        "Ze": np.concatenate(all_Ze),
        "Az": np.concatenate(all_Az),
        "Ne": np.concatenate(all_Ne),
        "Nmu": np.concatenate(all_Nmu),
        "run": np.concatenate(all_run),
        "idx_in_run": np.concatenate(all_idx_in_run),
        "timestamp": np.concatenate(all_timestamp),
    }


def main():
    t0 = time.time()

    mean, std, n_features = compute_normalization_stats()
    models = load_models(mean, std, n_features)

    sim_scores, sim_labels, sim_energies = score_sim_test(models, mean, std)
    print(
        f"  Sim: {int((sim_labels == 0).sum())} gammas, "
        f"{int((sim_labels == 1).sum())} hadrons"
    )

    real = score_real_data(models, mean, std)

    out_npz = Path(__file__).resolve().parent / "gamma_real_full_opus.npz"
    np.savez_compressed(
        out_npz,
        # Real data (all events passing quality cuts)
        real_score=real["score"],
        real_E=real["E"],
        real_Ze=real["Ze"],
        real_Az=real["Az"],
        real_Ne=real["Ne"],
        real_Nmu=real["Nmu"],
        real_run=real["run"],
        real_idx_in_run=real["idx_in_run"],
        real_timestamp=real["timestamp"],
        # Simulation test set (for calibration)
        sim_score=sim_scores.astype(np.float32),
        sim_label=sim_labels.astype(np.int8),
        sim_E=sim_energies.astype(np.float32),
    )
    print(f"\nSaved {out_npz}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
