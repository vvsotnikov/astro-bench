"""Score ALL real KASCADE events with Sonnet 4.6's best gamma classifier (v4).

Mirrors run_real_gamma_full_opus.py --- same cuts (Ze<30, Ne>4.8,
Age in (0.2,1.48), NO energy cut), same output structure (saves all scores
plus per-event metadata: E, Ze, Az, Ne, Nmu, run name, idx_in_run, timestamp).

The output file paper/gamma_real_full_sonnet.npz is event-aligned with
paper/gamma_real_full_opus.npz (same set of events in the same per-run order),
enabling top-K overlap analysis.

Uses Sonnet's model_v4_best.pt from gamma-sonnet-22mar-v2/.

Run with:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python paper/run_real_gamma_full_sonnet.py
"""
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from run_real_gamma_skymap import (  # noqa: E402
    CUTS,
    DATA_DIR,
    EXPERIMENT_DIR,
    GammaNet,
    REAL_TO_SIM,
    RealDataset,
    compute_derived_features,
    compute_normalization_stats,
)


DEVICE = torch.device("cuda:0")
OUT_NPZ = Path(__file__).resolve().parent / "gamma_real_full_sonnet.npz"


def score_sim_test(model, img_mean, img_std, feat_mean, feat_std):
    """Score the sim test set; returns (scores, labels, energies)."""
    print("Scoring simulation test set...", flush=True)
    X_test = np.load(DATA_DIR / "gamma_test" / "matrices.npy", mmap_mode="r")
    f_test = np.load(DATA_DIR / "gamma_test" / "features.npy", mmap_mode="r")
    y_test = np.load(DATA_DIR / "gamma_test" / "labels_gamma.npy", mmap_mode="r")

    f_test_arr = np.array(f_test, dtype=np.float32)
    features_10 = compute_derived_features(np.array(X_test), f_test_arr)
    ds = RealDataset(np.array(X_test), features_10, img_mean, img_std, feat_mean, feat_std)
    loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    scores = []
    with torch.no_grad():
        for img, feat in loader:
            img, feat = img.to(DEVICE), feat.to(DEVICE)
            logits = model(img, feat)
            scores.append(torch.softmax(logits, dim=1)[:, 0].cpu().numpy())
    scores = np.concatenate(scores).astype(np.float32)
    labels = np.array(y_test, dtype=np.int8)
    energies = f_test_arr[:, 0].astype(np.float32)
    print(f"  Sim test: {len(scores):,} events ({(labels==0).sum()} gamma, "
          f"{(labels==1).sum()} hadron)", flush=True)
    return scores, labels, energies


def score_real_data(model, img_mean, img_std, feat_mean, feat_std):
    """Score all real runs; same cut rules as run_real_gamma_full_opus.py."""
    real_dir = DATA_DIR / "real_kascade"
    run_files = sorted(glob(str(real_dir / "*_matrices.npz")))
    print(f"\nScoring {len(run_files)} runs (cuts: Ze<{CUTS['Ze_max']}, "
          f"Ne>{CUTS['Ne_min']}, {CUTS['Age_min']}<Age<{CUTS['Age_max']})...",
          flush=True)

    all_scores, all_E, all_Ze, all_Az = [], [], [], []
    all_Ne, all_Nmu, all_run, all_idx_in_run, all_timestamp = [], [], [], [], []

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

        features_10 = compute_derived_features(mat, feat_sim)
        ds = RealDataset(mat, features_10, img_mean, img_std, feat_mean, feat_std)
        loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)

        scores = []
        with torch.no_grad():
            for img, f in loader:
                img, f = img.to(DEVICE), f.to(DEVICE)
                logits = model(img, f)
                scores.append(torch.softmax(logits, dim=1)[:, 0].cpu().numpy())
        scores = np.concatenate(scores).astype(np.float32)

        ef = np.load(efpath, allow_pickle=True)["extra_features"][mask]
        timestamps = np.array([
            str(ef[i][8]) if ef[i][8] is not None else "" for i in range(len(ef))
        ])

        all_scores.append(scores)
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
            print(f"  {ri+1}/{len(run_files)} runs, {n_total:,} events "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\nTotal: {n_total:,} events scored ({time.time() - t0:.0f}s)", flush=True)

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

    print("Loading Sonnet v4 model...", flush=True)
    model = GammaNet(feat_dim=10).to(DEVICE)
    model.load_state_dict(torch.load(
        EXPERIMENT_DIR / "model_v4_best.pt",
        map_location=DEVICE, weights_only=True,
    ))
    model.eval()

    img_mean, img_std, feat_mean, feat_std = compute_normalization_stats()

    sim_scores, sim_labels, sim_E = score_sim_test(
        model, img_mean, img_std, feat_mean, feat_std
    )
    real = score_real_data(model, img_mean, img_std, feat_mean, feat_std)

    print(f"\nSaving to {OUT_NPZ}...", flush=True)
    np.savez_compressed(
        OUT_NPZ,
        real_score=real["score"],
        real_E=real["E"],
        real_Ze=real["Ze"],
        real_Az=real["Az"],
        real_Ne=real["Ne"],
        real_Nmu=real["Nmu"],
        real_run=real["run"],
        real_idx_in_run=real["idx_in_run"],
        real_timestamp=real["timestamp"],
        sim_score=sim_scores,
        sim_label=sim_labels,
        sim_E=sim_E,
    )
    print(f"Total: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
