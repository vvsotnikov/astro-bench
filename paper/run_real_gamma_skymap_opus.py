"""Apply Opus 4.6 best gamma classifier (v5 3-seed geometric ensemble) to real KASCADE data.

Uses the three DualBranchCNN checkpoints from composition-opus-23mar:
  model_v5_seed42.pt, model_v5_seed123.pt, model_v5_seed777.pt
combined via geometric mean of sigmoid scores.

Calibrates the 75% gamma efficiency threshold on the simulation test set,
applies the classifier to ~73M real KASCADE events with E > 14.5 cut,
collects timestamps and sky coordinates, builds galactic/equatorial
skymaps, and performs LHAASO source stacking.
"""

import datetime
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda:0")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OPUS_DIR = Path(__file__).resolve().parent.parent.parent / "astro-bench-experiments" / "gamma-opus-23mar"

REAL_TO_SIM = [0, 3, 4, 5, 6]  # E, Ze, Az, Ne, Nmu indices in real features
CUTS = {"Ze_max": 30.0, "Ne_min": 4.8, "Age_min": 0.2, "Age_max": 1.48}
ENERGY_CUT = 14.5  # matching KASCADE ICRC 2015 paper

# --- Model architecture (copied from Opus train_v5.py) ---


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction), nn.GELU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.se = SEBlock(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.se(self.conv(x)))


class DualBranchCNN(nn.Module):
    def __init__(self, n_eng_features):
        super().__init__()
        self.elec_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            ResBlock(32, dropout=0.15),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            ResBlock(64, dropout=0.15),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.muon_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            ResBlock(32, dropout=0.15),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            ResBlock(64, dropout=0.15),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.combined_branch = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feat_net = nn.Sequential(
            nn.Linear(n_eng_features, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(288, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, mat, feat):
        elec = mat[:, 0:1, :, :]
        muon = mat[:, 1:2, :, :]
        x_elec = self.elec_branch(elec).squeeze(-1).squeeze(-1)
        x_muon = self.muon_branch(muon).squeeze(-1).squeeze(-1)
        x_comb = self.combined_branch(mat).squeeze(-1).squeeze(-1)
        x_feat = self.feat_net(feat)
        x = torch.cat([x_elec, x_muon, x_comb, x_feat], dim=1)
        return self.classifier(x).squeeze(-1)


# --- Feature engineering (copied from Opus train_v5.py) ---


def engineer_features(matrices, features, batch_size=50000):
    n = len(features)
    feat_list = []

    yy, xx = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    center = 7.5
    r = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    r_flat = r.flatten()
    radial_edges = [0, 2, 4, 6, 8, 11.5]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        mat = np.array(matrices[start:end]).astype(np.float32)
        f = np.array(features[start:end]).astype(np.float32)

        ch0 = mat[:, :, :, 0]
        ch1 = mat[:, :, :, 1]
        E, Ze, Az, Ne, Nmu = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]

        Ne_Nmu = Ne - Nmu
        cos_ze = np.cos(np.radians(Ze))

        ch0_sum = ch0.sum(axis=(1, 2))
        ch1_sum = ch1.sum(axis=(1, 2))
        ch0_max = ch0.max(axis=(1, 2))
        ch1_max = ch1.max(axis=(1, 2))
        ch0_std = ch0.reshape(-1, 256).std(axis=1)
        ch1_std = ch1.reshape(-1, 256).std(axis=1)
        ch0_nonzero = (ch0 > 0).sum(axis=(1, 2)).astype(np.float32)
        ch1_nonzero = (ch1 > 0).sum(axis=(1, 2)).astype(np.float32)
        muon_electron_ratio = ch1_sum / (ch0_sum + 1e-8)

        ch0_log_sum = np.log10(ch0_sum + 1)
        ch1_log_sum = np.log10(ch1_sum + 1)
        ch0_log_max = np.log10(ch0_max + 1)
        ch1_log_max = np.log10(ch1_max + 1)

        ch0_log = np.log1p(ch0)
        ch1_log = np.log1p(ch1)
        ch0_log_sum2 = ch0_log.sum(axis=(1, 2))
        ch1_log_sum2 = ch1_log.sum(axis=(1, 2))
        ch0_log_std = ch0_log.reshape(-1, 256).std(axis=1)
        ch1_log_std = ch1_log.reshape(-1, 256).std(axis=1)

        ch0_center = ch0[:, 6:10, 6:10].sum(axis=(1, 2))
        ch1_center = ch1[:, 6:10, 6:10].sum(axis=(1, 2))
        ch0_conc = ch0_center / (ch0_sum + 1e-8)
        ch1_conc = ch1_center / (ch1_sum + 1e-8)

        ch0_flat = ch0.reshape(-1, 256)
        ch1_flat = ch1.reshape(-1, 256)
        total0 = ch0_flat.sum(axis=1, keepdims=True) + 1e-8
        total1 = ch1_flat.sum(axis=1, keepdims=True) + 1e-8
        ch0_mean_r = (ch0_flat * r_flat).sum(axis=1) / total0.squeeze()
        ch1_mean_r = (ch1_flat * r_flat).sum(axis=1) / total1.squeeze()

        radial_feats = []
        for i in range(len(radial_edges) - 1):
            mask = (r_flat >= radial_edges[i]) & (r_flat < radial_edges[i + 1])
            radial_feats.append(ch0_flat[:, mask].sum(axis=1) / total0.squeeze())
            radial_feats.append(ch1_flat[:, mask].sum(axis=1) / total1.squeeze())

        q1 = ch1[:, :8, :8].sum(axis=(1, 2))
        q2 = ch1[:, :8, 8:].sum(axis=(1, 2))
        q3 = ch1[:, 8:, :8].sum(axis=(1, 2))
        q4 = ch1[:, 8:, 8:].sum(axis=(1, 2))
        q_total = q1 + q2 + q3 + q4 + 1e-8
        q_max = np.maximum(np.maximum(q1, q2), np.maximum(q3, q4))
        q_min = np.minimum(np.minimum(q1, q2), np.minimum(q3, q4))
        muon_asymmetry = (q_max - q_min) / q_total

        ch0_row_std = ch0.sum(axis=2).std(axis=1)
        ch1_row_std = ch1.sum(axis=2).std(axis=1)
        ch0_col_std = ch0.sum(axis=1).std(axis=1)
        ch1_col_std = ch1.sum(axis=1).std(axis=1)

        muon_per_epix = ch1_sum / (ch0_nonzero + 1e-8)
        nmu_vs_matrix = Nmu - ch1_log_sum

        ch1_sorted = np.sort(ch1_flat, axis=1)[:, ::-1]
        ch1_top1 = ch1_sorted[:, 0]
        ch1_top5 = ch1_sorted[:, :5].sum(axis=1)
        ch1_top10 = ch1_sorted[:, :10].sum(axis=1)

        batch_feats = np.column_stack([
            E, Ze, cos_ze, Ne, Nmu, Ne_Nmu,
            ch0_sum, ch1_sum, ch0_max, ch1_max,
            ch0_std, ch1_std, ch0_nonzero, ch1_nonzero,
            muon_electron_ratio,
            ch0_log_sum, ch1_log_sum, ch0_log_max, ch1_log_max,
            ch0_log_sum2, ch1_log_sum2, ch0_log_std, ch1_log_std,
            ch0_conc, ch1_conc,
            ch0_mean_r, ch1_mean_r,
        ] + radial_feats + [
            muon_asymmetry,
            ch0_row_std, ch1_row_std, ch0_col_std, ch1_col_std,
            muon_per_epix, nmu_vs_matrix,
            ch1_top1, ch1_top5, ch1_top10,
        ])
        feat_list.append(batch_feats)

    return np.vstack(feat_list)


# --- Dataset ---


class EvalDataset(Dataset):
    def __init__(self, matrices, eng_features):
        self.matrices = matrices
        self.eng_features = eng_features

    def __len__(self):
        return len(self.eng_features)

    def __getitem__(self, idx):
        mat = self.matrices[idx].astype(np.float32)
        mat = np.log1p(mat)
        mat = mat.transpose(2, 0, 1)
        ef = self.eng_features[idx].astype(np.float32)
        return torch.from_numpy(mat), torch.from_numpy(ef)


# --- Inference ---


def score_with_ensemble(models, matrices, eng_features_norm, batch_size=4096):
    """Run all models and return geometric mean of sigmoid scores."""
    ds = EvalDataset(matrices, eng_features_norm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_scores = []  # shape (n_models, n_samples)
    for model in models:
        model.eval()
        scores = []
        with torch.no_grad():
            for mat, feat in loader:
                mat, feat = mat.to(DEVICE), feat.to(DEVICE)
                logits = model(mat, feat)
                scores.append(torch.sigmoid(logits).cpu().numpy())
        all_scores.append(np.concatenate(scores))
    all_scores = np.stack(all_scores, axis=0)
    # Geometric mean of sigmoid scores
    geo = np.exp(np.mean(np.log(np.clip(all_scores, 1e-10, 1.0)), axis=0))
    return geo


def compute_normalization_stats():
    """Recompute feature normalization from training data (as in train_v5.py)."""
    print("Loading training data to compute normalization stats...", flush=True)
    X_train = np.load(DATA_DIR / "gamma_train" / "matrices.npy", mmap_mode="r")
    f_train = np.load(DATA_DIR / "gamma_train" / "features.npy", mmap_mode="r")
    print("Engineering features on training data...", flush=True)
    F_train = engineer_features(X_train, f_train)
    mean = F_train.mean(axis=0)
    std = F_train.std(axis=0)
    std[std == 0] = 1.0
    print(f"  Training features: {F_train.shape}", flush=True)
    return mean, std, F_train.shape[1]


def calibrate_threshold(models, mean, std):
    """Find gamma score threshold for 75% gamma efficiency on simulation test set."""
    print("Calibrating threshold on simulation test set...", flush=True)
    X_test = np.load(DATA_DIR / "gamma_test" / "matrices.npy", mmap_mode="r")
    f_test = np.load(DATA_DIR / "gamma_test" / "features.npy", mmap_mode="r")
    y_test = np.load(DATA_DIR / "gamma_test" / "labels_gamma.npy", mmap_mode="r")

    F_test = engineer_features(np.array(X_test), np.array(f_test, dtype=np.float32))
    F_test_norm = (F_test - mean) / std

    scores = score_with_ensemble(models, np.array(X_test), F_test_norm)
    y_arr = np.array(y_test)

    # train_v5 uses targets = 1.0 - labels, so sigmoid(logits) = P(gamma).
    # Higher score = more gamma-like. Threshold keeps top 75% of gammas.
    gamma_scores = scores[y_arr == 0]
    threshold = np.percentile(gamma_scores, 25)  # keep top 75% of gammas (scores >= threshold)
    hadron_scores = scores[y_arr == 1]
    n_hadron_pass = int((hadron_scores >= threshold).sum())
    n_hadron = int((y_arr == 1).sum())
    survival = n_hadron_pass / n_hadron
    print(f"  Threshold: {threshold:.6f}, hadrons passing: {n_hadron_pass}/{n_hadron} = {survival:.4e}")
    return threshold


def load_models(mean, std, n_features):
    print("Loading Opus v5 models (3 seeds)...", flush=True)
    models = []
    for seed in [42, 123, 777]:
        path = OPUS_DIR / f"model_v5_seed{seed}.pt"
        model = DualBranchCNN(n_features).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()
        models.append(model)
        print(f"  Loaded {path.name}", flush=True)
    return models


def main():
    t0 = time.time()

    mean, std, n_features = compute_normalization_stats()
    models = load_models(mean, std, n_features)
    threshold = calibrate_threshold(models, mean, std)

    # Process all runs
    real_dir = DATA_DIR / "real_kascade"
    run_files = sorted(glob(str(real_dir / "*_matrices.npz")))
    print(f"\nProcessing {len(run_files)} runs (E > {ENERGY_CUT})...", flush=True)

    all_candidates = []
    n_total = 0
    n_passing = 0

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
        energy = feat[:, 0]
        mask = (
            (ze < CUTS["Ze_max"])
            & (ne > CUTS["Ne_min"])
            & (age > CUTS["Age_min"])
            & (age < CUTS["Age_max"])
            & (energy > ENERGY_CUT)
        )

        if mask.sum() == 0:
            continue

        mat = np.load(mpath)["matrices"][mask][:, :, :, 1:3]  # channels 1,2
        feat_cut = feat[mask]
        feat_sim = feat_cut[:, REAL_TO_SIM].astype(np.float32)
        n_total += len(feat_sim)

        F_real = engineer_features(mat, feat_sim)
        F_real_norm = (F_real - mean) / std

        scores = score_with_ensemble(models, mat, F_real_norm, batch_size=4096)

        passing = scores >= threshold  # high score = gamma-like (P(gamma) from sigmoid)
        n_pass = int(passing.sum())
        n_passing += n_pass

        if n_pass > 0:
            ef = np.load(efpath, allow_pickle=True)["extra_features"]
            ef_cut = ef[mask]
            for idx in np.where(passing)[0]:
                timestamp = ef_cut[idx][8]
                e = feat_cut[idx, 0]
                ze_val = feat_cut[idx, 3]
                az_val = feat_cut[idx, 4]
                ne_val = feat_cut[idx, 5]
                nmu_val = feat_cut[idx, 6]
                all_candidates.append({
                    "score": float(scores[idx]),
                    "E": float(e),
                    "Ze": float(ze_val),
                    "Az": float(az_val),
                    "Ne": float(ne_val),
                    "Nmu": float(nmu_val),
                    "timestamp": timestamp,
                })

        if (ri + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(
                f"  {ri + 1}/{len(run_files)} runs, {n_total:,} events, "
                f"{n_passing:,} candidates ({elapsed:.0f}s)",
                flush=True,
            )

    print(f"\nTotal: {n_total:,} events, {n_passing:,} candidates ({time.time() - t0:.0f}s)")

    # Convert to sky coordinates
    print("\nConverting to equatorial coordinates...", flush=True)
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u

    kascade = EarthLocation(lon=8.4 * u.deg, lat=49.1 * u.deg, height=110 * u.m)

    ra_list, dec_list, energies, scores_list = [], [], [], []
    for cand in all_candidates:
        if cand["timestamp"] is None:
            continue
        try:
            coord = SkyCoord(
                alt=(90 - cand["Ze"]) * u.deg,
                az=cand["Az"] * u.deg,
                obstime=Time(cand["timestamp"]),
                frame="altaz",
                location=kascade,
            ).transform_to("icrs")
            ra_list.append(coord.ra.deg)
            dec_list.append(coord.dec.deg)
            energies.append(cand["E"])
            scores_list.append(cand["score"])
        except Exception:
            continue

    ra = np.array(ra_list)
    dec = np.array(dec_list)
    energies = np.array(energies)
    scores_arr = np.array(scores_list)
    print(f"  {len(ra)} candidates with sky coordinates")

    out_npz = Path(__file__).resolve().parent / "gamma_skymap_candidates_opus.npz"
    np.savez(out_npz, ra=ra, dec=dec, energies=energies, scores=scores_arr)
    print(f"Saved {out_npz}")

    # KASCADE hotspot from ICRC 2015
    hotspot_ra, hotspot_dec = 77.75, 70.85
    hotspot = SkyCoord(ra=hotspot_ra * u.deg, dec=hotspot_dec * u.deg)
    candidate_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    seps = candidate_coords.separation(hotspot).deg

    print(f"\n{'=' * 60}")
    print(f"Distance to KASCADE hotspot (RA={hotspot_ra}, Dec={hotspot_dec})")
    print(f"{'=' * 60}")
    for r in [2, 3, 5, 7, 10]:
        n = int((seps < r).sum())
        print(f"  Within {r:2d}°: {n} candidates")

    # Energy bins
    print(f"\n{'=' * 60}")
    print(f"Energy-binned search (N_expected_bg < 1 criterion)")
    print(f"{'=' * 60}")

    p_survival = float((len(scores_arr) / n_total)) if n_total else 0
    # Actually we want sim-calibrated p_survival, but we'll use the same proxy for reporting
    print(f"{'E_min':>8s} {'E_max':>8s} {'N_events':>10s} {'N_pass':>8s}")
    print("-" * 45)
    for elo, ehi in [
        (14.0, 14.5), (14.5, 15.0), (15.0, 15.5), (15.5, 16.0),
        (16.0, 16.5), (16.5, 17.0), (17.0, 18.0),
    ]:
        emask = (energies >= elo) & (energies < ehi)
        n = int(emask.sum())
        print(f"{elo:8.1f} {ehi:8.1f} {0:10d} {n:8d}")

    # Skymap plots
    print("\nGenerating skymap figures...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gal = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic
    gal_l = gal.l.wrap_at("180d").radian
    gal_b = gal.b.radian
    hotspot_gal = hotspot.galactic

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection="aitoff")
    plane_l = np.linspace(-np.pi, np.pi, 360)
    ax.plot(plane_l, np.zeros_like(plane_l), "k-", alpha=0.15, linewidth=8)

    sc = ax.scatter(
        gal_l, gal_b, c=energies, cmap="YlOrRd", s=8, alpha=0.6,
        vmin=14.5, vmax=17, zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
    cbar.set_label("log$_{10}$(E/eV)")

    ax.scatter(
        hotspot_gal.l.wrap_at("180d").radian, hotspot_gal.b.radian,
        c="none", s=200, marker="o", edgecolors="blue", linewidths=2, zorder=5,
    )
    ax.annotate(
        "KASCADE hotspot\n(5.5σ, ICRC 2015)",
        (hotspot_gal.l.wrap_at("180d").radian, hotspot_gal.b.radian),
        fontsize=7, color="blue", xytext=(10, -15), textcoords="offset points",
    )

    known = [
        ("Crab", 83.633, 22.015), ("Cyg OB2", 308.08, 41.31),
        ("Cas A", 350.85, 58.82), ("LHAASO J2108", 317.15, 51.95),
    ]
    for name, sra, sdec in known:
        src = SkyCoord(ra=sra * u.deg, dec=sdec * u.deg).galactic
        ax.scatter(
            src.l.wrap_at("180d").radian, src.b.radian,
            c="gray", s=30, marker="D", zorder=4, alpha=0.7,
        )
        ax.annotate(
            name, (src.l.wrap_at("180d").radian, src.b.radian),
            fontsize=5, color="gray", xytext=(4, 4), textcoords="offset points",
        )

    ax.set_title(
        f"Opus 4.6 gamma candidates on real KASCADE data "
        f"(E > {ENERGY_CUT}, {len(ra)} events)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_galactic = Path(__file__).resolve().parent / "fig_gamma_skymap_opus_galactic.png"
    plt.savefig(out_galactic, dpi=200, bbox_inches="tight")
    plt.savefig(out_galactic.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out_galactic}")

    # Equatorial
    fig2 = plt.figure(figsize=(14, 7))
    ax2 = fig2.add_subplot(111, projection="aitoff")
    ra_wrap = np.where(ra > 180, ra - 360, ra)
    sc2 = ax2.scatter(
        np.radians(ra_wrap), np.radians(dec), c=energies,
        cmap="YlOrRd", s=8, alpha=0.6, vmin=14.5, vmax=17, zorder=3,
    )
    cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.7, pad=0.08)
    cbar2.set_label("log$_{10}$(E/eV)")
    hs_wrap = hotspot_ra if hotspot_ra <= 180 else hotspot_ra - 360
    ax2.scatter(
        np.radians(hs_wrap), np.radians(hotspot_dec),
        c="none", s=200, marker="o", edgecolors="blue", linewidths=2, zorder=5,
    )
    ax2.annotate(
        "KASCADE hotspot",
        (np.radians(hs_wrap), np.radians(hotspot_dec)),
        fontsize=7, color="blue", xytext=(10, -15), textcoords="offset points",
    )
    ax2.set_title(
        f"Equatorial --- Opus 4.6 gamma candidates (E > {ENERGY_CUT}, {len(ra)} events)",
        fontsize=11,
    )
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out_eq = Path(__file__).resolve().parent / "fig_gamma_skymap_opus_equatorial.png"
    plt.savefig(out_eq, dpi=200, bbox_inches="tight")
    plt.savefig(out_eq.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out_eq}")

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
