"""Build a full skymap of gamma candidates from real KASCADE data.

Runs Sonnet v4 classifier on all real data with E > 14.5 (matching
KASCADE ICRC 2015 point source search), collects timestamps from
extra_features, converts to equatorial coordinates, and generates
a skymap to compare with the known 5.5σ hotspot at RA=77.75°, Dec=70.85°.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from glob import glob
import time
import datetime

DEVICE = torch.device("cuda:0")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent.parent / "astro-bench-experiments" / "gamma-sonnet-22mar-v2"

REAL_TO_SIM = [0, 3, 4, 5, 6]
CUTS = {'Ze_max': 30.0, 'Ne_min': 4.8, 'Age_min': 0.2, 'Age_max': 1.48}
ENERGY_CUT = 14.5  # matching KASCADE ICRC 2015 paper


# --- Model definition (from train_v4.py) ---

class GammaNet(nn.Module):
    def __init__(self, feat_dim=10):
        super().__init__()
        self.elec_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ELU())
        self.muon_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ELU())
        self.joint = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ELU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ELU(),
            nn.AdaptiveAvgPool2d(3), nn.Flatten())
        self.cnn_proj = nn.Sequential(
            nn.Linear(2304, 512), nn.BatchNorm1d(512), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ELU())
        self.feat_net = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.BatchNorm1d(128), nn.ELU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ELU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ELU())
        self.head = nn.Sequential(
            nn.Linear(384, 256), nn.BatchNorm1d(256), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(128, 2))

    def forward(self, img, feat):
        elec = self.elec_conv(img[:, 0:1])
        muon = self.muon_conv(img[:, 1:2])
        cnn_out = self.cnn_proj(self.joint(torch.cat([elec, muon], dim=1)))
        feat_out = self.feat_net(feat)
        return self.head(torch.cat([cnn_out, feat_out], dim=1))


# --- Feature computation ---

def compute_derived_features(matrices, features_sim):
    n = len(features_sim)
    mat = matrices.astype(np.float32)
    total_elec = np.log1p(mat[:, :, :, 0].sum(axis=(1, 2)))
    total_muon = np.log1p(mat[:, :, :, 1].sum(axis=(1, 2)))
    tot = mat.sum(axis=(1, 2, 3)) + 1e-6
    muon_frac = mat[:, :, :, 1].sum(axis=(1, 2)) / tot
    muon_map = mat[:, :, :, 1].reshape(n, -1)
    muon_sum = muon_map.sum(axis=1, keepdims=True) + 1e-6
    muon_norm = muon_map / muon_sum
    muon_entropy = -(muon_norm * np.log(muon_norm + 1e-10)).sum(axis=1)
    ne_nmu = features_sim[:, 3] - features_sim[:, 4]
    return np.stack([
        features_sim[:, 0], features_sim[:, 1], features_sim[:, 2],
        features_sim[:, 3], features_sim[:, 4],
        ne_nmu, total_elec, total_muon, muon_frac, muon_entropy
    ], axis=1)


def compute_normalization_stats():
    print("Computing normalization stats from training data...", flush=True)
    X_train = np.load(DATA_DIR / "gamma_train" / "matrices.npy", mmap_mode='r')
    f_train = np.load(DATA_DIR / "gamma_train" / "features.npy", mmap_mode='r')
    rng = np.random.default_rng(42)
    n = len(f_train)

    sub_idx = rng.choice(n, size=min(200_000, n), replace=False)
    imgs = np.log1p(X_train[sub_idx].astype(np.float32))
    img_mean = imgs.mean(axis=0)
    img_std = imgs.std(axis=0)
    img_std[img_std == 0] = 1.0

    n_stat = min(300_000, n)
    stat_idx = rng.choice(n, size=n_stat, replace=False)
    feats_all = compute_derived_features(np.array(X_train[stat_idx]),
                                          np.array(f_train[stat_idx], dtype=np.float32))
    feat_mean = feats_all.mean(axis=0)
    feat_std = feats_all.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    return img_mean, img_std, feat_mean, feat_std


class RealDataset(Dataset):
    def __init__(self, matrices, features_10, img_mean, img_std, feat_mean, feat_std):
        self.matrices = matrices
        self.features_10 = features_10
        self.img_mean = img_mean
        self.img_std = img_std
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self):
        return len(self.features_10)

    def __getitem__(self, idx):
        mat = np.log1p(self.matrices[idx].astype(np.float32))
        mat = (mat - self.img_mean) / self.img_std
        mat = mat.transpose(2, 0, 1)
        feat = (self.features_10[idx] - self.feat_mean) / self.feat_std
        return torch.from_numpy(mat), torch.from_numpy(feat.astype(np.float32))


def calibrate_threshold(model, img_mean, img_std, feat_mean, feat_std):
    """Find gamma score threshold for 75% gamma efficiency on simulation test set."""
    print("Calibrating threshold from simulation test set...", flush=True)
    X_test = np.load(DATA_DIR / "gamma_test" / "matrices.npy", mmap_mode='r')
    f_test = np.load(DATA_DIR / "gamma_test" / "features.npy", mmap_mode='r')
    y_test = np.load(DATA_DIR / "gamma_test" / "labels_gamma.npy", mmap_mode='r')

    f_test_arr = np.array(f_test, dtype=np.float32)
    features_10_test = compute_derived_features(np.array(X_test), f_test_arr)
    ds = RealDataset(np.array(X_test), features_10_test, img_mean, img_std, feat_mean, feat_std)
    loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    scores = []
    with torch.no_grad():
        for img, feat in loader:
            img, feat = img.to(DEVICE), feat.to(DEVICE)
            logits = model(img, feat)
            scores.append(torch.softmax(logits, dim=1)[:, 0].cpu().numpy())
    scores = np.concatenate(scores)
    y_arr = np.array(y_test)

    gamma_scores = scores[y_arr == 0]
    threshold = np.percentile(gamma_scores, 25)
    h_pass = (scores[y_arr == 1] >= threshold).sum()
    print(f"  Threshold: {threshold:.6f}, hadrons passing: {h_pass}/{(y_arr==1).sum()}", flush=True)
    return threshold


def main():
    t0 = time.time()

    # Load model
    model = GammaNet(feat_dim=10).to(DEVICE)
    model.load_state_dict(torch.load(EXPERIMENT_DIR / "model_v4_best.pt",
                                      map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Loaded model", flush=True)

    img_mean, img_std, feat_mean, feat_std = compute_normalization_stats()
    threshold = calibrate_threshold(model, img_mean, img_std, feat_mean, feat_std)

    # Process all runs: load data, apply cuts, run inference, collect candidates with timestamps
    real_dir = DATA_DIR / "real_kascade"
    run_files = sorted(glob(str(real_dir / "*_matrices.npz")))
    print(f"\nProcessing {len(run_files)} runs (E > {ENERGY_CUT})...", flush=True)

    all_candidates = []  # (score, E, Ze, Az, timestamp)
    n_total = 0
    n_passing = 0

    for ri, mpath in enumerate(run_files):
        run_name = Path(mpath).name.replace("_matrices.npz", "")
        fpath = real_dir / f"{run_name}_features.npz"
        efpath = real_dir / f"{run_name}_extra_features.npz"

        feat = np.load(fpath)['features']
        if feat.ndim != 2 or len(feat) == 0:
            continue

        # Quality cuts + energy cut
        ze = feat[:, 3]
        ne = feat[:, 5]
        age = feat[:, 7]
        energy = feat[:, 0]

        mask = ((ze < CUTS['Ze_max']) & (ne > CUTS['Ne_min']) &
                (age > CUTS['Age_min']) & (age < CUTS['Age_max']) &
                (energy > ENERGY_CUT))

        if mask.sum() == 0:
            continue

        mat = np.load(mpath)['matrices'][mask][:, :, :, 1:3]  # channels 1,2
        feat_cut = feat[mask]
        feat_sim = feat_cut[:, REAL_TO_SIM].astype(np.float32)
        n_total += len(feat_sim)

        # Compute features and run inference
        features_10 = compute_derived_features(mat, feat_sim)
        ds = RealDataset(mat, features_10, img_mean, img_std, feat_mean, feat_std)
        loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)

        scores = []
        with torch.no_grad():
            for img, f in loader:
                img, f = img.to(DEVICE), f.to(DEVICE)
                logits = model(img, f)
                scores.append(torch.softmax(logits, dim=1)[:, 0].cpu().numpy())
        scores = np.concatenate(scores)

        # Collect candidates passing threshold
        passing = scores >= threshold
        n_pass = passing.sum()
        n_passing += n_pass

        if n_pass > 0:
            # Load extra_features for timestamps
            ef = np.load(efpath, allow_pickle=True)['extra_features']
            ef_cut = ef[mask]
            for idx in np.where(passing)[0]:
                timestamp = ef_cut[idx][8]  # datetime column
                e = feat_cut[idx, 0]
                ze_val = feat_cut[idx, 3]
                az_val = feat_cut[idx, 4]
                ne_val = feat_cut[idx, 5]
                nmu_val = feat_cut[idx, 6]
                all_candidates.append({
                    'score': float(scores[idx]),
                    'E': float(e),
                    'Ze': float(ze_val),
                    'Az': float(az_val),
                    'Ne': float(ne_val),
                    'Nmu': float(nmu_val),
                    'timestamp': timestamp,
                })

        if (ri + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {ri+1}/{len(run_files)} runs, {n_total:,} events, "
                  f"{n_passing:,} candidates ({elapsed:.0f}s)", flush=True)

    print(f"\nTotal: {n_total:,} events, {n_passing:,} candidates ({time.time()-t0:.0f}s)")

    # Convert to sky coordinates
    print("\nConverting to equatorial coordinates...", flush=True)
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz
    from astropy.time import Time
    import astropy.units as u

    kascade = EarthLocation(lon=8.4 * u.deg, lat=49.1 * u.deg, height=110 * u.m)

    ra_list = []
    dec_list = []
    energies = []
    scores_list = []
    n_no_time = 0

    for cand in all_candidates:
        if cand['timestamp'] is None:
            n_no_time += 1
            continue
        try:
            coord = SkyCoord(
                alt=(90 - cand['Ze']) * u.deg,
                az=cand['Az'] * u.deg,
                obstime=Time(cand['timestamp']),
                frame='altaz',
                location=kascade
            ).transform_to('icrs')
            ra_list.append(coord.ra.deg)
            dec_list.append(coord.dec.deg)
            energies.append(cand['E'])
            scores_list.append(cand['score'])
        except Exception as e:
            n_no_time += 1

    if n_no_time > 0:
        print(f"  Skipped {n_no_time} events without valid timestamps")

    ra = np.array(ra_list)
    dec = np.array(dec_list)
    energies = np.array(energies)
    scores_arr = np.array(scores_list)
    print(f"  {len(ra)} candidates with sky coordinates")

    # Save all candidate data
    np.savez("gamma_skymap_candidates.npz",
             ra=ra, dec=dec, energies=energies, scores=scores_arr)
    print("Saved gamma_skymap_candidates.npz")

    # KASCADE hotspot from ICRC 2015
    hotspot_ra, hotspot_dec = 77.75, 70.85

    # Check angular distances to hotspot
    hotspot = SkyCoord(ra=hotspot_ra * u.deg, dec=hotspot_dec * u.deg)
    candidate_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    seps = candidate_coords.separation(hotspot).deg

    print(f"\n{'='*60}")
    print(f"Distance to KASCADE hotspot (RA={hotspot_ra}, Dec={hotspot_dec})")
    print(f"{'='*60}")
    near = seps < 10
    print(f"  Within 10°: {near.sum()} candidates")
    near5 = seps < 5
    print(f"  Within 5°:  {near5.sum()} candidates")
    near2 = seps < 2
    print(f"  Within 2°:  {near2.sum()} candidates")

    if near.sum() > 0:
        print(f"\n  Candidates within 10° of hotspot:")
        for idx in np.where(near)[0]:
            print(f"    RA={ra[idx]:.2f} Dec={dec[idx]:.2f} E={energies[idx]:.2f} "
                  f"score={scores_arr[idx]:.4f} sep={seps[idx]:.1f}°")

    # Generate skymap
    print("\nGenerating skymap...", flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gal_coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).galactic
    gal_l = gal_coords.l.wrap_at('180d').radian
    gal_b = gal_coords.b.radian

    hotspot_gal = hotspot.galactic

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='aitoff')

    # Galactic plane
    plane_l = np.linspace(-np.pi, np.pi, 360)
    ax.plot(plane_l, np.zeros_like(plane_l), 'k-', alpha=0.15, linewidth=8)

    # All candidates colored by energy
    sc = ax.scatter(gal_l, gal_b, c=energies, cmap='YlOrRd', s=8, alpha=0.6,
                    vmin=14.5, vmax=17, zorder=3)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
    cbar.set_label('log$_{10}$(E/eV)')

    # KASCADE hotspot
    ax.scatter(hotspot_gal.l.wrap_at('180d').radian, hotspot_gal.b.radian,
               c='none', s=200, marker='o', edgecolors='blue', linewidths=2, zorder=5)
    ax.annotate('KASCADE hotspot\n(5.5σ, ICRC 2015)',
                (hotspot_gal.l.wrap_at('180d').radian, hotspot_gal.b.radian),
                fontsize=7, color='blue', xytext=(10, -15), textcoords='offset points')

    # Known sources
    known = [('Crab', 83.633, 22.015), ('Cyg OB2', 308.08, 41.31),
             ('Cas A', 350.85, 58.82), ('LHAASO J2108', 317.15, 51.95)]
    for name, sra, sdec in known:
        src = SkyCoord(ra=sra * u.deg, dec=sdec * u.deg).galactic
        ax.scatter(src.l.wrap_at('180d').radian, src.b.radian,
                   c='gray', s=30, marker='D', zorder=4, alpha=0.7)
        ax.annotate(name, (src.l.wrap_at('180d').radian, src.b.radian),
                    fontsize=5, color='gray', xytext=(4, 4), textcoords='offset points')

    ax.set_title(f'AI classifier gamma candidates on real KASCADE data '
                 f'(E > {ENERGY_CUT}, {len(ra)} events)', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('paper/fig_gamma_skymap_full.png', dpi=200, bbox_inches='tight')
    plt.savefig('paper/fig_gamma_skymap_full.pdf', bbox_inches='tight')
    print("Saved paper/fig_gamma_skymap_full.{png,pdf}")

    # Also make equatorial skymap (matching KASCADE paper format)
    fig2 = plt.figure(figsize=(14, 7))
    ax2 = fig2.add_subplot(111, projection='aitoff')

    ra_wrap = np.where(ra > 180, ra - 360, ra)
    sc2 = ax2.scatter(np.radians(ra_wrap), np.radians(dec), c=energies,
                      cmap='YlOrRd', s=8, alpha=0.6, vmin=14.5, vmax=17, zorder=3)
    cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.7, pad=0.08)
    cbar2.set_label('log$_{10}$(E/eV)')

    # Hotspot
    hs_ra_wrap = hotspot_ra if hotspot_ra <= 180 else hotspot_ra - 360
    ax2.scatter(np.radians(hs_ra_wrap), np.radians(hotspot_dec),
                c='none', s=200, marker='o', edgecolors='blue', linewidths=2, zorder=5)
    ax2.annotate('KASCADE hotspot\n(5.5σ)',
                 (np.radians(hs_ra_wrap), np.radians(hotspot_dec)),
                 fontsize=7, color='blue', xytext=(10, -15), textcoords='offset points')

    ax2.set_title(f'Equatorial coordinates — AI classifier gamma candidates '
                  f'(E > {ENERGY_CUT}, {len(ra)} events)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('paper/fig_gamma_skymap_equatorial.png', dpi=200, bbox_inches='tight')
    print("Saved paper/fig_gamma_skymap_equatorial.png")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
