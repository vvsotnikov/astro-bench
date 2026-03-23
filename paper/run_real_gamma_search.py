"""Apply Sonnet's v4 gamma classifier to real KASCADE data.

Searches for gamma-ray candidates using Kostunin's method:
choose energy cut E > E0 such that N_real × p_survival < 1,
then any surviving event is a gamma-ray candidate.

Uses model_v4_best.pt from gamma-sonnet-22mar-v2.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from glob import glob
import time

DEVICE = torch.device("cuda:0")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent.parent / "astro-bench-experiments" / "gamma-sonnet-22mar-v2"

# Real data feature columns: [E, Xc, Yc, Ze, Az, Ne, Nmu, Age]
# Sim feature columns:       [E, Ze, Az, Ne, Nmu]
REAL_TO_SIM = [0, 3, 4, 5, 6]  # indices into real features

# Quality cuts (matching test set)
CUTS = {'Ze_max': 30.0, 'Ne_min': 4.8, 'Age_min': 0.2, 'Age_max': 1.48}

# Survival rate from benchmark (v4 model)
P_SURVIVAL = 4.67e-04


# --- Model definition (from train_v4.py) ---

class GammaNet(nn.Module):
    def __init__(self, feat_dim=10):
        super().__init__()
        self.elec_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
        )
        self.muon_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
        )
        self.joint = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ELU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ELU(),
            nn.AdaptiveAvgPool2d(3),
            nn.Flatten(),
        )
        self.cnn_proj = nn.Sequential(
            nn.Linear(2304, 512), nn.BatchNorm1d(512), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ELU(),
        )
        self.feat_net = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.BatchNorm1d(128), nn.ELU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ELU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(384, 256), nn.BatchNorm1d(256), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, img, feat):
        elec = self.elec_conv(img[:, 0:1])
        muon = self.muon_conv(img[:, 1:2])
        cnn_out = self.cnn_proj(self.joint(torch.cat([elec, muon], dim=1)))
        feat_out = self.feat_net(feat)
        return self.head(torch.cat([cnn_out, feat_out], dim=1))


# --- Data handling ---

def compute_derived_features(matrices, features_sim):
    """Compute 10 features matching v4 training pipeline.
    features_sim: (N, 5) = [E, Ze, Az, Ne, Nmu]
    Returns: (N, 10) = [E, Ze, Az, Ne, Nmu, Ne-Nmu, log1p(total_elec), log1p(total_muon), muon_frac, muon_entropy]
    """
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
    """Compute image and feature normalization stats from training data (matching v4)."""
    print("Computing normalization stats from training data...", flush=True)
    X_train = np.load(DATA_DIR / "gamma_train" / "matrices.npy", mmap_mode='r')
    f_train = np.load(DATA_DIR / "gamma_train" / "features.npy", mmap_mode='r')

    # Image stats from random subset
    rng = np.random.default_rng(42)
    n = len(f_train)
    sub_idx = rng.choice(n, size=min(200_000, n), replace=False)
    imgs = np.log1p(X_train[sub_idx].astype(np.float32))
    img_mean = imgs.mean(axis=0)
    img_std = imgs.std(axis=0)
    img_std[img_std == 0] = 1.0

    # Feature stats: need to precompute derived features on a subset
    n_stat = min(300_000, n)
    stat_idx = rng.choice(n, size=n_stat, replace=False)
    f_sub = np.array(f_train[stat_idx], dtype=np.float32)
    mat_sub = np.array(X_train[stat_idx])
    feats_all = compute_derived_features(mat_sub, f_sub)
    feat_mean = feats_all.mean(axis=0)
    feat_std = feats_all.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    print(f"  Stats computed from {n_stat:,} training samples.", flush=True)
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
        mat = mat.transpose(2, 0, 1)  # (2, 16, 16)
        feat = (self.features_10[idx] - self.feat_mean) / self.feat_std
        return torch.from_numpy(mat), torch.from_numpy(feat.astype(np.float32))


def load_real_data_with_cuts(energy_cut=None):
    """Load all real KASCADE data, apply quality cuts and optional energy cut.
    Returns matrices (N, 16, 16, 2), features_sim (N, 5), raw_features (N, 8).
    """
    real_dir = DATA_DIR / "real_kascade"
    run_files = sorted(glob(str(real_dir / "*_matrices.npz")))
    print(f"Loading real data from {len(run_files)} runs...", flush=True)

    all_matrices = []
    all_features = []
    t0 = time.time()

    for i, mpath in enumerate(run_files):
        run_name = Path(mpath).name.replace("_matrices.npz", "")
        fpath = real_dir / f"{run_name}_features.npz"

        mat = np.load(mpath)['matrices']         # (N, 16, 16, 3)
        feat = np.load(fpath)['features']         # (N, 8)

        # Skip empty runs
        if feat.ndim != 2 or len(feat) == 0:
            continue

        # Quality cuts on real data
        # Real features: [E, Xc, Yc, Ze, Az, Ne, Nmu, Age]
        ze = feat[:, 3]
        ne = feat[:, 5]
        age = feat[:, 7]

        mask = (ze < CUTS['Ze_max']) & (ne > CUTS['Ne_min']) & \
               (age > CUTS['Age_min']) & (age < CUTS['Age_max'])

        if energy_cut is not None:
            mask &= (feat[:, 0] > energy_cut)

        if mask.sum() > 0:
            # Take channels [1, 2] to match simulation format (elec, muon)
            all_matrices.append(mat[mask][:, :, :, 1:3])
            all_features.append(feat[mask])

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(run_files)} runs loaded ({elapsed:.0f}s)", flush=True)

    matrices = np.concatenate(all_matrices)      # (N, 16, 16, 2)
    features_raw = np.concatenate(all_features)  # (N, 8)

    # Map to sim feature format: [E, Ze, Az, Ne, Nmu]
    features_sim = features_raw[:, REAL_TO_SIM].astype(np.float32)

    print(f"  Total after cuts: {len(matrices):,} events ({time.time()-t0:.0f}s)", flush=True)
    return matrices, features_sim, features_raw


def main():
    t0 = time.time()

    # Load model
    model = GammaNet(feat_dim=10).to(DEVICE)
    model_path = EXPERIMENT_DIR / "model_v4_best.pt"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Loaded model from {model_path}", flush=True)

    # Normalization stats from training data
    img_mean, img_std, feat_mean, feat_std = compute_normalization_stats()

    # First pass: load all data with quality cuts, no energy cut, to understand the dataset
    matrices, features_sim, features_raw = load_real_data_with_cuts(energy_cut=None)
    n_total = len(matrices)
    print(f"\nTotal real events after quality cuts: {n_total:,}")

    # Compute derived features
    print("Computing derived features for real data...", flush=True)
    features_10 = compute_derived_features(matrices, features_sim)

    # Run inference
    print("Running inference...", flush=True)
    ds = RealDataset(matrices, features_10, img_mean, img_std, feat_mean, feat_std)
    loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    all_scores = []
    with torch.no_grad():
        for img, feat in loader:
            img, feat = img.to(DEVICE), feat.to(DEVICE)
            logits = model(img, feat)
            gamma_prob = torch.softmax(logits, dim=1)[:, 0]
            all_scores.append(gamma_prob.cpu().numpy())
    scores = np.concatenate(all_scores)

    # Find threshold from simulation: the score threshold that gives 75% gamma efficiency
    # We need to calibrate this from the test set
    print("\nCalibrating threshold from simulation test set...", flush=True)
    X_test = np.load(DATA_DIR / "gamma_test" / "matrices.npy", mmap_mode='r')
    f_test = np.load(DATA_DIR / "gamma_test" / "features.npy", mmap_mode='r')
    y_test = np.load(DATA_DIR / "gamma_test" / "labels_gamma.npy", mmap_mode='r')

    f_test_arr = np.array(f_test, dtype=np.float32)
    features_10_test = compute_derived_features(np.array(X_test), f_test_arr)
    ds_test = RealDataset(np.array(X_test), features_10_test, img_mean, img_std, feat_mean, feat_std)
    loader_test = DataLoader(ds_test, batch_size=4096, shuffle=False, num_workers=4, pin_memory=True)

    test_scores = []
    with torch.no_grad():
        for img, feat in loader_test:
            img, feat = img.to(DEVICE), feat.to(DEVICE)
            logits = model(img, feat)
            gamma_prob = torch.softmax(logits, dim=1)[:, 0]
            test_scores.append(gamma_prob.cpu().numpy())
    test_scores = np.concatenate(test_scores)
    y_test_arr = np.array(y_test)

    # Find threshold for 75% gamma efficiency
    gamma_scores = test_scores[y_test_arr == 0]
    threshold_75 = np.percentile(gamma_scores, 25)  # keep top 75%
    hadron_scores = test_scores[y_test_arr == 1]
    n_hadron_pass = (hadron_scores >= threshold_75).sum()
    print(f"  Threshold for 75% gamma eff: {threshold_75:.6f}")
    print(f"  Hadrons passing: {n_hadron_pass} / {len(hadron_scores)} = {n_hadron_pass/len(hadron_scores):.4e}")

    # Apply to real data
    real_passing = scores >= threshold_75
    n_passing = real_passing.sum()
    print(f"\n{'='*60}")
    print(f"RESULTS: Real KASCADE data gamma-ray search")
    print(f"{'='*60}")
    print(f"Total events after quality cuts: {n_total:,}")
    print(f"Events passing gamma threshold: {n_passing:,}")
    print(f"Expected background (N × p_survival): {n_total * P_SURVIVAL:.1f}")

    # Energy-binned analysis (Kostunin's method)
    print(f"\n{'='*60}")
    print(f"Energy-binned search (N_expected_bg < 1 criterion)")
    print(f"{'='*60}")
    energies = features_raw[:, 0]  # log10(E/eV)
    e_bins = np.arange(14.0, 18.5, 0.5)

    print(f"{'E_min':>8s} {'E_max':>8s} {'N_events':>10s} {'N_pass':>8s} {'N_exp_bg':>10s} {'Significant?':>12s}")
    print("-" * 65)
    for i in range(len(e_bins) - 1):
        e_lo, e_hi = e_bins[i], e_bins[i+1]
        mask = (energies >= e_lo) & (energies < e_hi)
        n_bin = mask.sum()
        n_pass_bin = (real_passing & mask).sum()
        n_exp_bg = n_bin * P_SURVIVAL

        sig = "*** YES ***" if (n_pass_bin > 0 and n_exp_bg < 1) else \
              "candidate" if (n_pass_bin > 0 and n_exp_bg < 3) else ""
        print(f"{e_lo:8.1f} {e_hi:8.1f} {n_bin:10,} {n_pass_bin:8,} {n_exp_bg:10.2f} {sig:>12s}")

    # Also check cumulative from high energy
    print(f"\n{'='*60}")
    print(f"Cumulative from high energy (E > E_cut)")
    print(f"{'='*60}")
    print(f"{'E_cut':>8s} {'N_events':>10s} {'N_pass':>8s} {'N_exp_bg':>10s} {'Significant?':>12s}")
    print("-" * 55)
    for e_cut in [15.0, 15.5, 16.0, 16.5, 17.0]:
        mask = energies >= e_cut
        n_bin = mask.sum()
        n_pass_bin = (real_passing & mask).sum()
        n_exp_bg = n_bin * P_SURVIVAL
        sig = "*** YES ***" if (n_pass_bin > 0 and n_exp_bg < 1) else \
              "candidate" if (n_pass_bin > 0 and n_exp_bg < 3) else ""
        print(f"{e_cut:8.1f} {n_bin:10,} {n_pass_bin:8,} {n_exp_bg:10.2f} {sig:>12s}")

    # Save candidates
    if n_passing > 0:
        candidate_idx = np.where(real_passing)[0]
        np.savez("gamma_candidates.npz",
                 scores=scores[candidate_idx],
                 features=features_raw[candidate_idx],
                 energies=energies[candidate_idx])
        print(f"\nSaved {len(candidate_idx)} candidates to gamma_candidates.npz")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
