"""Run unfolding on real KASCADE data using our best model (v2_sam).

Steps:
1. Load the trained model (v2_sam)
2. Build confusion matrix from simulation test set
3. Stream real KASCADE data from S3, run inference
4. Unfold predicted fractions per energy bin using confusion matrix inversion
5. Plot energy spectra per mass group
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.metrics import confusion_matrix
import boto3, io, gc, sys, time
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUT_DIR = "submissions/opus-composition-mar14/matched_pipeline"
DEVICE = "cuda"
BUCKET = 'kascade-with-arr-times-npz'
PARTICLES = ['H', 'He', 'C', 'Si', 'Fe']
COLORS = ['#F44336', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']

# Quality cuts matching Kuznetsov et al.
CUTS_ZE = 18  # degrees (legacy uses 0-18 for spectra)
CUTS_NE = 4.8
CUTS_NMU = 3.6
CUTS_AGE_LO = 0.2
CUTS_AGE_HI = 1.48

# Physical constants from legacy code
DATA_PART = 0.185  # fraction of data that is unblinded
T_EFF = 792519034.6611552 * DATA_PART  # effective observation time [s]
AREA = np.pi * (91.0 ** 2)  # detector area [m²]

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 300})

def p(msg): print(msg, flush=True)

# --- Model (same as training scripts) ---
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch//r,8)), nn.ReLU(), nn.Linear(max(ch//r,8), ch), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class HybridModel(nn.Module):
    def __init__(self, n_feat=6, n_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ChannelAttention(64), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ChannelAttention(128), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            ChannelAttention(256), nn.AdaptiveAvgPool2d(1))
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_feat,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2))
        self.head = nn.Sequential(
            nn.Linear(256+128,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,n_classes))
    def forward(self, mat, feat):
        mat = mat.permute(0,3,1,2)
        return self.head(torch.cat([self.cnn(mat).flatten(1), self.feat_mlp(feat)], dim=1))


def unfolding_direct(trues, preds, pred_counts, labels=(0,1,2,3,4)):
    """Direct matrix inversion unfolding."""
    cm = confusion_matrix(trues, preds, normalize='true', labels=labels).T
    cm_inv = np.linalg.inv(cm)
    result = (cm_inv @ pred_counts).ravel()
    cov = cm_inv @ np.diag(pred_counts) @ cm_inv.T
    errors = np.sqrt(np.abs(np.diag(cov)))
    return result, errors


def main():
    t0 = time.time()

    # === 1. Load model ===
    p("Loading model (v2_sam)...")
    model = HybridModel(n_feat=6).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUT_DIR}/model_v2_sam.pt", weights_only=True))
    model.eval()

    # DE biases from v2_sam
    biases = np.array([-0.0549, -0.1059, -0.1677, -0.2252, -0.3169])

    # === 2. Build confusion matrix from simulation ===
    p("Building confusion matrix from simulation test set...")
    RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
    CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}

    raw_feat = np.load(f"{DATA_DIR}/qgs_spectra_features.npz")['features']
    raw_mat = np.load(f"{DATA_DIR}/qgs_spectra_matrices.npz")['matrices']
    reco = raw_feat[:, 1:].astype(np.float32)
    labels = raw_feat[:, 0].astype(np.int64) - 1
    mask = np.ones(len(reco), dtype=bool)
    for fn, (lo, hi) in CUTS.items():
        i = RECO_HEADERS.index(fn); mask &= (reco[:, i] > lo) & (reco[:, i] < hi)
    reco = reco[mask]; labels = labels[mask]; mat = raw_mat[mask]
    del raw_mat, raw_feat; gc.collect()

    # Same split
    n = len(labels); nv = int(n*0.3); nt = n-nv
    all_idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    train_idx = all_idx[:nt]; valid_idx = all_idx[nt:]
    vperm = torch.randperm(nv, generator=torch.Generator().manual_seed(42)).numpy()
    ntest = int(nv*0.3)
    test_idx = valid_idx[vperm[nv-ntest:]]

    # Run inference on sim test set to get confusion matrix
    test_mat = torch.log1p(torch.from_numpy(mat[test_idx][:, :, :, [1, 2]]).float())
    r = reco[test_idx]
    Ne=r[:,6];Nmu=r[:,7];Age=r[:,8];Ze=r[:,4];E=r[:,0]
    test_reco = torch.from_numpy(np.column_stack([
        (Ne-5.31)/0.5,(Nmu-4.3)/0.42,Age-1.0,Ze/60.0,
        (Ne-Nmu-0.8)/0.3,(E-15.5)/1.0]).astype(np.float32))
    test_labels = labels[test_idx]

    sim_preds = []
    BS = 2048
    with torch.no_grad():
        for i in range(0, len(test_labels), BS):
            a = test_mat[i:i+BS].to(DEVICE)
            r_ = test_reco[i:i+BS].to(DEVICE)
            with autocast(device_type='cuda'):
                out = model(a, r_)
            probs = F.softmax(out.float(), 1).cpu().numpy()
            preds = (np.log(probs + 1e-10) + biases).argmax(1)
            sim_preds.append(preds)
    sim_preds = np.concatenate(sim_preds)

    # Store sim predictions and labels for unfolding per energy bin
    sim_df_E = reco[test_idx, 0]  # energies
    p(f"  Sim test: {len(test_labels)} events, acc={(sim_preds==test_labels).mean():.4f}")
    del test_mat, test_reco, mat, reco; gc.collect()

    # === 3. Stream real data from S3 and run inference ===
    p("Downloading and classifying real KASCADE data...")
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    all_files = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET):
        all_files.extend([obj['Key'] for obj in page.get('Contents', [])])
    runs = sorted(set(f.split('_')[0] for f in all_files))
    runs = [r for r in runs if r[-1] in {'0', '1'}]
    p(f"  {len(runs)} runs available")

    real_energies = []
    real_predictions = []
    total_events = 0

    for run in tqdm(runs, desc="Processing"):
        try:
            # Download matrices
            buf = io.BytesIO()
            s3.download_fileobj(BUCKET, f'{run}_matrices.npz', buf)
            buf.seek(0)
            mat_run = np.load(buf, allow_pickle=True)['matrices']

            # Download features
            buf2 = io.BytesIO()
            s3.download_fileobj(BUCKET, f'{run}_features.npz', buf2)
            buf2.seek(0)
            feat_run = np.load(buf2, allow_pickle=True)['features'].astype(np.float64)
            # Real features: [E, Xc, Yc, Ze, Az, Ne, Nmu, Age]

            if len(mat_run) == 0:
                continue

            # Apply quality cuts (Ze < 18 for spectra, matching legacy)
            Ze = feat_run[:, 3]; Ne = feat_run[:, 5]; Nmu = feat_run[:, 6]; Age = feat_run[:, 7]
            cut_mask = (Ze < CUTS_ZE) & (Ne > CUTS_NE) & (Nmu > CUTS_NMU) & \
                       (Age > CUTS_AGE_LO) & (Age < CUTS_AGE_HI)
            mat_cut = mat_run[cut_mask]
            feat_cut = feat_run[cut_mask]

            if len(mat_cut) == 0:
                continue

            # Prepare inputs (channels 1,2 = electron + muon)
            x = torch.log1p(torch.from_numpy(mat_cut[:, :, :, [1, 2]].astype(np.float32)))
            E_real = feat_cut[:, 0]
            Ne_r = feat_cut[:, 5]; Nmu_r = feat_cut[:, 6]
            Age_r = feat_cut[:, 7]; Ze_r = feat_cut[:, 3]
            reco_real = torch.from_numpy(np.column_stack([
                (Ne_r-5.31)/0.5, (Nmu_r-4.3)/0.42, Age_r-1.0, Ze_r/60.0,
                (Ne_r-Nmu_r-0.8)/0.3, (E_real-15.5)/1.0
            ]).astype(np.float32))

            # Run inference
            preds_run = []
            with torch.no_grad():
                for i in range(0, len(x), BS):
                    a = x[i:i+BS].to(DEVICE)
                    r_ = reco_real[i:i+BS].to(DEVICE)
                    with autocast(device_type='cuda'):
                        out = model(a, r_)
                    probs = F.softmax(out.float(), 1).cpu().numpy()
                    preds = (np.log(probs + 1e-10) + biases).argmax(1)
                    preds_run.append(preds)

            preds_run = np.concatenate(preds_run)
            real_energies.append(E_real)
            real_predictions.append(preds_run)
            total_events += len(preds_run)

        except Exception as e:
            continue

    real_energies = np.concatenate(real_energies)
    real_predictions = np.concatenate(real_predictions)
    p(f"  Total real events after cuts: {total_events:,}")
    p(f"  [{time.time()-t0:.0f}s]")

    # === 4. Compute spectra (folded + unfolded) ===
    p("Computing spectra...")

    energy_bins = np.linspace(15, 17, 16)  # 15 bins from 10^15 to 10^17 eV
    centers = (10**energy_bins[1:] + 10**energy_bins[:-1]) / 2

    th_min, th_max = np.radians(0), np.radians(CUTS_ZE)
    exposure = AREA * T_EFF * np.pi * (np.cos(th_min)**2 - np.cos(th_max)**2)
    C = centers**2.7 / np.diff(10**energy_bins) / exposure

    # --- Folded spectra (no unfolding) ---
    folded = {}
    for cls in range(5):
        mask_cls = real_predictions == cls
        folded[cls], _ = np.histogram(real_energies[mask_cls], bins=energy_bins)
    folded['total'], _ = np.histogram(real_energies, bins=energy_bins)

    # --- Unfolded spectra (confusion matrix inversion per energy bin) ---
    # Use energy-dependent confusion matrices
    cm_bins = [0, 15.15, 15.5, 16.0, 16.5, 20]  # energy bins for CM
    unfolded = {cls: np.zeros(len(energy_bins)-1) for cls in range(5)}
    unfolded_err = {cls: np.zeros(len(energy_bins)-1) for cls in range(5)}

    for bin_idx in range(len(energy_bins)-1):
        emin, emax = energy_bins[bin_idx], energy_bins[bin_idx+1]
        emean = (emin + emax) / 2

        # Find appropriate CM energy bin
        cm_idx = np.searchsorted(cm_bins, emean) - 1
        cm_idx = max(0, min(cm_idx, len(cm_bins)-2))
        cm_emin, cm_emax = cm_bins[cm_idx], cm_bins[cm_idx+1]

        # Get sim predictions in this CM energy range
        sim_mask = (sim_df_E >= cm_emin) & (sim_df_E < cm_emax)
        if sim_mask.sum() < 100:
            sim_mask = np.ones(len(sim_df_E), dtype=bool)  # fallback to full

        # Get real prediction counts in this energy bin
        real_mask = (real_energies >= emin) & (real_energies < emax)
        pred_counts = np.array([np.sum(real_predictions[real_mask] == c) for c in range(5)])

        if pred_counts.sum() < 10:
            continue

        try:
            result, errors = unfolding_direct(
                test_labels[sim_mask], sim_preds[sim_mask], pred_counts)
            for cls in range(5):
                unfolded[cls][bin_idx] = max(result[cls], 0)
                unfolded_err[cls][bin_idx] = errors[cls]
        except Exception as e:
            p(f"  Unfolding failed at bin {bin_idx}: {e}")
            for cls in range(5):
                unfolded[cls][bin_idx] = folded.get(cls, np.zeros(len(energy_bins)-1))[bin_idx]

    # === 5. Plot ===
    p("Plotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Folded
    for cls in range(5):
        N = folded[cls]
        ax1.errorbar(centers, N * C, yerr=np.sqrt(np.maximum(N, 1)) * C,
                     fmt='o-', color=COLORS[cls], label=PARTICLES[cls],
                     markersize=4, linewidth=1, capsize=2)
    ax1.set_xlabel('Energy $E$ [eV]')
    ax1.set_ylabel(r'$E^{2.7} \cdot dN/dE$ [eV$^{1.7}$ m$^{-2}$ sr$^{-1}$ s$^{-1}$]')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('Folded spectra')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    # Unfolded
    for cls in range(5):
        N = unfolded[cls]
        N_err = unfolded_err[cls]
        valid = N > 0
        ax2.errorbar(centers[valid], N[valid] * C[valid],
                     yerr=N_err[valid] * C[valid],
                     fmt='o-', color=COLORS[cls], label=PARTICLES[cls],
                     markersize=4, linewidth=1, capsize=2)
    ax2.set_xlabel('Energy $E$ [eV]')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('Unfolded spectra (direct)')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    plt.suptitle(f'KASCADE Real Data — CNN+Attn+MLP (SAM+DE)\n'
                 f'{total_events:,} events, $\\theta < {CUTS_ZE}°$', fontsize=12)
    plt.tight_layout()
    plt.savefig('paper/fig_real_spectra.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_real_spectra.png', bbox_inches='tight')
    p(f"Saved paper/fig_real_spectra.pdf")
    p(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
