"""Run unfolding on real KASCADE data using local files + energy-binned CMs.

Fixes from v1:
- Reads from local data/real_kascade/ (no S3 streaming)
- Uses energy-binned confusion matrices (different CM per energy range)
- Proper ylim matching legacy plots
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.metrics import confusion_matrix
import os, gc, sys, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = "data"
REAL_DIR = "data/real_kascade"
OUT_DIR = "submissions/opus-composition-mar14/matched_pipeline"
DEVICE = "cuda"
PARTICLES = ['H', 'He', 'C', 'Si', 'Fe']
COLORS = ['#F44336', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']

# Quality cuts for spectra (Ze < 18, matching legacy)
CUTS_ZE = 18
CUTS_NE = 4.8
CUTS_NMU = 3.6
CUTS_AGE_LO = 0.2
CUTS_AGE_HI = 1.48

# Physical constants
DATA_PART = 0.185
T_EFF = 792519034.6611552 * DATA_PART
AREA = np.pi * (91.0 ** 2)

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 300})
def p(msg): print(msg, flush=True)

# --- Model ---
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
            nn.Linear(128,5))
    def forward(self, mat, feat):
        mat = mat.permute(0,3,1,2)
        return self.head(torch.cat([self.cnn(mat).flatten(1), self.feat_mlp(feat)], dim=1))


def unfolding_direct(cm_trues, cm_preds, pred_counts, labels=(0,1,2,3,4)):
    """Direct matrix inversion unfolding."""
    cm = confusion_matrix(cm_trues, cm_preds, normalize='true', labels=labels).T
    cm_inv = np.linalg.inv(cm)
    result = (cm_inv @ pred_counts).ravel()
    cov = cm_inv @ np.diag(np.maximum(pred_counts, 1).astype(float)) @ cm_inv.T
    errors = np.sqrt(np.abs(np.diag(cov)))
    return result, errors


def main():
    t0 = time.time()

    # === 1. Load model ===
    p("Loading model...")
    model = HybridModel(n_feat=6).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUT_DIR}/model_v2_sam.pt", weights_only=True))
    model.eval()
    biases = np.array([-0.0549, -0.1059, -0.1677, -0.2252, -0.3169])

    # === 2. Build energy-binned confusion matrices from simulation ===
    p("Building energy-binned confusion matrices...")
    RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
    SIM_CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}

    raw_feat = np.load(f"{DATA_DIR}/qgs_spectra_features.npz")['features']
    raw_mat = np.load(f"{DATA_DIR}/qgs_spectra_matrices.npz")['matrices']
    reco = raw_feat[:, 1:].astype(np.float32)
    labels = raw_feat[:, 0].astype(np.int64) - 1
    mask = np.ones(len(reco), dtype=bool)
    for fn, (lo, hi) in SIM_CUTS.items():
        i = RECO_HEADERS.index(fn); mask &= (reco[:, i] > lo) & (reco[:, i] < hi)
    reco = reco[mask]; labels = labels[mask]; mat = raw_mat[mask]
    del raw_mat, raw_feat; gc.collect()

    # Same split
    n = len(labels); nv = int(n*0.3); nt = n-nv
    all_idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    test_idx = all_idx[nt:]  # Use full validation+test for CM (more data)

    # Run inference on sim
    BS = 2048
    sim_energies = reco[test_idx, 0]
    sim_labels = labels[test_idx]
    sim_preds = []
    for i in range(0, len(test_idx), BS):
        idx = test_idx[i:i+BS]
        x = torch.log1p(torch.from_numpy(mat[idx][:, :, :, [1, 2]].astype(np.float32)))
        r = reco[idx]
        Ne=r[:,6];Nmu=r[:,7];Age=r[:,8];Ze=r[:,4];E=r[:,0]
        reco_t = torch.from_numpy(np.column_stack([
            (Ne-5.31)/0.5,(Nmu-4.3)/0.42,Age-1.0,Ze/60.0,
            (Ne-Nmu-0.8)/0.3,(E-15.5)/1.0]).astype(np.float32))
        with torch.no_grad():
            with autocast(device_type='cuda'):
                out = model(x.to(DEVICE), reco_t.to(DEVICE))
            probs = F.softmax(out.float(), 1).cpu().numpy()
            sim_preds.append((np.log(probs + 1e-10) + biases).argmax(1))
    sim_preds = np.concatenate(sim_preds)
    p(f"  Sim: {len(sim_labels)} events, acc={(sim_preds==sim_labels).mean():.4f}")

    # Energy bins for confusion matrices (matching legacy cm_buckets)
    cm_energy_bins = [0, 15.15, 15.5, 16.0, 16.5, 20]
    p(f"  CM energy bins: {cm_energy_bins}")
    del mat, reco; gc.collect()

    # === 3. Classify real data from local files ===
    p("Classifying real data from local files...")
    runs = sorted([f.replace('_matrices.npz', '') for f in os.listdir(REAL_DIR)
                   if f.endswith('_matrices.npz')])
    p(f"  {len(runs)} runs available locally")

    real_energies = []
    real_predictions = []
    total = 0

    for run in runs:
        try:
            mat_run = np.load(f"{REAL_DIR}/{run}_matrices.npz", allow_pickle=True)['matrices']
            feat_run = np.load(f"{REAL_DIR}/{run}_features.npz", allow_pickle=True)['features'].astype(np.float64)

            if len(mat_run) == 0: continue

            Ze = feat_run[:, 3]; Ne = feat_run[:, 5]; Nmu = feat_run[:, 6]; Age = feat_run[:, 7]
            cut = (Ze < CUTS_ZE) & (Ne > CUTS_NE) & (Nmu > CUTS_NMU) & \
                  (Age > CUTS_AGE_LO) & (Age < CUTS_AGE_HI)
            mat_cut = mat_run[cut]; feat_cut = feat_run[cut]
            if len(mat_cut) == 0: continue

            x = torch.log1p(torch.from_numpy(mat_cut[:, :, :, [1, 2]].astype(np.float32)))
            E_r = feat_cut[:, 0]; Ne_r = feat_cut[:, 5]; Nmu_r = feat_cut[:, 6]
            Age_r = feat_cut[:, 7]; Ze_r = feat_cut[:, 3]
            reco_r = torch.from_numpy(np.column_stack([
                (Ne_r-5.31)/0.5, (Nmu_r-4.3)/0.42, Age_r-1.0, Ze_r/60.0,
                (Ne_r-Nmu_r-0.8)/0.3, (E_r-15.5)/1.0
            ]).astype(np.float32))

            preds_run = []
            with torch.no_grad():
                for i in range(0, len(x), BS):
                    a = x[i:i+BS].to(DEVICE)
                    r_ = reco_r[i:i+BS].to(DEVICE)
                    with autocast(device_type='cuda'):
                        out = model(a, r_)
                    probs = F.softmax(out.float(), 1).cpu().numpy()
                    preds_run.append((np.log(probs + 1e-10) + biases).argmax(1))

            real_energies.append(E_r)
            real_predictions.append(np.concatenate(preds_run))
            total += len(feat_cut)
        except:
            continue

        if len(real_energies) % 200 == 0:
            p(f"  {len(real_energies)} runs, {total:,} events...")

    real_energies = np.concatenate(real_energies)
    real_predictions = np.concatenate(real_predictions)
    p(f"  Total: {total:,} events")

    # === 4. Compute spectra ===
    p("Computing spectra...")
    energy_bins = np.linspace(15, 17, 16)
    centers = (10**energy_bins[1:] + 10**energy_bins[:-1]) / 2
    th_min, th_max = np.radians(0), np.radians(CUTS_ZE)
    exposure = AREA * T_EFF * np.pi * (np.cos(th_min)**2 - np.cos(th_max)**2)
    C = centers**2.7 / np.diff(10**energy_bins) / exposure

    # Folded
    folded = {}
    for cls in range(5):
        folded[cls], _ = np.histogram(real_energies[real_predictions == cls], bins=energy_bins)

    # Unfolded with energy-binned CMs
    unfolded = {cls: np.zeros(len(energy_bins)-1) for cls in range(5)}
    unfolded_err = {cls: np.zeros(len(energy_bins)-1) for cls in range(5)}

    for bin_idx in range(len(energy_bins)-1):
        emin, emax = energy_bins[bin_idx], energy_bins[bin_idx+1]
        emean = (emin + emax) / 2

        # Find the CM energy bin
        cm_idx = np.searchsorted(cm_energy_bins, emean) - 1
        cm_idx = max(0, min(cm_idx, len(cm_energy_bins)-2))
        cm_emin, cm_emax = cm_energy_bins[cm_idx], cm_energy_bins[cm_idx+1]

        # Get sim data in this CM energy range
        sim_mask = (sim_energies >= cm_emin) & (sim_energies < cm_emax)
        n_sim = sim_mask.sum()
        if n_sim < 200:
            # Fallback to global CM
            sim_mask = np.ones(len(sim_energies), dtype=bool)

        # Predicted counts in this energy bin
        real_mask = (real_energies >= emin) & (real_energies < emax)
        pred_counts = np.array([np.sum(real_predictions[real_mask] == c) for c in range(5)])

        if pred_counts.sum() < 10:
            continue

        try:
            result, errors = unfolding_direct(
                sim_labels[sim_mask], sim_preds[sim_mask], pred_counts)
            for cls in range(5):
                unfolded[cls][bin_idx] = max(result[cls], 0)
                unfolded_err[cls][bin_idx] = errors[cls]
        except:
            for cls in range(5):
                unfolded[cls][bin_idx] = folded[cls][bin_idx]
                unfolded_err[cls][bin_idx] = np.sqrt(max(folded[cls][bin_idx], 1))

    # === 5. Plot ===
    p("Plotting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for cls in range(5):
        N = folded[cls]
        ax1.errorbar(centers, N * C, yerr=np.sqrt(np.maximum(N, 1)) * C,
                     fmt='.-', color=COLORS[cls], label=PARTICLES[cls],
                     markersize=6, linewidth=1, capsize=2)
    ax1.set_xlabel('Energy, [eV]')
    ax1.set_ylabel(r'$\sim$ Flux $\cdot E^{2.7}$, [eV$^{1.7}$ m$^{-2}$ sr$^{-1}$ s$^{-1}$]')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_ylim(1e17, None)
    ax1.set_title(r'Folded ($0° < \theta < 18°$)')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.2)

    for cls in range(5):
        N = unfolded[cls]; N_err = unfolded_err[cls]
        valid = N > 0
        ax2.errorbar(centers[valid], N[valid] * C[valid],
                     yerr=N_err[valid] * C[valid],
                     fmt='.-', color=COLORS[cls], label=PARTICLES[cls],
                     markersize=6, linewidth=1, capsize=2)
    ax2.set_xlabel('Energy, [eV]')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_ylim(1e17, None)
    ax2.set_title(r'Unfolded direct ($0° < \theta < 18°$)')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.2)

    plt.suptitle(f'KASCADE Real Data — CNN+Attn+MLP (SAM+DE)\n'
                 f'{total:,} events', fontsize=12)
    plt.tight_layout()
    plt.savefig('paper/fig_real_spectra.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_real_spectra.png', bbox_inches='tight')
    p(f"Saved paper/fig_real_spectra.pdf")
    p(f"Time: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
