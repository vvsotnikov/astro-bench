"""Generate publication-quality real data spectra figures.
Fig 1: Single-panel folded spectra (matching JINST Fig 17 style)
Fig 2: Per-component unfolded spectra (5 subplots, matching JCAP Fig 2 style)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.metrics import confusion_matrix
import os, gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = "data"
REAL_DIR = "data/real_kascade"
OUT_DIR = "submissions/opus-composition-mar14/matched_pipeline"
DEVICE = "cuda"
PARTICLES = ['p', 'He', 'C', 'Si', 'Fe']
COLORS = ['#F44336', '#2196F3', '#9C27B0', '#757575', '#FF9800']  # match JCAP style
MARKERS = ['o', 'D', 's', '^', 'v']

CUTS_ZE = 18; CUTS_NE = 4.8; CUTS_NMU = 3.6; CUTS_AGE_LO = 0.2; CUTS_AGE_HI = 1.48
DATA_PART = 0.185
T_EFF = 792519034.6611552 * DATA_PART
AREA = np.pi * (91.0 ** 2)

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 300})

# --- Model (same as all other scripts) ---
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
    cm = confusion_matrix(cm_trues, cm_preds, normalize='true', labels=labels).T
    cm_inv = np.linalg.inv(cm)
    result = (cm_inv @ pred_counts).ravel()
    cov = cm_inv @ np.diag(np.maximum(pred_counts, 1).astype(float)) @ cm_inv.T
    return result, np.sqrt(np.abs(np.diag(cov)))

def load_and_classify():
    """Load model, build CMs, classify real data. Returns energies, predictions, sim data for unfolding."""
    print("Loading model...", flush=True)
    model = HybridModel(n_feat=6).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUT_DIR}/model_v2_sam.pt", weights_only=True))
    model.eval()

    # Build sim CMs
    print("Building confusion matrices...", flush=True)
    RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
    SIM_CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, float('inf')), 'Nmu': (3.6, float('inf'))}
    raw_feat = np.load(f"{DATA_DIR}/qgs_spectra_features.npz")['features']
    raw_mat = np.load(f"{DATA_DIR}/qgs_spectra_matrices.npz")['matrices']
    reco = raw_feat[:, 1:].astype(np.float32); labels = raw_feat[:, 0].astype(np.int64) - 1
    mask = np.ones(len(reco), dtype=bool)
    for fn, (lo, hi) in SIM_CUTS.items():
        i = RECO_HEADERS.index(fn); mask &= (reco[:, i] > lo) & (reco[:, i] < hi)
    reco=reco[mask]; labels=labels[mask]; mat=raw_mat[mask]
    del raw_mat, raw_feat; gc.collect()

    n = len(labels); nv = int(n*0.3); nt = n-nv
    all_idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    test_idx = all_idx[nt:]

    BS = 2048
    sim_energies = reco[test_idx, 0]; sim_labels = labels[test_idx]
    sim_preds = []
    for i in range(0, len(test_idx), BS):
        idx = test_idx[i:i+BS]
        x = torch.log1p(torch.from_numpy(mat[idx][:, :, :, [1, 2]].astype(np.float32)))
        r = reco[idx]; Ne=r[:,6];Nmu=r[:,7];Age=r[:,8];Ze=r[:,4];E=r[:,0]
        reco_t = torch.from_numpy(np.column_stack([(Ne-5.31)/0.5,(Nmu-4.3)/0.42,Age-1.0,Ze/60.0,(Ne-Nmu-0.8)/0.3,(E-15.5)/1.0]).astype(np.float32))
        with torch.no_grad():
            with autocast(device_type='cuda'):
                out = model(x.to(DEVICE), reco_t.to(DEVICE))
            sim_preds.append(F.softmax(out.float(), 1).cpu().numpy().argmax(1))
    sim_preds = np.concatenate(sim_preds)
    del mat, reco; gc.collect()

    # Classify real data (no DE biases)
    print("Classifying real data...", flush=True)
    runs = sorted([f.replace('_matrices.npz', '') for f in os.listdir(REAL_DIR) if f.endswith('_matrices.npz')])
    real_energies, real_predictions = [], []
    total = 0
    for run in runs:
        try:
            mat_run = np.load(f"{REAL_DIR}/{run}_matrices.npz", allow_pickle=True)['matrices']
            feat_run = np.load(f"{REAL_DIR}/{run}_features.npz", allow_pickle=True)['features'].astype(np.float64)
            if len(mat_run) == 0: continue
            Ze=feat_run[:,3]; Ne=feat_run[:,5]; Nmu=feat_run[:,6]; Age=feat_run[:,7]
            cut = (Ze<CUTS_ZE) & (Ne>CUTS_NE) & (Nmu>CUTS_NMU) & (Age>CUTS_AGE_LO) & (Age<CUTS_AGE_HI)
            if cut.sum() == 0: continue
            mat_cut=mat_run[cut]; feat_cut=feat_run[cut]
            x = torch.log1p(torch.from_numpy(mat_cut[:,:,:,[1,2]].astype(np.float32)))
            E_r=feat_cut[:,0]; Ne_r=feat_cut[:,5]; Nmu_r=feat_cut[:,6]; Age_r=feat_cut[:,7]; Ze_r=feat_cut[:,3]
            reco_r = torch.from_numpy(np.column_stack([(Ne_r-5.31)/0.5,(Nmu_r-4.3)/0.42,Age_r-1.0,Ze_r/60.0,(Ne_r-Nmu_r-0.8)/0.3,(E_r-15.5)/1.0]).astype(np.float32))
            preds_run = []
            with torch.no_grad():
                for i in range(0, len(x), BS):
                    a=x[i:i+BS].to(DEVICE); r_=reco_r[i:i+BS].to(DEVICE)
                    with autocast(device_type='cuda'):
                        out=model(a,r_)
                    preds_run.append(F.softmax(out.float(),1).cpu().numpy().argmax(1))
            real_energies.append(E_r); real_predictions.append(np.concatenate(preds_run))
            total += len(feat_cut)
        except: continue
    real_energies = np.concatenate(real_energies)
    real_predictions = np.concatenate(real_predictions)
    print(f"  {total:,} events", flush=True)
    return real_energies, real_predictions, sim_energies, sim_labels, sim_preds


def main():
    real_E, real_pred, sim_E, sim_labels, sim_preds = load_and_classify()

    energy_bins = np.linspace(15, 17, 16)
    centers = (10**energy_bins[1:] + 10**energy_bins[:-1]) / 2
    th_min, th_max = np.radians(0), np.radians(CUTS_ZE)
    exposure = AREA * T_EFF * np.pi * (np.cos(th_min)**2 - np.cos(th_max)**2)
    C = centers**2.7 / np.diff(10**energy_bins) / exposure

    # === Figure 1: Folded spectra only (single panel, JINST style) ===
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for cls in range(5):
        N, _ = np.histogram(real_E[real_pred == cls], bins=energy_bins)
        ax.errorbar(centers, N * C, yerr=np.sqrt(np.maximum(N, 1)) * C,
                    fmt='.-', color=COLORS[cls], label=PARTICLES[cls],
                    markersize=6, linewidth=1, capsize=2)
    ax.set_xlabel('Energy, [eV]')
    ax.set_ylabel(r'$\sim$ Flux $\cdot E^{2.7}$, [eV$^{1.7}$ m$^{-2}$ sr$^{-1}$ s$^{-1}$]')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(1e18, None)
    ax.set_title(r'Folded spectra ($0° < \theta < 18°$)')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('paper/fig_real_spectra.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_real_spectra.png', bbox_inches='tight')
    print("fig_real_spectra saved", flush=True)
    plt.close()

    # === Figure 2: Per-component unfolded spectra (JCAP Fig 2 style) ===
    cm_energy_bins = [0, 15.15, 15.5, 16.0, 16.5, 20]

    # Compute unfolded spectra
    unfolded = {cls: np.zeros(len(energy_bins)-1) for cls in range(5)}
    unfolded_err = {cls: np.zeros(len(energy_bins)-1) for cls in range(5)}
    for bin_idx in range(len(energy_bins)-1):
        emin, emax = energy_bins[bin_idx], energy_bins[bin_idx+1]
        emean = (emin + emax) / 2
        cm_idx = max(0, min(np.searchsorted(cm_energy_bins, emean) - 1, len(cm_energy_bins)-2))
        cm_emin, cm_emax = cm_energy_bins[cm_idx], cm_energy_bins[cm_idx+1]
        sim_mask = (sim_E >= cm_emin) & (sim_E < cm_emax)
        if sim_mask.sum() < 200: sim_mask = np.ones(len(sim_E), dtype=bool)
        real_mask = (real_E >= emin) & (real_E < emax)
        pred_counts = np.array([np.sum(real_pred[real_mask] == c) for c in range(5)])
        if pred_counts.sum() < 10: continue
        try:
            result, errors = unfolding_direct(sim_labels[sim_mask], sim_preds[sim_mask], pred_counts)
            for cls in range(5):
                unfolded[cls][bin_idx] = max(result[cls], 0)
                unfolded_err[cls][bin_idx] = errors[cls]
        except:
            for cls in range(5):
                N_cls, _ = np.histogram(real_E[(real_pred == cls) & real_mask], bins=[emin, emax])
                unfolded[cls][bin_idx] = N_cls[0]
                unfolded_err[cls][bin_idx] = np.sqrt(max(N_cls[0], 1))

    # 5-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(10, 7), dpi=300)
    axes = axes.flatten()

    for cls in range(5):
        ax = axes[cls]
        N = unfolded[cls]; N_err = unfolded_err[cls]
        valid = N > 0
        ax.errorbar(centers[valid], N[valid] * C[valid], yerr=N_err[valid] * C[valid],
                    fmt='o', color=COLORS[cls], markersize=5, capsize=2, linewidth=1,
                    label=f'{PARTICLES[cls]} (This work)')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_ylim(1e17, None)
        ax.set_title(PARTICLES[cls], fontsize=14, fontweight='bold')
        ax.set_xlabel('Energy, [eV]')
        if cls % 3 == 0:
            ax.set_ylabel(r'$\sim$ Flux $\cdot E^{2.7}$')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    # Hide 6th subplot
    axes[5].set_visible(False)

    plt.suptitle(r'Unfolded mass component spectra ($0° < \theta < 18°$)', fontsize=13)
    plt.tight_layout()
    plt.savefig('paper/fig_unfolded_components.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_unfolded_components.png', bbox_inches='tight')
    print("fig_unfolded_components saved", flush=True)


if __name__ == '__main__':
    main()
