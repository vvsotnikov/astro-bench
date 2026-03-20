"""Run unfolding on real KASCADE data: compare Raw vs P99AE-DE biases."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from sklearn.metrics import confusion_matrix
import os, gc, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = "data"
REAL_DIR = "data/real_kascade"
OUT_DIR = "submissions/opus-composition-mar14/matched_pipeline"
DEVICE = "cuda"
PARTICLES = ['H', 'He', 'C', 'Si', 'Fe']
COLORS = ['#F44336', '#2196F3', '#9C27B0', '#757575', '#FF9800']

CUTS_ZE = 18; CUTS_NE = 4.8; CUTS_NMU = 3.6; CUTS_AGE_LO = 0.2; CUTS_AGE_HI = 1.48
DATA_PART = 0.185
T_EFF = 792519034.6611552 * DATA_PART
AREA = np.pi * (91.0 ** 2)

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 300})
def p(msg): print(msg, flush=True)

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

def classify_with_biases(model, biases_dict):
    """Classify sim and real data, returning results for each bias configuration."""
    BS = 2048
    RECO_HEADERS = ['E', 'Xc', 'Yc', 'Core_distance', 'Ze', 'Az', 'Ne', 'Nmu', 'Age']
    SIM_CUTS = {'Ze': (0, 30), 'Age': (0.2, 1.48), 'Ne': (4.8, np.inf), 'Nmu': (3.6, np.inf)}

    # Load sim data
    p("Loading simulation data...")
    raw_feat = np.load(f"{DATA_DIR}/qgs_spectra_features.npz")['features']
    raw_mat = np.load(f"{DATA_DIR}/qgs_spectra_matrices.npz")['matrices']
    reco = raw_feat[:, 1:].astype(np.float32)
    labels = raw_feat[:, 0].astype(np.int64) - 1
    mask = np.ones(len(reco), dtype=bool)
    for fn, (lo, hi) in SIM_CUTS.items():
        i = RECO_HEADERS.index(fn); mask &= (reco[:, i] > lo) & (reco[:, i] < hi)
    reco = reco[mask]; labels = labels[mask]; mat = raw_mat[mask]
    del raw_mat, raw_feat; gc.collect()

    n = len(labels); nv = int(n*0.3); nt = n-nv
    all_idx = torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
    test_idx = all_idx[nt:]

    # Get sim probabilities (once)
    p("Running sim inference...")
    sim_energies = reco[test_idx, 0]
    sim_labels = labels[test_idx]
    sim_log_probs = []
    for i in range(0, len(test_idx), BS):
        idx = test_idx[i:i+BS]
        x = torch.log1p(torch.from_numpy(mat[idx][:, :, :, [1, 2]].astype(np.float32)))
        r = reco[idx]; Ne=r[:,6];Nmu=r[:,7];Age=r[:,8];Ze=r[:,4];E=r[:,0]
        reco_t = torch.from_numpy(np.column_stack([
            (Ne-5.31)/0.5,(Nmu-4.3)/0.42,Age-1.0,Ze/60.0,
            (Ne-Nmu-0.8)/0.3,(E-15.5)/1.0]).astype(np.float32))
        with torch.no_grad():
            with autocast(device_type='cuda'):
                out = model(x.to(DEVICE), reco_t.to(DEVICE))
            sim_log_probs.append(np.log(F.softmax(out.float(), 1).cpu().numpy() + 1e-10))
    sim_log_probs = np.concatenate(sim_log_probs)
    del mat, reco; gc.collect()

    # Get real data probabilities (once)
    p("Classifying real data...")
    runs = sorted([f.replace('_matrices.npz', '') for f in os.listdir(REAL_DIR) if f.endswith('_matrices.npz')])
    real_energies_list, real_log_probs_list = [], []
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
            reco_r = torch.from_numpy(np.column_stack([
                (Ne_r-5.31)/0.5,(Nmu_r-4.3)/0.42,Age_r-1.0,Ze_r/60.0,
                (Ne_r-Nmu_r-0.8)/0.3,(E_r-15.5)/1.0]).astype(np.float32))
            lp_run = []
            with torch.no_grad():
                for i in range(0, len(x), BS):
                    a=x[i:i+BS].to(DEVICE); r_=reco_r[i:i+BS].to(DEVICE)
                    with autocast(device_type='cuda'):
                        out=model(a,r_)
                    lp_run.append(np.log(F.softmax(out.float(),1).cpu().numpy() + 1e-10))
            real_energies_list.append(E_r)
            real_log_probs_list.append(np.concatenate(lp_run))
            total += len(feat_cut)
        except: continue
    real_energies = np.concatenate(real_energies_list)
    real_log_probs = np.concatenate(real_log_probs_list)
    p(f"  {total:,} real events")

    # Now apply each bias configuration
    results = {}
    for name, biases in biases_dict.items():
        p(f"  Applying biases: {name}")
        sim_preds = (sim_log_probs + biases).argmax(1)
        real_preds = (real_log_probs + biases).argmax(1)
        results[name] = {
            'sim_energies': sim_energies, 'sim_labels': sim_labels, 'sim_preds': sim_preds,
            'real_energies': real_energies, 'real_preds': real_preds,
        }
    return results


def compute_spectra(res, energy_bins):
    centers = (10**energy_bins[1:] + 10**energy_bins[:-1]) / 2
    th_min, th_max = np.radians(0), np.radians(CUTS_ZE)
    exposure = AREA * T_EFF * np.pi * (np.cos(th_min)**2 - np.cos(th_max)**2)
    C = centers**2.7 / np.diff(10**energy_bins) / exposure
    cm_energy_bins = [0, 15.15, 15.5, 16.0, 16.5, 20]

    folded = {}
    unfolded = {cls: np.zeros(len(energy_bins)-1) for cls in range(5)}
    unfolded_err = {cls: np.zeros(len(energy_bins)-1) for cls in range(5)}

    for cls in range(5):
        folded[cls], _ = np.histogram(res['real_energies'][res['real_preds'] == cls], bins=energy_bins)

    for bin_idx in range(len(energy_bins)-1):
        emin, emax = energy_bins[bin_idx], energy_bins[bin_idx+1]
        emean = (emin + emax) / 2
        cm_idx = max(0, min(np.searchsorted(cm_energy_bins, emean) - 1, len(cm_energy_bins)-2))
        cm_emin, cm_emax = cm_energy_bins[cm_idx], cm_energy_bins[cm_idx+1]
        sim_mask = (res['sim_energies'] >= cm_emin) & (res['sim_energies'] < cm_emax)
        if sim_mask.sum() < 200:
            sim_mask = np.ones(len(res['sim_energies']), dtype=bool)
        real_mask = (res['real_energies'] >= emin) & (res['real_energies'] < emax)
        pred_counts = np.array([np.sum(res['real_preds'][real_mask] == c) for c in range(5)])
        if pred_counts.sum() < 10: continue
        try:
            result, errors = unfolding_direct(res['sim_labels'][sim_mask], res['sim_preds'][sim_mask], pred_counts)
            for cls in range(5):
                unfolded[cls][bin_idx] = max(result[cls], 0)
                unfolded_err[cls][bin_idx] = errors[cls]
        except:
            for cls in range(5):
                unfolded[cls][bin_idx] = folded[cls][bin_idx]
                unfolded_err[cls][bin_idx] = np.sqrt(max(folded[cls][bin_idx], 1))

    return folded, unfolded, unfolded_err, centers, C


def main():
    t0 = time.time()
    model = HybridModel(n_feat=6).to(DEVICE)
    model.load_state_dict(torch.load(f"{OUT_DIR}/model_v2_sam.pt", weights_only=True))
    model.eval()

    biases_dict = {
        'Raw (no DE)': np.zeros(5),
        'DE-P99AE': np.array([0.4963, 0.4553, 0.2797, 0.0406, -0.2102]),
    }

    results = classify_with_biases(model, biases_dict)

    energy_bins = np.linspace(15, 17, 16)

    # Load published CNN for overlay
    pub_files = ['internal/kg-nn/p_data.csv', 'internal/kg-nn/he_data.csv',
                 'internal/kg-nn/c_data.csv', 'internal/kg-nn/si_data.csv', 'internal/kg-nn/fe_data.csv']
    pub_E = {}
    for cls, f in enumerate(pub_files):
        data = np.genfromtxt(f, delimiter=',', skip_header=1)
        pub_E[cls] = data[data[:, 1] < CUTS_ZE, 0]

    # === Plot: 2-column comparison (Raw vs P99AE) ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    MARKERS = ['o', 'D', 's', '^', 'v']

    for col, (name, res) in enumerate(results.items()):
        folded, unfolded, unfolded_err, centers, C = compute_spectra(res, energy_bins)

        # Top row: folded
        ax = axes[0, col]
        for cls in range(5):
            N = folded[cls]
            ax.errorbar(centers * 1.01, N * C, yerr=np.sqrt(np.maximum(N, 1)) * C,
                        fmt='o', color=COLORS[cls], label=f'{PARTICLES[cls]} (This work)',
                        markersize=5, linewidth=1, capsize=2)
            N_pub, _ = np.histogram(pub_E[cls], bins=energy_bins)
            ax.errorbar(centers * 0.99, N_pub * C, yerr=np.sqrt(np.maximum(N_pub, 1)) * C,
                        fmt='s', markerfacecolor='none', markeredgecolor=COLORS[cls],
                        label=f'{PARTICLES[cls]} (Published CNN)',
                        markersize=5, linewidth=0.8, capsize=2, alpha=0.6)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e18, None)
        ax.set_xlabel('Energy [eV]')
        if col == 0: ax.set_ylabel(r'$\sim$ Flux $\cdot E^{2.7}$ [eV$^{1.7}$ m$^{-2}$ sr$^{-1}$ s$^{-1}$]')
        ax.set_title(f'Folded — {name}')
        ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.2)

        # Bottom row: unfolded
        ax = axes[1, col]
        for cls in range(5):
            N = unfolded[cls]; N_err = unfolded_err[cls]
            valid = N > 0
            ax.errorbar(centers[valid] * 1.01, N[valid] * C[valid], yerr=N_err[valid] * C[valid],
                        fmt='o', color=COLORS[cls], label=f'{PARTICLES[cls]} (This work)',
                        markersize=5, linewidth=1, capsize=2)
            N_pub, _ = np.histogram(pub_E[cls], bins=energy_bins)
            ax.errorbar(centers * 0.99, N_pub * C, yerr=np.sqrt(np.maximum(N_pub, 1)) * C,
                        fmt='s', markerfacecolor='none', markeredgecolor=COLORS[cls],
                        label=f'{PARTICLES[cls]} (Published CNN)',
                        markersize=5, linewidth=0.8, capsize=2, alpha=0.6)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e18, None)
        ax.set_xlabel('Energy [eV]')
        if col == 0: ax.set_ylabel(r'$\sim$ Flux $\cdot E^{2.7}$ [eV$^{1.7}$ m$^{-2}$ sr$^{-1}$ s$^{-1}$]')
        ax.set_title(f'Unfolded — {name}')
        ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.2)

    plt.suptitle(r'Real KASCADE spectra ($0° < \theta < 18°$): Raw vs P99AE-optimized', fontsize=13)
    plt.tight_layout()
    plt.savefig('paper/fig_unfolding_compare.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_unfolding_compare.png', bbox_inches='tight')
    p(f"Saved paper/fig_unfolding_compare.{{pdf,png}}")
    p(f"Time: {(time.time()-t0)/60:.1f} min")

if __name__ == '__main__':
    main()
