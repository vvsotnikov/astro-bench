"""Apply the best Opus composition ensemble (v34) to real KASCADE data.

Uses the 5-model ensemble from train_v34.py:
  v8 (NetV8, DS8), v15 (NetV8, DS8), v26 (NetV26, DS11),
  v33 (NetV26, DS11), v30 (NetConvNeXt, DS11)
with 16x TTA, averaged softmax probabilities.

Produces folded per-class spectra and unfolded energy spectra.
"""

import gc
import sys
import time
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda:0"
BATCH_SIZE = 1024

OPUS_DIR = Path(__file__).resolve().parent.parent.parent / "astro-bench-experiments" / "composition-opus-2apr"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REAL_DIR = DATA_DIR / "real_kascade"
OUT_DIR = Path(__file__).resolve().parent

PARTICLES = ["p", "He", "C", "Si", "Fe"]
COLORS = ["#F44336", "#2196F3", "#9C27B0", "#757575", "#FF9800"]
MARKERS = ["o", "D", "s", "^", "v"]

# Quality cuts matching published analysis (Kuznetsov et al. JINST 2024)
CUTS_ZE = 18  # degrees (tighter than benchmark's 30, matching published spectra)
CUTS_NE = 4.8
CUTS_NMU = 3.6
CUTS_AGE_LO = 0.2
CUTS_AGE_HI = 1.48

plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 300})

# ===== Model definitions (from Opus train_v31.py) =====

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)

class NetV8(nn.Module):
    def __init__(self, n_classes=5, n_reco=8):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer1 = nn.Sequential(ResBlock(32), ResBlock(32), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64), ResBlock(64), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128), nn.AdaptiveAvgPool2d(1))
        self.feat_net = nn.Sequential(nn.Linear(n_reco, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))
        self.cnn_head = nn.Linear(128, n_classes)
    def forward(self, x, xr, return_both=False):
        x = x.permute(0, 3, 1, 2).float()
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        c = x.flatten(1); f = self.feat_net(xr)
        o = self.classifier(torch.cat([c, f], 1))
        if return_both: return o, self.cnn_head(c)
        return o

class NetV26(nn.Module):
    def __init__(self, n_classes=5, n_reco=11):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer1 = nn.Sequential(ResBlock(32), ResBlock(32), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64), ResBlock(64), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128), nn.AdaptiveAvgPool2d(1))
        self.feat_net = nn.Sequential(nn.Linear(n_reco, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n_classes))
        self.cnn_head = nn.Linear(128, n_classes)
    def forward(self, x, xr, return_both=False):
        x = x.permute(0, 3, 1, 2).float()
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        c = x.flatten(1); f = self.feat_net(xr)
        o = self.classifier(torch.cat([c, f], 1))
        if return_both: return o, self.cnn_head(c)
        return o

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * mult)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * mult, dim)
    def forward(self, x):
        r = x; x = self.dwconv(x); x = x.permute(0, 2, 3, 1)
        x = self.norm(x); x = self.pwconv1(x); x = self.act(x); x = self.pwconv2(x)
        return x.permute(0, 3, 1, 2) + r

class NetConvNeXt(nn.Module):
    def __init__(self, n_classes=5, n_reco=11):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU())
        self.stage1 = nn.Sequential(ConvNeXtBlock(32), ConvNeXtBlock(32))
        self.down1 = nn.Sequential(nn.Conv2d(32, 64, 2, stride=2), nn.BatchNorm2d(64))
        self.stage2 = nn.Sequential(ConvNeXtBlock(64), ConvNeXtBlock(64))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 2, stride=2), nn.BatchNorm2d(128))
        self.stage3 = nn.Sequential(ConvNeXtBlock(128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_net = nn.Sequential(nn.Linear(n_reco, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n_classes))
        self.cnn_head = nn.Linear(128, n_classes)
    def forward(self, x, xr, return_both=False):
        x = x.permute(0, 3, 1, 2).float()
        x = self.stem(x); x = self.stage1(x); x = self.down1(x); x = self.stage2(x)
        x = self.down2(x); x = self.stage3(x); c = self.pool(x).flatten(1)
        f = self.feat_net(xr)
        o = self.classifier(torch.cat([c, f], 1))
        if return_both: return o, self.cnn_head(c)
        return o

# ===== Dataset classes (from Opus train_v31.py) =====

class DS8(Dataset):
    def __init__(self, matrices, features, labels=None, augment=False):
        self.matrices = matrices; self.features = features
        self.labels = labels if labels is not None else np.zeros(len(matrices), dtype=int)
        self.augment = augment
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        mat = torch.from_numpy(self.matrices[idx].copy().astype(np.float32))
        feat = self.features[idx].astype(np.float32)
        label = int(self.labels[idx])
        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            if k > 0: mat = torch.rot90(mat, k, [0, 1])
            if torch.rand(1) < 0.5: mat = torch.flip(mat, dims=[0])
            if torch.rand(1) < 0.5: mat = torch.flip(mat, dims=[1])
        reco = torch.tensor([(feat[0]-16)/1, feat[1]/30, feat[2]/360,
            (feat[3]-5.31)/0.5, (feat[4]-4.3)/0.42, feat[5]-1.0,
            feat[3]-feat[4], feat[3]/(feat[4]+1e-6)], dtype=torch.float32)
        return mat, reco, label

class DS11(Dataset):
    def __init__(self, matrices, features, labels=None, augment=False):
        self.matrices = matrices; self.features = features
        self.labels = labels if labels is not None else np.zeros(len(matrices), dtype=int)
        self.augment = augment
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        mat = torch.from_numpy(self.matrices[idx].copy().astype(np.float32))
        feat = self.features[idx].astype(np.float32)
        label = int(self.labels[idx])
        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            if k > 0: mat = torch.rot90(mat, k, [0, 1])
            if torch.rand(1) < 0.5: mat = torch.flip(mat, dims=[0])
            if torch.rand(1) < 0.5: mat = torch.flip(mat, dims=[1])
        E = (feat[0]-16)/1; Ne = (feat[3]-5.31)/0.5; Nmu = (feat[4]-4.3)/0.42
        Ne_Nmu = feat[3]-feat[4]
        reco = torch.tensor([E, feat[1]/30, feat[2]/360, Ne, Nmu, feat[5]-1.0,
            Ne_Nmu, feat[3]/(feat[4]+1e-6), Ne*Ne, Nmu*Nmu, Ne_Nmu*Ne_Nmu], dtype=torch.float32)
        return mat, reco, label


def predict_tta(model, ds_class, X, f, n_tta=1):
    """Predict with TTA, return averaged softmax probabilities."""
    model.eval()
    dummy_labels = np.zeros(len(f), dtype=int)
    all_probs = []
    for t in range(n_tta):
        ds = ds_class(X, f, dummy_labels, augment=(t > 0))
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        pp = []
        with torch.no_grad():
            for mat, reco, _ in ld:
                mat, reco = mat.to(DEVICE), reco.to(DEVICE)
                pp.append(F.softmax(model(mat, reco), dim=1).cpu().numpy())
        all_probs.append(np.concatenate(pp))
    return np.mean(all_probs, axis=0)


def load_models():
    """Load the 5 models from v34 ensemble."""
    cfgs = [
        ("v8-adamw-val", NetV8, DS8, "model_v8.pt", 8),
        ("v15-adamw-full", NetV8, DS8, "model_v15_ep100.pt", 8),
        ("v26-adamw-deepmlp", NetV26, DS11, "model_v26_s42.pt", 11),
        ("v33-sgd", NetV26, DS11, "model_v33.pt", 11),
        ("v30-convnext", NetConvNeXt, DS11, "model_v30.pt", 11),
    ]
    models = []
    for name, mcls, dcls, path, nreco in cfgs:
        model = mcls(n_reco=nreco).to(DEVICE)
        model.load_state_dict(torch.load(OPUS_DIR / path, map_location=DEVICE, weights_only=True))
        model.eval()
        models.append((name, model, dcls))
        print(f"  Loaded {name} ({path})", flush=True)
    return models


def classify_real_data(models, n_tta=1):
    """Stream over real KASCADE data, classify all events passing quality cuts."""
    run_files = sorted(glob(str(REAL_DIR / "*_matrices.npz")))
    print(f"\nClassifying {len(run_files)} runs (Ze<{CUTS_ZE}, Ne>{CUTS_NE}, "
          f"Nmu>{CUTS_NMU}, {CUTS_AGE_LO}<Age<{CUTS_AGE_HI})...", flush=True)

    all_predictions = []
    all_energies = []
    n_total = 0
    t0 = time.time()

    for ri, mpath in enumerate(run_files):
        run_name = Path(mpath).name.replace("_matrices.npz", "")
        fpath = REAL_DIR / f"{run_name}_features.npz"

        feat = np.load(fpath)["features"]
        if feat.ndim != 2 or len(feat) == 0:
            continue

        # Quality cuts (tighter Ze < 18 for spectra, matching published)
        ze = feat[:, 3]; ne = feat[:, 5]; nmu = feat[:, 6]; age = feat[:, 7]
        mask = (ze < CUTS_ZE) & (ne > CUTS_NE) & (nmu > CUTS_NMU) & \
               (age > CUTS_AGE_LO) & (age < CUTS_AGE_HI)
        if mask.sum() == 0:
            continue

        mat = np.load(mpath)["matrices"][mask][:, :, :, 1:3]  # channels 1,2
        feat_cut = feat[mask]
        # Composition features: [E, Ze, Az, Ne, Nmu, Age] = indices [0,3,4,5,6,7]
        feat_comp = feat_cut[:, [0, 3, 4, 5, 6, 7]].astype(np.float32)
        energies = feat_cut[:, 0].astype(np.float32)

        # Ensemble prediction: average softmax across models
        ensemble_probs = None
        for name, model, ds_class in models:
            probs = predict_tta(model, ds_class, mat, feat_comp, n_tta=n_tta)
            if ensemble_probs is None:
                ensemble_probs = probs
            else:
                ensemble_probs += probs
        ensemble_probs /= len(models)

        predictions = ensemble_probs.argmax(axis=1)
        all_predictions.append(predictions)
        all_energies.append(energies)
        n_total += len(predictions)

        if (ri + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {ri+1}/{len(run_files)} runs, {n_total:,} events ({elapsed:.0f}s)",
                  flush=True)

    all_predictions = np.concatenate(all_predictions)
    all_energies = np.concatenate(all_energies)
    print(f"\nDone: {n_total:,} events classified in {(time.time()-t0)/60:.1f} min")
    return all_predictions, all_energies


def compute_confusion_matrix(models, n_tta=1):
    """Compute confusion matrix on composition test set for unfolding."""
    print("Computing confusion matrix on sim test set...", flush=True)
    X_test = np.load(DATA_DIR / "composition_test" / "matrices.npy", mmap_mode="r")
    f_test = np.load(DATA_DIR / "composition_test" / "features.npy", mmap_mode="r")
    y_test = np.load(DATA_DIR / "composition_test" / "labels_composition.npy", mmap_mode="r")

    X_arr = np.array(X_test)
    f_arr = np.array(f_test, dtype=np.float32)
    y_arr = np.array(y_test)

    ensemble_probs = None
    for name, model, ds_class in models:
        probs = predict_tta(model, ds_class, X_arr, f_arr, n_tta=n_tta)
        if ensemble_probs is None:
            ensemble_probs = probs
        else:
            ensemble_probs += probs
    ensemble_probs /= len(models)

    predictions = ensemble_probs.argmax(axis=1)
    cm = confusion_matrix(y_arr, predictions, labels=range(5))
    # Normalize rows
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print(f"  Accuracy: {(predictions == y_arr).mean():.4f}")
    print(f"  Confusion matrix (normalized):")
    for i, name in enumerate(PARTICLES):
        print(f"    {name}: {cm_norm[i]}")
    return cm_norm, np.array(f_test)[:, 0]  # return energies too


def plot_folded_spectra(predictions, energies):
    """Plot folded per-class spectra."""
    e_bins = np.arange(14.5, 18.01, 0.1)
    e_centers = (e_bins[:-1] + e_bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in range(5):
        counts, _ = np.histogram(energies[predictions == cls], bins=e_bins)
        ax.plot(e_centers, counts, f"-{MARKERS[cls]}", color=COLORS[cls],
                label=PARTICLES[cls], markersize=3, linewidth=1)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(E/\mathrm{eV})$")
    ax.set_ylabel("Events per bin")
    ax.set_title("Folded mass composition spectra (real KASCADE data, Opus v34)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_real_spectra.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig_real_spectra.png", bbox_inches="tight")
    print("Saved fig_real_spectra.{pdf,png}")


def plot_unfolded_spectra(predictions, energies, cm_norm):
    """Plot unfolded per-component spectra using confusion matrix inversion."""
    e_bins = np.arange(14.5, 18.01, 0.2)
    e_centers = (e_bins[:-1] + e_bins[1:]) / 2
    n_classes = 5

    fig, axes = plt.subplots(1, n_classes, figsize=(18, 4), sharex=True, sharey=True)
    cm_inv = np.linalg.inv(cm_norm)

    for ebin_idx in range(len(e_bins) - 1):
        mask = (energies >= e_bins[ebin_idx]) & (energies < e_bins[ebin_idx + 1])
        if mask.sum() == 0:
            continue
        pred_counts = np.bincount(predictions[mask], minlength=n_classes)[:n_classes].astype(float)
        true_counts = cm_inv @ pred_counts
        true_err = np.sqrt(np.abs(cm_inv) @ pred_counts)

        for cls in range(n_classes):
            if true_counts[cls] > 0:
                axes[cls].errorbar(
                    e_centers[ebin_idx], true_counts[cls],
                    yerr=true_err[cls],
                    fmt=MARKERS[cls], color=COLORS[cls], markersize=4, capsize=2,
                )

    for cls in range(n_classes):
        axes[cls].set_yscale("log")
        axes[cls].set_title(PARTICLES[cls])
        axes[cls].set_xlabel(r"$\log_{10}(E/\mathrm{eV})$")
        axes[cls].grid(alpha=0.3)
        axes[cls].set_ylim(1, None)
    axes[0].set_ylabel("Unfolded events per bin")

    fig.suptitle("Unfolded mass composition spectra (real KASCADE data, Opus v34)", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_unfolded_components.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig_unfolded_components.png", bbox_inches="tight")
    print("Saved fig_unfolded_components.{pdf,png}")


def main():
    t0 = time.time()
    print("Loading Opus v34 ensemble models...", flush=True)
    models = load_models()

    cm_norm, _ = compute_confusion_matrix(models, n_tta=1)
    predictions, energies = classify_real_data(models, n_tta=1)

    np.savez(OUT_DIR / "real_spectra_opus.npz",
             predictions=predictions, energies=energies)
    print(f"Saved real_spectra_opus.npz ({len(predictions):,} events)")

    plot_folded_spectra(predictions, energies)
    plot_unfolded_spectra(predictions, energies, cm_norm)

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
