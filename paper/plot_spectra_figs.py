"""Reproduce Figures 10 and 17 from Kuznetsov et al. (JINST 2024) with Opus v34 ensemble.

Figure 10: 5x5 confusion matrix on QGSJet-II.04 sim test set.
Figure 17: Folded mass composition spectra on real KASCADE data (Ze<18, full archive)
           with flux units dJ/dE * E^2.7 [m^-2 sr^-1 s^-1 eV^1.7].

Requires: paper/real_spectra_opus.npz (produced by run_real_spectra_opus.py).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
NPZ = HERE / "real_spectra_opus.npz"

PARTICLES = ["p", "He", "C", "Si", "Fe"]

# Per-class colors matching JINST Figure 17 (approximate)
FIG17_COLORS = {
    "p": "#1f77b4",
    "He": "#ff7f0e",
    "C": "#2ca02c",
    "Si": "#d62728",
    "Fe": "#9467bd",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "cm",
    "figure.dpi": 300,
})

# Confusion matrix computed by run_real_spectra_opus.py on the QGSJet-II.04 sim test set
# (115,066 events; same cuts as training). Row-normalized: rows = true, cols = predicted.
# Overall accuracy: 0.5145
CM_SIM = np.array([
    [0.63914929, 0.25750909, 0.08491168, 0.01517760, 0.00325234],
    [0.28714236, 0.40947566, 0.22666513, 0.06308170, 0.01363514],
    [0.04447037, 0.25104993, 0.40373308, 0.23131125, 0.06943537],
    [0.00191571, 0.04533844, 0.25186249, 0.41985951, 0.28102384],
    [0.00011513, 0.00356917, 0.04755052, 0.26267918, 0.68608600],
])


def plot_confusion_matrix():
    """Figure 10 style: single 5x5 matrix with Oranges colormap and value annotations."""
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.2), tight_layout=True)

    im = ax.imshow(CM_SIM, cmap="Oranges", vmin=0, vmax=1, aspect="equal")

    for i in range(5):
        for j in range(5):
            v = CM_SIM[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=11)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(PARTICLES)
    ax.set_yticklabels(PARTICLES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.tick_params(axis="both", which="both", length=0)

    for out in ["fig_confusion_matrix.pdf", "fig_confusion_matrix.png"]:
        fig.savefig(HERE / out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig_confusion_matrix.pdf/png  (accuracy = {np.diag(CM_SIM).mean():.4f})")


def plot_folded_spectra():
    """Figure 17 style: 5 species, log-log, y = dJ/dE * E^2.7 [m^-2 sr^-1 s^-1 eV^1.7]."""
    d = np.load(NPZ)
    preds = d["predictions"]
    logE = d["energies"]

    # Binning matches the published figure: log10(E/eV) in [15, 17], 15 bins.
    emin, emax, nbins = 15.0, 17.0, 15
    edges = np.linspace(emin, emax, nbins + 1)

    # Count events per class per bin
    counts = np.zeros((5, nbins), dtype=np.int64)
    for c in range(5):
        m = preds == c
        counts[c], _ = np.histogram(logE[m], bins=edges)

    # Exposure: the 1308 KCDC-released runs span May 1998 - Nov 2012,
    # essentially the full operational period of KASCADE (1996-2013, nominal
    # T_eff = 7.925e8 s per the legacy code). The published CNN analysis used
    # only the 2000-2004 window (data_part = 0.185); here we use data_part = 1.0
    # consistent with our full-archive coverage. Same area/solid-angle formula
    # as legacy code (mass composition.md line 4438).
    data_part = 1.0
    T_eff = 7.925190346611552e8 * data_part  # seconds (~25 years)
    area = np.pi * 91.0**2  # m^2  (fiducial circular area, r=91 m)
    th_min, th_max = np.radians(0.0), np.radians(18.0)
    solid_angle = np.pi * (np.cos(th_min) ** 2 - np.cos(th_max) ** 2)
    exposure = area * T_eff * solid_angle

    # Flux conversion: dJ/dE * E^2.7 = N / (Delta_E * exposure) * E^2.7
    energy_pow = 2.7
    centers_eV = (10.0 ** edges[1:] + 10.0 ** edges[:-1]) / 2.0
    deltaE_eV = np.diff(10.0 ** edges)
    C = centers_eV**energy_pow / deltaE_eV / exposure

    flux = counts * C[None, :]
    flux_err = np.sqrt(counts) * C[None, :]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), tight_layout=True)

    for c, name in enumerate(PARTICLES):
        ax.errorbar(
            centers_eV, flux[c], yerr=flux_err[c],
            fmt=".-", color=FIG17_COLORS[name], label=name,
            markersize=6, linewidth=1.2, capsize=0,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy, [eV]")
    ax.set_ylabel(r"$dJ/dE \cdot E^{2.7}$, [m$^{-2}$ sr$^{-1}$ s$^{-1}$ eV$^{1.7}$]")
    ax.set_xlim(10**emin, 10**emax)
    ax.legend(ncol=2, frameon=True, loc="upper right")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)

    for out in ["fig_folded_spectra.pdf", "fig_folded_spectra.png"]:
        fig.savefig(HERE / out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig_folded_spectra.pdf/png  (exposure = {exposure:.3e} m^2 s sr)")

    # Log flux values for comparison with Figure 17
    print("\nFlux values at bin centers (dJ/dE * E^2.7 in m^-2 sr^-1 s^-1 eV^1.7):")
    print(f"  {'log10(E/eV)':>11s}  " + "  ".join(f"{p:>9s}" for p in PARTICLES))
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        vals = "  ".join(f"{flux[c, i]:9.3e}" for c in range(5))
        print(f"  {(lo + hi) / 2:11.3f}  " + vals)


if __name__ == "__main__":
    plot_confusion_matrix()
    plot_folded_spectra()
