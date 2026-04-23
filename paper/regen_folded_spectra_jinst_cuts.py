"""Regenerate the folded spectrum applying the JINST 2024 cut set:
   x^2 + y^2 < 91 m  (fiducial core-position cut)
   log10(E/eV) > 15.15

Our existing paper/real_spectra_opus.npz has predictions for 1,572,349 events
that passed the looser cut set (Ze<18, Ne>4.8, Nmu>3.6, 0.2<Age<1.48).

Here we:
  1. Re-iterate over the 1308 runs in the same sorted order as the original scoring,
     apply the ORIGINAL mask, and extract per-event (Xc, Yc, E) aligned to the
     existing predictions array.
  2. Apply the additional JINST cuts (r_core<91 m, log10 E>15.15).
  3. Re-normalize flux with the correct exposure for the resulting subset.
  4. Plot and save.

Real features column order (verified): [E, Xc, Yc, Ze, Az, Ne, Nmu, Age].
"""
from pathlib import Path
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "real_kascade"
PREDS_NPZ = HERE / "real_spectra_opus.npz"
OUT_PDF = HERE / "fig_folded_spectra.pdf"
OUT_PNG = HERE / "fig_folded_spectra.png"

# Original cuts used in run_real_spectra_opus.py
CUTS_ZE = 18
CUTS_NE = 4.8
CUTS_NMU = 3.6
CUTS_AGE_LO = 0.2
CUTS_AGE_HI = 1.48

# JINST/JCAP additional cuts
R_CORE_MAX = 91.0           # metres — matches area = pi * 91^2
E_MIN_LOG10 = 15.15         # log10 E/eV

PARTICLES = ["p", "He", "C", "Si", "Fe"]
FIG17_COLORS = {
    "p": "#1f77b4",
    "He": "#ff7f0e",
    "C": "#2ca02c",
    "Si": "#d62728",
    "Fe": "#9467bd",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "cm", "figure.dpi": 300,
})


def extract_aligned_features():
    """Re-iterate 1308 runs with the ORIGINAL mask and return per-event
    (E, Xc, Yc) aligned 1:1 with the saved predictions array.
    """
    preds = np.load(PREDS_NPZ)
    predictions = preds["predictions"]
    energies_saved = preds["energies"]
    n_saved = len(predictions)
    print(f"Loaded {n_saved:,} predictions from {PREDS_NPZ.name}")

    run_files = sorted(glob(str(DATA_DIR / "*_matrices.npz")))
    print(f"Re-iterating {len(run_files)} runs...")

    E_all, Xc_all, Yc_all = [], [], []
    for ri, mpath in enumerate(run_files):
        run_name = Path(mpath).name.replace("_matrices.npz", "")
        fpath = DATA_DIR / f"{run_name}_features.npz"
        feat = np.load(fpath)["features"]
        if feat.ndim != 2 or len(feat) == 0:
            continue
        E = feat[:, 0]
        Xc = feat[:, 1]
        Yc = feat[:, 2]
        ze = feat[:, 3]
        ne = feat[:, 5]
        nmu = feat[:, 6]
        age = feat[:, 7]
        mask = (ze < CUTS_ZE) & (ne > CUTS_NE) & (nmu > CUTS_NMU) & \
               (age > CUTS_AGE_LO) & (age < CUTS_AGE_HI)
        if mask.sum() == 0:
            continue
        E_all.append(E[mask])
        Xc_all.append(Xc[mask])
        Yc_all.append(Yc[mask])

    E = np.concatenate(E_all).astype(np.float32)
    Xc = np.concatenate(Xc_all).astype(np.float32)
    Yc = np.concatenate(Yc_all).astype(np.float32)
    print(f"Re-derived {len(E):,} events under original mask")
    assert len(E) == n_saved, f"mismatch: {len(E)} vs {n_saved}"

    # Verify against saved energies
    assert np.allclose(E, energies_saved, atol=1e-3), \
        "energy mismatch — iteration order differs from original scoring"
    print("Alignment verified against saved energies.")
    return predictions, E, Xc, Yc


def apply_jinst_cuts(predictions, E, Xc, Yc):
    r = np.sqrt(Xc**2 + Yc**2)
    mask = (r < R_CORE_MAX) & (E > E_MIN_LOG10)
    print(f"\nApplying JINST cuts: r<{R_CORE_MAX} m, log10(E)>{E_MIN_LOG10}")
    print(f"  Before: {len(E):,} events")
    print(f"  After r_core < 91 m:  {(r < R_CORE_MAX).sum():,}  "
          f"({100 * (r < R_CORE_MAX).mean():.1f}%)")
    print(f"  After log10(E) > 15.15: {(E > E_MIN_LOG10).sum():,}  "
          f"({100 * (E > E_MIN_LOG10).mean():.1f}%)")
    print(f"  After both: {mask.sum():,}  ({100 * mask.mean():.1f}%)")
    print(f"\n  Reference (JINST 2024, unblind 20% subset):")
    print(f"    with cuts (no E>15.15):  ~1.6M")
    print(f"    with all cuts (inc E>15.15): ~0.7M")
    print(f"  Reference (JINST/JCAP, full blind+unblind, all cuts): ~3.5M")
    return predictions[mask], E[mask]


def plot_folded(predictions, logE, data_part, label_tag):
    emin, emax, nbins = 15.15, 17.0, 15
    edges = np.linspace(emin, emax, nbins + 1)

    counts = np.zeros((5, nbins), dtype=np.int64)
    for c in range(5):
        m = predictions == c
        counts[c], _ = np.histogram(logE[m], bins=edges)

    T_eff = 7.925190346611552e8 * data_part
    area = np.pi * 91.0**2
    th_min, th_max = np.radians(0.0), np.radians(18.0)
    solid_angle = np.pi * (np.cos(th_min)**2 - np.cos(th_max)**2)
    exposure = area * T_eff * solid_angle

    energy_pow = 2.7
    centers_eV = (10**edges[1:] + 10**edges[:-1]) / 2
    deltaE_eV = np.diff(10**edges)
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

    for out in [OUT_PDF, OUT_PNG]:
        fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT_PDF.name}, {OUT_PNG.name}")
    print(f"\nFlux values at bin centres (dJ/dE E^2.7, m^-2 sr^-1 s^-1 eV^1.7), data_part={data_part}:")
    print(f"  {'log10(E)':>9s}  " + "  ".join(f"{p:>9s}" for p in PARTICLES))
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        vals = "  ".join(f"{flux[c, i]:9.3e}" for c in range(5))
        print(f"  {(lo + hi) / 2:9.3f}  " + vals)


def main():
    predictions, E, Xc, Yc = extract_aligned_features()
    preds_cut, E_cut = apply_jinst_cuts(predictions, E, Xc, Yc)

    # Try data_part = 0.185 first (matches JINST exposure convention for the
    # unblind 18.5% / 20% subset). Our event count after cuts will tell us if
    # we need to adjust.
    print(f"\n{'=' * 60}")
    print(f"FLUX with data_part = 0.185 (JINST unblind-subset convention)")
    print(f"{'=' * 60}")
    plot_folded(preds_cut, E_cut, data_part=0.185, label_tag="0.185")


if __name__ == "__main__":
    main()
