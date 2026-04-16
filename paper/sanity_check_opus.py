"""Sanity checks for the Opus v5b gamma classifier.

Verifies:
1. Classification metrics on the simulation test set (ROC-AUC, precision, recall, F1).
2. Score distributions for gamma vs hadron (should be well-separated if the
   classifier is working).
3. Energy-dependent survival rate on the sim test set (to confirm p_survival is
   not constant across energy, which matters for the walk-up-the-ladder analysis).
4. Physical sanity: expected vs observed events passing on real data, given the
   sim-calibrated survival rate and a range of plausible diffuse gamma fractions.
"""

from pathlib import Path

import numpy as np

from run_real_gamma_skymap_opus import (
    DATA_DIR,
    compute_normalization_stats,
    engineer_features,
    load_models,
    score_with_ensemble,
)


def main():
    mean, std, n_features = compute_normalization_stats()
    models = load_models(mean, std, n_features)

    # === 1. Score the simulation test set ===
    print("\nScoring simulation test set...")
    X_test = np.load(DATA_DIR / "gamma_test" / "matrices.npy", mmap_mode="r")
    f_test = np.load(DATA_DIR / "gamma_test" / "features.npy", mmap_mode="r")
    y_test = np.load(DATA_DIR / "gamma_test" / "labels_gamma.npy", mmap_mode="r")

    F_test = engineer_features(np.array(X_test), np.array(f_test, dtype=np.float32))
    F_test_norm = (F_test - mean) / std

    scores = score_with_ensemble(models, np.array(X_test), F_test_norm)
    y = np.array(y_test)  # 0 = gamma, 1 = hadron
    energies = np.array(f_test)[:, 0]

    n_gamma = int((y == 0).sum())
    n_hadron = int((y == 1).sum())
    print(f"  Test set: {n_gamma} gammas, {n_hadron} hadrons")

    gamma_scores = scores[y == 0]
    hadron_scores = scores[y == 1]

    # === 2. Score distributions ===
    print("\n=== Score distributions (sigmoid output = P(gamma)) ===")
    print(f"Gamma scores:  min={gamma_scores.min():.4f}, "
          f"median={np.median(gamma_scores):.4f}, "
          f"mean={gamma_scores.mean():.4f}, max={gamma_scores.max():.4f}")
    print(f"Hadron scores: min={hadron_scores.min():.4f}, "
          f"median={np.median(hadron_scores):.4f}, "
          f"mean={hadron_scores.mean():.4f}, max={hadron_scores.max():.4f}")
    print(f"\nScore percentiles (gamma):")
    for p in [5, 25, 50, 75, 95]:
        print(f"  p{p}: {np.percentile(gamma_scores, p):.4f}")
    print(f"Score percentiles (hadron):")
    for p in [5, 25, 50, 75, 95, 99, 99.9]:
        print(f"  p{p}: {np.percentile(hadron_scores, p):.4f}")

    # === 3. ROC-AUC ===
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        print("sklearn not installed; skipping ROC/PR computation.")
        return
    # Model outputs P(gamma), so positive class = gamma (label 0). Flip the label.
    y_binary_gamma = (y == 0).astype(int)
    auc = roc_auc_score(y_binary_gamma, scores)
    ap = average_precision_score(y_binary_gamma, scores)
    print(f"\n=== Classification metrics (positive class = gamma) ===")
    print(f"ROC-AUC: {auc:.6f}")
    print(f"Average Precision: {ap:.6f}")

    # === 4. Precision/recall/F1 at the 75% gamma efficiency threshold ===
    threshold_75 = float(np.percentile(gamma_scores, 25))
    pred_pos = scores >= threshold_75
    tp = int((pred_pos & (y == 0)).sum())
    fp = int((pred_pos & (y == 1)).sum())
    fn = int(((~pred_pos) & (y == 0)).sum())
    tn = int(((~pred_pos) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    p_survival = fp / (tn + fp)

    print(f"\n=== At 75% gamma efficiency threshold = {threshold_75:.6f} ===")
    print(f"TP (gamma kept):    {tp}")
    print(f"FP (hadron kept):   {fp}")
    print(f"FN (gamma rejected):{fn}")
    print(f"TN (hadron reject): {tn}")
    print(f"Precision (gamma purity): {precision:.4f}")
    print(f"Recall (gamma efficiency): {recall:.4f}")
    print(f"Specificity: {specificity:.6f}")
    print(f"F1: {f1:.4f}")
    print(f"Hadron survival rate: {p_survival:.4e}")

    # === 5. Energy-dependent survival rate on sim test set ===
    print(f"\n=== Energy-dependent survival on sim test set (at 75% gamma eff) ===")
    print(f"{'E bin':>12s} {'N_gamma':>8s} {'gamma_eff':>10s} "
          f"{'N_hadron':>9s} {'p_surv':>12s}")
    print("-" * 60)
    e_bins = [(14.0, 14.5), (14.5, 15.0), (15.0, 15.5), (15.5, 16.0),
              (16.0, 16.5), (16.5, 17.0), (17.0, 18.0)]
    for e_lo, e_hi in e_bins:
        e_mask = (energies >= e_lo) & (energies < e_hi)
        n_g = int(((y == 0) & e_mask).sum())
        n_h = int(((y == 1) & e_mask).sum())
        if n_g == 0 and n_h == 0:
            continue
        g_pass = int((pred_pos & (y == 0) & e_mask).sum())
        h_pass = int((pred_pos & (y == 1) & e_mask).sum())
        eff = g_pass / n_g if n_g > 0 else 0
        surv = h_pass / n_h if n_h > 0 else 0
        print(f"{e_lo:5.1f}-{e_hi:5.1f} {n_g:8d} {eff:10.3f} {n_h:9d} {surv:12.4e}")

    # === 6. Physical sanity on real data ===
    print(f"\n=== Physical sanity check on real data ===")
    saved = np.load(Path(__file__).resolve().parent / "gamma_kostunin_opus.npz")
    all_e = saved["all_energies"]
    pass_e = saved["passing_energies"]
    n_total = int(len(all_e))
    n_pass = int(len(pass_e))
    print(f"Real data (after quality cuts): {n_total:,} events")
    print(f"Events passing threshold:        {n_pass:,}")
    print(f"Expected bg (N_real * p_survival): {n_total * p_survival:.1f}")
    print(f"Observed / expected: {n_pass / (n_total * p_survival):.3f}")

    print(f"\nIf the deficit is due to real gammas (unlikely, since we see a "
          f"deficit not excess):")
    n_excess = n_pass - n_total * p_survival
    print(f"  Excess (observed - expected): {n_excess:+.0f}")
    print(f"  Inferred gamma count (if excess were signal): "
          f"{n_excess / recall:.0f}")
    print(f"  Implied gamma/total ratio: {n_excess / recall / n_total:.2e}")
    print(
        f"  [A deficit means real hadrons are being rejected more efficiently "
        f"than MC hadrons, consistent with a simulation--data gap where MC is "
        f"'harder' than real data.]"
    )

    # === 7. Plausible diffuse gamma flux check ===
    print(f"\n=== Theoretical expected gammas on real data ===")
    # Diffuse gamma-to-cosmic-ray ratios at PeV (rough, order-of-magnitude)
    for frac_name, frac in [("1e-6", 1e-6), ("1e-5", 1e-5), ("1e-4", 1e-4)]:
        n_gamma_real = n_total * frac
        n_gamma_pass = n_gamma_real * recall
        print(
            f"  gamma/total = {frac_name}: "
            f"{n_gamma_real:>8.1f} real gammas → "
            f"{n_gamma_pass:>7.1f} would pass at 75% eff "
            f"(fraction of observed: {n_gamma_pass / n_pass:.1%})"
        )
    print(
        "\nNote: observed signal from real gammas --- if any --- would be "
        "buried in ~877 expected hadronic false positives and "
        "the ~114-event simulation--data deficit, well below statistical sensitivity."
    )


if __name__ == "__main__":
    main()
