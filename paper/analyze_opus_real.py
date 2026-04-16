"""Proper analysis of Opus v5b gamma scores on real KASCADE data.

Uses the full score database from run_real_gamma_full_opus.py.

For each energy bin:
  1. Score distribution comparison: real data vs sim gamma vs sim hadron
  2. Per-bin threshold at a fixed gamma efficiency (75%)
  3. Top-K highest-scoring real events with full features
  4. Per-bin candidate criterion: real score above the empirical sim-hadron
     99.9th percentile AND within the sim-gamma score range

Physics sanity check at the end.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPZ = HERE / "gamma_real_full_opus.npz"

# Energy bins (log10 E/eV)
EBINS = [(14.0, 14.5), (14.5, 15.0), (15.0, 15.5), (15.5, 16.0),
         (16.0, 16.5), (16.5, 17.0), (17.0, 18.0)]


def main():
    d = np.load(NPZ, allow_pickle=True)
    real_score = d["real_score"]
    real_E = d["real_E"]
    real_Ze = d["real_Ze"]
    real_Az = d["real_Az"]
    real_Ne = d["real_Ne"]
    real_Nmu = d["real_Nmu"]
    real_run = d["real_run"]
    real_ts = d["real_timestamp"]

    sim_score = d["sim_score"]
    sim_label = d["sim_label"]  # 0 = gamma, 1 = hadron
    sim_E = d["sim_E"]

    n_real = len(real_score)
    n_gamma = int((sim_label == 0).sum())
    n_hadron = int((sim_label == 1).sum())
    print(f"Real data: {n_real:,} events (after quality cuts)")
    print(f"Sim test:  {n_gamma:,} gammas, {n_hadron:,} hadrons")

    # === Overall score statistics ===
    print("\n=== Global score distributions ===")
    print(f"Real:        min={real_score.min():.4f}, "
          f"median={np.median(real_score):.4f}, "
          f"p99={np.percentile(real_score, 99):.4f}, "
          f"p99.9={np.percentile(real_score, 99.9):.4f}, "
          f"max={real_score.max():.4f}")
    g = sim_score[sim_label == 0]
    h = sim_score[sim_label == 1]
    print(f"Sim gamma:   min={g.min():.4f}, median={np.median(g):.4f}, "
          f"p25={np.percentile(g, 25):.4f}, "
          f"p5={np.percentile(g, 5):.4f}")
    print(f"Sim hadron:  min={h.min():.4f}, median={np.median(h):.4f}, "
          f"p99={np.percentile(h, 99):.4f}, "
          f"p99.9={np.percentile(h, 99.9):.4f}, max={h.max():.4f}")

    # === Per-bin analysis ===
    print("\n\n=== Per-energy-bin analysis ===\n")

    header = (
        f"{'E bin':>12s} {'N_real':>10s} {'N_gamma':>8s} {'N_had':>8s} "
        f"{'gamma_med':>10s} {'had_p99.9':>10s} {'thr75':>9s} "
        f"{'N_real>thr':>11s} {'N_exp_bg':>10s}"
    )
    print(header)
    print("-" * len(header))

    candidates_all = []

    for elo, ehi in EBINS:
        m_real = (real_E >= elo) & (real_E < ehi)
        m_g = (sim_label == 0) & (sim_E >= elo) & (sim_E < ehi)
        m_h = (sim_label == 1) & (sim_E >= elo) & (sim_E < ehi)

        n_r = int(m_real.sum())
        n_g_bin = int(m_g.sum())
        n_h_bin = int(m_h.sum())

        if n_g_bin == 0 or n_h_bin == 0:
            print(f"{elo:5.1f}-{ehi:5.1f} {n_r:10,} {n_g_bin:8d} {n_h_bin:8d}"
                  f"  (insufficient sim statistics)")
            continue

        g_bin = sim_score[m_g]
        h_bin = sim_score[m_h]
        r_bin = real_score[m_real]

        # Per-bin 75% gamma efficiency threshold
        thr75 = float(np.percentile(g_bin, 25))  # keep top 75% of gammas
        n_pass_real = int((r_bin >= thr75).sum())
        p_surv_bin = float((h_bin >= thr75).mean())
        exp_bg_bin = n_r * p_surv_bin

        gamma_median = float(np.median(g_bin))
        hadron_p999 = float(np.percentile(h_bin, 99.9))

        print(
            f"{elo:5.1f}-{ehi:5.1f} {n_r:10,} {n_g_bin:8d} {n_h_bin:8d} "
            f"{gamma_median:10.4f} {hadron_p999:10.4f} {thr75:9.4f} "
            f"{n_pass_real:11d} {exp_bg_bin:10.2f}"
        )

        # Per-bin candidate criterion:
        #   score > p99.9(sim hadron in this bin) AND
        #   score >= p5(sim gamma in this bin)
        # (the upper end of the score distribution where almost no hadrons sit,
        #  and where at least 95% of sim gammas sit)
        gamma_p5 = float(np.percentile(g_bin, 5))
        cand_thr = max(hadron_p999, gamma_p5)
        cand_mask = r_bin >= cand_thr
        cand_idx_global = np.where(m_real)[0][cand_mask]

        for idx in cand_idx_global:
            candidates_all.append({
                "E": float(real_E[idx]),
                "Ze": float(real_Ze[idx]),
                "Az": float(real_Az[idx]),
                "Ne": float(real_Ne[idx]),
                "Nmu": float(real_Nmu[idx]),
                "score": float(real_score[idx]),
                "run": str(real_run[idx]),
                "timestamp": str(real_ts[idx]),
                "e_bin": (elo, ehi),
                "bin_cand_thr": cand_thr,
                "bin_gamma_med": gamma_median,
                "bin_had_p999": hadron_p999,
            })

        # Top-5 real events in this bin
        top_k = 5
        top_idx = np.argsort(r_bin)[::-1][:top_k]
        print(f"  top-{top_k} real events (E_lo={elo}, E_hi={ehi}):")
        print(
            f"    {'score':>8s} {'E':>7s} {'Ze':>6s} {'Ne':>6s} "
            f"{'Nmu':>6s} {'Ne-Nmu':>7s}  "
            f"(gamma_score_p25={np.percentile(g_bin, 25):.3f}, "
            f"p5={np.percentile(g_bin, 5):.3f})"
        )
        m_real_idx = np.where(m_real)[0]
        for rank, local_idx in enumerate(top_idx):
            gi = m_real_idx[local_idx]
            s = real_score[gi]
            # Where does this score sit in the sim hadron distribution?
            hadron_pct = (h_bin < s).mean() * 100
            # Where does it sit in the sim gamma distribution?
            gamma_pct = (g_bin < s).mean() * 100
            print(
                f"    {s:8.4f} {real_E[gi]:7.3f} {real_Ze[gi]:6.1f} "
                f"{real_Ne[gi]:6.3f} {real_Nmu[gi]:6.3f} "
                f"{real_Ne[gi] - real_Nmu[gi]:7.3f}  "
                f"[hadron pct: {hadron_pct:5.2f}, gamma pct: {gamma_pct:5.2f}]"
            )
        print()

    # === Candidates meeting per-bin criterion ===
    print("\n=== Events passing per-bin candidate criterion ===")
    print("(score > p99.9 of sim hadrons in bin AND >= p5 of sim gammas in bin)")
    print(f"Total candidates: {len(candidates_all)}")

    if candidates_all:
        candidates_all.sort(key=lambda c: c["E"], reverse=True)
        print(f"\nTop 20 highest-energy candidates:")
        print(
            f"{'E':>7s} {'Ze':>6s} {'Az':>7s} {'Ne':>6s} {'Nmu':>6s} "
            f"{'score':>8s} {'bin_had_p999':>14s} {'bin_gamma_med':>14s}"
        )
        for c in candidates_all[:20]:
            print(
                f"{c['E']:7.3f} {c['Ze']:6.1f} {c['Az']:7.1f} "
                f"{c['Ne']:6.3f} {c['Nmu']:6.3f} {c['score']:8.4f} "
                f"{c['bin_had_p999']:14.4f} {c['bin_gamma_med']:14.4f}"
            )

    # === High-energy tail: top real events at E > 15.5 regardless of threshold ===
    print("\n\n=== TOP 20 REAL EVENTS AT E > 15.5 (any score) ===")
    print(
        "The crucial question: do any high-energy events have scores comparable to "
        "sim gammas?"
    )
    m_he = real_E >= 15.5
    he_idx = np.where(m_he)[0]
    he_scores = real_score[m_he]
    n_he = len(he_idx)
    print(f"\nTotal real events at E > 15.5: {n_he:,}")
    # Sim statistics at E > 15.5
    sim_he_g = sim_score[(sim_label == 0) & (sim_E >= 15.5)]
    sim_he_h = sim_score[(sim_label == 1) & (sim_E >= 15.5)]
    print(f"Sim at E > 15.5: {len(sim_he_g)} gammas "
          f"(score median={np.median(sim_he_g):.4f}, "
          f"p5={np.percentile(sim_he_g, 5):.4f}, p1={np.percentile(sim_he_g, 1):.4f}), "
          f"{len(sim_he_h)} hadrons "
          f"(score max={sim_he_h.max():.4f}, "
          f"p99.9={np.percentile(sim_he_h, 99.9):.4f})")

    top_idx = np.argsort(he_scores)[::-1][:20]
    print(
        f"\n{'score':>8s} {'E':>7s} {'Ze':>6s} {'Az':>7s} {'Ne':>6s} "
        f"{'Nmu':>6s} {'Ne-Nmu':>7s} {'run':>10s}  "
        f"{'had_pct':>8s} {'gamma_pct':>10s}"
    )
    for local in top_idx:
        gi = he_idx[local]
        s = real_score[gi]
        had_pct = (sim_he_h < s).mean() * 100
        gamma_pct = (sim_he_g < s).mean() * 100
        print(
            f"{s:8.4f} {real_E[gi]:7.3f} {real_Ze[gi]:6.1f} {real_Az[gi]:7.1f} "
            f"{real_Ne[gi]:6.3f} {real_Nmu[gi]:6.3f} "
            f"{real_Ne[gi] - real_Nmu[gi]:7.3f} {str(real_run[gi]):>10s}  "
            f"{had_pct:8.2f} {gamma_pct:10.2f}"
        )

    # Same but for E > 16.0 (the PeVatron regime)
    print("\n\n=== TOP 20 REAL EVENTS AT E > 16.0 (PeVatron regime) ===")
    m_pev = real_E >= 16.0
    pev_idx = np.where(m_pev)[0]
    pev_scores = real_score[m_pev]
    print(f"Total real events at E > 16.0: {len(pev_idx):,}")
    sim_pev_g = sim_score[(sim_label == 0) & (sim_E >= 16.0)]
    sim_pev_h = sim_score[(sim_label == 1) & (sim_E >= 16.0)]
    print(f"Sim at E > 16.0: {len(sim_pev_g)} gammas "
          f"(median={np.median(sim_pev_g):.4f}, p1={np.percentile(sim_pev_g, 1):.4f}), "
          f"{len(sim_pev_h)} hadrons "
          f"(max={sim_pev_h.max():.4f}, p99.9={np.percentile(sim_pev_h, 99.9):.4f})")
    top_pev = np.argsort(pev_scores)[::-1][:20]
    print(
        f"\n{'score':>8s} {'E':>7s} {'Ze':>6s} {'Az':>7s} {'Ne':>6s} "
        f"{'Nmu':>6s} {'Ne-Nmu':>7s} {'run':>10s}  "
        f"{'had_pct':>8s} {'gamma_pct':>10s}"
    )
    for local in top_pev:
        gi = pev_idx[local]
        s = real_score[gi]
        had_pct = (sim_pev_h < s).mean() * 100
        gamma_pct = (sim_pev_g < s).mean() * 100
        print(
            f"{s:8.4f} {real_E[gi]:7.3f} {real_Ze[gi]:6.1f} {real_Az[gi]:7.1f} "
            f"{real_Ne[gi]:6.3f} {real_Nmu[gi]:6.3f} "
            f"{real_Ne[gi] - real_Nmu[gi]:7.3f} {str(real_run[gi]):>10s}  "
            f"{had_pct:8.2f} {gamma_pct:10.2f}"
        )


if __name__ == "__main__":
    main()
