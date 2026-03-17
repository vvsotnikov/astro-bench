"""Generate chronological research trajectory figures for both tasks.
Omits runs that used incorrect metrics:
- haiku-gamma-mar8: used survival@99% (wrong, should be @75%)
- haiku-gamma-mar9: used survival@99% (wrong)
- haiku-mar8 composition: used accuracy (no fraction error evaluated)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'figure.dpi': 300,
})

COLORS = {
    'haiku': '#FF9800',
    'opus': '#2196F3',
    'baseline': '#999999',
}

def fig_trajectory_composition():
    """Chronological trajectory for mass composition task."""
    # Chronological order — only valid experiments on matched pipeline
    experiments = [
        # --- Mar 14: SOTA reproduction (matched pipeline) ---
        ('Repro\nLeNet', 0.1079, 'baseline', 'valid'),

        # --- Mar 14: Opus on matched pipeline ---
        ('CNN+Attn\nseed 42', 0.1051, 'opus', 'valid'),
        ('CNN+Attn\nseed 7', 0.1048, 'opus', 'valid'),
        ('CNN+Aug\nseed 2026', 0.1047, 'opus', 'valid'),
        ('GNN', 0.1058, 'opus', 'valid'),
        ('ViT', 0.1064, 'opus', 'valid'),
        ('HGB', 0.1066, 'opus', 'valid'),

        # --- Mar 15: Training strategies ---
        ('log1p', 0.1050, 'opus', 'valid'),
        ('SAM', 0.1045, 'opus', 'valid'),
        ('TTA', 0.1050, 'opus', 'valid'),
        ('Focal', 0.1046, 'opus', 'valid'),
        ('SWA', 0.1046, 'opus', 'valid'),
        ('Distill', 0.1047, 'opus', 'valid'),
        ('CrossAttn', 0.1050, 'opus', 'valid'),
        ('MultiTask', 0.1049, 'opus', 'valid'),
        ('Regress', 0.1049, 'opus', 'valid'),
    ]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    labels_all = []
    x_pts = []
    y_pts = []
    c_pts = []

    for i, (label, fe, agent, status) in enumerate(experiments):
        labels_all.append(label)
        x_pts.append(i)
        y_pts.append(fe)
        c_pts.append(COLORS[agent])

    # Plot all points
    for xi, yi, ci in zip(x_pts, y_pts, c_pts):
        ax.scatter(xi, yi, c=ci, s=60, zorder=5, edgecolors='black', linewidths=0.5)

    # Running best
    running_best = 1.0
    bx, by = [], []
    for i, (_, fe, _, _) in enumerate(experiments):
        running_best = min(running_best, fe)
        bx.append(i); by.append(running_best)
    ax.plot(bx, by, '-', color='gray', linewidth=1, alpha=0.7)

    # Reference lines
    ax.axhline(y=0.107, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(0, 0.1072, 'Published baseline', ha='left', fontsize=7, color='gray')

    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels([e[0] for e in experiments], rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Fraction Error (+ DE) ↓')
    ax.set_xlabel('Experiment (chronological)')
    ax.set_ylim(0.1035, 0.112)
    ax.grid(axis='y', alpha=0.3)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['opus'],
               markersize=8, label='Opus 4.6', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['baseline'],
               markersize=8, label='LeNet repro.', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], color='gray', linewidth=1, alpha=0.5, label='Running best'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('paper/fig_trajectory.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_trajectory.png', bbox_inches='tight')
    print("fig_trajectory (composition) saved")


def fig_trajectory_gamma():
    """Chronological trajectory for gamma/hadron task.
    Omits haiku-gamma-mar8 and haiku-gamma-mar9 (used wrong @99% metric).
    Only shows haiku-gamma-mar9-v2 and v3 (correct @75% metric).
    """
    # v2 key results (23 experiments, correct metric)
    v2_experiments = [
        ('v1\nreweight', 3.15e-03, 'discard'),
        ('v2\nlonger', 1.31e-03, 'keep'),
        ('v3\nregression', 9.05e-04, 'keep'),
        ('v5\nens v2+v3', 7.89e-04, 'keep'),
        ('v9\nmultiseed', 7.51e-04, 'keep'),
        ('v14\nens opt', 6.94e-04, 'keep'),
        ('v18\nbest ens', 6.43e-04, 'keep'),
    ]

    # v3 key results (41+ experiments, correct metric)
    v3_experiments = [
        ('v1\nseeds', 6.43e-04, 'keep'),
        ('v3\nAttnCNN', 5.84e-04, 'keep'),
        ('v8\ndeeper', 6.13e-04, 'discard'),
        ('v9\nAttn+feat', 3.50e-04, 'keep'),
        ('v11\nmultiseed', 4.97e-04, 'discard'),
        ('v16\npure CNN', 5.26e-04, 'discard'),
        ('v20\nViT', 6.72e-04, 'discard'),
        ('v27b\nViT tuned', 5.55e-04, 'keep'),
        ('v34\nAE+feat', 5.55e-04, 'keep'),
        ('v38\nResNet', 3.80e-04, 'keep'),
        ('v41\nensemble', 3.21e-04, 'keep'),
    ]

    all_exp = []
    sessions = []
    # Session 1: v2
    start = 0
    for label, surv, status in v2_experiments:
        all_exp.append((label, surv, 'haiku', status))
    sessions.append(('Run 3 (v2)', start, len(all_exp)-1))
    # Session 2: v3
    start = len(all_exp)
    for label, surv, status in v3_experiments:
        all_exp.append((label, surv, 'haiku', status))
    sessions.append(('Run 4 (v3)', start, len(all_exp)-1))

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Only plot kept experiments
    kept_x, kept_y = [], []
    for i, (label, surv, agent, status) in enumerate(all_exp):
        if status == 'keep':
            ax.scatter(i, surv, c=COLORS[agent], s=60, zorder=5, edgecolors='black', linewidths=0.5)
            kept_x.append(i); kept_y.append(surv)

    # Running best (kept only)
    best = 1.0
    bx, by = [], []
    for i, (_, surv, _, status) in enumerate(all_exp):
        if status == 'keep':
            best = min(best, surv)
            bx.append(i); by.append(best)
    ax.plot(bx, by, '-', color='gray', linewidth=1, alpha=0.7)

    # (session separators removed for clarity)

    # Reference
    ax.axhline(y=1e-3, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(0, 1.1e-3, 'Published baseline', ha='left', fontsize=7, color='gray')

    ax.set_xticks(range(len(all_exp)))
    ax.set_xticklabels([e[0] for e in all_exp], rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Hadronic Survival @ 75% γ eff ↓')
    ax.set_xlabel('Experiment (chronological)')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['haiku'],
               markersize=8, label='Haiku 4.5', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], color='gray', linewidth=1, alpha=0.5, label='Running best'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('paper/fig_trajectory_gamma.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_trajectory_gamma.png', bbox_inches='tight')
    print("fig_trajectory_gamma saved")


if __name__ == '__main__':
    fig_trajectory_composition()
    fig_trajectory_gamma()
    print("All trajectory figures generated.")
