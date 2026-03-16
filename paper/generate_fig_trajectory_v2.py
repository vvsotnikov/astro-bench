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
    'baseline': '#F44336',
    'published': '#999999',
}

def fig_trajectory_composition():
    """Chronological trajectory for mass composition task."""
    # Chronological order of ALL composition runs (including failures)
    experiments = [
        # (label, FE or None, agent, status)
        # --- Mar 8: Haiku composition (accuracy only, no FE) ---
        ('Haiku\nCNN v4\n(acc only)', None, 'haiku', 'invalid'),  # best was 50.86% acc but no FE

        # --- Mar 8-12: Opus on old pipeline (wrong data, no Age, different splits) ---
        ('Opus\nRF v1\n(old pipe)', 0.1195, 'opus', 'invalid'),
        ('Opus\nHGB v2\n(old pipe)', 0.1182, 'opus', 'invalid'),
        ('Opus\nCNN v8\n(old pipe)', 0.1080, 'opus', 'invalid'),
        ('Opus\nDE opt\n(old pipe)', 0.1060, 'opus', 'invalid'),

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

    x_valid = []
    y_valid = []
    c_valid = []
    x_invalid = []
    y_invalid = []
    c_invalid = []
    labels_all = []

    for i, (label, fe, agent, status) in enumerate(experiments):
        labels_all.append(label)
        if status == 'invalid' or fe is None:
            if fe is not None:
                x_invalid.append(i)
                y_invalid.append(fe)
                c_invalid.append(COLORS[agent])
            # Skip None (no FE computed)
        else:
            x_valid.append(i)
            y_valid.append(fe)
            c_valid.append(COLORS[agent])

    # Plot invalid runs with X markers
    for xi, yi, ci in zip(x_invalid, y_invalid, c_invalid):
        ax.scatter(xi, yi, c=ci, s=50, marker='x', zorder=5, linewidths=2, alpha=0.4)

    # Plot valid runs with circles
    for xi, yi, ci in zip(x_valid, y_valid, c_valid):
        ax.scatter(xi, yi, c=ci, s=60, zorder=5, edgecolors='black', linewidths=0.5)

    # Running best for valid only
    best_so_far = []
    running_best = 1.0
    for i, (_, fe, _, status) in enumerate(experiments):
        if status == 'valid' and fe is not None:
            running_best = min(running_best, fe)
        if i in x_valid:
            best_so_far.append((i, running_best))

    if best_so_far:
        bx, by = zip(*best_so_far)
        ax.plot(bx, by, 'k-', linewidth=1, alpha=0.5)

    # Reference lines
    ax.axhline(y=0.107, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(len(experiments)-1, 0.1072, 'Published SOTA', ha='right', fontsize=7, color='red')

    # Separator between phases
    ax.axvline(x=4.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(2, 0.121, 'Wrong pipeline', ha='center', fontsize=7, color='gray', style='italic')
    ax.text(13, 0.121, 'Matched pipeline', ha='center', fontsize=7, color='gray', style='italic')

    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels([e[0] for e in experiments], rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Fraction Error (+ DE) ↓')
    ax.set_xlabel('Experiment (chronological)')
    ax.set_ylim(0.1035, 0.123)
    ax.grid(axis='y', alpha=0.3)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['opus'],
               markersize=8, label='Opus (valid)', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['baseline'],
               markersize=8, label='Baseline', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='x', color=COLORS['opus'], markersize=8,
               label='Invalid metric/pipeline', linewidth=0, markeredgewidth=2, alpha=0.4),
        Line2D([0], [0], marker='x', color=COLORS['haiku'], markersize=8,
               label='Haiku (no FE)', linewidth=0, markeredgewidth=2, alpha=0.4),
        Line2D([0], [0], color='black', linewidth=1, alpha=0.5, label='Running best'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7, ncol=2)

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

    for i, (label, surv, agent, status) in enumerate(all_exp):
        color = COLORS[agent]
        if status == 'discard':
            ax.scatter(i, surv, c=color, s=40, zorder=5, edgecolors='black',
                      linewidths=0.5, alpha=0.4)
        else:
            ax.scatter(i, surv, c=color, s=60, zorder=5, edgecolors='black', linewidths=0.5)

    # Running best
    best = 1.0
    bx, by = [], []
    for i, (_, surv, _, _) in enumerate(all_exp):
        best = min(best, surv)
        bx.append(i); by.append(best)
    ax.plot(bx, by, 'k-', linewidth=1, alpha=0.5)

    # Session separators
    for name, s, e in sessions:
        ax.axvspan(s-0.5, e+0.5, alpha=0.05, color=COLORS['haiku'])
        ax.text((s+e)/2, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 3.5e-3,
                name, ha='center', fontsize=7, color=COLORS['haiku'], style='italic')

    # Reference
    ax.axhline(y=1e-3, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(len(all_exp)-1, 1.1e-3, 'Published ~10⁻³', ha='right', fontsize=7, color='gray')

    ax.set_xticks(range(len(all_exp)))
    ax.set_xticklabels([e[0] for e in all_exp], rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Hadronic Survival @ 75% γ eff ↓')
    ax.set_xlabel('Experiment (chronological)')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['haiku'],
               markersize=8, label='Haiku (kept)', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['haiku'],
               markersize=8, label='Haiku (discarded)', markeredgecolor='black',
               markeredgewidth=0.5, alpha=0.4),
        Line2D([0], [0], color='black', linewidth=1, alpha=0.5, label='Running best'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.savefig('paper/fig_trajectory_gamma.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_trajectory_gamma.png', bbox_inches='tight')
    print("fig_trajectory_gamma saved")


if __name__ == '__main__':
    fig_trajectory_composition()
    fig_trajectory_gamma()
    print("All trajectory figures generated.")
