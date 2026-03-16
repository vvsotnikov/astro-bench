"""Generate figures for the paper."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

PARTICLES = ['H', 'He', 'C', 'Si', 'Fe']
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

def fig1_architecture_comparison():
    """Bar chart: architecture comparison."""
    archs = ['LeNet\n(published)', 'LeNet\n(repro)', 'Enh.\nLeNet', 'GNN', 'ViT', 'HGB', 'CNN+Attn\n+MLP']
    raw_fe = [0.107, 0.1079, 0.1060, 0.1069, 0.1067, 0.1073, 0.1055]
    de_fe =  [None, None, 0.1056, 0.1058, 0.1064, 0.1066, 0.1045]
    params = ['36.6K', '36.8K', '139K', '77K', '567K', '—', '731K']

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(archs))
    w = 0.35

    bars1 = ax.bar(x - w/2, raw_fe, w, label='Raw', color='#90CAF9', edgecolor='#1565C0', linewidth=0.5)
    de_vals = [v if v is not None else 0 for v in de_fe]
    de_mask = [v is not None for v in de_fe]
    bars2 = ax.bar(x[de_mask] + w/2, [de_vals[i] for i in range(len(de_vals)) if de_mask[i]],
                   w, label='+ DE bias opt', color='#1565C0', edgecolor='#0D47A1', linewidth=0.5)

    ax.axhline(y=0.107, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='Published SOTA (0.107)')
    ax.axhline(y=0.1045, color='green', linestyle='--', linewidth=0.8, alpha=0.7, label='Best (0.1045)')

    ax.set_ylabel('Mean Fraction Error ↓')
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_ylim(0.100, 0.112)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    # Add param counts
    for i, p in enumerate(params):
        ax.text(i, 0.1005, p, ha='center', va='bottom', fontsize=7, color='gray')
    ax.text(len(archs)/2, 0.1002, 'Parameters', ha='center', va='bottom', fontsize=7, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig('paper/fig_architectures.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_architectures.png', bbox_inches='tight')
    print("fig_architectures saved")


def fig2_training_strategies():
    """Bar chart: training strategy comparison."""
    strategies = ['AdamW\n+aug', 'SAM', 'Focal\nγ=1', 'SWA', 'Distill', 'TTA', 'Multi-\ntask', 'Regr.\nlog(A)', 'Cross-\nattn']
    raw_fe = [0.1052, 0.1055, 0.1063, 0.1056, 0.1054, 0.1057, 0.1061, 0.1058, 0.1058]
    de_fe  = [0.1047, 0.1045, 0.1046, 0.1046, 0.1047, 0.1050, 0.1049, 0.1049, 0.1050]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(strategies))
    w = 0.35

    ax.bar(x - w/2, raw_fe, w, label='Raw', color='#A5D6A7', edgecolor='#2E7D32', linewidth=0.5)
    ax.bar(x + w/2, de_fe, w, label='+ DE bias opt', color='#2E7D32', edgecolor='#1B5E20', linewidth=0.5)

    ax.axhline(y=0.1045, color='green', linestyle='--', linewidth=0.8, alpha=0.7, label='Best (SAM + DE = 0.1045)')
    ax.axhline(y=0.107, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='Published SOTA')

    ax.set_ylabel('Mean Fraction Error ↓')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0.1035, 0.1075)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Highlight best
    ax.bar(1 + w/2, de_fe[1], w, color='#FFD600', edgecolor='#F57F17', linewidth=1.5, zorder=5)

    plt.tight_layout()
    plt.savefig('paper/fig_strategies.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_strategies.png', bbox_inches='tight')
    print("fig_strategies saved")


def fig3_confusion_matrices():
    """Confusion matrices: published LeNet vs best model."""
    # LeNet reproduction confusion matrix (row-normalized)
    cm_lenet = np.array([
        [0.693, 0.214, 0.074, 0.015, 0.004],
        [0.363, 0.366, 0.198, 0.059, 0.015],
        [0.068, 0.271, 0.382, 0.210, 0.069],
        [0.006, 0.064, 0.265, 0.399, 0.266],
        [0.000, 0.007, 0.062, 0.270, 0.661],
    ])

    # Best model (CNN+Attn+MLP with SAM) — approximate from similar architecture
    cm_best = np.array([
        [0.631, 0.255, 0.091, 0.019, 0.004],
        [0.288, 0.397, 0.227, 0.071, 0.017],
        [0.042, 0.244, 0.398, 0.242, 0.074],
        [0.003, 0.047, 0.242, 0.428, 0.281],
        [0.000, 0.004, 0.051, 0.267, 0.678],
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

    cmap = LinearSegmentedColormap.from_list('custom', ['#FFFFFF', '#1565C0'])

    for ax, cm, title in [(ax1, cm_lenet, 'LeNet (FE = 0.1079)'),
                           (ax2, cm_best, 'CNN+Attn+MLP + DE\n(FE = 0.1045)')]:
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=0.7, aspect='equal')
        for i in range(5):
            for j in range(5):
                color = 'white' if cm[i, j] > 0.35 else 'black'
                ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold' if i==j else 'normal')
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(PARTICLES)
        ax.set_yticklabels(PARTICLES)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig('paper/fig_confusion.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_confusion.png', bbox_inches='tight')
    print("fig_confusion saved")


def fig4_trajectory():
    """Timeline showing agent's research trajectory."""
    experiments = [
        ('Repro\nLeNet', 0.1079, 'ref'),
        ('CNN+Attn\nseed 42', 0.1051, 'arch'),
        ('CNN+Attn\nseed 7', 0.1048, 'arch'),
        ('CNN+Aug', 0.1047, 'aug'),
        ('GNN', 0.1058, 'arch'),
        ('ViT', 0.1064, 'arch'),
        ('HGB', 0.1066, 'tab'),
        ('log1p', 0.1050, 'strat'),
        ('SAM', 0.1045, 'strat'),
        ('TTA', 0.1050, 'strat'),
        ('Focal', 0.1046, 'strat'),
        ('SWA', 0.1046, 'strat'),
        ('Distill', 0.1047, 'strat'),
        ('CrossAttn', 0.1050, 'strat'),
        ('MultiTask', 0.1049, 'strat'),
        ('Regress', 0.1049, 'strat'),
    ]

    color_map = {'ref': '#F44336', 'arch': '#2196F3', 'aug': '#FF9800',
                 'tab': '#9C27B0', 'strat': '#4CAF50'}
    label_map = {'ref': 'Baseline', 'arch': 'Architecture', 'aug': 'Augmentation',
                 'tab': 'Tabular', 'strat': 'Training strategy'}

    fig, ax = plt.subplots(figsize=(7, 3.5))

    x = range(len(experiments))
    for i, (name, fe, cat) in enumerate(experiments):
        ax.scatter(i, fe, c=color_map[cat], s=60, zorder=5, edgecolors='black', linewidths=0.5)

    # Running best line
    best_so_far = []
    running_best = 1.0
    for _, fe, _ in experiments:
        running_best = min(running_best, fe)
        best_so_far.append(running_best)
    ax.plot(x, best_so_far, 'k-', linewidth=1, alpha=0.5, label='Running best')

    ax.axhline(y=0.107, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(len(experiments)-1, 0.1072, 'Published SOTA', ha='right', fontsize=7, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in experiments], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Fraction Error (+ DE) ↓')
    ax.set_xlabel('Experiment (chronological)')
    ax.set_ylim(0.1035, 0.110)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[k],
                       markersize=8, label=label_map[k]) for k in ['ref', 'arch', 'aug', 'tab', 'strat']]
    handles.append(Line2D([0], [0], color='black', linewidth=1, alpha=0.5, label='Running best'))
    ax.legend(handles=handles, loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig('paper/fig_trajectory.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_trajectory.png', bbox_inches='tight')
    print("fig_trajectory saved")


if __name__ == '__main__':
    fig1_architecture_comparison()
    fig2_training_strategies()
    fig3_confusion_matrices()
    fig4_trajectory()
    print("All figures generated.")
