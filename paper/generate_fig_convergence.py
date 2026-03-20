"""Generate convergence and trajectory figures from results.tsv files.

Fig 1: Best@N convergence curves — all agents on one plot
Fig 2: Detailed trajectory — single agent, every attempt labeled
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import csv

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 300})

AGENT_COLORS = {
    'Haiku 4.5': '#FF9800',
    'Sonnet 4.6': '#2196F3',
    'Opus 4.6': '#9C27B0',
    'GPT Codex': '#4CAF50',
    'Qwen': '#F44336',
    'Kimi': '#607D8B',
}


def load_results(tsv_path):
    """Load results.tsv, return list of (attempt, metric, description)."""
    results = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            attempt = int(row['attempt'])
            if attempt == 0:  # skip baseline
                continue
            # Handle both composition (frac_error) and gamma (survival_75)
            if 'frac_error' in row:
                metric = float(row['frac_error'])
            elif 'survival_75' in row:
                metric = float(row['survival_75'])
            else:
                continue
            desc = row.get('description', '')
            results.append((attempt, metric, desc))
    return results


def compute_best_at_n(results):
    """Compute running best (Best@N) from results list."""
    if not results:
        return [], []
    attempts = []
    bests = []
    best_so_far = float('inf')
    for attempt, metric, _ in sorted(results):
        best_so_far = min(best_so_far, metric)
        attempts.append(attempt)
        bests.append(best_so_far)
    return attempts, bests


def fig_convergence(task='gamma'):
    """Fig 1: Best@N convergence curves for all agents."""
    experiments_dir = Path(f'../experiments')
    if not experiments_dir.exists():
        experiments_dir = Path('v2/experiments')

    fig, ax = plt.subplots(figsize=(8, 5))

    # Find all results for this task
    found = False
    for exp_dir in sorted(experiments_dir.glob(f'{task}-*')):
        tsv = exp_dir / 'results.tsv'
        if not tsv.exists():
            continue
        results = load_results(tsv)
        if not results:
            continue

        # Extract agent name from directory name (e.g., gamma-haiku-19mar-v2 → Haiku 4.5)
        dirname = exp_dir.name
        if 'haiku' in dirname:
            agent = 'Haiku 4.5'
        elif 'sonnet' in dirname:
            agent = 'Sonnet 4.6'
        elif 'opus' in dirname:
            agent = 'Opus 4.6'
        elif 'codex' in dirname:
            agent = 'GPT Codex'
        elif 'qwen' in dirname:
            agent = 'Qwen'
        elif 'kimi' in dirname:
            agent = 'Kimi'
        else:
            agent = dirname

        color = AGENT_COLORS.get(agent, '#999999')
        attempts, bests = compute_best_at_n(results)
        ax.plot(attempts, bests, '-o', color=color, label=agent,
                markersize=4, linewidth=2, alpha=0.9)
        found = True

    if not found:
        print(f"No results found for task={task}")
        return

    # Baseline reference
    if task == 'gamma':
        ax.axhline(y=1e-2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.text(1, 1.1e-2, 'Published baseline', fontsize=8, color='gray')
        ax.set_ylabel('Hadronic survival @ 75% γ eff ↓')
        ax.set_yscale('log')
    else:
        ax.axhline(y=0.107, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.text(1, 0.1075, 'Published baseline (0.107)', fontsize=8, color='gray')
        ax.set_ylabel('Fraction error ↓')

    ax.set_xlabel('Attempt number')
    ax.set_xlim(0.5, 50.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_title(f'Convergence: {task.capitalize()} task (Best@N)')

    plt.tight_layout()
    plt.savefig(f'fig_convergence_{task}.pdf', bbox_inches='tight')
    plt.savefig(f'fig_convergence_{task}.png', bbox_inches='tight')
    print(f"Saved fig_convergence_{task}.{{pdf,png}}")


def fig_trajectory(tsv_path, agent_name='Agent', task='gamma'):
    """Fig 2: Detailed trajectory with labeled attempts (Karpathy-style)."""
    results = load_results(tsv_path)
    if not results:
        print(f"No results in {tsv_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    attempts = [r[0] for r in results]
    metrics = [r[1] for r in results]
    descriptions = [r[2] for r in results]

    # Compute running best
    best_so_far = float('inf')
    running_best = []
    is_improvement = []
    for m in metrics:
        improved = m < best_so_far
        is_improvement.append(improved)
        best_so_far = min(best_so_far, m)
        running_best.append(best_so_far)

    # Plot all attempts
    for i, (a, m, desc, imp) in enumerate(zip(attempts, metrics, descriptions, is_improvement)):
        if imp:
            ax.scatter(a, m, c='#4CAF50', s=60, zorder=5, edgecolors='black', linewidths=0.5)
            # Label improvements
            short_desc = desc[:40] + '...' if len(desc) > 40 else desc
            ax.annotate(short_desc, (a, m), fontsize=5, rotation=30,
                       ha='left', va='bottom', color='#2E7D32',
                       xytext=(3, 5), textcoords='offset points')
        else:
            ax.scatter(a, m, facecolors='none', s=40, zorder=4,
                      edgecolors='#999999', linewidths=1, alpha=0.6)

    # Running best line
    # Connect only the improvement points
    imp_attempts = [a for a, imp in zip(attempts, is_improvement) if imp]
    imp_metrics = [m for m, imp in zip(running_best, is_improvement) if imp]
    # Add step-wise connection
    step_x, step_y = [], []
    for i, (a, m) in enumerate(zip(imp_attempts, imp_metrics)):
        if i > 0:
            step_x.append(a)
            step_y.append(imp_metrics[i-1])  # horizontal line at previous best
        step_x.append(a)
        step_y.append(m)
    if step_x:
        ax.plot(step_x, step_y, '-', color='#4CAF50', linewidth=1.5, alpha=0.7)

    # Baseline
    if task == 'gamma':
        ax.axhline(y=1e-2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_ylabel('Hadronic survival @ 75% γ eff ↓')
        ax.set_yscale('log')
    else:
        ax.axhline(y=0.107, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_ylabel('Fraction error ↓')

    ax.set_xlabel('Attempt number')
    n_attempts = len(results)
    n_improvements = sum(is_improvement)
    ax.set_title(f'{agent_name}: {n_attempts} attempts, {n_improvements} improvements')

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
               markersize=8, label='Improvement', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markersize=8, label='No improvement', markeredgecolor='#999999', markeredgewidth=1),
        Line2D([0], [0], color='#4CAF50', linewidth=1.5, alpha=0.7, label='Running best'),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    safe_name = agent_name.lower().replace(' ', '_').replace('.', '')
    plt.savefig(f'fig_trajectory_{task}_{safe_name}.pdf', bbox_inches='tight')
    plt.savefig(f'fig_trajectory_{task}_{safe_name}.png', bbox_inches='tight')
    print(f"Saved fig_trajectory_{task}_{safe_name}.{{pdf,png}}")


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Generate convergence plots for available data
    for task in ['gamma', 'composition']:
        fig_convergence(task)

    # Generate detailed trajectory for each agent
    for dirname, agent_name in [
        ('gamma-haiku-19mar-v2', 'Haiku 4.5'),
        ('gamma-sonnet-19mar', 'Sonnet 4.6'),
    ]:
        tsv = Path(f'../experiments/{dirname}/results.tsv')
        if tsv.exists():
            fig_trajectory(tsv, agent_name, 'gamma')
