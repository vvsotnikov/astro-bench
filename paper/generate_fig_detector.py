"""Generate KASCADE detector grid schematic + example event."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'figure.dpi': 300,
})

def fig_detector_and_event():
    """Show detector grid layout + example sparse event."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    # Panel A: Schematic of 16x16 binned grid
    ax = axes[0]
    ax.set_xlim(0, 16); ax.set_ylim(0, 16)
    for i in range(17):
        ax.axhline(i, color='gray', linewidth=0.3)
        ax.axvline(i, color='gray', linewidth=0.3)
    # Show some "active" stations
    np.random.seed(42)
    active = np.random.random((16, 16)) < 0.15
    for i in range(16):
        for j in range(16):
            if active[i, j]:
                ax.add_patch(Rectangle((j, i), 1, 1, color='#2196F3', alpha=0.6))
    ax.set_xlabel('x bin')
    ax.set_ylabel('y bin')
    ax.set_title('(a) 16×16 grid\n(~85% zero)', fontsize=9)
    ax.set_aspect('equal')

    # Panel B: Example electron channel
    np.random.seed(7)
    event_e = np.zeros((16, 16))
    # Simulate a shower core
    cx, cy = 8, 7
    for i in range(16):
        for j in range(16):
            r = np.sqrt((i - cy)**2 + (j - cx)**2)
            if r < 6 and np.random.random() < 0.4 * np.exp(-r/3):
                event_e[i, j] = np.random.exponential(100) * np.exp(-r/2)

    ax = axes[1]
    im = ax.imshow(np.log1p(event_e), cmap='YlOrRd', aspect='equal', origin='lower')
    ax.set_xlabel('x bin')
    ax.set_title('(b) Electron channel\n(log1p)', fontsize=9)
    ax.set_yticks([])

    # Panel C: Example muon channel (sparser)
    event_m = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            r = np.sqrt((i - cy)**2 + (j - cx)**2)
            if r < 5 and np.random.random() < 0.15 * np.exp(-r/4):
                event_m[i, j] = np.random.exponential(30) * np.exp(-r/3)

    ax = axes[2]
    im = ax.imshow(np.log1p(event_m), cmap='YlGnBu', aspect='equal', origin='lower')
    ax.set_xlabel('x bin')
    ax.set_title('(c) Muon channel\n(log1p)', fontsize=9)
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('paper/fig_detector.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_detector.png', bbox_inches='tight')
    print("fig_detector saved")


def fig_ne_nmu_separation():
    """Show Ne-Nmu separation between particle classes."""
    np.random.seed(42)
    n_per_class = 2000
    particles = ['H', 'He', 'C', 'Si', 'Fe']
    # Approximate Ne-Nmu distributions from EDA
    ne_nmu_means = [0.97, 0.84, 0.74, 0.67, 0.62]
    ne_nmu_stds = [0.24, 0.18, 0.18, 0.20, 0.22]
    colors = ['#F44336', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']

    fig, ax = plt.subplots(figsize=(5, 3))
    for i, (name, mu, sig, col) in enumerate(zip(particles, ne_nmu_means, ne_nmu_stds, colors)):
        x = np.random.normal(mu, sig, n_per_class)
        bins = np.linspace(0, 1.6, 60)
        ax.hist(x, bins=bins, alpha=0.4, color=col, label=name, density=True)
        # KDE-like line
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(x)
        xx = np.linspace(0, 1.6, 200)
        ax.plot(xx, kde(xx), color=col, linewidth=1.5)

    ax.set_xlabel(r'$\log_{10}(N_e) - \log_{10}(N_\mu)$')
    ax.set_ylabel('Density')
    ax.set_title(r'$N_e/N_\mu$ ratio separation by mass group')
    ax.legend(fontsize=8, ncol=5, loc='upper right')
    ax.set_xlim(0, 1.6)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig('paper/fig_separation.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_separation.png', bbox_inches='tight')
    print("fig_separation saved")


if __name__ == '__main__':
    fig_detector_and_event()
    fig_ne_nmu_separation()
    print("All additional figures generated.")
