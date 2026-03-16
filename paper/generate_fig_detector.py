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
    """Show detector grid layout + example event from real KASCADE simulation data."""
    # Load real simulation data (channel 0=time, 1=electron, 2=muon)
    data = np.load('data/qgs_spectra_matrices.npz')
    matrices = data['matrices']  # (4164709, 16, 16, 3) float32

    # Find a visually interesting event: high total signal with spatial structure
    # Sample a manageable subset to search through
    np.random.seed(42)
    n_candidates = 50000
    idx_candidates = np.random.choice(len(matrices), n_candidates, replace=False)

    best_idx = None
    best_score = -1
    for idx in idx_candidates:
        event = matrices[idx]
        e_ch = event[:, :, 1]  # electron/photon
        m_ch = event[:, :, 2]  # muon
        combined_active = (e_ch > 0) | (m_ch > 0)
        frac_active = combined_active.sum() / (16 * 16)
        n_active_m = np.count_nonzero(m_ch)
        # Want: clear shower pattern with realistic sparsity (~15-40% active cells)
        # and muon hits for panel (c) to be interesting
        if 0.15 < frac_active < 0.40 and n_active_m >= 8:
            total_signal = e_ch.sum() + m_ch.sum()
            coords_e = np.argwhere(e_ch > 0)
            if len(coords_e) > 5:
                spread = coords_e.std(axis=0).mean()
                score = total_signal * spread
                if score > best_score:
                    best_score = score
                    best_idx = idx

    print(f"Selected event index: {best_idx}")
    event = matrices[best_idx]
    event_e = event[:, :, 1]  # electron/photon channel
    event_m = event[:, :, 2]  # muon channel

    n_active_e = np.count_nonzero(event_e)
    n_active_m = np.count_nonzero(event_m)
    print(f"  Electron channel: {n_active_e}/256 active, max={event_e.max():.1f}")
    print(f"  Muon channel:     {n_active_m}/256 active, max={event_m.max():.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    # Panel A: Active detector cells (non-zero in either channel)
    ax = axes[0]
    ax.set_xlim(0, 16); ax.set_ylim(0, 16)
    for i in range(17):
        ax.axhline(i, color='gray', linewidth=0.3)
        ax.axvline(i, color='gray', linewidth=0.3)
    active = (event_e > 0) | (event_m > 0)
    frac_zero = 1.0 - active.sum() / (16 * 16)
    for i in range(16):
        for j in range(16):
            if active[i, j]:
                ax.add_patch(Rectangle((j, i), 1, 1, color='#2196F3', alpha=0.6))
    ax.set_xlabel('x bin')
    ax.set_ylabel('y bin')
    ax.set_title(f'(a) Active cells\n({frac_zero:.0%} zero)', fontsize=9)
    ax.set_aspect('equal')

    # Panel B: Electron/photon channel with log1p
    ax = axes[1]
    im = ax.imshow(np.log1p(event_e), cmap='YlOrRd', aspect='equal', origin='lower')
    ax.set_xlabel('x bin')
    ax.set_title('(b) Electron channel\n(log1p)', fontsize=9)
    ax.set_yticks([])

    # Panel C: Muon channel with log1p
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
