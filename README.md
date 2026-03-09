# astro-bench

**Can AI agents build better cosmic ray classifiers than physicists?**

This is a benchmark for AI agents and humans building ML classifiers on real astrophysics data from the [KASCADE experiment](https://web.ikp.kit.edu/KASCADE/) — a 200×200m detector array in Karlsruhe, Germany that measured cosmic ray air showers for ~25 years.

Two tasks, two leaderboards. Read the full challenge description: [challenge.md](challenge.md)

## Leaderboard: Mass Composition (5-class)

Classify cosmic ray primaries into proton, helium, carbon, silicon, iron.

| Rank | Accuracy | Author | Agent? | Architecture | Link |
|------|----------|--------|--------|--------------|------|
| — | ~51% | Kuznetsov, Petrov et al. | No | CNN (LeNet-5), QGS-only | [JINST 2024](https://doi.org/10.1088/1748-0221/19/01/P01025) |
| — | 50.86% | Claude Haiku 4.5 | Yes | CNN+MLP hybrid (622K params) | [haiku-mar8/](submissions/haiku-mar8/) |
| — | 49.9% | Claude Opus 4.6 (supervised) | Yes | MLP (512×2, ELU+BN) | [baselines/](submissions/baselines/) |
| — | 29.5% | baseline | — | RandomForest (5 features) | this repo |

## Leaderboard: Gamma/Hadron Separation (binary)

Distinguish gamma rays from hadronic cosmic rays. Key metric: hadronic survival rate at 99% gamma efficiency (lower is better).

| Rank | Survival rate | Author | Agent? | Architecture | Link |
|------|---------------|--------|--------|--------------|------|
| — | 10⁻² – 10⁻³ | Kostunin et al. | No | RF regressor | [ICRC 2021](https://arxiv.org/abs/2108.03407) |
| — | 7.8×10⁻¹ | Claude Opus 4.6 (supervised) | Yes | MLP (512×2, class weights) | [baselines/](submissions/baselines/) |
| — | 8.4×10⁻¹ | Claude Haiku 4.5 | Yes | MLP (517→512→512, class weights) | [haiku-gamma-mar8/](submissions/haiku-gamma-mar8/) |

## Quick Start

```bash
uv sync
uv run python download_data.py            # ~8.6 GB from S3
uv run python verify.py submissions/X/predictions.npz            # composition
uv run python verify.py --task gamma submissions/X/predictions.npz  # gamma
```

## How It Works

1. `download_data.py` downloads pre-split, memory-mappable `.npy` files
2. You (or your agent) build a classifier — any tools/frameworks, no constraints
3. Produce `predictions.npz` and run `verify.py` to score
4. Submit via Issue or PR

See [challenge.md](challenge.md) for data format, physics background, and submission details.

## What Makes This Different

Most ML benchmarks ask "what's the best model?" We also ask: **which AI agent builds the best model, how does it approach the problem, and what does it cost?**

The leaderboard tracks both what was achieved and how — making this a benchmark for AI agents as autonomous ML researchers.

## Context

Related work:
- [AI Agents for Ground-Based Gamma Astronomy](https://arxiv.org/abs/2503.00821) (Kostunin, Sotnikov et al., 2025)
- [New insights from old cosmic rays](https://arxiv.org/abs/2108.03407) (Kostunin, Plokhikh et al., ICRC 2021) — foundational analysis: RF composition + gamma search
- [Methods of ML for cosmic rays mass composition](https://arxiv.org/abs/2311.06893) (Kuznetsov, Petrov, Plokhikh, Sotnikov, JINST 2024) — CNN/MLP/RF comparison
- [Energy spectra of elemental groups of cosmic rays](https://arxiv.org/abs/2312.08279) (Kuznetsov, Petrov, Plokhikh, Sotnikov, JCAP 2024) — mass spectra results
- [autoresearch](https://github.com/karpathy/autoresearch) (Karpathy, 2026) — autonomous AI agents doing ML research overnight
- [Addition Under Pressure](https://dimitrisp.substack.com/p/addition-under-pressure) (Papailiopoulos, 2026) — comparing agent research paths

## License

MIT
