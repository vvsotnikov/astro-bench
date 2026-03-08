# astro-bench

**Can AI agents build better cosmic ray classifiers than physicists?**

This is a benchmark for AI agents and humans building ML classifiers on real astrophysics data from the [KASCADE experiment](https://web.ikp.kit.edu/KASCADE/) — a 200×200m detector array in Karlsruhe, Germany that measured cosmic ray air showers for ~25 years.

Two tasks, two leaderboards. Read the full challenge description: [challenge.md](challenge.md)

## Leaderboard: Mass Composition (5-class)

Classify cosmic ray primaries into proton, helium, carbon, silicon, iron.

| Rank | Accuracy | Author | Agent? | Architecture | Link |
|------|----------|--------|--------|--------------|------|
| — | ~47% | Kuznetsov, Petrov et al. | No | DNN (ELU+BN), QGS-only | [JCAP 2024](https://doi.org/10.1088/1475-7516/2024/08/025) |
| — | 29.5% | baseline | — | RandomForest (5 features) | this repo |

## Leaderboard: Gamma/Hadron Separation (binary)

Distinguish gamma rays from hadronic cosmic rays. Key metric: hadronic survival rate at 99% gamma efficiency (lower is better).

| Rank | Survival rate | Author | Agent? | Architecture | Link |
|------|---------------|--------|--------|--------------|------|
| — | 10⁻⁶ – 3×10⁻⁵ | Petrov et al. | No | published | [Chinese Physics C 2023](https://doi.org/10.1088/1674-1137/acd4f1) |

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
- [Energy spectra of cosmic rays with KASCADE and ML](https://doi.org/10.1088/1475-7516/2024/08/025) (Kuznetsov, Petrov, Plokhikh, Sotnikov, JCAP 2024)
- [Addition Under Pressure](https://dimitrisp.substack.com/p/addition-under-pressure) (Papailiopoulos, 2026)

## License

MIT
