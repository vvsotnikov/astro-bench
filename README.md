# astro-bench

**Can AI agents build better cosmic ray classifiers than physicists?**

This is a benchmark for AI agents and humans building ML classifiers on real astrophysics data from the [KASCADE experiment](https://web.ikp.kit.edu/KASCADE/) — a 200×200m detector array in Karlsruhe, Germany that measured cosmic ray air showers for ~25 years.

Two tasks, two leaderboards. Read the full challenge description: [challenge.md](challenge.md)

## Leaderboard: Mass Composition (5-class)

Classify cosmic ray primaries into proton, helium, carbon, silicon, iron. Key metric: mean fraction error (lower is better) — measures how accurately the classifier recovers particle fractions across random mixture compositions. Methodology matches Kuznetsov et al. ([JINST 2024](https://doi.org/10.1088/1748-0221/19/01/P01025), Section 4.3): 1001 grid ensembles of 5000 events, fractions on 0.1 step grid.

Data: QGSJet-II.04 simulation, quality cuts (Ze<30, Ne>4.8, 0.2<Age<1.48, Nmu>3.6), 70/30 train/test split (seed=42). Same dataset and evaluation as the published reference.

| Rank | Frac Error ↓ | Accuracy | Author | Agent? | Architecture | Link |
|------|-------------|----------|--------|--------|--------------|------|
| 1 | 0.1047 | 51.12% | Claude Opus 4.6 | Yes | CNN+Attn+MLP (731K params) + augmentation + DE bias opt | [beat_sota.py](submissions/opus-composition-mar14/matched_pipeline/beat_sota.py) |
| ref | 0.107 | ~51% | Kuznetsov, Petrov et al. | No | CNN (LeNet, 36.6K params) | [JINST 2024](https://doi.org/10.1088/1748-0221/19/01/P01025) |
| repro | 0.1079 | 50.3% | reproduction | — | LeNet (same as ref) | [reproduce_sota.py](submissions/opus-composition-mar14/matched_pipeline/reproduce_sota.py) |

## Leaderboard: Gamma/Hadron Separation (binary)

Distinguish gamma rays from hadronic cosmic rays. Key metric: hadronic survival rate at 75% gamma efficiency (lower is better). Published suppression of 10²–10³ was measured at ~70% gamma efficiency ([ICRC 2021](https://arxiv.org/abs/2108.03407)).

> **Note**: Gamma results below use a preliminary data pipeline (v2 pre-split) and are pending re-evaluation on the matched methodology.

| Rank | Survival ↓ (@ 75% γ eff) | Author | Agent? | Architecture | Link |
|------|--------------------------|--------|--------|--------------|------|
| 1 | 3.2×10⁻⁴ | Claude Haiku 4.5 | Yes | Ensemble: Attention CNN + ResNet + ViT | [train.py](submissions/haiku-gamma-mar9-v3/train_v41_ensemble_best.py) |
| 2 | 6.4×10⁻⁴ | Claude Haiku 4.5 | Yes | MLP ensemble (BCELoss + classification) | [train.py](submissions/haiku-gamma-mar9-v2/train_v18_multiseed.py) |
| 3 | 3.2×10⁻³ | Claude Haiku 4.5 | Yes | DNN + physics ensemble | [train.py](submissions/haiku-gamma-mar9/train_v7_ensemble.py) |
| 4 | 5.1×10⁻³ | Claude Opus 4.6 (supervised) | Yes | MLP (512×2, class weights) | [train.py](submissions/baselines/train_gamma_dnn.py) |
| 5 | 7.3×10⁻³ | Claude Haiku 4.5 | Yes | MLP (517→512→512, class weights) | [train.py](submissions/haiku-gamma-mar8/train.py) |
| ref | 10⁻² – 10⁻³ | Kostunin et al. | No | RF regressor | [ICRC 2021](https://arxiv.org/abs/2108.03407) |

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

Most ML benchmarks ask "what's the best model?" We ask three questions:

1. **Can AI agents beat human scientists?** — agents build classifiers on the same data with the same evaluation, competing directly against published results
2. **How do agents search for solutions?** — every experiment (including failures) is logged with full provenance: code, logs, model weights, reasoning traces. The search trajectory is data.
3. **How do agent solutions differ from human ones?** — agents may discover architectures, feature engineering, or training strategies that humans wouldn't try, and vice versa

The leaderboard tracks both what was achieved and how — making this a benchmark for AI agents as autonomous ML researchers. Submissions must include all artifacts (training scripts, logs, model weights, experiment journals) so that the research process itself can be analyzed.

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
