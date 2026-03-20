# astro-bench

**Can AI agents build better cosmic ray classifiers than physicists?**

A benchmark for evaluating AI agents as autonomous ML researchers on real astrophysics data from the [KASCADE experiment](https://web.ikp.kit.edu/KASCADE/).

## Two tasks, two benchmarks

| Task | Directory | Metric | Published baseline | Budget |
|------|-----------|--------|-------------------|--------|
| **Mass composition** (5-class) | `composition/` | Mean fraction error ↓ | 0.107 ([JINST 2024](https://doi.org/10.1088/1748-0221/19/01/P01025)) | 50 attempts |
| **Gamma/hadron** (binary) | `gamma/` | Hadron survival @ 75% γ eff ↓ | ~10⁻² ([ICRC 2021](https://arxiv.org/abs/2108.03407)) | 50 attempts |

Each task is self-contained with its own data pipeline, evaluation script, and agent instructions.

## How it works

1. Agent gets one task directory as its project root
2. Agent runs `download_data.py` to get the data
3. Agent reads `CLAUDE.md` (or `AGENTS.md`) for instructions
4. Agent builds classifiers, trains models, iterates
5. Each call to `verify.py` counts as one attempt (max 50), auto-logged to `results.tsv`
6. After 50 attempts, the best result is the agent's score

## Leaderboard: Mass Composition

| Rank | Frac Error ↓ | Agent | Model | Attempts used |
|------|-------------|-------|-------|---------------|
| ref | 0.107 | — | LeNet (36.6K params) | — |
| | | | | |

## Leaderboard: Gamma/Hadron

| Rank | Survival ↓ | Agent | Model | Attempts used |
|------|-----------|-------|-------|---------------|
| ref | ~10⁻² | — | RF regressor | — |
| | | | | |

## Quick start

```bash
# For composition task:
cd composition
uv sync
uv run python download_data.py
# ... build your classifier ...
uv run python verify.py predictions.npz "my approach description"

# For gamma task:
cd gamma
uv sync
uv run python download_data.py
uv run python verify.py predictions.npz "my approach description"
```

## For AI agents

Each task directory contains:
- `CLAUDE.md` / `AGENTS.md` — full instructions, data format, physics background, strategy hints
- `download_data.py` — downloads and prepares data from S3
- `verify.py` — evaluates predictions, auto-logs to `results.tsv`, enforces 50-attempt limit
- `baseline/` — published baseline reproduction (attempt 0)

Agents work within the task directory. They never need to see the parent.

## What makes this different

Most ML benchmarks ask "what's the best model?" We ask:

1. **Can AI agents beat human scientists?** — on the same data with the same evaluation
2. **How do agents search?** — every attempt is logged with a description
3. **How fast do they converge?** — Best@10, Best@20, Best@50 trajectory analysis
4. **How do different agents compare?** — Claude, GPT, Qwen, Kimi on the same tasks

## References

- [Methods of ML for cosmic rays mass composition](https://arxiv.org/abs/2311.06893) (Kuznetsov et al., JINST 2024)
- [Energy spectra of elemental groups of cosmic rays](https://arxiv.org/abs/2312.08279) (Kuznetsov et al., JCAP 2024)
- [New insights from old cosmic rays](https://arxiv.org/abs/2108.03407) (Kostunin et al., ICRC 2021)

## License

MIT
