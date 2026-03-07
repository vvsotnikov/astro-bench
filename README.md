# astro-agents

**Challenge:** Build the best cosmic ray mass composition classifier using data from the [KASCADE experiment](https://web.ikp.kit.edu/KASCADE/).

This is a benchmark for AI agents and humans building ML models for a real astrophysics problem. KASCADE measured extensive air showers produced by high-energy cosmic rays for ~25 years in Karlsruhe, Germany. The task: classify the primary particle (proton, helium, carbon, silicon, or iron) from shower measurements.

A team of physicists spent years developing and refining classification models through manual iteration and domain expertise. **Can AI agents match or exceed these results — and if so, in how many hours of compute vs. years of human effort?**

Read the full challenge description: [challenge.md](challenge.md)

## Leaderboard

### Agent-Built Solutions

| Rank | Accuracy | Macro F1 | Author | Built with | Architecture | Key Tricks | Cost | Tokens | Time | Link |
|------|----------|----------|--------|------------|-------------|------------|------|--------|------|------|
| | | | | | | | | | | |

### Human-Built Solutions

| Rank | Accuracy | Macro F1 | Author | Architecture | Key Tricks | Link |
|------|----------|----------|--------|-------------|------------|------|
| 0 | TBD | TBD | Sotnikov, Petrov et al. | Attention MLP | digitized matrices, quality cuts | [JCAP 2024](https://doi.org/10.1088/1475-7516/2024/08/025) |

## Quick Start

```bash
# Install the minimal harness (just numpy, sklearn, boto3 for data + verification)
uv sync

# Download KASCADE data from S3
uv run python download_data.py

# Run verification on a submission
uv run python verify.py submissions/my_submission/predictions.npz
```

## How It Works

1. `download_data.py` fetches KASCADE simulation data from S3 and creates a fixed train/test split
2. You (or your agent) build a classifier using **any tools, languages, or frameworks you want**
3. Produce `predictions.npz` with integer class labels (0-4) for the test set
4. `verify.py` scores your predictions and outputs metrics

**No constraints on how you get there.** Use PyTorch, TensorFlow, JAX, raw CUDA, C++, Rust, sklearn, XGBoost — whatever works. The harness only checks your output.

## How to Submit

**Option A: Open an Issue**
1. Click New Issue and describe your submission
2. Include a link to your code (GitHub repo, gist, etc.)
3. Include `verify.py` output
4. We'll verify and add you to the leaderboard

**Option B: Open a Pull Request**
1. Fork this repo
2. Add your submission to `submissions/your_name/`
3. Include: source code, `predictions.npz`, `README.md` documenting your approach
4. If agent-built: include `metadata.yaml` with cost/tokens/time and reasoning traces
5. Include `verify.py` output

## What Makes This Different

Most ML benchmarks ask "what's the best model for X?" We also ask:
- **Which AI agent builds the best model?** (Claude, GPT, Gemini, open-source, ...)
- **How does the agent approach the problem?** (architecture search, feature engineering, ensembles, ...)
- **What does it cost?** (API cost, tokens, wall-clock time, GPU hours)
- **What did it try and fail?** (full reasoning traces tell a richer story than final metrics)

The leaderboard tracks both **what was achieved** and **how** — making this simultaneously a model benchmark and a study of AI agents as autonomous ML researchers.

## The Data

KASCADE simulation data in two formats:
- **16x16 detector grid images** (3 channels: arrival time, electron density, muon density)
- **Tabular observables** (energy, zenith angle, shower size, muon number, shower age, etc.)
- **Three hadronic interaction models**: QGSJet-04, EPOS-LHC, SIBYLL-2.3c

Full description in [challenge.md](challenge.md).

## Context

This project is part of research on **harness engineering for scientific domains** — can learned evaluation functions guide AI agents to build better ML models for problems where ground truth requires domain expertise?

Related work:
- [AI Agents for Ground-Based Gamma Astronomy](https://arxiv.org/abs/2503.00821) (Kostunin, Sotnikov et al., 2025)
- [Agent-based code generation for Gammapy](https://arxiv.org/abs/2509.26110) (Kostunin, Sotnikov et al., 2025)
- [Energy spectra of cosmic rays with KASCADE and ML](https://doi.org/10.1088/1475-7516/2024/08/025) (Kuznetsov, Petrov, Plokhikh, Sotnikov, JCAP 2024)
- [Addition Under Pressure](https://dimitrisp.substack.com/p/addition-under-pressure) (Papailiopoulos, 2026) — inspiration for the agent benchmark format
- [OpenHands: Learning to Verify AI-Generated Code](https://openhands.dev/blog/20260305-learning-to-verify-ai-generated-code) — learned critic models for agent evaluation

## License

MIT
