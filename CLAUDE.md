# astro-bench — Paper Writing & Orchestration

You are helping write a scientific paper about AI agents as autonomous ML researchers on the KASCADE cosmic ray dataset. You also help orchestrate agent experiments.

## Project structure

- `composition/` — 5-class mass composition task (template)
- `gamma/` — binary gamma/hadron separation task (template)
- `experiments/` — agent run directories (one per agent × task)
- `create_experiment.sh` — creates new experiment dirs with symlinked data
- `prompts.md` — standardized agent launch prompts
- `paper/` — LaTeX paper and figures
- `internal/` — legacy code, agent logs, analysis (not public)
- `legacy/` — original KASCADE notebooks
- `old/` — deprecated files from v1 of the benchmark
- `data/` — raw data files (shared, not copied per experiment)

## The paper

The paper analyzes how different AI agents (Claude Opus, Sonnet, Haiku; GPT Codex; Qwen; Kimi) approach ML research tasks on real astrophysics data.

Key points:
- Each agent gets 50 attempts to beat the published baseline
- The metric is "Best @ 50" — best result achieved within the budget
- We analyze convergence speed, architecture diversity, failure modes
- Two tasks: composition (5-class) and gamma/hadron (binary)
- Published baselines from Kuznetsov et al. (JINST 2024) and Kostunin et al. (ICRC 2021)

## Working with the paper

- Paper source: `paper/main.tex`
- Figures: `paper/fig_*.pdf` and generation scripts `paper/generate_fig_*.py`
- Agent results: `experiments/<task>-<agent>/results.tsv`
- Use `uv run` for all Python commands

## Launching agent experiments

```bash
# Create experiment directory
./create_experiment.sh gamma haiku-20mar

# Launch agent (use prompts from prompts.md)
```

## Critical rules for orchestration

- NEVER send shutdown requests to agents before they've used all 50 attempts
- NEVER modify files in an active experiment directory
- Let agents run to completion or context exhaustion
