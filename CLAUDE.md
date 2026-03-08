# KASCADE Cosmic Ray Classification — Agent Instructions

You are an autonomous ML researcher. Your goal: build the best possible classifier for the KASCADE cosmic ray dataset. You work independently, iterating until you're stopped.

## Setup

1. Read `challenge.md` for the full task description, data format, and physics background.
2. Run `uv run python download_data.py` to get the data (if `data/` doesn't exist).
3. Pick a task to work on (start with **composition** — 5-class mass classification).
4. Create your working directory: `submissions/<run_tag>/` where `<run_tag>` is today's date (e.g. `mar8`).

## Available tools

- **Python**: `uv run python your_script.py`
- **Dependencies**: numpy, torch, scikit-learn (already in pyproject.toml). Do not add new dependencies.
- **GPU**: You have GPU access. Use `CUDA_VISIBLE_DEVICES=0` if needed.
- **Verification**: `uv run python verify.py submissions/<run_tag>/predictions.npz`

## The data

Pre-split, memory-mappable `.npy` files:

```python
import numpy as np

# Task 1: Mass composition (5-class)
X_train = np.load('data/composition_train/matrices.npy', mmap_mode='r')   # (5.5M, 16, 16, 2) float16
f_train = np.load('data/composition_train/features.npy', mmap_mode='r')   # (5.5M, 5) float16
y_train = np.load('data/composition_train/labels_composition.npy', mmap_mode='r')  # (5.5M,) int8

# Task 2: Gamma/hadron (binary)
X_train = np.load('data/gamma_train/matrices.npy', mmap_mode='r')         # (1.5M, 16, 16, 2) float16
y_train = np.load('data/gamma_train/labels_gamma.npy', mmap_mode='r')     # (1.5M,) int8
```

Features: E (energy), Ze (zenith), Az (azimuth), Ne (electron number), Nmu (muon number).
Matrix channels: electron/photon density + muon density on a 16×16 detector grid (~85% zeros, sparse).

Test sets have quality cuts pre-applied (Ze<30, Ne>4.8, 0.2<Age<1.48). Train sets have no cuts — apply them or not as a modeling decision.

## What to submit

Your `submissions/<run_tag>/` directory must contain:
1. `predictions.npz` — with key `predictions` (int array, classes 0-4) for composition, or `gamma_scores` (float array) for gamma task
2. `train.py` (or equivalent) — your training code
3. `README.md` — what you tried, what worked, what didn't

## The experiment loop

Work iteratively. Each cycle:

1. **Think**: What architecture/approach should I try? Consider the data structure (small 16×16 images + scalar features).
2. **Build**: Write or modify your training script.
3. **Train**: Run training, redirect output to a log file: `uv run python train.py > run.log 2>&1`
4. **Evaluate**: Run `uv run python verify.py submissions/<run_tag>/predictions.npz` and read the accuracy.
5. **Log**: Record the result in `submissions/<run_tag>/results.tsv`:
   ```
   experiment	accuracy	description
   baseline_rf	0.2134	RandomForest on 5 features only
   cnn_v1	0.3891	Simple CNN on 16x16x2 matrices
   ```
6. **Iterate**: If accuracy improved, keep the code. If not, revert and try something else.

## Strategy hints

These are suggestions, not requirements. You decide the approach.

- **Start simple**: A random forest or logistic regression on the 5 features gives you a quick baseline.
- **Use both inputs**: The 16×16×2 matrices and 5 scalar features are complementary. Models that use both tend to do better.
- **The physics**: Ne/Nmu ratio is the strongest single discriminant. Light particles (protons) have fewer muons; heavy particles (iron) have more.
- **Architecture ideas**: CNNs for the spatial data, concatenated with an MLP for scalar features. Or flatten everything into a single MLP. Or try something creative.
- **Scale matters**: With 5.5M training events, you can train large models. But start small and scale up.
- **Known baselines**: Dense NN on flattened matrices + 2 features → ~44-47% accuracy. You should aim to match or beat this.

## Rules

- Do NOT modify `verify.py`, `download_data.py`, or `challenge.md`.
- Do NOT look at the test labels. Only use `data/*_train/` for training.
- Do NOT install additional packages. Use numpy, torch, and scikit-learn.
- Do NOT pause to ask if you should continue. Work autonomously until stopped.
- Log everything. Your results.tsv and README.md are part of the submission.
