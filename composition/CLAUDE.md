# KASCADE Mass Composition Classification

You are an autonomous ML researcher. Your goal: build the best possible 5-class cosmic ray classifier for the KASCADE experiment dataset.

You have **50 attempts**. Each call to `verify.py` counts as one attempt. Make them count.

You are fully autonomous. Make your own decisions about what to try next. Do NOT ask for permission or strategy guidance. Do NOT write shell orchestration scripts, monitoring scripts, or polling loops. Your workflow is simple: write ONE training script → run it → wait for it to finish → look at the result → decide what to try next → repeat. Each script must be fully self-contained — do NOT depend on outputs from other scripts.

50 attempts means 50 calls to `evaluate()` — NOT 50 scripts. Write a script, run it, evaluate the result, then decide the next experiment based on what you learned.

## Setup

1. Run `uv run python download_data.py` to get the data (if `data/` doesn't exist).
2. Read this document fully before starting.
3. Write all your scripts in this directory (next to `load_data.py` and `verify.py`).

## The task

Classify cosmic ray primary particles into 5 mass groups:

| Class | Particle | Description |
|-------|----------|-------------|
| 0 | Proton (H) | Lightest, fewest muons |
| 1 | Helium (He) | |
| 2 | Carbon (C) | |
| 3 | Silicon (Si) | |
| 4 | Iron (Fe) | Heaviest, most muons |

## The data

Pre-split `.npy` files from QGSJet-II.04 simulation. Quality cuts pre-applied to both train and test (Ze<30, Ne>4.8, Age∈(0.2,1.48), Nmu>3.6). Same data and splits as the published baseline (Kuznetsov et al., JINST 2024).

```python
from load_data import load_train, load_test

matrices, features, labels = load_train()   # (268K, 16,16,2) float32, (268K, 6) float32, (268K,) int8
X_test, f_test, y_test = load_test()        # (115K, 16,16,2), (115K, 6), (115K,)
```

Use `load_data.py` to load data — it resolves paths correctly regardless of where your script runs from.

**Matrices** (16×16×2): detector grid images.
- Channel 0: electron/photon densities
- Channel 1: muon densities
- ~85% of cells are zero (sparse)

**Features** (6 columns):

| Index | Feature | Description | Range |
|-------|---------|-------------|-------|
| 0 | E | log10(energy/eV) | 14–18 |
| 1 | Ze | Zenith angle (degrees) | 0–30 |
| 2 | Az | Azimuth angle (degrees) | 0–360 |
| 3 | Ne | log10(electron number) | 4.8–7.5 |
| 4 | Nmu | log10(muon number) | 3.6–6.5 |
| 5 | Age | Shower age | 0.2–1.48 |

**Published feature normalization** (from the baseline model):
- Ne: `(Ne - 5.31) / 0.5`
- Nmu: `(Nmu - 4.3) / 0.42`
- Age: `Age - 1.0`
- Ze: `Ze / 60.0`

## The metric

**Mean fraction error** (lower is better). Measures how well your classifier recovers true particle fractions across 1001 random mixture compositions of 5000 events each.

Published baseline: **0.107** (LeNet CNN, 36.6K params). Your target: beat this.

The baseline reproduction is in `baseline/` (attempt 0 in results.tsv).

## How to evaluate

From the command line:
```bash
uv run python verify.py predictions.npz "description of what you tried"
```

Or from your training script (recommended — no need to save predictions.npz first):
```python
from verify import evaluate
metric = evaluate(predictions, "v3: CNN + attention, seed=42")
```

Each call counts as one attempt. You have 50. Results auto-log to `results.tsv`.

## The physics

- **Ne/Nmu ratio** is the strongest discriminant. Iron nuclei produce ~3× more muons than protons at the same energy.
- **Zenith angle** affects shower development through atmosphere.
- **Energy** is reconstructed from Ne and Nmu, so not independent.
- **The hard part**: neighboring mass groups overlap heavily (H/He and Si/Fe). Most errors come from these confusions.

## Available tools

- **Python**: `uv run python your_script.py`
- **Dependencies**: numpy, torch, scikit-learn. Do not add new dependencies.
- **GPU**: Available. Use `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (GPU 0 may be occupied).

## Critical rules

- **verify.py is the ONLY official metric.** Do NOT compute fraction error in your training scripts. Do NOT implement your own evaluation. The ONLY way to measure your model's performance is `uv run python verify.py predictions.npz "description"`. Every model you train should be evaluated this way. Internal metrics are meaningless for this benchmark.
- **Data is float32.** Do NOT convert to float16.
- **Use a validation set.** Split your training data into train/val for checkpoint selection. Do NOT select checkpoints based on test metrics.
- **Do NOT look at test labels.** Only use `data/composition_train/` for training.
- **Do NOT modify verify.py or download_data.py.**
- **Do NOT use differential evolution (DE) bias optimization.** Biases tuned on balanced simulation distort real data. All predictions must be raw argmax.
- **Redirect training output to log files**: `uv run python train.py > run.log 2>&1`. Do NOT flood your context.
- **One GPU job at a time.** Call `check_gpu_free()` from `load_data` before training. NEVER run multiple training scripts simultaneously — they will crash or produce bad results. Wait for each to finish before starting the next.
- **Training can take up to 24 hours.** This is normal. Do NOT kill long-running training jobs. Do NOT write polling/monitoring scripts. Just wait and check the output after it finishes.

## Strategy hints

- Start simple and build complexity. Understand why each approach works or fails.
- You have both scalar features and 16×16×2 spatial data — decide how to use them.
- Feel free to try everything: linear models, random forests, MLPs, CNNs, transformers, GNNs, autoencoders, diffusion models, ensembles — whatever you think might work.
- Think about the physics. What distinguishes light from heavy cosmic ray primaries?
- When stuck, look at the confusion matrix to understand where errors come from.

## Experiment discipline

After every experiment:
1. Save the training script (never overwrite — create new files: `train_v1.py`, `train_v2.py`, ...).
2. Save the log file.
3. Save model weights and predictions.
4. Run `verify.py` to get the official metric.
5. Decide: keep or discard. Move on.

Track what you tried and why in a `journal.md` file. This is your external memory.
