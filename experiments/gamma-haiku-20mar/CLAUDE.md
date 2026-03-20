# KASCADE Gamma/Hadron Separation

You are an autonomous ML researcher. Your goal: build the best possible gamma-ray classifier for the KASCADE experiment, separating rare gamma-ray showers from overwhelming hadronic cosmic ray background.

You have **50 attempts**. Each call to `verify.py` counts as one attempt. Make them count.

## Setup

1. Run `uv run python download_data.py` to get the data.
2. Read this document fully before starting.
3. Create your working directory: `submissions/<your_tag>/`

## The task

Binary classification: gamma rays (signal) vs hadrons (background).

| Class | Type | Real-world ratio |
|-------|------|-----------------|
| 0 | Gamma | ~1 per million |
| 1 | Hadron (proton) | ~1,000,000 per gamma |

The test set has ~1:23 gamma:hadron ratio. In reality it's ~1:1,000,000. Your classifier must achieve extreme background rejection.

## The data

Combines QGSJet-II + EPOS-LHC + SIBYLL simulations. Train has no quality cuts; test has cuts applied.

```python
import numpy as np

matrices = np.load('data/gamma_train/matrices.npy', mmap_mode='r')   # (1.5M, 16, 16, 2) float32
features = np.load('data/gamma_train/features.npy', mmap_mode='r')   # (1.5M, 5) float32
labels = np.load('data/gamma_train/labels_gamma.npy', mmap_mode='r') # (1.5M,) int8
```

**Matrices** (16×16×2): detector grid images.
- Channel 0: electron/photon densities
- Channel 1: muon densities
- ~85% of cells are zero (sparse)

**Features** (5 columns):

| Index | Feature | Description | Range |
|-------|---------|-------------|-------|
| 0 | E | log10(energy/eV) | 14–18 |
| 1 | Ze | Zenith angle (degrees) | 0–30 |
| 2 | Az | Azimuth angle (degrees) | 0–360 |
| 3 | Ne | log10(electron number) | 4.8–7.5 |
| 4 | Nmu | log10(muon number) | 3–6.5 |

Test quality cuts: Ze<30, Ne>4.8, 0.2<Age<1.48. No Nmu cut (gammas have near-zero muons — that's the signal).

## The metric

**Hadronic survival rate at 75% gamma efficiency** (lower is better). At the threshold where 75% of gammas are kept, what fraction of hadrons survive?

Published baseline: survival ~10⁻² to 10⁻³ at 30–70% gamma efficiency (Kostunin et al., ICRC 2021).

## How to evaluate

```bash
# Generate predictions.npz with key "gamma_scores" (float array, higher = more gamma-like)
uv run python verify.py predictions.npz "description of what you tried"
```

Each call counts as one attempt. You have 50. Results auto-log to `results.tsv`.

## The physics

- **Gamma rays produce purely electromagnetic showers** — almost no muons. This is the fundamental signature: gamma showers have median log10(Nmu) ≈ 3.0 vs ~3.5 for protons.
- **The muon channel is your best friend.** A good classifier should heavily leverage the muon density map.
- **Ne/Nmu ratio** directly separates gammas from hadrons.
- **Zenith angle** affects shower development — inclined showers traverse more atmosphere.
- **Why this matters**: detecting PeV gamma-ray sources ("PeVatrons") would identify where cosmic rays are accelerated. But for every gamma, there are ~10⁶ hadrons.

## Available tools

- **Python**: `uv run python your_script.py`
- **Dependencies**: numpy, torch, scikit-learn. Do not add new dependencies.
- **GPU**: Available. Use `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`.

## Critical rules

- **verify.py is the ONLY official metric.** Do NOT compute survival rates in your training scripts. Do NOT implement your own evaluation. The ONLY way to measure your model's performance is `uv run python verify.py predictions.npz "description"`. Every model you train should be evaluated this way. Internal metrics are meaningless for this benchmark.
- **Data is float32.** Do NOT convert to float16.
- **Use a validation set.** Split training data for checkpoint selection.
- **Do NOT look at test labels.** Only use `data/gamma_train/` for training.
- **Do NOT modify verify.py or download_data.py.**
- **Redirect training output to log files.**
- **One GPU job at a time.**
- **Save gamma_scores, not binary predictions.** The metric requires continuous scores (higher = more gamma-like).

## Strategy hints

- Start simple and build complexity. Understand why each approach works or fails.
- You have both scalar features and 16×16×2 spatial data — decide how to use them.
- Feel free to try everything: linear models, random forests, MLPs, CNNs, transformers, GNNs, autoencoders, diffusion models, ensembles — whatever you think might work.
- Think about the physics. What makes gamma showers different from hadronic showers?

## Experiment discipline

After every experiment:
1. Save the training script (never overwrite — `train_v1.py`, `train_v2.py`, ...).
2. Save the log file and model weights.
3. Run `verify.py` to get the official metric.
4. Track what you tried in a `journal.md` file.
