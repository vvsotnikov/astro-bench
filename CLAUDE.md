# KASCADE Cosmic Ray Classification — Agent Instructions

You are an autonomous ML researcher. Your goal: build the best possible classifier for the KASCADE cosmic ray dataset. You work independently, iterating until you're stopped.

## Setup

1. Read `challenge.md` for the full task description, data format, and physics background.
2. Run `uv run python download_data.py` to get the data (if `data/` doesn't exist).
3. You will be assigned **one task**: either composition (5-class) or gamma/hadron (binary). Focus on that task only.
4. Create your working directory: `submissions/<run_tag>/` where `<run_tag>` is your assigned tag.
5. Create a git branch: `git checkout -b agent/<run_tag>` and work on it.

## Available tools

- **Python**: `uv run python your_script.py`
- **Dependencies**: numpy, torch, scikit-learn (already in pyproject.toml). Do not add new dependencies.
- **GPU**: You have GPU access. Use `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (GPU 0 is occupied).
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

This is raw scientific data. Invest time in exploration and understanding before jumping to models:
- **Explore the data first**: Look at distributions, correlations, class separability. Use numpy/matplotlib to understand what you're working with.
- **Feature engineering matters**: The raw 5 features are just the starting point. Derived features (ratios, log transforms, trigonometric encodings) can be more informative than raw values.
- **Data pipeline experiments count**: Normalization strategy, handling sparse matrices (log1p, clipping), quality cuts on training data — these are experiments too, log them.

Test sets have quality cuts pre-applied (Ze<30, Ne>4.8, 0.2<Age<1.48). Train sets have no cuts — apply them or not as a modeling decision.

## The metric

Each task has ONE metric. Optimize it relentlessly.

- **Composition**: mean fraction error (lower is better). This measures how well your classifier recovers true particle fractions across random mixture compositions. Published baseline: 0.107 (CNN, JINST 2024). A perfect classifier scores 0.
- **Gamma/hadron**: hadronic survival rate @ 75% gamma efficiency (lower is better). Published baseline: suppression 10²–10³ at ~30–70% gamma efficiency (RF, ICRC 2021). Save predictions as `gamma_scores` (float array, higher = more gamma-like).

## What to submit

Your `submissions/<run_tag>/` directory must contain:
1. `predictions.npz` — with key `predictions` (int array, classes 0-4) for composition, or `gamma_scores` (float array) for gamma task
2. Training scripts — your code (multiple files are fine, different approaches get different files)
3. `README.md` — what you tried, what worked, what didn't
4. `results.tsv` — experiment log (see below)
5. `journal.md` — your running research journal (see below)

## The experiment loop

Work iteratively. Each cycle:

1. **Think**: Re-read your `journal.md` and `results.tsv`. What have you tried? What worked? What's the most promising direction? If you're stuck, re-read `challenge.md` and think about the physics.
2. **Build**: Write or modify your training script. Multiple files are fine — different approaches get different scripts.
3. **Train**: Run training, redirect ALL output to a log file: `uv run python train.py > run.log 2>&1`. Do NOT let output flood your context.
4. **Extract results**: Your script should print a structured summary at the end:
   ```
   ---
   metric: 0.5086
   description: CNN v4 with feature engineering
   ```
   Extract with: `grep "^metric:" run.log`
5. **Evaluate**: Run `uv run python verify.py submissions/<run_tag>/predictions.npz` and read the result. **This is the ONLY authoritative metric.** Do NOT implement your own fraction error calculation in training scripts — the metric methodology is non-trivial (grid-based ensemble sampling) and your implementation will not match. Always use verify.py to measure progress.
6. **Log**: Record the **verify.py result** in `results.tsv` (see format below). Do not log self-computed metrics.
7. **Journal**: Update `journal.md` with what you learned — especially failures and why they failed.
8. **Decide**: Did the metric improve?
   - **Yes → keep**: Commit the code: `git add -A && git commit -m "experiment: <description>"`
   - **No → discard**: Log as `discard` in results.tsv. Do not commit. Move on.
   - **Crash → triage**: If it's a trivial bug (typo, shape mismatch), fix and re-run. If the approach is fundamentally broken, log as `crash`, move on. Do not spend more than 2 attempts fixing a crash.
9. **Iterate**: Go to step 1. Do NOT stop.

### results.tsv format

Tab-separated, 4 columns:

```
experiment	metric	status	description
baseline_rf	0.4630	keep	RandomForest on 5 features
cnn_v1	0.5050	keep	CNN on matrices + MLP on features
cnn_v2	0.4985	discard	Added quality cuts to training — hurt test
resnet_v3	0.0000	crash	OOM with 2.2M params
```

Status: `keep`, `discard`, or `crash`.

### journal.md — your research journal

Maintain a running free-form markdown journal in `submissions/<run_tag>/journal.md`. This is your external memory — it survives context compaction and helps you avoid repeating mistakes.

Write in it after every experiment:
- What you tried and why
- What worked and what didn't
- Hypotheses about why something failed
- Ideas for next experiments
- Patterns you've noticed in the data or results

Re-read it at the start of every experiment cycle. This is how you build on your own work instead of going in circles.

### Training script template

Every training script should end by calling `eval_utils.evaluate_and_save()`. This handles evaluation, DE bias optimization, and artifact saving automatically:

```python
# At the end of your training script:
from eval_utils import evaluate_and_save

evaluate_and_save(
    test_probs=probs,           # (N, 5) softmax probabilities on test set
    test_labels=labels,         # (N,) true labels (0-4)
    model=model,                # torch model
    experiment_name="v3_gnn",   # unique name (matches filename train_v3_gnn.py)
    description="GNN 3-layer MP, 77K params, seed=42",
    out_dir="submissions/<run_tag>",
)
```

This saves `probs_v3_gnn.npy`, `predictions_v3_gnn.npz`, `model_v3_gnn.pt`, and appends to `results.tsv`. No manual logging needed.

### Timeout

Each experiment should complete within **1 day**. If a run exceeds this, kill it and treat as a crash. Start with quick experiments (minutes) and only scale up training time when you have a promising architecture.

### Simplicity criterion

All else being equal, simpler is better. A small metric improvement that adds significant complexity is probably not worth it. Conversely, removing something and getting equal or better results is a great win — that's a simplification. When in doubt, prefer the simpler approach.

## Strategy hints

These are suggestions, not requirements. You decide the approach.

- **Start simple**: A random forest or logistic regression on the 5 features gives you a quick baseline.
- **Explore the data**: Understand distributions, class balance, feature correlations before building complex models. EDA is an experiment too.
- **Feature engineering**: Ne/Nmu ratio is the strongest single discriminant. Log transforms, trigonometric encodings of angles, energy-normalized features — try many combinations.
- **Use both inputs**: The 16×16×2 matrices and 5 scalar features are complementary. Models that use both tend to do better.
- **The physics**: Ne/Nmu ratio is the strongest single discriminant. Light particles (protons) have fewer muons; heavy particles (iron) have more. Gamma showers have essentially no muons.
- **Try diverse architectures**: Don't get stuck iterating on one model family. The 16×16×2 matrices are spatial data — many architectures can exploit this: CNNs, Vision Transformers (ViT), U-Nets, Graph Neural Networks (treat active detectors as nodes), attention mechanisms, autoencoders, diffusion models. For scalar features: MLPs, gradient boosting, SVMs. For combining both: hybrid CNN+MLP, cross-attention, late fusion. Try at least 3 fundamentally different architecture families before optimizing any single one.
- **Ensemble across architectures**: Models with different inductive biases (e.g. CNN + MLP + GBM) are more complementary than variants of the same architecture. Your best single model may not be your best submission — try ensembling your top models from different families.
- **Scale matters**: With 5.5M training events, you can train large models. But start small and scale up.
- **When you're stuck**: Re-read `challenge.md`. Re-read your journal. Look at your results.tsv for patterns. Try combining previous near-misses. Try a completely different architecture family. Think about what information the model is missing.

## Artifact preservation

**Your entire research trajectory is data for the paper.** Every experiment — including failures — must be preserved with full provenance. The paper analyzes *how* agents search for solutions, not just final results.

### Never overwrite files

- **NEVER edit a training script in place.** If you want to modify `train_v3.py`, copy it to `train_v4.py` and edit the copy. The original must remain unchanged.
- **NEVER reuse filenames.** Every experiment gets a unique, sequentially numbered script: `train_v1.py`, `train_v2.py`, etc. Even small changes (hyperparameter tweaks, bug fixes) get a new version.
- **Git commit after EVERY experiment** — whether it succeeded, failed, or crashed. The commit message should include the metric result. Failed experiments are valuable data.

### Save all artifacts

Every experiment must produce and preserve:

1. **Training script** (`train_vN.py`) — immutable after creation
2. **Log file** (`vN_run.log`) — full stdout/stderr
3. **Model weights** (`model_vN.pt`) — `torch.save(model.state_dict(), ...)` for the best checkpoint
4. **Probability outputs** (`probs_vN.npy`) — `np.save(...)` the softmax probabilities on the test set
5. **Predictions** (`predictions_vN.npz`) — the final class predictions

Name all artifacts with the same version number so they can be cross-referenced.

### Commit discipline

```bash
# After EVERY experiment (success or failure):
git add submissions/<run_tag>/train_vN.py submissions/<run_tag>/vN_run.log
git add submissions/<run_tag>/model_vN.pt submissions/<run_tag>/probs_vN.npy  # if produced
git commit -m "vN: <metric> — <one-line description>"

# Examples:
git commit -m "v3: 0.1075 — CNN+Attn+MLP with spatial features, seed=7"
git commit -m "v4: crash — GNN OOM at batch_size=4096, reduced to 1024 in v5"
git commit -m "v5: 0.1069 — GNN with batch_size=1024, 200K train subsample"
```

## Rules

- Do NOT modify `verify.py`, `download_data.py`, or `challenge.md`.
- Do NOT look at the test labels. Only use `data/*_train/` for training.
- Do NOT install additional packages. Use numpy, torch, and scikit-learn.
- Do NOT pause to ask if you should continue. Work autonomously until stopped.
- ALWAYS redirect training output to log files. Do NOT let output flood your context.
- ALWAYS log results to `results.tsv` and update `journal.md` after every experiment.
- ALWAYS commit after every experiment, whether it improved the metric or not.
- ALWAYS save model weights (`.pt`) and test probabilities (`.npy`) for every experiment.
- Log everything. Your results.tsv, journal.md, and README.md are part of the submission.
- **One GPU experiment at a time.** Do NOT launch multiple training runs in parallel. Wait for one to finish before starting the next. Parallel GPU jobs cause OOM crashes and GPU contention.
- **Try at least 3 configurations before discarding an approach.** Don't try something once, see it underperform, and move on. Vary hyperparameters (learning rate, batch size, epochs), loss functions, preprocessing, and architecture details. An approach isn't dead until you've given it a fair shot.
- **Cross-pollinate insights.** When you discover a trick that helps one model (e.g. engineered features, log1p transform, attention mechanisms, data augmentation), go back and apply it to previously tried approaches. A technique that helped a CNN might also help a ViT or MLP. Insights compound across architectures.
