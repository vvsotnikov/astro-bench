# Comprehensive Exploration Plan: All Dimensions

## Current Status
- **Best**: v1 @ 50.52% (Attention CNN + 8 features)
- **Target**: 50.71% (haiku-mar8) or 51% (published SOTA)
- **v17**: Failed (hung at epoch 20, didn't complete inference) — KILLED
- **GPU**: Using only GPU 1 (CUDA_VISIBLE_DEVICES=1) — NO parallelism
- **Timeout**: 2 hours per experiment (v17 was 4 blocks × 35s/epoch × 30 epochs ≈ 30 min, plus I/O)

---

## DIMENSION 1: DATA PROCESSING & FEATURE ENGINEERING

### Quality Cuts Strategy (TRAIN DATA)
- **v-qcut-train-all**: Apply quality cuts (Ze<30, Ne>4.8) to TRAINING data
  - Hypothesis: Cleaner training data = better generalization
  - vs v1: removes ~5-10% of train data

- **v-qcut-none**: Remove ALL quality cuts (even from test)
  - Hypothesis: Maybe quality cuts are removing useful edge cases
  - vs baseline: unfiltered training, no cuts on val/test

- **v-qcut-partial**: Apply cuts only to val, not train
  - Current v17 approach

### Feature Engineering Variants
- **v-feat-minimal**: Only 5 raw features (E, Ze, Az, Ne, Nmu)
  - Strip all engineered features
  - See if neural net learns better without hints

- **v-feat-expanded-12d**: Add more engineered features
  - E, Ze, Az, Ne, Nmu (raw)
  - Ne-Nmu, Ne/E, Nmu/E (ratios)
  - log(E), log(Ne+1), log(Nmu+1) (logs)
  - cos(Ze), sin(Ze), cos(Az), sin(Az) (angles)
  - Total: 16D features

- **v-feat-engineered-only**: Only engineered features, no raw
  - Hypothesis: engineered features have signal, raw don't

- **v-feat-selective**: Features selected by importance
  - Train baseline, extract feature importance
  - Keep only top-K features

### Normalization Strategies
- **v-norm-none**: No normalization
  - Let BatchNorm handle it

- **v-norm-log-first**: log1p BEFORE z-score
  - vs current: z-score directly

- **v-norm-per-channel**: Normalize each detector channel separately
  - vs current: global normalization

### Data Augmentation
- **v-aug-rotation**: 90° rotations (detector is rotationally symmetric)

- **v-aug-flip**: Horizontal/vertical flips

- **v-aug-noise**: Gaussian noise on matrices (σ=0.01, 0.05, 0.1)

- **v-aug-mixup**: Mixup on event pairs

- **v-aug-cutmix**: CutMix (cut patches from two events)

---

## DIMENSION 2: MODEL ARCHITECTURES

### Current CNN Variants
- v1: Attention CNN (3 blocks, 50.52%)
- v17: haiku-mar8 replica (4 blocks, running)
- v-cnn-6block: Even deeper (6 blocks)
- v-cnn-inception: Inception blocks instead of conv
- v-cnn-depthwise: Depthwise separable convolutions

### Vision Transformers (ViT)
- **v-vit-patch1**: 1×1 patches (one per pixel, 256 tokens)
  - Ultra-fine granularity

- **v-vit-patch2**: 2×2 patches (64 tokens)
  - Started as v20

- **v-vit-patch4**: 4×4 patches (16 tokens)
  - Original ViT size

- **v-vit-hybrid**: CNN backbone + ViT head
  - CNN extracts, ViT attends

### Graph Neural Networks (GNN)
- **v-gnn-node-per-cell**: 16×16 grid = 256 nodes
  - Edges: 4-connected neighborhood
  - Features: matrix value + electron + muon density

- **v-gnn-sparse-active**: Only active detectors are nodes
  - Edges: distance-based (k-NN or radius)
  - Features: detection strength
  - Hypothesis: sparsity matters (85% zeros)

- **v-gnn-multi-graph**: Separate graphs for e/γ and μ channels
  - Two graph inputs
  - Fuse at end

### U-Net / Encoder-Decoder
- **v-unet-standard**: Classic U-Net
  - Encoder: 3 blocks → bottleneck
  - Decoder: 3 blocks → classification head
  - Hypothesis: skip connections help

- **v-unet-deep**: Deeper U-Net (6 blocks)

### Capsule Networks
- **v-capsnet**: CapsuleNet for composition
  - 16×16×2 → capsule layers → routing
  - Hypothesis: capsules capture part-whole relationships

### Diffusion Models
- **v-diffusion-score**: Score-based diffusion
  - Train to denoise corrupted events
  - Use final state for classification

- **v-diffusion-latent**: VAE + diffusion
  - Compress to latent, diffuse, classify

### Hybrid Architectures
- **v-hybrid-cnn-gnn**: CNN on matrices + GNN on sparse features

- **v-hybrid-cnn-vit**: CNN initial + ViT refinement

- **v-hybrid-multi-path**: 3 paths (CNN, GNN, classical ML) → fusion

### Classical ML Baselines (Properly Implemented)
- **v-xgboost-safe**: XGBoost with 32 engineered features

- **v-catboost**: CatBoost (handles categoricals, raw features)

- **v-svm-rbf**: SVM with RBF kernel

- **v-svm-poly**: SVM with polynomial kernel

- **v-knn-weighted**: Weighted KNN (k=5, 10, 20)

---

## DIMENSION 3: TRAINING PIPELINES

### Loss Functions
- **v-loss-focal**: Focal loss (γ=2, 3, 4)
  - Focus on hard examples

- **v-loss-labelsmooth-variants**: Different smoothing (0.05, 0.1, 0.2)

- **v-loss-class-weighted**: Inverse frequency weighting per class

- **v-loss-margin**: Margin loss (push classes apart)

### Optimizers
- **v-opt-sgd**: SGD with momentum (0.9)

- **v-opt-sgd-nesterov**: SGD Nesterov

- **v-opt-adamw**: AdamW (current)

- **v-opt-lamb**: LAMB optimizer (large batch)

- **v-opt-adadelta**: Adadelta

### Learning Rate Schedules
- **v-lr-cosine**: Cosine annealing (current v1)

- **v-lr-onecycle**: OneCycleLR (current v17)

- **v-lr-linear**: Linear warmup → decay

- **v-lr-exponential**: Exponential decay

- **v-lr-plateau**: Reduce on plateau

### Batch Size Sensitivity
- **v-batch-512**: Very small batches

- **v-batch-1024**: Small

- **v-batch-2048**: Current v1

- **v-batch-4096**: Current v17

- **v-batch-8192**: Large

### Regularization
- **v-reg-dropout-high**: Dropout 0.5 everywhere

- **v-reg-dropout-low**: Dropout 0.1 only

- **v-reg-l1**: L1 weight penalty

- **v-reg-l2**: L2 weight penalty (current)

- **v-reg-mixup**: Mixup regularization

- **v-reg-cutout**: Random cutout of matrix patches

---

## DIMENSION 4: ENSEMBLE STRATEGIES

### Single-Architecture Ensembles
- **v-ensemble-5seed**: Average 5 models trained with different seeds

- **v-ensemble-snapshot**: Snapshots of training at different epochs

### Multi-Architecture Ensembles
- **v-ensemble-cnn-vit**: Weighted average (CNN + ViT)

- **v-ensemble-cnn-gnn**: Weighted average (CNN + GNN)

- **v-ensemble-cnn-classical**: Neural + XGBoost

- **v-ensemble-3best**: Top 3 performers from different families

### Stacking
- **v-stacking-meta-mlp**: Meta-learner MLP on top 3 models

- **v-stacking-meta-logistic**: Meta-learner logistic regression

---

## PRIORITY ORDER

### Phase A: QUICK WINS (Run in parallel if possible)
1. **v-qcut-train-all**: Apply cuts to train (maybe helps?)
2. **v-feat-expanded-12d**: More features (log, ratios)
3. **v-aug-rotation**: 90° rotations (symmetric detector)
4. **v-opt-sgd**: Try different optimizer family

### Phase B: ARCHITECTURE DIVERSITY (Run sequentially)
1. **v-gnn-sparse-active**: GNN on sparse detectors (novel approach)
2. **v-vit-patch1**: Ultra-fine ViT (different inductive bias)
3. **v-unet-standard**: Skip connections (different from CNN)
4. **v-hybrid-cnn-gnn**: Combine spatial + relational learning

### Phase C: ADVANCED (If Phase A/B don't beat 51%)
1. **v-capsnet**: Capsule networks (part-whole relationships)
2. **v-diffusion-score**: Generative approach
3. **v-ensemble-3best**: Combine winners

### Phase D: CLASSICAL ML (Parallel, fast)
1. **v-xgboost-safe**: Tree ensemble
2. **v-catboost**: Better tree model
3. **v-knn-weighted**: KNN baseline

---

## TESTING STRATEGY

For each architecture/approach:
1. **Variant 1**: Base version with current best hyperparameters
2. **Variant 2**: 2-3 key hyperparameter changes
3. **Variant 3**: Different data processing variant

Example for GNN:
- v-gnn-sparse-active (base)
- v-gnn-sparse-active-larger (bigger hidden dims)
- v-gnn-sparse-active-qcut (with quality cuts)

---

## SUCCESS METRICS

- **Minimum beat v1**: 50.52% → 50.60%
- **Beat haiku-mar8**: 50.52% → 50.71%
- **Beat published SOTA**: 50.52% → 51%+
- **Find complementary model**: Any model uncorrelated with v1 (for ensemble)

---

## Hypothesis Summary

**Data Processing**:
- Quality cuts on train may help (cleaner signal)
- More engineered features may capture more patterns
- Augmentation helps with 5.5M training examples

**Architectures**:
- CNNs good at spatial patterns (edges, shapes)
- ViT good at global structure (long-range dependencies)
- GNN good at relational structure (topology matters)
- Hybrid approaches combine strengths

**Training**:
- Different optimizers find different minima
- Ensemble diversity > single model

**Target**: Beat 50.71% using combination of:
- Better data processing
- Novel architecture (GNN or ViT)
- Ensemble of diverse models
