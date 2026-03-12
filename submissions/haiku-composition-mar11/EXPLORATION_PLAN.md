# Systematic Exploration Plan for Mass Composition (5-class)

## Current Status
- **Baseline to beat**: haiku-mar8 @ 50.71%
- **Best so far**: v1 @ 50.52% (attention CNN + 8 features)
- **Total experiments**: 8 (v1-v8, with v5-v8 crashes, v6b/v7b partial)

## CATEGORIZED APPROACHES

### CATEGORY A: DATA PIPELINE / PREPROCESSING (UNTESTED)
These are fundamentally different ways to prepare input data.

#### A1: No normalization (raw values)
- Try raw matrices without z-score normalization
- Hypothesis: BatchNorm handles it automatically, normalization might lose signal

#### A2: Per-channel normalization
- Normalize each channel (electron, muon) independently
- Current approach: global normalization on all values

#### A3: Log1p with better numerical stability
- log1p without infinity issues
- Add clipping before log: log1p(np.clip(mat, 0, max_val))

#### A4: Matrix scaling by event energy
- Divide matrices by E^p (energy normalization)
- Hypothesis: spatial patterns scale with energy

#### A5: Sparse matrix handling
- Use sparse tensor representations
- Apply sparsity-aware operations

#### A6: Augmentation strategies
- Random 90° rotations (has physics meaning - detector is symmetric)
- Flips and crops
- Mixup/CutMix on event pairs

### CATEGORY B: MODEL ARCHITECTURE (PARTIALLY TESTED)

#### B1: CNN Depth/Width Variations (NEED 3+ TRIALS)
Current: 3 blocks (32→64→128)
- B1a: haiku-mar8 exact replica (4 blocks: 32→32→64→64→128→128→256) [v9 running]
- B1b: 5 blocks (32→64→64→128→128→256) [NOT TRIED]
- B1c: Wider 3-block (32→64→128→256) [NOT TRIED]
- B1d: MobileNet-style (depthwise separable) [NOT TRIED]

#### B2: Attention Mechanisms (NEED 3+ TRIALS)
Current: Self-attention in CNN
- B2a: Current attention blocks [v1]
- B2b: Multi-head attention (4 heads) [NOT TRIED]
- B2c: Channel attention (SE-Net style) [NOT TRIED]
- B2d: No attention baseline [v2 @ 49.47%]

#### B3: Non-CNN Neural Architectures (NEED 3+ TRIALS)
Current: Only 1 attempt each
- B3a: Vision Transformer (4×4 patches) [v12 running]
- B3b: Vision Transformer (8×8 patches) [NOT TRIED]
- B3c: Vision Transformer (2×2 patches) [NOT TRIED]
- B3d: Pure MLP on flattened matrices [v11 running]
- B3e: MLP on downsampled matrices [NOT TRIED]
- B3f: U-Net (encoder-decoder) [NOT TRIED]
- B3g: Autoencoder + head [NOT TRIED]

#### B4: Tree-Based Models (NEED 3+ TRIALS)
Current: All crashed with infinity issues (v6, v6b, v8)
- B4a: RandomForest with safe preprocessing [NOT PROPERLY TRIED]
- B4b: GradientBoosting (sklearn) [NOT TRIED]
- B4c: LightGBM [NOT TRIED]
- B4d: HistGradientBoosting [NOT TRIED]

#### B5: Other Classical ML (NEED 3+ TRIALS)
Current: Only logistic (v7b @ 37.99%)
- B5a: Logistic Regression [v7b @ 37.99%]
- B5b: SVM (RBF kernel) [v13 running]
- B5c: SVM (poly kernel) [NOT TRIED]
- B5d: KNN (k=5,10,20) [NOT TRIED]

#### B6: Ensemble Methods (AFTER FINDING 3+ GOOD MODELS)
- B6a: Voting ensemble (hard)
- B6b: Weighted ensemble (soft)
- B6c: Stacking (meta-learner)
- B6d: Boosting on weak learners

### CATEGORY C: LOSS FUNCTIONS / REGULARIZATION (NEED 3+ TRIALS)
Current: CrossEntropyLoss with label_smoothing=0.02

#### C1: Label Smoothing Variations
- C1a: label_smoothing=0.02 [v1, v16 running]
- C1b: label_smoothing=0.1 [v16 running]
- C1c: label_smoothing=0.05 [NOT TRIED]
- C1d: label_smoothing=0.2 [NOT TRIED]

#### C2: Class Weighting (NEED 3+ TRIALS)
Current: No weights
- C2a: Inverse frequency weights [v15 running]
- C2b: Sqrt inverse frequency [NOT TRIED]
- C2c: Focal loss (gamma=2) [NOT TRIED]

#### C3: Alternative Loss Functions
- C3a: CrossEntropyLoss [current]
- C3b: Focal Loss [NOT TRIED]
- C3c: LabelSmoothing + FocalLoss combo [NOT TRIED]
- C3d: Contrastive losses [NOT TRIED]

### CATEGORY D: TRAINING HYPERPARAMETERS (NEED 3+ TRIALS)
Current: lr=1e-3, 30 epochs, batch_size=2048, cosine_annealing

#### D1: Learning Rate Strategies (NEED 3+ TRIALS)
- D1a: lr=1e-3 + cosine [v1]
- D1b: lr=3e-4 + cosine [v14 running]
- D1c: OneCycleLR (peak_lr=2e-3) [haiku-mar8, NOT REPLICATED]
- D1d: OneCycleLR (peak_lr=1e-3) [NOT TRIED]
- D1e: Exponential decay [NOT TRIED]

#### D2: Batch Size Effects (NEED 3+ TRIALS)
- D2a: batch_size=2048 [current]
- D2b: batch_size=4096 [haiku-mar8, NOT TRIED]
- D2c: batch_size=1024 [NOT TRIED]

#### D3: Epoch/Patience Variations
- D3a: 30 epochs, patience=10 [v1]
- D3b: 100 epochs, patience=15 [v14 running]
- D3c: 50 epochs, patience=20 [NOT TRIED]

#### D4: Optimizer Variants (NEED 3+ TRIALS)
- D4a: AdamW [current]
- D4b: SGD with momentum [NOT TRIED]
- D4c: LAMB [NOT TRIED]

### CATEGORY E: FEATURE ENGINEERING (NEED 3+ TRIALS)
Current: 8 features (E, Ze, Az, Ne, Nmu, Ne-Nmu, cos(Ze), sin(Ze))
haiku-mar8: 7 features (E, cos(Ze), sin(Az), cos(Az), Ne, Nmu, Ne-Nmu)

#### E1: Feature Subsets
- E1a: haiku-mar8 exact (7 features, NO sin(Ze)) [NOT TRIED]
- E1b: Current (8 features) [v1]
- E1c: Minimal (5 raw only) [NOT TRIED]
- E1d: Extended (12+ features with logs/polys) [NOT TRIED]

#### E2: Feature Transformations
- E2a: Log transforms (log(E), log(Ne+1)) [NOT TRIED]
- E2b: Polynomial features (E^2, Ze^2) [NOT TRIED]
- E2c: Interaction terms [NOT TRIED]
- E2d: Ratios (Ne/E, Nmu/E) [NOT TRIED]

### CATEGORY F: VALIDATION STRATEGY (UNTESTED)
Current: 80/20 train/val split with seed=42

#### F1: Cross-validation
- F1a: 5-fold CV ensemble [NOT TRIED]
- F1b: Stratified K-fold [NOT TRIED]

#### F2: Test-time augmentation
- F2a: Rotate 4 times, average predictions [NOT TRIED]

---

## SYSTEMATIC TESTING SCHEDULE

### PHASE 1: Current v14-v16 Completions (3+ v1 variants)
- v14: v1 + longer training (100 epochs, lr=3e-4) — RUNNING
- v15: v1 + class weights — RUNNING
- v16: v1 + label_smoothing=0.1 — RUNNING

**Decision point**: If ANY of these beat 50.71%, iterate on that one more. Otherwise, move to Phase 2.

### PHASE 2: Deep Architecture Search (B1: CNN depth)
Since v1 is close but not beating haiku-mar8, try exact replica:
- v17: Exact haiku-mar8 replica (4 CNN blocks, OneCycleLR, batch=4096, 7 features)
- v18: Deeper CNN variant (5 blocks)
- v19: Wider CNN variant (extra channels)

### PHASE 3: Non-CNN Architectures (B3: Vision Transformer, MLP)
- v20: ViT 4×4 patches (v12 alternative)
- v21: ViT 2×2 patches (finer granularity)
- v22: Pure MLP on flattened (v11 variant)

### PHASE 4: Tree Models (B4) with Safe Preprocessing
- v23: RandomForest with np.nan_to_num + engineered features
- v24: LightGBM
- v25: HistGradientBoosting

### PHASE 5: Loss Function Tuning (C2, C3)
- v26: Focal loss (gamma=2, alpha=0.5)
- v27: Different class weight scaling

### PHASE 6: Ensemble of Best Models
- v28: Weighted ensemble of top 3 performers

---

## Key Metrics to Track
- Test accuracy (primary metric)
- Validation accuracy progression
- Training time
- Parameter count
- Feature importance (for tree models)

## Success Criteria
- Beat 50.71% (haiku-mar8 baseline)
- Published CNN ~51% (stretch goal)

## Principles
1. Try EACH approach at least 3 times with variations before discarding
2. Cross-pollinate insights (if A works, try A on other architectures)
3. Keep results.tsv and journal.md updated after every experiment
4. Only commit code that improves the metric
