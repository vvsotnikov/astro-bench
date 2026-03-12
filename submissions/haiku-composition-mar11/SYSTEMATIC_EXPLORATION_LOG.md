# Systematic Composition Exploration Log

## Principle
Per team lead guidance: try EVERY approach at least 3 times with variations before discarding. Apply insights from working approaches to other architectures (cross-pollination).

## Current Status (March 12, 2026)

### Baseline
- **Target**: Beat 50.86% (haiku-mar8's best from gamma task)
- **haiku-mar8 composition**: 50.71% with exact replication
- **Published CNN (JINST 2024)**: ~51%

### Running Experiments (Queued on GPU 1)
All v14-v23 are queued to run sequentially:

#### PHASE 1: v1 Variations (3 hyperparameter trials)
- **v14**: v1 + long training (100 epochs, lr=3e-4) — RUNNING
- **v15**: v1 + class-weighted loss — QUEUED (will run after v14)
- **v16**: v1 + label_smoothing=0.1 — QUEUED

**Expected**: One of these should beat 50.71% or show clear direction

#### PHASE 2: CNN Architecture Search (3 depth/width variants)
- **v17**: Exact haiku-mar8 replica (4 blocks, OneCycleLR, batch=4096, 7 features) — QUEUED
- **v18**: Deeper CNN (5 blocks instead of 4) — QUEUED
- **v19**: Wider CNN (more channels per layer) — QUEUED

**Hypothesis**: haiku-mar8's 50.71% comes from exact architecture + training setup. If v17 matches haiku-mar8, we learn whether it's replicable. v18-v19 test if deeper/wider helps.

#### PHASE 3: Non-CNN Architectures (3 different families)
- **v20**: Vision Transformer (2×2 patches) — QUEUED
- **v21**: Pure MLP (flattened 512D + 7 features) — QUEUED
- **(v11, v12, v13 from earlier runs)*

**Hypothesis**: Different inductive biases might capture patterns CNNs miss.

#### PHASE 4: Tree-Based Models (3 variants with safe preprocessing)
- **v22**: RandomForest (7 scalar + 4 spatial features, safe preprocessing) — QUEUED
- **(v6b, v7b from earlier: crashed or underperformed)**

**Hypothesis**: With proper feature engineering and infinity/nan handling, trees might be competitive.

#### PHASE 5: Loss Functions & Regularization (3 variants)
- **v23**: Focal loss (gamma=2.0) — QUEUED
- **(v14-v16 already cover label smoothing and class weights)**

**Hypothesis**: Different loss functions handle hard examples differently.

---

## Systematic Approach Checklist

### MODEL ARCHITECTURES ✓
- [x] CNN (3-block): v1, v2, v3
- [x] CNN with attention: v1, v4 (ResNet)
- [x] Deeper CNN (3→4 blocks): v9, v17 (exact haiku-mar8), v18 (5 blocks)
- [x] Wider CNN: v19
- [x] ViT (Transformer): v20
- [x] MLP (flattened): v21
- [x] RandomForest: v6/v6b/v22
- [x] Logistic Regression: v7/v7b
- [ ] Gradient Boosting (LightGBM) — could add
- [ ] SVM — v13 running from earlier
- [ ] Autoencoder — could add

### DATA PIPELINES ✓
- [x] No normalization: checked (BN handles it)
- [x] log1p on matrices: v3, v9-v23 use this
- [x] Feature engineering: 8D (v1) vs 7D (haiku-mar8) — v17-v23 match haiku-mar8
- [x] Safe preprocessing for trees: v22
- [ ] Per-channel normalization — not tried
- [ ] Augmentation (90° rotations) — not tried
- [ ] Sparse tensor handling — not tried

### TRAINING PIPELINES ✓
- [x] Cosine annealing: v1-v4, v9-v10
- [x] OneCycleLR: v17-v23 (matching haiku-mar8)
- [x] Different LR: v14 (3e-4 vs 1e-3)
- [x] Different batch sizes: v17-v23 use 4096 (haiku-mar8), v1-v4 use 2048
- [x] Different epochs: v14 (100 vs 30)
- [x] Different patience: v14 (15 vs 10)
- [ ] SGD with momentum — not tried
- [ ] LAMB optimizer — not tried
- [ ] Gradient accumulation — not needed at batch=4096

### LOSS FUNCTIONS ✓
- [x] CrossEntropyLoss with label_smoothing=0.02: v1-v4, v17-v19
- [x] CrossEntropyLoss with label_smoothing=0.1: v16
- [x] CrossEntropyLoss with class_weights: v15
- [x] Focal loss: v23
- [ ] LabelSmoothing + FocalLoss combo — could add
- [ ] Contrastive losses — unlikely to help classification

### FEATURE ENGINEERING ✓
- [x] 8D (E, Ze, Az, Ne, Nmu, Ne-Nmu, cos(Ze), sin(Ze)): v1, v9-v10
- [x] 7D (haiku-mar8 exact: E, cos(Ze), sin(Az), cos(Az), Ne, Nmu, Ne-Nmu): v17-v23
- [ ] Extended (12+ with logs/ratios) — could try
- [ ] Interaction terms — could try
- [ ] Polynomial features — could try

---

## What We'll Learn from This Batch

### From v14-v16 (v1 variations):
- If longer training + lower LR helps (optimization trajectory)
- If class weights help (imbalance problem)
- If stronger label smoothing helps (regularization)

### From v17-v19 (CNN variants):
- If exact haiku-mar8 replication is achievable
- If depth helps (v18) or hurts (overfitting)
- If width helps (v19) or adds unnecessary parameters

### From v20-v21 (non-CNN):
- If spatial structure (CNN inductive bias) is necessary
- If global patch structure (ViT) captures patterns locally-oriented CNN misses
- If flattened MLP can compete despite losing spatial structure

### From v22 (trees):
- If proper preprocessing makes trees viable
- If spatial statistics can replace CNN learning

### From v23 (focal loss):
- If hard example focus helps class-imbalanced problem

---

## Decision Tree (After Results)

```
If v14-v16 beat 50.71%:
  → Iterate 3 more variations on that winner
  → If still improving, deep-dive on hyperparameters

If v17 (exact haiku-mar8) beats 50.71%:
  → It's replicable! All variants after should reference it
  → Try variants of v17 (v18-v19 in Phase 2)

If v17 < 50.71%:
  → haiku-mar8 might have used different data or preprocessing
  → Investigate haiku-mar8 code more carefully

If v17-v19 all underperform v1 (50.52%):
  → Depth/width not the limiting factor
  → Move focus to v20-v21 (non-CNN) or v22-v23 (loss/preprocessing)

If v20-v21 underperform v1:
  → CNN inductive bias is valuable
  → Focus on improving CNN variants

If v22 (trees) competitive (>50%):
  → Ensemble v22 with best neural net
  → Feature engineering is carrying the signal

After all v14-v23:
  → Identify top 3 performers from different architecture families
  → Try weighted ensemble of top 3
  → Test on diversity (complementary error modes)
```

---

## Commit Plan

1. Create all training scripts (✓ done: v14-v23)
2. Queue all experiments (✓ done)
3. Monitor and log results in results.tsv after each completes
4. Commit only scripts that improve metric
5. Update journal.md with findings
6. After Phase 5 completes, decide on Phase 6 (ensemble or next direction)

---

## Success Criteria

- **Phase 1 (v14-v16)**: At least one beats 50.71% OR shows clear direction
- **Phase 2 (v17-v19)**: v17 replicates haiku-mar8 (50.71%) OR one variant beats it
- **Phase 3 (v20-v21)**: Competitive with best CNN (within 0.5%)
- **Phase 4 (v22)**: Viable baseline (>48%), potentially ensembleable
- **Phase 5 (v23)**: Different loss function tested, results logged
- **Overall**: Beat 50.86% or identify best architecture + hyperparameter combination

---

## Notes
- All experiments use absolute GPU path: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 (GPU 1)
- v22 runs on CPU (RandomForest)
- Each experiment has timeout=3600s (1 hour)
- Results logged to results.tsv in TAB format
- Predictions saved as .npz (required for verify.py)
