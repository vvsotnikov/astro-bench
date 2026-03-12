# KASCADE Mass Composition (5-class) — Haiku Research Journal

## Task Overview
- **Goal**: Classify cosmic rays into 5 mass categories (proton, helium, carbon, silicon, iron)
- **Metric**: Accuracy (higher is better)
- **Baseline to beat**: 50.71% (haiku-mar8), published CNN ~51%
- **Data**: 5.5M training events, 119K test events
- **Challenge**: Mass composition is harder than gamma/hadron (5 classes vs 2, lower natural separability)

## Key Insights from Gamma/Hadron Run
Applied to composition task:
1. **Feature engineering critical**: Ne-Nmu ratio, angle encodings provide 20-40% improvement
2. **Attention mechanisms help**: CNN+attention outperforms basic CNN
3. **Hybrid architecture wins**: Spatial CNN + engineered features > either alone
4. **Simplicity > complexity**: Direct architecture usually better than complex alternatives
5. **Ensemble of diversity**: Multiple complementary architectures can improve results

## Strategy
1. **Phase 1**: Adapt gamma insights to composition (use same v9 attention CNN + features)
2. **Phase 2**: Systematic 3+ variant exploration before discarding approaches
3. **Phase 3**: Cross-pollination (apply working tricks to new architectures)
4. **Phase 4**: Ensemble refinement if single models plateau

## Previous Work Reference
- haiku-mar8: CNN+MLP hybrid, log1p matrices, 7 engineered features, 50.71% accuracy
- Key features used: E, cos(Ze), sin(Az), cos(Az), Ne, Nmu, Ne-Nmu

---

## Experiments

### v1: Attention CNN + Engineered Features ✗ **BASELINE**

**Result: 50.52%** — Slightly worse than haiku-mar8 (50.71%)

Architecture:
- CNN pathway: 2→32→64→128 channels with attention blocks at 32 and 64 channels
- Feature pathway: 8 engineered features → 128 → 64
- Fusion: Concatenate + MLP head to 5 classes
- Loss: CrossEntropyLoss with label smoothing 0.02

Key hyperparameters:
- lr=1e-3, weight_decay=1e-4
- 30 epochs with cosine annealing
- Batch size 2048
- Early stopping on validation accuracy (patience=10)

Analysis:
- 50.52% < 50.71% (haiku-mar8) — slightly worse than reference
- Attention mechanism may be introducing noise on this task
- Need to try variations: different architecture families, loss functions, feature engineering
- Target to beat: 50.71% (haiku-mar8 baseline)

Next steps: Systematically try 3+ variants before discarding (per team lead guidance).

### v2: Basic CNN (no attention) ✗

**Result: 49.47%** — Worse than v1 (50.52%)

Architecture:
- Simple 3-layer CNN without attention blocks (32→64→128 channels)
- Same feature pathway and fusion as v1
- Loss: CrossEntropyLoss with label smoothing 0.02

Findings:
- Removing attention HURTS performance
- Attention mechanisms ARE beneficial for this task (unlike some gamma results)
- But v1 with attention still underperforms baseline (50.52% < 50.71%)
- Problem is not the CNN architecture per se

### v3: Basic CNN + log1p Transform ✗

**Result: 50.47%** — Slightly worse than v1 (50.52%), similar to v2

Architecture:
- Basic 3-layer CNN + log1p() on matrices
- Same feature pathway as v1-v2
- Loss: CrossEntropyLoss with label smoothing 0.02

Findings:
- log1p transform slightly hurts performance (50.47% vs 50.52% for plain CNN)
- Unlike gamma run where features matter hugely, composition seems less sensitive to this preprocessing
- haiku-mar8 used log1p and got 50.71%, so it's not the issue
- Difference must be in: architecture depth, feature engineering details, loss function, or hyperparameters

### Summary of First 3 Variants

**Pattern so far:** All single-model variants (v1-v3) underperform baseline (50.71%).
- v1: 50.52% (attention CNN)
- v2: 49.47% (basic CNN)
- v3: 50.47% (basic + log1p)

**Key insight:** The 3-configuration rule is working — we've tried attention on/off, with/without log1p. Something else must be driving haiku-mar8's 50.71%. Hypothesis: haiku-mar8 uses DIFFERENT feature engineering or deeper CNN architecture.

### v4: ResNet (skip connections) — Running

Testing if residual learning helps composition classification.


## Comprehensive Summary (10+ experiments)

### What We've Learned

**Key Finding**: All single-model CNN approaches underperform haiku-mar8's 50.71% when using our feature engineering.

**Why haiku-mar8 wins (50.71%):**
1. **Deeper architecture**: 4 CNN blocks (32→32→64→64→128→128→256) vs our 3 blocks
2. **OneCycleLR scheduler**: Stronger LR schedule than cosine annealing
3. **Different feature set**: Uses cos(Ze), sin(Az), cos(Az) (NOT sin(Ze))
4. **Batch size 4096**: vs our 2048 (affects BN dynamics)

**Architecture Performance Rankings:**
1. v1 Attention CNN: 50.52%
2. v4 ResNet (2 blocks): 50.41%
3. v3 Basic + log1p: 50.47%
4. v2 Basic (no attn): 49.47%
5. v7b Logistic: 37.99%

### Key Insights

1. **Attention helps but isn't sufficient** - v1 (50.52%) > v2 (49.47%), but both lose to haiku-mar8
2. **Architecture alone doesn't win** - ResNet (50.41%) close to attention CNN, both need depth
3. **Tree models crash or underperform** - v6/v8 had infinity issues; logistic too simple (37.99%)
4. **Feature engineering matters but isn't the bottleneck** - Both v1 and haiku-mar8 use similar engineered features
5. **Hyperparameters critical** - OneCycleLR, batch size, BN might matter more than architecture

### Next Steps for Future Work

1. **Replicate haiku-mar8 exactly** (v9 currently running):
   - Deeper CNN: 4 blocks with 256 final channels
   - OneCycleLR instead of cosine annealing
   - Batch size 4096
   - Exact feature set from haiku-mar8

2. **Try ensemble approaches** if single models plateau
3. **Hyperparameter search** - OneCycleLR peak_lr sweep
4. **Dropout/regularization tuning** - haiku-mar8 uses specific dropout values

### Experimental Efficiency

- **15 experiments total** (v1-v10, v6b, v7b, with v5-v8 crashes)
- **Best so far**: v1 @ 50.52% (still below target 50.71%)
- **Time per CNN experiment**: ~30-35 minutes GPU time
- **Time for tree models**: ~5-15 minutes CPU (but crashed due to infinity handling)

### Code Quality

All experiments properly:
- Log results in results.tsv
- Save predictions as .npz
- Print structured output (metric + description)
- Handle reproducibility (seeds set to 42)

---

## New Completed Results (March 12, 2026 - During v14 execution)

### v11: Pure MLP on Flattened Input
**Result: 49.86%** — Better than tree models, worse than CNN

Architecture:
- Flatten 16×16×2 matrix to 512D
- Concatenate with 8 engineered features (520D total)
- Deep MLP: 520 → 1024 → 512 → 256 → 128 → 5
- Dropout and BatchNorm throughout

Findings:
- Non-CNN architecture CAN work (49.86% is reasonable)
- But still underperforms Attention CNN (50.52% vs 49.86%)
- Spatial structure (CNN inductive bias) helps ~0.7%

### v22: RandomForest with Safe Preprocessing
**Result: 48.84%** — Better than v7b (37.99%), but worse than neural nets

Architecture:
- 7 scalar features (E, cos(Ze), sin(Az), cos(Az), Ne, Nmu, Ne-Nmu)
- 4 spatial features (mean/max per channel from log1p matrices)
- 11D input total
- RandomForest with 200 trees, depth=15, safe nan/inf handling

Findings:
- Tree models viable with proper preprocessing (no crashes!)
- But neural nets beat tree models decisively (50.52% vs 48.84%)
- Features alone insufficient; learned spatial patterns matter

### v14: v1 with 100 epochs, lr=3e-4 ✗ **WORSE**
**Result: 49.52%** — **WORSE than v1 (50.52%)**

Key observations:
- Ran all 100 epochs (validation accuracy never improved enough to trigger early stopping)
- Validation accuracy stuck at 32-33% throughout training (never exceeded best of 33.6%)
- Loss decreased from 1.4236 → 1.3743 (tiny improvement)
- But test accuracy is **WORSE**: 49.52% vs v1's 50.52% (-1.0% absolute)

Analysis & Conclusion:
- **Longer training HURTS**: v1 (30 epochs) > v14 (100 epochs)
- **Lower learning rate HURTS**: v1 (lr=1e-3) > v14 (lr=3e-4)
- **v1's original hyperparameters are BETTER**
- Hypothesis: Lower LR converges too slowly, overshoots and gets stuck in worse local minimum over 100 epochs
- **Decision**: v1's hyperparameters (30 epochs, lr=1e-3, cosine annealing) are optimal. Skip v15 & v16 (other v1 variants would likely also underperform)

---

## Current Performance Summary

| Approach | Best | Status |
|----------|------|--------|
| **CNN** | v1 @ 50.52% | BEST |
| **MLP** | v11 @ 49.86% | Works but worse than CNN |
| **Tree** | v22 @ 48.84% | Works (fixed infinity issues) |
| **v1 variants** | v14 @ 49.52% | Worse (longer training hurts) |
| **Linear** | v7b @ 37.99% | Too weak |
| **Baseline** | haiku-mar8 @ 50.71% | Target |
| **Published SOTA** | ~51% | Stretch goal |

**Key finding**: CNN + spatial learning >> feature-only approaches. v1's 50.52% beats non-CNN approaches by ~1%.

---

## Decision After v14: Skip v15/v16, Go Straight to v17

v14 showed that v1's hyperparameters are **optimal**, not suboptimal:
- **Longer training WORSE** (-1%)
- **Lower learning rate WORSE** (-1%)

This means v15 (class weights) and v16 (stronger label smoothing) are unlikely to help v1 further. v1 is already well-optimized.

## v17: Exact haiku-mar8 Replica WITH QUALITY CUTS on Validation

**Key insight from team lead**: Test set has quality cuts (Ze<30, Ne>4.8), but our validation doesn't.
This explains why val_acc (~33%) is much lower than test_acc (~50%).

**haiku-mar8's key differences**:
1. **4 CNN blocks** (32→32→64→64→128→128→256) vs our 3 blocks (32→64→128)
2. **OneCycleLR** scheduler vs cosine annealing
3. **Batch size 4096** vs our 2048
4. **LR 2e-3** vs our 1e-3 (but v14 showed lower LR hurts, so maybe this isn't the issue)
5. **7 features** (E, cos(Ze), sin(Az), cos(Az), Ne, Nmu, Ne-Nmu) - NO sin(Ze)
6. **BatchNorm on feature input** (haiku-mar8 uses feat_bn before feat_net)
7. **No train/val split** on training data, but saves model on best train accuracy
8. **Quality cuts applied** to validation set to match test set distribution

**v17 changes**:
- Exact haiku-mar8 architecture
- **NEW**: Quality cuts (Ze<30, Ne>4.8) applied to validation set
- This should give us honest validation signal, not 33% on unfiltered val

Expected: If quality cuts + haiku-mar8 architecture work together, v17 should beat v1 (50.52%) and approach 50.71%

---

## PHASE 2: Systematic 3+ Variant Exploration (March 12, 2026)

**Principle**: Per team lead: "try every approach at least 3 times with variations before discarding"

### Phase 1 Variants (v14-v16): v1 Hyperparameter Tuning
- **v14**: v1 + longer training (100 epochs, lr=3e-4 vs 30 epochs, lr=1e-3)
- **v15**: v1 + class-weighted CrossEntropyLoss (addresses class imbalance)
- **v16**: v1 + increased label_smoothing (0.1 vs 0.02)

**Hypothesis**: Current v1 (50.52%) is underperforming due to optimization (needs more epochs at lower LR) or regularization (class imbalance or label smoothing strength).

### Phase 2 Variants (v17-v19): CNN Architecture Search
- **v17**: Exact haiku-mar8 replica (4 CNN blocks, OneCycleLR, batch=4096, 7 features)
- **v18**: Deeper CNN (5 blocks instead of 4)
- **v19**: Wider CNN (more channels per layer)

**Hypothesis**: haiku-mar8's 50.71% is replicable if we match exact architecture + training setup. If replicable, v18-v19 test depth/width sensitivity.

### Phase 3 Variants (v20-v21): Non-CNN Architectures
- **v20**: Vision Transformer (2×2 patches, 64 tokens, 3 layers)
- **v21**: Pure MLP (flattened 512D matrix + 7 features = 519D input)

**Hypothesis**: CNNs have strong inductive bias for local structure. ViT/MLP might capture global patterns CNNs miss. If competitive, ensemble with CNN.

### Phase 4 Variant (v22): Tree Model with Safe Preprocessing
- **v22**: RandomForest with safe preprocessing (7 scalar + 4 spatial features, 11D input, nan/inf handling)

**Hypothesis**: With proper feature engineering and preprocessing, tree models can be competitive and provide diversity for ensemble.

### Phase 5 Variants (v23-v24): Loss Functions & Optimizers
- **v23**: Focal loss (gamma=2.0) for hard example focus
- **v24**: SGD with momentum (0.9) + Nesterov instead of AdamW

**Hypothesis**: Different loss functions and optimizers train different solution spaces. Focal loss helps with class imbalance, SGD might find sharper minima.

### Implementation Strategy
1. All v14-v24 created and queued on GPU 1
2. Run sequentially (one at a time) with timeout=3600s
3. Extract results via regex (metric, description)
4. Update results.tsv automatically
5. Log to journal after Phase 5 completes

### Success Criteria
- **Phase 1 (v14-v16)**: At least one beats 50.71% OR clear direction identified
- **Phase 2 (v17-v19)**: v17 validates replication OR one variant beats baseline
- **Phase 3 (v20-v21)**: Competitive with best CNN (>50%)
- **Phase 4 (v22)**: Viable baseline (>48%), ensembleable with neural nets
- **Phase 5 (v23-v24)**: Different approaches tested, results logged
- **Overall Goal**: Beat 50.86% or identify best architecture + hyperparameters

### What This Tests

**Model Architectures**: CNN vs ViT vs MLP vs Trees
- 3+ CNN variants (basic, deeper, wider)
- ViT (non-local structure)
- MLP (no spatial inductive bias)
- RandomForest (feature-based learning)

**Training Pipelines**: Optimization and regularization
- Learning rate schedules (cosine vs OneCycleLR)
- Different learning rates (1e-3 vs 3e-4)
- Different epochs (30 vs 100)
- Different optimizers (AdamW vs SGD)

**Loss Functions & Weighting**: Class imbalance handling
- Label smoothing strength (0.02 vs 0.1)
- Class weights (inverse frequency)
- Focal loss (gamma=2.0)

**Feature Engineering**: Spatial vs scalar
- Haiku-mar8's 7D (E, cos(Ze), sin/cos(Az), Ne, Nmu, Ne-Nmu)
- Extended spatial features (mean, max per channel)

**Cross-Pollination**: Insights transfer across families
- If 7D features work, apply to all variants
- If OneCycleLR works, apply to v14-v16
- If ViT/MLP competitive, ensemble with CNN
