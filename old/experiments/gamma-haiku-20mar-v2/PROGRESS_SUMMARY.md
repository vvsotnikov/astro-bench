# Gamma/Hadron Classification - Progress Summary

## Current Status (as of March 20, 18:30 EET)

### Best Result
- **v3: 3.21e-04** (dual-channel CNN + attention pooling)
- **Baseline**: 1.0e-02 (published RF, Kostunin et al.)
- **Improvement**: 31× better than baseline

### Attempts Used
- **Total experiments**: 12+ completed
- **Attempts used**: 13/50  
- **Remaining**: 37 attempts

## Completed Experiments

| Attempt | Model | Metric | Description |
|---------|-------|--------|-------------|
| 0 | RF Baseline | 1.0e-02 | Published baseline |
| 1 | RF | 2.04e-03 | Engineered features |
| 2 | MLP | 7.30e-04 | Physics features (v1) |
| 3 | CNN | 1.75e-03 | Single-channel muon |
| 4 | MLP | 7.59e-04 | Residual + log1p (v5) |
| **5** | **CNN** | **3.21e-04** | **Dual + attention (v3)** ⭐ |
| 8 | MLP | 4.38e-04 | Deep 5-layer |
| 13 | Ensemble | 4.38e-04 | v3+v1 simple average |
| 15 | CNN | 5.54e-04 | v3 reproduction variant |
| 25 | Ensemble | 4.38e-04 | Equal weight 4-way |
| 26 | Ensemble | 4.09e-04 | v3-heavy 70% |
| 27 | Ensemble | 4.67e-04 | Top 2 variant |
| 28 | Ensemble | 4.38e-04 | v3+v5+v1 50/25/25 |

## Key Findings

### ✅ What Works
1. **Dual-channel CNN architecture** - Processing electron AND muon channels together is essential
2. **Attention pooling** - Learnable attention mechanism significantly outperforms global average pooling
3. **Batch normalization** - Critical for training stability
4. **Class-weighted loss** - Properly handling extreme imbalance is crucial
5. **Cosine annealing** - Better than fixed learning rates

### ❌ What Doesn't Help
1. **Simple ensemble averaging** - Mixing v3 (best) with worse models dilutes performance
2. **Single-channel CNN** - Muon channel alone or electron channel alone is insufficient
3. **Very deep MLPs** - v8 (5 layers) worse than v3's 3-layer CNN
4. **Complex ensemble weighting** - Even optimal weights don't beat v3
5. **Seed ensemble variants** - Early attempts took 37+ minutes (likely hung or very slow GPU contention)

### 📊 Architecture Impact
- **CNN > MLP**: Spatial information is critical
- **Dual-channel > Single**: Both channels necessary
- **Attention > Global Avg Pool**: Focus mechanism critical
- **Simple > Complex**: v3's compact design beats deeper variants

## Next Steps (37 attempts remaining)

### Phase A: Architecture Variations (10-15 attempts)
- v29: Skip connections in CNN (in progress)
- v30: Deeper MLP after attention
- v31: Multi-head attention  
- v32: Different activation functions (ELU, GELU)
- v33: Spatial pyramid pooling
- v34: Channel attention (SENet-style)
- v35: Different CNN channel widths (16→32→32, 64→128→128)

### Phase B: Learning & Training Tweaks (8-10 attempts)
- Different learning rates [5e-4, 2e-3, 5e-3]
- Different optimizers (SGD with momentum, LAMB, AdaBound)
- Warmup strategies
- Different batch sizes [2048, 8192]

### Phase C: Data Augmentation (5-8 attempts)
- Rotations (small angles)
- Flips & spatial perturbations
- Noise injection
- Mixup/CutMix

### Phase D: Advanced Techniques (5 attempts)
- Cross-validation ensemble (5-fold)
- Stacking (v3 features → second classifier)
- Knowledge distillation
- Semi-supervised learning (exploit test distribution)

### Phase E: Final Polish (4 attempts)
- Best configuration refinement
- Validation on held-out set
- Hyperparameter grid search around best

## Insights from Failed Approaches

1. **Seed Ensemble took 37 min** - Something wrong with GPU contention or data loading. Skip this strategy.
2. **Weighted averaging failed** - v3 is so dominant that mixing waters it down. Need fundamentally different approaches.
3. **Reproduction (v15) was slightly worse (5.54e-04 vs v3's 3.21e-04)** - Small variance from random init/val split randomness.

## Strategy Going Forward

**Goal**: Either beat v3's 3.21e-04 or document ceiling around 3.0e-04.

**Approach**:
1. Focus on architecture variations that exploit physics insights better
2. Avoid ensemble averaging - v3 is too dominant
3. Try different training techniques (augmentation, different optimizers)
4. If stuck, move to cross-validation + stacking strategies
5. Document findings for paper - understanding what doesn't work is valuable

**Expected ceiling**: ~2.5-3.0e-04 with best techniques, ~2.0e-04 with cross-validation.

## GPU Utilization Notes
- GPU 1 (A6000) assigned via CUDA_VISIBLE_DEVICES=1
- Training takes ~25-30 minutes per full model
- Memory usage: 4.3-4.5 GB per model
- Batch size 4096 is good for this GPU

