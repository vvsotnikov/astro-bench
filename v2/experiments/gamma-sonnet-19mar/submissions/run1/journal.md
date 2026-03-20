# Gamma/Hadron Classification Journal

## Task
Binary classification: gamma (0) vs hadron (1). Metric: hadronic survival rate at 75% gamma efficiency (lower is better). Published baseline ~10^-2 to 10^-3.

## Physics Key Points
- Gamma rays produce electromagnetic showers — almost NO muons
- Ne/Nmu ratio is the key discriminant: gammas have much higher ratio
- Muon density maps are the best feature
- Test quality cuts: Ze<30, Ne>4.8, 0.2<Age<1.48

## Data
- matrices.npy: (1.5M, 16, 16, 2) — channel 0: electron/photon, channel 1: muon density
- features.npy: (1.5M, 5) — [E, Ze, Az, Ne, Nmu]
- Class imbalance: ~5% gamma in training data

## Key Findings

### What works
- **CNN + MLP on engineered features** (v2) is the backbone: 4.09e-04
- **Geometric ensemble** consistently improves over individual models
- **Cross-channel attention** (v9) provides good ensemble diversity: 3.79e-04 solo
- **Best ensemble** (ens2): v2+v7+v8+v9 with weights 0.45,0.15,0.15,0.25 = 2.92e-04
- Feature engineering (11 features from 5 raw) is crucial

### What doesn't work
- Focal loss (v3, v10): overfits without careful tuning
- Quality cuts on training (v5): hurts performance
- Multi-seed ensembles of same model (v6, v11): no diversity benefit
- ViT (v13): 4.96e-04 - attention-based but doesn't outperform CNN
- Physics-only LLR (v14): 9.92e-04 - discriminant but much worse
- Multi-task learning (v16): 5.25e-04 - doesn't help
- Pairwise ranking loss (v17): 5.25e-04 - no better than CE
- Random Forest (v18): 9.92e-04 - completely different approach but much worse

### Physical limit
- The 10 surviving hadrons (at 75% gamma threshold) have mu_nnz=34 hits, mu_sum=721
- Compare with gammas: bottom 25% have mu_nnz=47, mu_sum=1285
- These are genuinely gamma-like hadrons — the physical limit

## Experiment Log

### v1 — Simple MLP baseline
- MLP on 5 scalar features + engineered Ne/Nmu ratio: 4.55e-03
- Result: Good baseline, Ne/Nmu clearly discriminates

### v2 — CNN + MLP (BEST SINGLE MODEL)
- CNN on 16x16x2 matrices (32→64→128 channels) + 11-feature MLP: 4.09e-04
- Result: 10x better than v1. The spatial muon maps matter a lot.

### v3 — Deep ResNet + focal loss
- 3-layer ResNet + channel attention + focal loss: 5.54e-04
- Result: Overfitted. Too complex.

### v4 — First ensemble
- Geometric mean of v1+v2: 3.50e-04
- Result: Ensemble helps even with weak model (v1)

### v5 — Quality cuts on training
- CNN v2 + Ze<30, Ne>4.8 cuts applied to training: 5.54e-04
- Result: Hurts! Training distribution shift is harmful.

### v6 — Multi-seed ensemble
- 3 seeds of CNN v2: 4.09e-04
- Result: Same as single model. Seeds converge to same solution.

### v7 — MLP with matrix statistics
- MLP on scalar features + matrix stats (mu sum, el sum, nnz, etc.): 4.67e-04
- Result: Worse than CNN. But provides diversity for ensemble!

### v8 — Larger CNN + matrix stats
- CNN (48→96→192 channels) + matrix stats features: 4.96e-04
- Result: Provides ensemble diversity

### ens1 — First good ensemble
- v2+v7+v8 geometric mean (0.5, 0.15, 0.35): 3.21e-04

### v9 — Cross-channel attention CNN
- Separate CNN per channel (el and mu), then cross-attention: 3.79e-04
- Result: Great ensemble component!

### ens2 — Best ensemble (CURRENT BEST)
- v2+v7+v8+v9 geometric mean (0.45, 0.15, 0.15, 0.25): 2.92e-04

### v10-v18 — Various attempts
- None improved the ensemble beyond 2.92e-04
- Focal loss, deeper models, ViT, MTL, ranking loss, RF all tried

## Current State (attempt 13/50)
- Best: 2.92e-04 (ens2)
- 37 attempts remaining
- Key question: what new architecture/approach provides complementary predictions?

## Ideas for Next Experiments

### High priority
1. **v19**: ResNet with skip connections, heavy regularization — different from v2/v9
2. **v20**: Multiple independent CNN v2 models with different random seeds but different ARCHITECTURES — try CNN with strided convolutions instead of maxpool
3. **v21**: Ensemble with optimal re-weighting using held-out validation data (Nelder-Mead)
4. **v22**: Deeper feature engineering — add energy-normalized muon features, shower age
5. **v23**: Data augmentation — rotate 90/180/270, flip horizontally/vertically (hexagonal KASCADE grid has symmetry)

### Medium priority
1. CNN with attention pooling instead of global average pooling
2. Two-stage classifier: first identify "clearly gamma" (mu=0), then use CNN for the rest
3. Isotonic regression calibration of ensemble scores
4. Add more models to ensemble with larger weights

### Key insight
The ensemble works best when models have different failure modes. v2 (vanilla CNN), v7 (MLP+stats), v8 (larger CNN), v9 (cross-attn) are architecturally diverse. The next improvement needs a model that makes different errors from these 4.
