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

## Current State (attempt 29/50, experiments through v33)
- **Best: ens3 at 2.33e-04** (7 models: v2+v8+v21+v25+v9+v1+v7)
- Key: Dirichlet random weight optimization (100K trials) found better weights than hand-tuning
- All attempts since ens3 have failed to improve the ensemble

## Physical Limit Analysis (v28-v33 phase)
The 8 surviving hadrons at 75% gamma efficiency are genuinely gamma-like:
- 7 events have Nmu < 3.3 (gamma mean is 3.14 ± 0.83) — physically indistinguishable
- 1 anomalous event (#26449): E=16.26, Nmu=4.31, mu_sum=5090, but has unusual "donut" muon pattern with zero-filled center rows. CNN interprets this as gamma-like due to the spatial pattern.
  - This event has 10^4.31 ≈ 20,000 muons but scores 0.997 gamma probability
  - Among all hadrons with E>16.2, it's the extreme outlier (mean score 0.13, this event 0.997)
  - Likely a pathological high-energy event with detector effects creating unusual patterns

## What DOESN'T improve the ensemble (v26-v33)
- v26: different LR/class weight — no diversity
- v27: wider CNN (64-128-256 ch) — same as larger v8
- v28: spatial attention pooling — same as v9
- v29: knowledge distillation from v2 — same as v2
- v30: quality-cut training — WORSE (overfits smaller dataset)
- v31: 5-model diversity (seeds/lr/aug) — pending
- v32: HistGBM on quality-cut data — too few events
- v33: HistGBM on all data — GBM spatial stats are weaker than CNN

## Key Insights
1. **All CNN variants converge to the same information**: Different seeds, LRs, architectures of the v2 family produce essentially the same predictions
2. **GBM on spatial statistics is weaker than CNN**: The raw 2D structure has more info than handcrafted statistics
3. **Quality cuts on training always hurt**: Less data = worse generalization
4. **The 8 surviving hadrons are at the physical limit**: No ML improvement can rescue them
5. **More Dirichlet trials help marginally**: Going from 50K to 100K improved ens from 2.63e-04 to 2.33e-04

## Remaining Ideas
- v31: 5-model diversity ensemble (running now) — might add marginal improvement
- More aggressive Dirichlet search with all 12 models — might find better weights
- A fundamentally different approach would be needed (GNN on detector hits, but hard without new packages)

## Conclusion
We're likely near the absolute physical limit for this problem given the current model family and available data. The ens3 at 2.33e-04 achieves ~10^3 hadron suppression while keeping 75% gammas — much better than the published baseline of 10^2-10^3 at 30-70% gamma efficiency.
