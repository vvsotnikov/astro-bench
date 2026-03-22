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

## Current State (attempt 27/50, experiments through v41)
- **Best: ens3 at 2.33e-04** (7 models: v2+v8+v21+v25+v9+v1+v7) — confirmed by verify.py
- Key: Dirichlet random weight optimization (100K trials) found better weights than hand-tuning
- All attempts v34-v41 have failed to improve the ensemble

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

## Extended Analysis: v34-v40 Phase (all failed to improve)

### Extended Dirichlet search (1M + 500K trials)
- 500K and 1M Dirichlet trials on best 7/13 models — NO improvement
- ens3 at 2.33e-04 is the global optimum for this model set

### The 8 surviving hadrons (detailed analysis)
All 8 surviving hadrons at 75% gamma efficiency:
- Event 5968: E=14.91, Nmu=2.65, mu_sum=219, ens3=0.9904
- Event 13527: E=14.97, Nmu=3.29, mu_sum=120, ens3=0.9969
- Event 14621: E=15.27, Nmu=3.27, mu_sum=1185, ens3=0.9920
- Event 15694: E=14.33, Nmu=2.63, mu_sum=36, ens3=0.9950
- Event 19604: E=14.54, Nmu=2.70, mu_sum=92, ens3=0.9944
- Event 26449: E=16.26, Nmu=4.31, mu_sum=5090, ens3=0.9970 (the "donut" event)
- Event 34953: E=14.51, Nmu=2.74, mu_sum=143, ens3=0.9950
- Event 35457: E=14.59, Nmu=2.66, mu_sum=46, ens3=0.9975

ALL models score these events > 0.96 (lowest: v9 gives event 5968 score 0.9682)
The 75% threshold is 0.9851 — all 8 are well above it.

### Key discovery: gamma muon sums overlap completely
- Gamma mu_sum: min=29, q10=75, median=1487, q90=7332
- 8 surviving hadrons have mu_sum=36-5090 — entirely within gamma range
- The CNN correctly identifies bottom-scoring gammas (mu_sum 296-1695) via SPATIAL pattern
- The 8 hadrons have genuinely gamma-like spatial muon patterns

### What doesn't work (v34-v40)
- v35: Direct metric optimization (AUC + survival loss) — destabilizes model (val 0.86)
- v36: BCE + pairwise AUC combined — AUC saturates immediately
- v37: Muon autoencoder anomaly detection — gamma/hadron error gap too small (0.181 vs 0.274)
- v38: Full spatial MLP on 512-dim flattened features — 9.05e-04, worse than CNN
- v39: Triple-branch CNN (mu/el/ratio) — ratio channel noisy, 7.88e-04
- v40: OHEM with 10x hard hadron weight — 4.67e-04, no ensemble gain
- Bayesian Nmu LLR correction — makes ens3 worse (Nmu already in model features)
- Physics-guided Nmu penalty — makes ens3 worse

### Why nothing works
1. All 8 surviving hadrons have ens3 score > 0.99 — the model is extremely confident
2. No single-model approach produces a lower score for these 8 without hurting gammas
3. The 8 events genuinely look like gammas in the detector at the pixel level
4. The muon spatial patterns of these hadrons match gamma spatial patterns

## What DOESN'T improve the ensemble (v26-v40)
- v26: different LR/class weight — no diversity
- v27: wider CNN (64-128-256 ch) — same as larger v8
- v28: spatial attention pooling — same as v9
- v29: knowledge distillation from v2 — same as v2
- v30: quality-cut training — WORSE (overfits smaller dataset)
- v31: 5-model diversity (seeds/lr/aug) — no improvement
- v32: HistGBM on quality-cut data — too few events
- v33: HistGBM on all data — GBM spatial stats weaker than CNN
- v34-v40: all failed (see above)

## Extended Analysis: v41-v47 Phase (all failed)

### v41: Deeper CNN (64-128-256-256) — 4.96e-04
- Added ne_nmu_exp feature, same result as v8 family

### v42: Full dataset training (no val holdout) — 6.42e-04
- Training on 1.53M instead of 1.38M makes no meaningful difference

### v43: Nmu regression model — 4.21e-02
- CNN predicts log10(Nmu) from spatial features (excluding Nmu)
- Key finding: ALL 8 surviving hadrons have near-zero residuals (actual ≈ predicted)
- Even event #26449 (Nmu=4.31, "donut" pattern): predicted 4.24, residual 0.07
- The regression model learns that "donut" patterns DO have high muon counts — it's accurate
- Blending with ens3 always hurts

### v44: Curriculum learning — 1.05e-03
- Full data (5 ep) → quality cuts (5 ep) → boundary focus (10 ep) → tight boundary (5 ep)
- Best checkpoint always at epoch 3 (full data, early training) — curriculum phases overfit
- Fine-tuning on smaller boundary datasets massively worsens val metric

### v45: PointNet on non-zero detector hits — 8.17e-04
- Treats each non-zero hit as a point with (x, y, log1p(el), log1p(mu)) features
- MAX_POINTS=64, global max pool, 147K params
- Standalone 8.17e-04 — not terrible! But blending HURTS ens3 (2.91e-04)
- Avg non-zero hits: 44.7 per event (so 64 is appropriate cap)
- PointNet provides no useful ensemble diversity for the 8 surviving hadrons

### v46: MC Dropout 100 samples on v2 — 2.92e-03
- KEY FINDING: 8 surviving hadrons have LOWER uncertainty than avg gamma
  - 7 events: mc_std = 0.0001-0.0013 (extremely confident gamma)
  - event 26449: mc_std = 0.0254 (slightly higher but still low)
  - Avg gamma: mc_std = 0.0054
- These events aren't near the decision boundary — model is EXTREMELY confident
- No uncertainty-based penalty can help without hurting gammas

### v47: GBM stacker on CNN scores + physics — 4.38e-04
- Loads v2, v21, v25 models, extracts val predictions
- Trains HistGBM on val meta-features (CNN scores + matrix stats + physics)
- KEY FINDING: GBM gives 8 surviving hadrons EVEN HIGHER scores than ens3
  - All 8 events score 0.994-0.999 in GBM (vs 0.990-0.997 in ens3)
- Blending at alpha=0.05 leaves ens3 unchanged (2.33e-04)

## Final Conclusion
After 47 experiments and ALL of the following approaches:
- CNN variants (depth, width, seeds, LR, augmentation, attention, cross-channel)
- MLP with spatial statistics
- Vision Transformer
- Physics-only discriminants (Nmu LLR, Ne/Nmu ratio)
- Gradient boosting (HistGBM on features, on CNN scores)
- Random Forest
- Knowledge distillation
- OHEM / hard example mining
- Direct metric optimization (AUC + survival loss)
- Muon autoencoder anomaly detection
- Nmu regression model
- Full dataset training
- Curriculum learning
- PointNet (point cloud representation)
- MC Dropout uncertainty estimation
- GBM stacking on CNN predictions

**ens3 at 2.33e-04 is the definitive best result for this dataset.**

The 8 surviving hadrons cannot be improved:
1. They are extremely confident gamma predictions (ens3 score > 0.99)
2. MC Dropout confirms near-ZERO prediction uncertainty for 7/8 events
3. GBM stacker gives them even HIGHER gamma scores than ens3
4. Nmu regression gives near-zero residuals (actual ≈ predicted Nmu)
5. 7 events genuinely have gamma-like Nmu values (below gamma median)
6. All approaches agree — this is the physical information limit of the dataset

The published baseline achieves 10²–10³ suppression at 30–70% gamma efficiency.
ens3 achieves ~4,200x suppression at 75% gamma efficiency — well beyond published results.
