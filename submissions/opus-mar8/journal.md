# Opus-mar8 Research Journal

## Experiment Plan
- Task: 5-class mass composition (proton, helium, carbon, silicon, iron)
- Metric: mean fraction error (lower is better). Current best to beat: 0.1080
- Published CNN baseline: 0.107

### Understanding the metric
The fraction error metric samples random mixtures of 5 classes and checks how well predicted fractions match true fractions. This means:
- Systematic biases matter more than random errors
- A well-calibrated confusion matrix is critical
- Ne-Nmu ratio is the strongest discriminant (physics: iron has more muons)
- Energy dependence is key -- classification is harder at low energies

### Strategy
1. Start with RF on engineered features (quick baseline)
2. Hybrid CNN+MLP (proven approach from prior runs)
3. Try to optimize for fraction error specifically
4. Ensemble methods

---

## Experiment Log

### v1_rf (0.1195 frac error, 46.2% acc)
- RF 500 trees on 16 engineered features
- Applied quality cuts (Ze<30, Ne>4.8) to train -- only 8.6% of data passes (no Age feature)
- Quick baseline (0.8 min)

### v2_hgb (0.1182, 46.4% acc)
- HistGradientBoosting, 2M train no cuts
- Slightly better than RF on fraction error
- Interesting: HGB better on fraction error despite similar accuracy

### v3-v4 (killed)
- v3: Data loading issues with mmap fancy indexing (hung for 10+ min)
- v4: 3M samples (1.5M QGS + 1.5M EPOS), killed due to slow epochs (2.5 min/ep)

### v5_cnn_attn (0.1203, 48.1% acc) -- DISCARDED
- CNN+Attn+MLP with AMP, 2M train (1M each sim), 30 epochs
- Higher accuracy than HGB but WORSE fraction error (0.1203 vs 0.1182)
- KEY INSIGHT: accuracy and fraction error are not perfectly correlated
- Helium fraction error was very high (0.1638)

### v6 (killed)
- Tried mmap random access for full 5.5M data -- way too slow due to page cache thrashing
- Each item read from mmap disk when shuffled

### v7 (killed)
- Float16 preload with per-item conversion -- still slow
- The per-item .float() conversion in __getitem__ was the bottleneck

### v8_cnn_full (0.1080, 50.75% acc) -- BEST
- CNN+Attn+MLP with AMP, ALL 5.5M data as float32 TensorDataset
- 11.4GB RAM for matrices, ~136s/epoch, 20 epochs, 46 min total
- Label smoothing 0.05
- Matches prior run's best (0.1080) and published CNN baseline (~0.107)
- Per-class fraction errors: pr=0.0862, he=0.1279, ca=0.1260, si=0.1225, ir=0.0774

### v11_long_cosine (DISCARDED, ~50.70% acc)
- Same architecture as v8 but 40 epochs with cosine annealing
- Peaked at epoch 7 with 50.70% acc, then plateaued/oscillated
- Killed at epoch 27 -- no improvement trend
- LEARNING: longer training with same architecture doesn't help

### v13_calibration (0.1075 frac error, 50.82% acc) -- MARGINAL
- Temperature scaling on v8 probabilities: no effect (argmax invariant to temperature)
- Per-class bias optimization via differential evolution: tiny improvement (0.1080 -> 0.1075)
- Confusion matrix analysis: helium is the weakest class (36% confused as proton, 20% as carbon)
- Fraction bias: proton over-predicted (+1.1%), helium under-predicted (-2.4%), iron over-predicted (+1.6%)
- LEARNING: calibration alone can't fix fundamental classification errors

### v14_ensemble (0.1079 frac error, 50.82% acc) -- DISCARDED
- Fine-grained sweep of CNN+HGB linear mix: best at w_cnn=0.89
- Tried: geometric mean, rank fusion, LR stacking, HGB stacking
- None beat the CNN alone on fraction error
- LEARNING: CNN and HGB make similar errors; different architectures needed for ensemble gain

### v15_hgb_deep (0.1186 frac error, 46.52% acc) -- DISCARDED
- HGB with 33 extended features (zenith-corrected, quadratic, log ratios)
- 3M train, depth=10, 1500 iterations
- Extra features actually hurt -- noise added, proton over-predicted even more
- LEARNING: more features != better for HGB; the 16 original features were already good

### v16_focal (killed at ep3, 49.65% peak)
- Focal loss gamma=2 with inverse-frequency class weights
- Accuracy declining: 49.65% -> 48.79% -> 45.99% after 3 epochs
- LEARNING: focal loss too aggressive for this problem, class weights near-uniform anyway

### v17_wider (killed at ep4, 49.94% peak)
- Wider CNN: 64-128-256-512 channels (3M params vs 1M), LR=5e-4, bs=2048
- Slow convergence (259s/epoch), only reached 49.94% in 4 epochs
- LEARNING: wider CNN + lower LR converges too slowly; 16x16 grids don't need huge CNNs

### v18_tta (0.1081 frac error, 50.80% acc) -- DISCARDED
- Test-time augmentation: 4 rotations + 2 flips on v8 model
- Rotated versions give similar accuracy (50.57-50.75%), averaging helps marginally
- LEARNING: TTA barely helps; the spatial patterns are already well-captured

### Retroactive verification of all saved models
- v8: 0.1080, v9: 0.1090, v11: 0.1084, v16: 0.1120
- v8+v11 ensemble: 0.1080 (tied), v8+v9+v11: 0.1081
- Same-architecture models are too correlated for effective ensembling

### v19_multiseed (0.1078 frac error, 50.84% acc) -- DISCARDED
- 3-seed ensemble: v8 (seed42, 50.75%) + seed123 (50.70%) + seed7 (50.64%)
- Best combo: v8*2 + s123 + s7 weighted = 0.1078
- All combos in 0.1078-0.1080 range
- LEARNING 12: MULTI-SEED ENSEMBLES DON'T HELP: same arch different seeds too correlated

## Key Learnings
1. DATA QUANTITY MATTERS: 2M samples -> 48% acc; 5.5M -> 50.7% acc
2. BOTH SIMULATIONS NEEDED: First 2M are all QGSJet, need to include EPOS too
3. ACCURACY != FRACTION ERROR: v5 had higher accuracy but worse fraction error than HGB
4. FLOAT32 TENSORDATASET is the fastest approach for full data training
5. Train accuracy (~37%) << test accuracy (~50%) because train has no quality cuts
6. LONGER TRAINING DOESN'T HELP: v11 peaked at epoch 7, same as v8 peaked early
7. CALIBRATION IS MARGINAL: Post-hoc bias adjustment gives <0.001 improvement
8. HELIUM IS THE BOTTLENECK: 2.4% fraction bias, worst per-class accuracy
9. CNN+HGB ENSEMBLE NOT EFFECTIVE: models make too-similar errors
10. WIDER/DEEPER != BETTER: 16x16 grids are small; v8's 1M params is already sufficient
11. ALWAYS RUN VERIFY.PY: accuracy is a poor proxy for fraction error
12. MULTI-SEED ENSEMBLES DON'T HELP: same architecture, different seeds too correlated

## Next Steps (Phase 2 continued)
- v23 bias optimization already beats published baseline (0.1061 vs 0.107)
- Key insight: bias optimization is the most effective post-processing technique
- Need to train a fundamentally better base model for bigger gains
- Ensemble diversity matters more than ensemble size
- Try MixUp augmentation (proven calibration improvement)
- Try deeper feature engineering specific to each class
- Try training with class-balanced sampling
- Cross-validate bias optimization to check for overfitting

### v20_resnet_cbam (CRASHED -- killed)
- ResNet with CBAM (channel+spatial attention), 3.3M params
- 500s/epoch (vs 136s for v8), way too slow for 30 epochs
- LEARNING: Spatial attention on 16x16 grids adds compute without benefit

### v21_resnet_lite (CRASHED -- killed)
- ResNet with channel attention only, 1.7M params
- OneCycleLR max_lr=2e-3 too aggressive, test acc crashed from 49.6% to 44.5%
- LEARNING: OneCycleLR needs careful max_lr tuning

### v23_bias_opt (0.1061 frac error, 50.83% acc) -- NEW BEST
- v8+v11 average probabilities + per-class bias optimization
- Biases: [-0.154, -0.084, -0.043, -0.008, 0.006]
- Reduces proton and helium prediction rates (they were over-predicted)
- Improvement: 0.1080 -> 0.1061 (1.8% relative improvement)
- KEY INSIGHT: Bias correction > model improvement > ensembling
- Per-class: pr=0.0819, he=0.1239, ca=0.1251, si=0.1224, ir=0.0770

### v24_ordinal (0.1092 frac error, 50.72% acc) -- DISCARDED
- Same v8 architecture but with ordinal soft labels (sigma=0.5) + KL loss
- Similar accuracy to v8 but worse fraction error
- The soft labels spread probability to neighbors, which may hurt calibration
- LEARNING: Ordinal structure doesn't help fraction error metric

## Key Learnings (updated)
13. BIAS OPTIMIZATION IS POWERFUL: DE optimization of per-class logit biases gave 0.1080->0.1061
14. ENSEMBLE + BIAS BEATS INDIVIDUAL BIAS: v8+v11 ensemble is a better base than v8 alone
15. ORDINAL SOFT LABELS DON'T HELP: fraction error metric not improved by ordinal structure

## Broad Exploration Plan (Phase 2)
18 experiments done, best still v8 at 0.1080. Need radical diversification.

### Architecture families to try (3+ configs each before discarding)
1. ResNet with skip connections
2. Vision Transformer (ViT) with patch embeddings
3. Pure MLP on matrix statistics + scalar features
4. Depthwise separable CNN (MobileNet-style)
5. 1D CNN on flattened matrix
6. Graph Neural Network (active detectors as nodes)

### Training strategy variations
1. MixUp / CutMix augmentation (improves calibration)
2. Quality cuts on train data (match test distribution)
3. Spatial augmentation (rotations/flips during training)
4. Ordinal regression (exploit mass ordering)
5. Curriculum learning (easy classes first)
6. Class-balanced sampling

### EDA needed
1. Per-energy-bin confusion matrices
2. Feature importance for helium vs proton
3. Train vs test distribution shift
4. Spatial patterns by class
