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

## Key Learnings
1. DATA QUANTITY MATTERS: 2M samples -> 48% acc; 5.5M -> 50.7% acc
2. BOTH SIMULATIONS NEEDED: First 2M are all QGSJet, need to include EPOS too
3. ACCURACY != FRACTION ERROR: v5 had higher accuracy but worse fraction error than HGB
4. FLOAT32 TENSORDATASET is the fastest approach for full data training
5. Train accuracy (~37%) << test accuracy (~50%) because train has no quality cuts

## Next Steps
- Try longer training (40+ epochs) -- v8 was still improving at epoch 20
- Try ensembling CNN predictions with HGB
- Try ordinal regression (mass is ordered: H < He < C < Si < Fe)
- Try larger CNN / deeper MLP
- Try different loss functions (focal loss for hard examples)
- Optimize for fraction error directly rather than accuracy
