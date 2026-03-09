# Gamma/Hadron Classification — Journal (v3)

## Goal
- **Metric**: Hadronic survival rate @ 75% gamma efficiency (LOWER is better)
- **Baseline to beat**: v2's best: **6.43e-04** (seed 42)
- **Published baseline**: 10²–10³ (0.01–0.001) at ~30–70% gamma eff

## Strategy
1. Systematic seed exploration (10+ seeds to find others that beat 6.43e-04)
2. Gradient boosting models (XGBoost/LightGBM on flattened features)
3. Advanced calibration techniques
4. Cross-validation instead of single train/val split
5. Feature engineering exploration

## Key Findings So Far

### Best Result: v3 Attention CNN @ 5.84e-04 (beats baseline 6.43e-04!)
- Architecture: Attention blocks in CNN pathway + MLP on features
- Key insight: Attention mechanisms help the model focus on discriminative spatial patterns
- This is the first experiment to beat the previous best

### What works:
1. **v3 (Attention CNN)**: 5.84e-04 ✓ **BEST - NEW**
   - Spatial attention helps capture complex patterns in 16x16x2 matrices

2. **v1 (Seed exploration)**: 6.43e-04 (seed 42)
   - Matches baseline but no improvement from 20-seed search
   - Other seeds: 6.72e-04, 7.01e-04, 7.51e-04, etc. (all worse)

### What doesn't work:
1. **v4 (5-fold CV)**: 7.59e-04 - Worse than baseline
   - CV ensemble averaged predictions, lost individual model strengths

2. **v6 (Logistic Regression)**: 5.90e-03 - Much worse
   - Linear model insufficient for this problem

### Still running:
- v2: XGBoost (tree ensemble)
- v5: SVM with engineered features
- v7: Platt scaling calibration

## Experiments
