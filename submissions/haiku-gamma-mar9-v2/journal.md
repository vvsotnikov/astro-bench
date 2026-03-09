# Gamma/Hadron Classification — Journal

## Goal
- **Metric**: Hadronic survival rate @ 75% gamma efficiency (LOWER is better)
- **Baseline to beat**: Previous run (mar9) achieved **3.15e-03**
- **Published baseline**: 10²–10³ (0.01–0.001) at ~30–70% gamma eff

## Physics Understanding
From previous run:
- **Gammas**: median log₁₀(Nmu) ≈ 2.83, Ne-Nmu ≈ 2.65
- **Hadrons**: median log₁₀(Nmu) ≈ 4.25, Ne-Nmu ≈ 0.96
- Ne-Nmu ratio is the strongest single discriminant

## Approach Strategy
The previous agent tried:
1. Classification DNN (CrossEntropy)
2. Physics baseline (Ne-Nmu)
3. Ensemble (0.73 * DNN + 0.27 * physics) → **3.15e-03**

Ideas to beat this:
1. **Calibration**: Post-train isotonic regression on DNN scores
2. **Better ensemble weights**: Cross-validated grid search
3. **Feature engineering**: Log transforms, polynomial features
4. **Energy-dependent models**: Separate classifiers per energy bin
5. **Anomaly detection**: Isolation forest approach
6. **Better DNN**: Try different architectures (wider, deeper, residual)
7. **SVM/RBF**: Different score ranking function
8. **Threshold optimization**: Data-driven threshold selection beyond ensemble

## Experiments

### v1: Re-weighted ensemble (DISCARD)
- Approach: Grid search over ensemble weights (α ∈ [0,1])
- Result: 3.15e-03 (α=1.00, pure DNN)
- Status: No improvement over baseline

### v2: DNN with longer training (KEEP)
- Approach: 60 epochs, higher dropout (0.3), lower LR (5e-4)
- Result: **1.31e-03** (much better!)
- Key insight: Longer training helps; validation metric drops steadily
- Status: Major improvement, 2.4× better than baseline

### v3: Regression-based DNN (BEST!)
- Approach: BCELoss with sigmoid output (direct score regression)
- Result: **9.05e-04** (excellent!)
- Key insight: Regression loss is better than classification for tail metric
- Status: Best so far, 3.5× improvement over baseline

### v4: SVM on physics features (in progress)
- Will try RBF kernel SVM on engineered features

## Key Findings
1. **Regression beats classification**: BCELoss directly optimizes ranking, not decision boundary
2. **Longer training helps**: 60 epochs with early stopping on validation metric
3. **Current best: v3 regression at 9.05e-04** (was baseline 3.15e-03)
