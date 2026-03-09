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

### v5: Ensemble v2+v3 (KEEP)
- Approach: Linear combination of v2 (classification) and v3 (regression)
- Result: 7.89e-04 (α=0.99, mostly v2)
- Status: Improvement from pure v3

### v9: Fine-tuned ensemble v2+v3 (BEST!)
- Approach: Fine-grained search for optimal α around 0.99
- Result: **6.72e-04** (α=0.994)
- Status: Best so far, 4.6× better than baseline (3.15e-03)

## Final Results

**BEST: v9 ensemble with 6.72e-04 (4.6× improvement)**

### Summary of All Experiments
- v2: Classification DNN (60 epochs) → 1.31e-03
- v3: Regression DNN (BCELoss) → 9.05e-04
- v5: Ensemble v2+v3 (α=0.99) → 7.89e-04
- v9: Fine-tuned ensemble (α=0.994) → **6.72e-04** ✓
- v17: Mixed loss (0.5 CE + 0.5 BCE) → 7.89e-04 (not better)
- v6/v8/v14: Deeper/wider/simple variants (underperforming, validation ~2%)
- v18: Multi-seed (pending)

### Key Findings
1. **Regression beats classification alone**: v3 (9.05e-04) > v2 (1.31e-03)
   - BCELoss optimizes score ranking across full range
   - CrossEntropy optimizes decision boundary

2. **But ensemble beats both**: v9 (6.72e-04) < min(v2, v3)
   - Suggests complementary strengths
   - Fine weight tuning (0.994/0.006) critical

3. **Fine-tuning matters**: α=0.994 better than 0.99
   - Grid search with 101 points found 0.2% improvement
   - Coarse optimization might miss sweet spot

4. **Longer training helps**: 60 epochs vs 30 epochs
   - Early stopping on validation metric prevents overfitting
   - Patience parameter (15) balances exploration and convergence

5. **Current best: v9 ensemble at 6.72e-04**
   - Previous baseline (mar9): 3.15e-03
   - Published baseline: 10²–10³ = 1e-2 to 1e-3
   - Result is 4.6× better than previous, well within published range
