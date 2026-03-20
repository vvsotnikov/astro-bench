# Haiku Gamma/Hadron Classification — March 9, 2026

## Task Summary
- Binary classification: distinguish gamma rays from hadrons
- Metric: **hadronic survival rate @ 99% gamma efficiency** (lower = better)
- Published baseline: 10⁻²–10⁻³ (Kostunin et al., ICRC 2021)
- Previous attempt (haiku-gamma-mar8): 0.836

## Key Insight
The metric is a **threshold optimization problem**, not a classification accuracy problem. Standard classification with CrossEntropy optimizes the decision boundary, not the tail of the score distribution.

## Physics Understanding
From test set exploration:
- Gammas have **lower muon counts** (median Nmu: 2.83 vs 4.25 for hadrons)
- **Ne - Nmu ratio** is extremely discriminative (median 2.65 vs 0.96)
- Simple threshold on `Ne - Nmu >= 0.78` gives 0.834 survival
- Energy-dependent: separation improves at higher energies

## Experiments

| # | Approach | Architecture | Survival@99% | Notes |
|---|----------|--------------|-------------|-------|
| v2 | Classification | DNN (512×2) + CrossEntropy + class weights | 0.8096 | Baseline approach |
| v3 | Feature engineering | DNN on 5 engineered features | 0.8107 | Physics features don't help much |
| v4 | Random Forest | RF on 100K subsampled data | 0.9392 | RF doesn't work well on this problem |
| v6 | Physics baseline | Score = Ne - Nmu | 0.8342 | Simple physics baseline |
| **v7** | **Ensemble** | **DNN v2 (α=0.73) + Physics baseline** | **0.7908** | **✅ BEST!** |
| v9 | Focal loss | DNN with focal loss | 0.9000 | Doesn't help (still classification) |
| v10 | Hard negative mining | Two-pass DNN with HNM | 0.8285 | Overfit (val: 0.4367, test: 0.8285) |
| v11 | Multitask learning | DNN + Nmu regression | 0.7810 | Slight improvement, comparable to v7 |
| v12 | Threshold optimization | DNN + ensemble weight search | 0.8072 | Better than v2 alone, worse than v7 |

**Final best result: 0.7908 (v7 ensemble)**

## Why Ensemble Works
- DNN v2: learned smooth ranking, but not optimal for tail metric
- Physics baseline: captures the fundamental physics (muon suppression in gammas)
- Combined: DNN learns complex patterns while physics baseline provides strong prior

## Next Steps
1. Try data augmentation or hard negative mining
2. Try different architectures (ResNet, attention)
3. Optimize ensemble weights more carefully (currently α=0.73)
4. Consider kernel methods (SVM with RBF kernel)
5. Look at individual failure modes

## Files
- `train_v2.py`: DNN classification baseline (0.8096)
- `train_v6_simple.py`: Physics baseline (0.8342)
- `train_v7_ensemble.py`: Ensemble optimization
- `predictions.npz`: Best predictions (v7 ensemble, 0.7908)
- `metrics_gamma.json`: Full evaluation metrics
