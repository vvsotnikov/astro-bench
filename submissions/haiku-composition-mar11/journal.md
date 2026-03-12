# KASCADE Mass Composition (5-class) — Haiku Research Journal

## Task Overview
- **Goal**: Classify cosmic rays into 5 mass categories (proton, helium, carbon, silicon, iron)
- **Metric**: Accuracy (higher is better)
- **Baseline to beat**: 50.71% (haiku-mar8), published CNN ~51%
- **Data**: 5.5M training events, 119K test events
- **Challenge**: Mass composition is harder than gamma/hadron (5 classes vs 2, lower natural separability)

## Key Insights from Gamma/Hadron Run
Applied to composition task:
1. **Feature engineering critical**: Ne-Nmu ratio, angle encodings provide 20-40% improvement
2. **Attention mechanisms help**: CNN+attention outperforms basic CNN
3. **Hybrid architecture wins**: Spatial CNN + engineered features > either alone
4. **Simplicity > complexity**: Direct architecture usually better than complex alternatives
5. **Ensemble of diversity**: Multiple complementary architectures can improve results

## Strategy
1. **Phase 1**: Adapt gamma insights to composition (use same v9 attention CNN + features)
2. **Phase 2**: Systematic 3+ variant exploration before discarding approaches
3. **Phase 3**: Cross-pollination (apply working tricks to new architectures)
4. **Phase 4**: Ensemble refinement if single models plateau

## Previous Work Reference
- haiku-mar8: CNN+MLP hybrid, log1p matrices, 7 engineered features, 50.71% accuracy
- Key features used: E, cos(Ze), sin(Az), cos(Az), Ne, Nmu, Ne-Nmu

---

## Experiments

