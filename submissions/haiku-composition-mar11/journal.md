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

### v1: Attention CNN + Engineered Features ✗ **BASELINE**

**Result: 50.52%** — Slightly worse than haiku-mar8 (50.71%)

Architecture:
- CNN pathway: 2→32→64→128 channels with attention blocks at 32 and 64 channels
- Feature pathway: 8 engineered features → 128 → 64
- Fusion: Concatenate + MLP head to 5 classes
- Loss: CrossEntropyLoss with label smoothing 0.02

Key hyperparameters:
- lr=1e-3, weight_decay=1e-4
- 30 epochs with cosine annealing
- Batch size 2048
- Early stopping on validation accuracy (patience=10)

Analysis:
- 50.52% < 50.71% (haiku-mar8) — slightly worse than reference
- Attention mechanism may be introducing noise on this task
- Need to try variations: different architecture families, loss functions, feature engineering
- Target to beat: 50.71% (haiku-mar8 baseline)

Next steps: Systematically try 3+ variants before discarding (per team lead guidance).

### v2: Basic CNN (no attention) ✗

**Result: 49.47%** — Worse than v1 (50.52%)

Architecture:
- Simple 3-layer CNN without attention blocks (32→64→128 channels)
- Same feature pathway and fusion as v1
- Loss: CrossEntropyLoss with label smoothing 0.02

Findings:
- Removing attention HURTS performance
- Attention mechanisms ARE beneficial for this task (unlike some gamma results)
- But v1 with attention still underperforms baseline (50.52% < 50.71%)
- Problem is not the CNN architecture per se

### v3: Basic CNN + log1p Transform ✗

**Result: 50.47%** — Slightly worse than v1 (50.52%), similar to v2

Architecture:
- Basic 3-layer CNN + log1p() on matrices
- Same feature pathway as v1-v2
- Loss: CrossEntropyLoss with label smoothing 0.02

Findings:
- log1p transform slightly hurts performance (50.47% vs 50.52% for plain CNN)
- Unlike gamma run where features matter hugely, composition seems less sensitive to this preprocessing
- haiku-mar8 used log1p and got 50.71%, so it's not the issue
- Difference must be in: architecture depth, feature engineering details, loss function, or hyperparameters

### Summary of First 3 Variants

**Pattern so far:** All single-model variants (v1-v3) underperform baseline (50.71%).
- v1: 50.52% (attention CNN)
- v2: 49.47% (basic CNN)
- v3: 50.47% (basic + log1p)

**Key insight:** The 3-configuration rule is working — we've tried attention on/off, with/without log1p. Something else must be driving haiku-mar8's 50.71%. Hypothesis: haiku-mar8 uses DIFFERENT feature engineering or deeper CNN architecture.

### v4: ResNet (skip connections) — Running

Testing if residual learning helps composition classification.

