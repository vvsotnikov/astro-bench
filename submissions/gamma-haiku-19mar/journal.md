# Gamma/Hadron Separation — Research Journal

## Task
Binary classification of gamma-ray vs hadronic showers from KASCADE cosmic ray data.
- **Metric**: Hadronic survival rate @ 75% gamma efficiency (lower is better)
- **Published baseline**: ~10⁻² to 10⁻³ (RF on scalar features)
- **Data**: 1.5M training events (5% gamma, 95% hadron), 36K test events
- **Attempt budget**: 50 calls to verify.py

## Physics intuition
- Gamma showers are purely electromagnetic → almost no muons (median Nmu ≈ 3.0)
- Hadron showers (protons) produce muons (median Nmu ≈ 3.5)
- **Key discriminant**: Ne/Nmu ratio + muon channel of 16×16×2 matrices
- Zenith angle affects shower development
- Energy affects separation (better at higher E)

## Exploration strategy
1. Start with simple baselines (logistic regression, RF on features)
2. Understand data distributions and class separability
3. Try diverse architectures: CNN, MLP, ViT, GNN, autoencoders
4. Use both spatial (matrices) and scalar (features) data
5. Ensemble models with complementary architectures
6. Cross-pollinate insights across models

## Experiments log

### v1: Logistic Regression
- **Metric**: 5.46e-03 (hadron survival @ 75% gamma eff)
- **Approach**: Logistic regression on 5 scalar features (E, Ze, Az, Ne, Nmu)
- **Result**: KEEP — beats published baseline (10⁻²-10⁻³)!
- **Observations**: Very simple, already extremely good performance

### v2: Random Forest
- **Metric**: 4.32e-03 (hadron survival @ 75% gamma eff)
- **Approach**: Random Forest (200 trees) on 5 scalar features
- **Result**: KEEP — marginal improvement over logistic regression
- **Observations**:
  - Feature importance: Nmu dominates (0.527), then Ze (0.208), Ne (0.124), E (0.109)
  - Confirms Nmu as key discriminant

### v3: MLP (matrices + features)
- **Metric**: 1.43e-03 (hadron survival @ 75% gamma eff) **[MAJOR IMPROVEMENT]**
- **Approach**: MLP on flattened 16×16×2 matrices + 5 features (517 dims)
- **Architecture**: Linear(517, 512) → BN → ELU → Dropout(0.2) → Linear(512, 256) → BN → ELU → Dropout(0.2) → Linear(256, 2)
- **Training**: 20 epochs, val-based checkpoint selection, CrossEntropyLoss with class weights
- **Result**: KEEP — 3x better than RF!
- **Key insight**: Spatial structure of detector matrices adds significant value
- **Next steps**: Try CNN (exploit 2D structure), vision transformer, feature engineering

