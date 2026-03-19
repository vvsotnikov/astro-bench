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

### v1: Logistic Regression (baseline)
- **Metric**: 5.46e-03 (hadron survival @ 75% gamma eff)
- **Approach**: Logistic regression on 5 scalar features (E, Ze, Az, Ne, Nmu)
- **Result**: KEEP — already beats published baseline (10⁻²-10⁻³)!
- **Observations**:
  - Very simple model, no hyperparameter tuning
  - Already achieves extreme suppression at 75% gamma eff
  - At 50% gamma eff: 1.05e-03 (even better)
  - Weak energy dependence: good at low E, perfect at high E
  - Suggests features contain strong gamma/hadron signal
- **Next steps**: Try models with spatial data (matrices), different architectures

