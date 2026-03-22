# Gamma/Hadron Classification Journal

## Attempt 1: Simple MLP Baseline
- **Metric**: 1.14e-03
- **Architecture**: DNN with 2 hidden layers (512 units each), batch norm, ELU, dropout
- **Input**: Flattened 16×16×2 matrices + 5 scalar features = 517 total features
- **Training**: 30 epochs, AdamW, cosine annealing LR, class-weighted CE loss
- **Result**: GOOD baseline! 1.14e-03 survival at 75% gamma efficiency

## Key Insights
- The MLP baseline is solid. The published baseline achieves 10²–10³ at 30–70% efficiency, so 1.14e-03 is competitive.
- Gamma rays are rare (4.4% of training data), so class weighting helps significantly.
- The 16×16×2 spatial data matters—flattening and feeding to MLP works.
- Need to try: (1) architectural diversity (CNN, ViT, etc.), (2) feature engineering, (3) ensemble approaches.

## Next Experiments
1. **CNN-based approach**: Leverage spatial structure of the 16×16×2 matrices
2. **Feature engineering**: Engineered features like Ne/Nmu, log transforms, etc.
3. **Ensemble**: Combine multiple architectures
4. **Hybrid input**: Use CNN on matrices + separate path for features
