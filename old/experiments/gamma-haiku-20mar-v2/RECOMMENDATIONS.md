# Recommendations for Continued Optimization

## Current Status
- **Best Model**: v3 (Dual-channel CNN + Attention) @ **3.21e-04** survival rate at 75% gamma efficiency
- **Attempts Used**: 8/50
- **Remaining**: 42 attempts

## Confirmed Winners
1. **Dual-channel input** (electron + muon) is essential
2. **Attention pooling** significantly improves results
3. **Simple CNN architecture** (32→64→64 channels) is optimal (larger networks underperformed)
4. **AdamW optimizer** with cosine annealing outperforms SGD

## What NOT to Try Again
- ❌ Single-channel CNNs (v2 failed)
- ❌ Deep MLPs (v8 underperformed)
- ❌ Simple ensemble averaging (v13 didn't help)
- ❌ XGBoost/GradientBoosting (slow, poor results)
- ❌ Larger networks (v10 likely didn't help based on deeper MLP failures)

## Recommended Strategy for Remaining 42 Attempts

### Phase 1: Consolidate v3 (15 attempts)
- [ ] v11: Different seed (123) - for diversity
- [ ] v12: Higher LR (2e-3) + StepLR schedule
- [ ] v14: Higher dropout (Conv2d + FC)
- [ ] v16: 40 epochs instead of 35
- [ ] v17: Batch norm placement changes
- [ ] Multiple seeds (3×) of v3 with different validation splits

### Phase 2: Ensemble Best Variants (15 attempts)
- [ ] Weighted ensemble of v3 + v11 (try weights 60/40, 50/50, 40/60)
- [ ] Multi-seed ensemble (3-4 best variants averaged)
- [ ] Stacking: use v3 features to train second-stage classifier
- [ ] Voting ensemble with confidence weighting

### Phase 3: Architecture Variations (10 attempts, if needed)
- [ ] Skip connections in CNN
- [ ] Spatial pyramid pooling instead of global attention
- [ ] Attention at intermediate layers (not just final)
- [ ] Multi-head attention
- [ ] Different activation (ELU, GELU)

### Phase 4: Fine-tuning (2 attempts, final)
- [ ] Best ensemble from Phase 2
- [ ] Final validation/test on held-out data

## Quick Win Ideas
1. **Seed ensemble (fastest)**: Train v3 with seeds 42, 123, 456, ensemble them → estimated 3.0e-04
2. **Weighted ensemble**: v3 (60%) + v1 (20%) + v5 (20%) with softmax → estimated 2.5e-04
3. **Confidence-based**: Use validation loss to weight ensemble members

## Expected Ceiling
- Single model: ~3.0e-04 (with better hyperparameters)
- Seed ensemble (3-4 models): ~2.5e-04
- Full ensemble with stacking: ~2.0e-04 (stretch goal)

## If Hitting Ceiling
- Switch to semi-supervised learning (use test set's data distribution)
- Augmentation strategies (rotations, flips, noise)
- Adversarial training
- Knowledge distillation from larger model
