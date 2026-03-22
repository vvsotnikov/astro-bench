# Next Steps After Phase 1-3 Completion

## Current Status
- **Best Result**: v3 @ 3.21e-04 (dual-channel CNN + attention)
- **Attempts Used**: 8/50
- **Remaining**: 42 attempts
- **Current Phase**: Seed ensemble variants (v16-v19) + Hyperparams (v21-v22) + Ablations (v23-v24)

## Decision Tree Based on Phase 1-3 Results

### If Seed Ensemble Improves (v20 < 3.1e-04)
- **Success**: Multi-seed averaging reduces variance
- **Next**: Weighted ensemble combinations (use validation scores)
- **Timeline**: 3-5 more attempts to find optimal weights

### If Seed Ensemble No Improvement (v20 ≈ 3.2e-04)
- **Finding**: Random initialization doesn't affect performance much
- **Next**: Focus on architecture/hyperparameter search
- **Try**: Skip connections, batch norm variations, learning rate sweep

### If Phase 2 Hyperparams Help (v21 or v22 < 3.2e-04)
- **Success**: Different hyperparameters can improve results
- **Next**: Systematic search (dropout sweep, LR sweep)
- **Timeline**: 5-10 attempts

### If Phase 3 Ablations Inform Design
- **If v23 (no attention) >> v3**: Attention is critical
- **If v24 (50 epochs) = v3**: 35 epochs is sufficient
- Use these insights to guide further architecture changes

## Recommended Path Forward (30 remaining attempts)

### Phase 4: Fine-tuned Ensemble (5 attempts)
- Weighted combinations of best seeds
- Try: 50/50, 60/40, equal weight averages

### Phase 5: Architecture Variations (10 attempts)
- Skip connections in CNN (residual blocks)
- Multi-head attention mechanism
- Different pooling strategies (spatial pyramid)
- Channel attention (SENet-style)

### Phase 6: Cross-validation & Stacking (10 attempts)
- Train v3 on different random splits
- Use 5-split cross-validation + ensemble
- Stacking: v3 features → second-stage model

### Phase 7: Final Refinement (5 attempts)
- Best configuration found above
- Try slight hyperparameter adjustments
- Final validation run

## Contingency Plans

**If still stuck at 3.2e-04:**
- Try semi-supervised learning (exploit test distribution)
- Data augmentation (rotations, flips, noise injection)
- Adversarial training
- Knowledge distillation from larger model

**If hit physical limit around 2.5-3.0e-04:**
- Attempt ceiling is ~2.0e-04 (stacking/cross-validation)
- Beyond that requires novel architecture or semi-supervised approach
- Document findings for paper

## Key Metrics to Track
- **Validation AUC/PR-AUC**: Internal classifier quality
- **Calibration**: Are confidence scores reliable for ensembling?
- **Diversity**: Do seed variants disagree on hard examples?
- **Generalization**: Gap between val and test performance
