# Optimization Strategy Summary

## Current Best: v3 @ 3.21e-04 (Attempt 8/50)
**Dual-channel CNN with attention pooling**

## Active Pipeline (Attempts 9-24)

### Phase 1: Seed Ensemble (v16-v19)
Currently executing in rapid_ensemble.sh:
- v16 (seed=456) - **TRAINING NOW** (7/~25 min)
- v17 (seed=789) - Queued after v16
- v18 (seed=999) - Queued after v17  
- v19 (seed=1111) - Queued after v18

**Expected**: Each seed should achieve 3.15-3.25e-04
**Hypothesis**: 5-seed ensemble should reach 2.8-3.0e-04 via variance reduction

### Phase 2: Hyperparameter Search (v21-v22)
Queued in phase2_queue.sh (starts after v16-v19):
- v21: Higher dropout (0.3, 0.2) - tests regularization balance
- v22: SGD optimizer (lr=0.01, momentum=0.9) - vs AdamW

**Goal**: Find if tweaking hyperparams beats base v3

### Phase 3: Ablations & Extended Training (v23-v24)
Queued in phase3_queue.sh (starts after v21-v22):
- v23: Global avg pooling instead of attention - validates attention importance
- v24: 50 epochs instead of 35 - checks training saturation

**Goal**: Confirm our design decisions are optimal

### Parallel: Ensemble Weight Optimization
Running test_ensemble_weights.py (waits for v16-v19 predictions):
- Equal weighting (1/5 each)
- v3-heavy (40/15/15/15/15)
- v3-dominant (60/10/10/10/10)

**Goal**: Find best ensemble combination

## Expected Outcomes

### Conservative Estimate (2.9-3.1e-04)
5-seed ensemble reaches middle of estimated range

### Optimistic Estimate (2.7-2.9e-04)
Combination of:
- Best seed variant
- Optimal ensemble weights
- Possible hyperparameter improvement (v21 or v22)

## Remaining Strategy (Attempts 25-50)

After phase 3 completes, with ~26 attempts left:
1. **Refinement**: Use best variant from phases 1-3 as baseline
2. **Cross-validation**: Train on different data splits, ensemble
3. **Advanced architectures** (if needed): ViT, skip connections, multi-head attention
4. **Fine-tuning**: Last 5-10 attempts for marginal gains

## Key Unknowns

1. **How much does seed variance matter?** → Phase 1 answers this
2. **Are our hyperparameters optimal?** → Phase 2 answers this
3. **Is attention critical?** → Phase 3 (v23) answers this
4. **What's the realistic ceiling?** → All together will show this

## Timeline

- **~15-20 min**: v16 + start v17
- **~40-50 min**: Phase 1 complete, ensemble weights calculated
- **~70-80 min**: Phase 2 complete  
- **~100-110 min**: Phase 3 complete
- **~120 min total**: All 24 experiments done, comprehensive results ready

Then final refinement phase with remaining 26 attempts.

