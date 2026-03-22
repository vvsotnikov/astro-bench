# Gamma/Hadron Classification Journal

## Runs completed

### v4: Random Forest (attempt 1)
- 15 engineered features
- RF n_estimators=100, max_depth=20, class weights
- **Result: 2.04e-03** survival @ 75% γ eff
- 5× better than baseline (1.0e-02)
- Fast to train (CPU), good reference point

### v1: MLP with physics features (attempt 2)
- 23 engineered features (Ne/Nmu, muon stats, spatial moments, centroids)
- 3-layer MLP: 23 → 256 → 256 → 128 → 2
- 40 epochs, cosine annealing, class weights
- **Result: 7.30e-04** survival @ 75% γ eff
- **NEW BEST** — 3× better than v4
- Key: advanced feature engineering critical for physics-informed learning

## Strategy

### Phase 1: Foundation (v1-v5)
Establish baseline approaches with physics-informed feature engineering.

**v1: MLP with engineered features (23 features)**
- Physics features: Ne/Nmu ratio, muon channel sum/max/variance, electron stats
- Spatial moments: centroids of electron/muon distributions
- Architecture: 3-layer MLP (23 → 256 → 256 → 128 → 2)
- Training: 40 epochs, cosine annealing, class weights
- Expected: moderate improvement over raw baseline

**v2: CNN on muon channel (primary discriminator)**
- Key insight: Gammas → Nmu ≈ 3.0, Hadrons → Nmu ≈ 3.5
- Muon spatial information is key discriminator
- Scalar features: Ne/Nmu, zenith, log(Nmu)
- Architecture: Conv2d (muon) → pooling → MLP
- Expected: improved due to spatial structure exploitation

**v3: Dual-channel CNN with attention**
- Both electron and muon channels
- Learnable attention pooling (focus on discriminative regions)
- Scalar features: Ne/Nmu, zenith, energy, log(Ne)
- Architecture: 3-layer Conv2d → attention → MLP
- Expected: best of phase 1 (~5e-04 range)

**v4: Random Forest (classical baseline)**
- 15 engineered features (same physics-informed set as v1)
- n_estimators=100, max_depth=20, class weights
- Fast, no GPU required
- Expected: competitive with MLP, reference point

**v5: Residual MLP with log1p features**
- Advanced feature engineering (log1p normalization for rare features)
- Deeper residual blocks: 256 → 256 (residual) → 256 → 128 → 2
- Architecture: better feature balance
- Expected: 4-6e-04 range

### Phase 2: Ensemble & Cross-pollination (v6-v15)
- Combine top performers from phase 1
- Apply successful tricks (attention, log1p) to previously underperforming approaches
- Try 2-3 seed variants of top performers for stability

### Phase 3: Specialized architectures (v16+)
- Vision Transformers (spatial transformer approach)
- Graph Neural Networks (neighbor correlations in detector grid)
- Autoencoder + downstream classifier
- Knowledge distillation from ensemble

## Key physics insights
- **Muon channel is primary discriminator**: Gammas produce almost no muons (EM showers only)
- **Ne/Nmu ratio directly separates classes**: log10(Ne) - log10(Nmu) ≈ 0-1 difference between classes
- **Zenith angle matters**: Affects shower development, inclined showers more muonic
- **Sparse data**: ~85% of detector cells are zero, CNN must handle sparsity
- **Extreme imbalance**: Test set ~1:23 gamma:hadron, real world ~1:1,000,000

## Runs completed (updated)

### v2: CNN on muon channel (attempt 3)
- **Result: 1.75e-03** — worse than MLP
- Suggests spatial structure alone isn't enough

### v5: Residual MLP with log1p (attempt 4)
- **Result: 7.59e-04** — competitive with v1
- Log-normalized features helped slightly

### v3: Dual-channel CNN + attention (attempt 5) ⭐ NEW BEST
- **Result: 3.21e-04** — 2.3× better than v1!
- Key success: using both channels (e+μ) + attention pooling
- Architecture: Conv (32→64 channels) → attention → MLP
- This is the breakthrough approach

### Interpretation
- **Spatial information matters**: v3 >> v1/v5 (MLPs)
- **Dual channels crucial**: v3 uses electron+muon together
- **Attention is key**: Focus mechanism helps identify discriminative regions
- **Feature engineering has limits**: v1's 23 engineered features underperforms v3's raw spatial data

## Completed Experiments Summary (Mar 20, 18:30)

**Best Result**: v3 @ 3.21e-04 (dual-channel CNN + attention, 31× better than baseline)
**Attempts Used**: 13/50
**Remaining**: 37 attempts

### Key Learnings
1. **Ensemble averaging hurts performance**: Weighted combinations of v3+weaker models all worse than v3 alone
   - v25 (equal): 4.38e-04
   - v26 (v3-heavy): 4.09e-04
   - v27 (top 2): 4.67e-04
   - v28 (v3+v5+v1): 4.38e-04
   - All worse than v3's 3.21e-04!

2. **Seed ensemble abandoned**: v16 took 37+ minutes (likely GPU contention/hung)

3. **Architecture matters most**: CNN+attention >> MLP, dual-channel >> single

### Next Phase: Architecture Variations (v29+)
Focus on architectural tweaks that might beat v3, not ensemble strategies.
- v29: Skip connections in CNN (in progress, ~25 min remaining)
- v30: Deeper MLP after attention
- v31+: Different architectures (multi-head attention, spatial pyramid, etc.)

Avoid: seed ensembles (too slow), weighted averaging (doesn't help)

## Attempt accounting & Strategy (UPDATED)

- **Total budget**: 50 attempts
- **Completed**: 8 attempts (v4, v1, v2, v5, v3, v8, v13)
- **In pipeline**: 16 experiments queued in 3 phases
  - Phase 1 (Seed variants): v16, v17, v18, v19 - **RUNNING**
  - Phase 2 (Hyperparams): v21 (higher dropout), v22 (SGD optimizer) - **QUEUED**
  - Phase 3 (Ablations): v23 (no attention), v24 (50 epochs) - **QUEUED**
  - Parallel: Ensemble weight optimization - **RUNNING**
- **Expected after all 24**: ~26 attempts remaining for refinement

## Three-Phase Optimization Pipeline

### Phase 1: Seed Ensemble Diversity (v16-v19)
- Same architecture as v3, different random seeds (456, 789, 999, 1111)
- Goal: Reduce variance through ensemble averaging
- Expected individual results: 3.15-3.25e-04 each
- Expected 5-seed ensemble (v3+v16+v17+v18+v19): **~2.8-3.0e-04**

### Phase 2: Hyperparameter Search (v21-v22)
- v21: Higher dropout (0.3, 0.2) vs baseline (0.2, 0.1)
- v22: SGD with momentum (0.01, 0.9) vs AdamW
- Expected: One may outperform base v3

### Phase 3: Ablations + Training Duration (v23-v24)
- v23: Global avg pooling instead of attention (confirms importance of attention)
- v24: 50 epochs instead of 35 (tests convergence saturation)
- Expected: Should be similar to v3, confirming design choices

### Parallel: Weighted Ensemble Testing
- Equal weighting (1/5 each)
- v3-heavy (40% v3, 15% each seed)
- v3-dominant (60% v3, 10% each seed)
- Tests optimal ensemble combination

## Results summary
- **Best: v3 @ 3.21e-04** (dual-channel CNN + attention)
- v8: 4.38e-04 (deep MLP)
- v13: 4.38e-04 (ensemble v3+v1, simple average didn't help)
- v1: 7.30e-04 (MLP with engineered features)
- v5: 7.59e-04 (residual MLP with log1p)
- v2: 1.75e-03 (CNN on muon channel)
- v4: 2.04e-03 (RF on engineered features)
- Baseline: 1.0e-02 (published RF)
  - v3 variants (larger, more epochs, different seeds)
  - Ensembles of v3 with others
  - Vision Transformers or other architectures
  - Fine-tuning and ablations

## Key Findings & Lessons

### Architecture: CNNs Beat MLPs for Spatial Data
- **v3 (CNN + attention): 3.21e-04** ✓ BEST
- v8 (Deep MLP): 4.38e-04 (10x worse than v3)
- v1, v5 (MLPs with engineered features): 7-7.6e-04

**Insight**: Raw spatial data processed by CNN >> hand-engineered features in MLP. Attention mechanism is critical for success.

### What DIDN'T help
- Simple ensemble (v13: v3 + v1 averaged): 4.38e-04 (worse!)
- CNN on single channel (v2, muon-only): 1.75e-03 (4.6x worse)
- Deep MLPs with dropout: still worse than shallow CNN

### Physics Intuition Confirmed
- Dual channels (electron + muon) is essential
- Attention helps focus on discriminative spatial regions
- Scalar features (Ne/Nmu, zenith) help but aren't sufficient alone

### Remaining Experiments (in queue)
- v11: v3 with seed=123 (diversity for ensemble)
- v12: v3 with higher LR and StepLR schedule
- v14: v3 with higher dropout
- v15: v3 reproduction check
- Batch: v14_v15 to follow

### Recommendations for Final Phase (42 attempts left)
1. **Ensemble variations**: Weighted combinations of v3 variants (v11, v12, v14, v15)
2. **Cross-validation**: Train v3 on different random splits, ensemble predictions
3. **Hyperparameter sweep**: Try learning rates [0.5e-3, 2e-3], dropout [0.05-0.35]
4. **Architecture variants**: Add skip connections, batch norm, or attention at different layers
5. **If time**: Vision Transformer, GNN on spatial correlations

## Notes
- verify.py is ONLY official metric
- Validation split used for checkpoint selection
- All training redirected to log files
- One GPU job at a time (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 → cuda:0 in PyTorch)
