# Mass Composition Classification: Experiment Rundown

**Task**: 5-class cosmic ray classification (proton, helium, carbon, silicon, iron)
**Metric**: Accuracy (higher is better)
**Baseline to beat**: 50.71% (haiku-mar8)
**Published SOTA**: ~51% (CNN, JINST 2024)

---

## COMPLETED EXPERIMENTS

### Early Batch (v1-v8, v6b, v7b): Initial Exploration
**Status**: 8 completed, 4 crashed
**Best**: v1 @ 50.52% (Attention CNN + 8 features)

| # | Approach | Result | Notes |
|---|----------|--------|-------|
| v1 | Attention CNN + 8 eng. features | **50.52%** | Baseline. Close but not beating haiku-mar8 |
| v2 | Basic CNN (no attention) | 49.47% | Worse. Attention IS helpful |
| v3 | Basic CNN + log1p | 50.47% | Similar to v1, log1p doesn't help |
| v4 | ResNet (skip connections) | 50.41% | Similar to v1, skip connections don't help |
| v5 | haiku-mar8 replica attempt | CRASH | Dimension mismatch in transpose |
| v6 | RandomForest | CRASH | Infinity values in feature pipeline |
| v6b | RandomForest (safe preprocessing) | CRASH | Still had infinity issues |
| v7 | Logistic Regression | CRASH | Infinity in features |
| v7b | Logistic Regression (safe) | 37.99% | Too simple, far worse |
| v8 | XGBoost | CRASH | Infinity values |

**Key findings**:
- Attention helps (v1 50.52% > v2 49.47%)
- All CNN approaches ~50.4-50.5% (close to baseline but not beating it)
- Tree models had infinity issues (poor feature handling)
- Linear models too weak (v7b 37.99%)
- **Difference from haiku-mar8 must be in**: deeper architecture, different LR schedule, or different feature engineering

---

## CURRENTLY RUNNING

### v14: v1 + Longer Training, Lower LR
**Configuration**:
- Architecture: Same as v1 (Attention CNN + 8 features)
- 100 epochs (vs v1's 30)
- lr=3e-4 (vs v1's 1e-3)
- CosineAnnealingLR scheduler
- patience=15 (vs v1's 10)

**Hypothesis**: v1 may be undertraining at too high a learning rate. Longer training at lower LR might find better optima.

**Status**: Running on GPU 1, currently at epoch 6. ETA: ~25 more minutes.

---

## PLANNED NEXT EXPERIMENTS (Order TBD based on v14 results)

### If v14 Beats 50.71% (Next: v15 & v16)
Continue exploring v1 hyperparameter variants

### If v14 ~50.52% (Similar to v1)
Move to **v17 (Exact haiku-mar8 replica)**:
- 4 CNN blocks (32→32→64→64→128→128→256)
- OneCycleLR scheduler
- Batch size 4096
- 7 features (haiku-mar8 set, NO sin(Ze))
- log1p matrices

**Why**: haiku-mar8 got 50.71% with these exact settings. If v17 replicates it, we've identified the key differences. If not, there's something else (seed, preprocessing, etc).

### Other Queued Variants (If Exploring Further)
- **v15**: v1 + class-weighted loss (address class imbalance)
- **v16**: v1 + label_smoothing=0.1 (stronger regularization)
- **v17**: Exact haiku-mar8 replica (4 blocks, OneCycleLR, batch=4096, 7 features)
- **v18**: Deeper CNN (5 blocks)
- **v19**: Wider CNN (more channels)
- **v20**: Vision Transformer (non-CNN architecture test)
- **v21**: Pure MLP (no spatial structure test)
- **v22**: RandomForest with safe preprocessing
- **v23**: Focal loss variant
- **v24**: SGD optimizer variant
- **v26**: Data augmentation (90° rotations)

---

## Summary of Key Insights

### What Works
1. **Attention mechanisms help**: v1 (attention, 50.52%) > v2 (no attention, 49.47%)
2. **Engineered features help**: 8D features > raw scalars
3. **Hybrid architecture helpful**: CNN pathway + feature pathway + fusion
4. **Batch normalization important**: Standard in all working models

### What Doesn't Work
1. **Tree models crash on this data**: Feature infinity/NaN issues need careful handling
2. **Linear models too weak**: v7b (37.99%) far below neural nets
3. **Simple CNNs insufficient**: v2, v3, v4 all ~50.4% - not enough to beat 50.71%
4. **ResNet vs Attention similar**: v4 (50.41%) ≈ v1 (50.52%) - architecture family matters less than depth/training

### What We Don't Know Yet
1. **Is 50.71% replicable?** Need to try exact haiku-mar8 setup (v17)
2. **Can deeper CNNs help?** (v18 with 5 blocks)
3. **Can non-CNN architectures compete?** (v20 ViT, v21 MLP)
4. **Do tree models work with proper handling?** (v22 with safe preprocessing)
5. **Can loss function changes help?** (v23 focal loss)

---

## Lessons Learned

### Approach Evolution
1. **Early mistake**: Tried tree models without handling infinities → all crashed
2. **Early mistake**: Created 26 experiment scripts and tried to run them all at once → GPU contention/OOM
3. **Fixed**: Now running one experiment at a time, examining results before planning next one

### Correct Approach
1. Run ONE experiment
2. Wait for completion
3. Extract result, examine carefully
4. Log findings in journal
5. Decide next experiment based on findings
6. Repeat

### GPU Execution
- **Wrong**: Shell backgrounding with `&` (all processes run parallel)
- **Right**: Synchronous execution or Python `subprocess.wait()` (process blocks until completion)
- **Current**: Using synchronous bash: `command && echo DONE`

---

## Performance vs Baseline

```
Target: 50.71% (haiku-mar8)
Published SOTA: 51%

Current best: v1 @ 50.52%
  ├─ 0.19% below haiku-mar8
  └─ 0.48% below published SOTA

All CNN variants cluster around 50.4-50.5%
Linear models: 37.99% (too weak)
Tree models: Crashed (infinity handling)
```

---

## Data & Setup
- **Training data**: 5.5M cosmic ray events
- **Test data**: 119K events (with quality cuts pre-applied)
- **Classes**: 5 (proton, helium, carbon, silicon, iron)
- **Features**:
  - Spatial: 16×16×2 detector matrices (electron + muon density)
  - Scalar: Energy, zenith angle, azimuth angle, electron count, muon count
- **Engineered features** (8D):
  - E, Ze, Az, Ne, Nmu (raw)
  - Ne-Nmu (difference)
  - cos(Ze), sin(Ze) (angle encodings)
- **Alternative features** (7D, haiku-mar8):
  - E, cos(Ze), sin(Az), cos(Az), Ne, Nmu, Ne-Nmu
  - (Note: NO sin(Ze), has sin/cos of azimuth)

---

## Code Locations

**Training scripts**:
- `/submissions/haiku-composition-mar11/train_v1.py` through `train_v26.py`

**Results tracking**:
- `results.tsv` - Tabular results
- `journal.md` - Research journal with detailed notes
- `EXPERIMENT_RUNDOWN.md` - This file

**Utilities**:
- `run_sequential.py` - Proper sequential runner (not used now, but available)
- `monitor_experiments.py` - Auto-extract results from logs

---

## Next Immediate Step

**Waiting for v14 to complete** (currently at epoch 6 of 100, ~25 min remaining)

Once done:
1. Extract accuracy from `train_v14.log`
2. Compare to v1 baseline (50.52%) and haiku-mar8 (50.71%)
3. Log findings
4. **Decision**:
   - If v14 > 50.71% → iterate on this winner
   - If v14 ≈ 50.52% → try v17 (exact haiku-mar8 replica)
   - If v14 < 50.4% → something went wrong, investigate

This is the right approach: one step at a time, examining results carefully.
