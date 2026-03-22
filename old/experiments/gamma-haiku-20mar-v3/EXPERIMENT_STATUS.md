# Gamma Hadron Classification - Wide Exploration Status

## Current Best Model
**v2: ResNet with muon emphasis @ 4.67e-04** (14x better than published baseline 1.00e-02)

## Attempts Used: 8/50

## Current Training Queue (Attempts 9-14+)

### CPU Models (Parallel Execution)
| Model | Architecture | Status | Est. Time |
|-------|---|---|---|
| v15 | GradientBoosting (sklearn) | Training | ~10 min |
| v20 | MuonCNN (seed=123) | Training | ~30 min |
| v21 | ResNet (seed=456) | Training | ~30 min |

### GPU Pipeline (Sequential)
| Model | Architecture | Status | Est. Time |
|-------|---|---|---|
| v23 | Wider ResNet (80/160 filters) | Training | ~15 min |
| v24 | Deeper CNN (3 conv blocks) | Queued | ~15 min |
| v25 | GELU activation variant | Queued | ~15 min |
| v26 | ResNet (lr=3e-4, 50 epochs) | Queued | ~25 min |
| v27 | ResNet (batch_size=256) | Queued | ~15 min |

### Ready for Additional Phases
- **v28**: Equal-weight ensemble of (v2, v4, v1)
- **v29**: ResNet with SGD optimizer (vs Adam)
- **v30**: Skip-rich CNN (UNet-style architecture)
- **v9**: DeepMuonCNN (3-layer conv)
- **v18**: Deep flattened MLP
- **v19**: Multi-seed ensemble of v2

## Exploration Strategy: WIDE (Not Deep)

Rather than optimizing a single architecture, testing diverse approaches:
1. **Model capacity**: wider (v23) vs deeper (v24)
2. **Nonlinearity**: GELU (v25) vs ReLU
3. **Optimization**: Adam (v2) vs SGD (v29), different LR/batch size (v26-v27)
4. **Training data**: multi-seed variants (v20, v21)
5. **Ensemble**: top-3 equal weight (v28)

## Key Learning from 8 Completed Experiments
✓ Physics-informed architecture (separate muon/electron paths) crucial
✓ ResNet > basic CNN by 35%
✓ Attention mechanisms don't help here
✓ Engineered features hurt (11x worse than CNN)
✓ Standard CrossEnt loss > Focal Loss

## Next Phase Triggers
- **If any v23-v27 beats v2**: investigate why and create variants
- **If all underperform**: focus on ensemble of top models
- **Target metric**: 3.5e-04 (matches mar9-v3 composition baseline)

## Monitoring
All models have automated monitoring running. Results auto-log to `results.tsv` as they complete.
