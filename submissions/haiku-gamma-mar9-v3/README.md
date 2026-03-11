# KASCADE Gamma/Hadron Classification — Haiku-Gamma-Mar9-v3

**Result**: 3.21×10⁻⁴ hadronic survival @ 75% gamma efficiency (50% better than baseline)

## Quick Links

- **Best Model**: v41 ensemble (`train_v41_ensemble_best.py`)
- **Best Single Model**: v9 Attention CNN (`train_v9_attention_features.py`)
- **Full Analysis**: `EXPERIMENT_SUMMARY.md` (89 experiments, lessons learned)
- **Data Analysis**: `DATA_PIPELINE_ANALYSIS.md` (distribution shifts, feature separability)
- **EDA Report**: `eda_report.txt` (class imbalance, sparsity, physics regimes)
- **Research Journal**: `journal.md` (detailed experiment log, phase-by-phase progress)
- **Experiment Results**: `results.tsv` (all 89 experiments with metrics and status)

## Model Architecture

### v41: 3-Model Weighted Ensemble ← **USE THIS**

```python
# Ensemble weights (from grid search optimization)
predictions = 0.70 × v9_score + 0.10 × v38_score + 0.20 × v27b_score
```

**Component Models**:

1. **v9** (70% weight): Attention CNN + Engineered Features
   - Spatial: Conv2d(2→32→64→128) with attention pooling
   - Features: 8D engineered vector (E, Ze, Az, Ne, Nmu, Ne-Nmu, cos(Ze), sin(Ze))
   - Result: 3.50×10⁻⁴ single-model

2. **v38** (10% weight): ResNet-style CNN + Skip Connections
   - Spatial: Conv2d with residual shortcuts
   - Features: Same 8D engineered vector
   - Result: 3.80×10⁻⁴ single-model

3. **v27b** (20% weight): Vision Transformer (2×2 patches)
   - Spatial: 64 patch tokens → 3-layer Transformer → global context
   - Features: Same 8D engineered vector
   - Result: 5.55×10⁻⁴ single-model

**Why This Works**:
- **Complementary inductive biases**: CNN learns local patterns, ResNet residual highways, ViT global context
- **Error diversity**: Each architecture misses different cases → ensemble corrects
- **Feature engineering**: Ne-Nmu ratio critical (40% improvement across all models)

## Key Findings from 89 Experiments

### What Worked
1. **Feature Engineering** (Ne-Nmu: +40% across all architectures)
2. **Ensemble Diversity** (3 models > any single, +8% improvement)
3. **Simplicity** (v9's straightforward design beats alternatives)
4. **Spatial Learning** (CNNs 100× better than tree models)

### What Failed
- Single-model tweaks (focal loss, augmentation, longer training)
- Radical reformulations (Nmu prediction 54× worse!)
- Complex paradigms (adversarial, MoE, multi-task)

## Verification

```bash
uv run python verify.py --task gamma submissions/haiku-gamma-mar9-v3/predictions_v41.npz
# Expected: metric: 3.21e-04
```

## Files

- `predictions_v41.npz` — Final submission (gamma_scores key)
- `train_v41_ensemble_best.py` — Ensemble code
- `train_v9_attention_features.py` — Best single model
- `EXPERIMENT_SUMMARY.md` — Complete analysis of 89 experiments
- `FINAL_ANALYSIS.md` — Architecture decisions and lessons learned
- `DATA_PIPELINE_ANALYSIS.md` — Feature analysis and data insights
- `eda_comprehensive.py` / `eda_report.txt` — Exploratory data analysis
- `results.tsv` — All experiment metrics
- `journal.md` — Detailed research log

