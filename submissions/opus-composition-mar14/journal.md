# Opus Composition Mar14 — Research Journal

## Goal
Beat 0.1060 fraction error (previous best with bias optimization).
Leaderboard shows 0.1080 (raw CNN+RF ensemble).

## Strategy
1. Energy-conditional bias optimization on existing v8+v11 probs
2. Train improved base models (better architecture, augmentation)
3. Sophisticated post-processing (confusion matrix correction, energy-conditional)

## Key insights from previous runs
- 0.1060 is hard floor with per-class biases (5 params)
- Helium is bottleneck: 2.4% fraction bias, worst per-class accuracy
- Accuracy != fraction error (v5 higher acc but worse frac err)
- Data quantity matters: 5.5M >> 2M
- Cross-arch ensembles don't break through 0.1060
- 8.6% of train data matches test distribution (quality cuts)

---

## Experiment Log

### EDA Findings
- Train/test energy shift: 75% of train is E<15, 93% of test is E>=15
- Class balance shift: test has more proton (27.6%) vs train (~21%)
- Ne-Nmu separation is much clearer in test (quality-cut, higher E)
- proton-helium: d_ratio=0.26, silicon-iron: d_ratio=0.16 (worst pairs)

### v1b_energy_bias (0.1060 → 0.105990)
- Energy-conditional bias optimization with Nelder-Mead
- 8 equal-count energy bins, each with 5 biases
- Marginal improvement: 0.106022 → 0.105990
- The biases across bins are nearly identical — model is already well-calibrated
- LEARNING: Energy-conditional biases are saturated. The model's probability
  calibration doesn't have energy-dependent systematic bias to correct.

### Confusion Matrix Analysis (KEY INSIGHT)
- **Theoretical floor: raw=0.1078, bias-opt=0.1059**
- Actual scores: raw=0.1080, bias-opt=0.1060
- The model is AT ITS THEORETICAL LIMIT given its confusion matrix
- No post-processing can improve beyond the confusion matrix floor
- Must improve the confusion matrix itself to make progress
- Helium recall: 36-40%, with 29-37% confused as proton
- v8 vs v11 agree 89%, equally poor on disagreements (37-38% correct)

### v2_augment_finetune (0.1078, 50.81%)
- CNN+Attn+MLP with random 90° rotations + 18 engineered features
- Phase 1: 20 epochs on all 5.5M (best 50.81%)
- Phase 2: 10 epochs fine-tune on quality-cut events (no improvement)
- Augmentation caused training instability but final accuracy matched v8
- LEARNING: Random rotations are roughly neutral on 16x16 KASCADE grids
- LEARNING: Fine-tuning on quality-cut subset doesn't help (too few events, same features)

### v4_spatial (RUNNING)
- CNN+Attn + explicit spatial feature extraction from matrices
- Features: center of mass, spread, kurtosis, non-zero fraction per channel
- Different seed (7) for ensemble diversity

### v3_confmat_correction (RUNNING)
- Full 5x5 matrix correction on log probabilities
- Per-class temperature scaling
- Energy-conditional matrix correction
- Energy-polynomial correction

## Key Learnings
1. Model is at theoretical confusion matrix floor (0.1059 vs actual 0.1060)
2. Energy-conditional biases are saturated — model well-calibrated across E
3. Random rotations are neutral augmentation for KASCADE grids
4. Fine-tuning on quality-cut subset doesn't help
5. To break barrier: need fundamentally better confusion matrix
6. Helium discrimination is the bottleneck (~37% recall)
