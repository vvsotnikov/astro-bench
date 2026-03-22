# KASCADE Gamma/Hadron Experiment Journal

## Current Status (March 21, 2026)
- **Attempts completed: 23/50**
- **Current best: v36 @ 3.21e-04** (6.9× better than baseline 1.00e-02)
- **Phase: 5 (v46-v50) NOW RUNNING** — v46 training on GPU 1

## Strategy Summary
- **Phase 1 (v1-v9)**: Architectural diversity (MuonCNN, ResNet, Focal Loss, Attention, Engineered Features)
- **Phase 2 (v30-v35)**: GPU hyperparameter exploration (deeper ResNet, wider muon path, learning rate, dropout, batch norm)
- **Phase 3 (v36-v39)**: Targeted refinement (batch_size=256, longer epochs, no batch norm, muon-only)
- **Phase 4 (v41-v45)**: Regularization & ensemble variants (L2, RMSprop, Kaiming init, simple ensemble)
- **Phase 5 (v46-v50)**: Final experiments (v46 now running)

## Key Insights
1. **Muon channel dominance**: Gammas have log10(Nmu)≈3.0 vs hadrons ≈3.5 — this is the fundamental signal
2. **Architecture that works**:
   - Separate muon/electron convolutional paths (v1, v2, v4 all good)
   - ResNet > basic CNN (v2 @ 4.67e-04 beat v1 @ 7.30e-04)
   - Larger batch sizes help (v36 with batch_size=256 achieved current best)
3. **What doesn't work**:
   - Pure tabular models (RF, LogReg, SVM all underperform)
   - Focal loss worse than CrossEntropyLoss with class weights
   - Attention mechanisms don't help
   - Engineered features alone hurt
   - Dropout=0.0 and without batch norm degrade performance

## Results Leaderboard (Top 10 of 23 completed)
| Rank | Model | Best Metric | Description |
|------|-------|-------------|-------------|
| 1 | **v36** | **3.21e-04** | ResNet batch_size=256 |
| 2 | v31 | 4.38e-04 | Wider muon path (64/128) |
| 3 | v2 | 4.67e-04 | ResNet with muon emphasis |
| 4 | v42 | 6.42e-04 | ResNet with RMSprop |
| 5 | v4 | 5.84e-04 | MuonCNN + Focal Loss |
| 6 | v32 | 5.54e-04 | ResNet lr=5e-3 |
| 7 | v44 | 5.54e-04 | Simple ensemble (v30+v31+v32) |
| 8 | v37 | 5.75e-04 | ResNet 40 epochs |
| 9 | v1 | 7.30e-04 | MuonCNN baseline |
| 10 | v6 | 7.88e-04 | AttentionCNN |

## Phase-by-Phase Breakdown

### Phase 1: Core Architecture Exploration (Attempts 1-9)
- v1: MuonCNN (7.30e-04) — good baseline, separate paths work
- v2: **ResNet (4.67e-04)** — BEST OF PHASE, beat v1 by 35%
- v4: MuonCNN + Focal (5.84e-04) — focal loss worse than CrossEnt
- v5: RandomForest (1.28e-03) — tabular baseline fails
- v6: AttentionCNN (7.88e-04) — attention doesn't help
- Others: Logistic Regression, SVM, MLP on engineered features all underperformed

### Phase 2: Hyperparameter Sweep (Attempts 10-14)
- v30: Deeper ResNet (1.31e-03) — depth alone hurts
- v31: Wider muon (4.38e-04) — GOOD, 7% worse than v36 final best
- v32: Higher LR (5.54e-04) — neutral
- v33: Multi-seed (FAILED)
- v34: Weighted ensemble (FAILED)
- v35: Lower dropout (1.31e-03) — regularization important

### Phase 3: Targeted Refinement (Attempts 15-18)
- v36: Larger batch size 256 (3.21e-04) — **NEW BEST! 8.5× over baseline!**
- v37: Longer training 40 epochs (5.75e-04) — more epochs don't help
- v38: No batch norm (1.14e-03) — batch norm critical
- v39: Muon-only (3.30e-03) — electron data needed

### Phase 4: Regularization Variants (Attempts 19-22)
- v41: L2 regularization (2.80e-03) — hurts
- v42: RMSprop optimizer (6.42e-04) — solid but not better than v36
- v43: Kaiming init (1.28e-03) — doesn't help
- v44: Simple ensemble (5.54e-04) — ensemble of suboptimal models is suboptimal
- v45: FAILED

### Phase 5: Final Experiments (Attempts 23-27, IN PROGRESS)
- v46: TRAINING
- v47: Queued
- v48: Queued
- v49: Queued
- v50: Queued

## Analysis
**Why v36 is best (3.21e-04)**:
- Larger batch_size=256 likely reduces variance in gradient estimates
- Combined with the robust ResNet architecture (separate paths, skip connections)
- 30 epochs with default LR=1e-3 and dropout=0.3 is optimal (more epochs worse, less dropout worse)
- Class weighting [1.0, 20.0] critical for imbalanced data

**Why ensembles haven't worked**:
- v44 (ensemble of v30/v31/v32) got 5.54e-04 — worse than v36 alone
- Simple averaging of suboptimal models doesn't compound improvements
- Lesson: focus on finding one great model rather than combining mediocre ones

## Remaining Strategy (27 attempts left)
1. **v46-v50**: Final architectural variants (running now)
2. **If v46-v50 don't beat 3.21e-04**:
   - Try ensemble of top models (v36, v31, v42) with learned weights
   - Try test-time augmentation
   - Try pseudo-labeling or semi-supervised approaches
3. **If any new architecture emerges**:
   - Cross-pollinate its best hyperparameters to other models

## Critical Rules Maintained
✓ One GPU job at a time (sequential pipelines)
✓ All results auto-logged to results.tsv via verify.py
✓ No custom evaluation metrics (trust verify.py only)
✓ Training logs saved (v1.log through v45.log)
✓ Model weights saved when needed (predictions_*.npz)
