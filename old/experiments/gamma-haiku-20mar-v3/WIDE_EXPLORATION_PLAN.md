# Wide Exploration Campaign — Gamma/Hadron Binary Classification

**Objective:** Test diverse approaches across 42 remaining attempts (8/50 used)  
**Current Best:** v2 ResNet @ 4.67e-04 (14× better than published baseline 1.00e-02)  
**Target:** Beat v2 OR build strong ensemble reaching ~3.5e-04

---

## PHASE 1: In Progress (Attempts 9-14)

**Status:** 6 models training now
- GPU will complete in order: v23 → v24 → v25 → v26 → v27 (each ~15-25 min)
- CPU completing in parallel: v15 (GradientBoosting), v20, v21

| Model | Type | Key Hypothesis | ETA |
|-------|------|---|---|
| v23 | GPU | Wider model (more capacity) | ~15 min |
| v24 | GPU | Deeper model (vs width) | ~15 min |
| v25 | GPU | GELU activation (vs ReLU) | ~15 min |
| v26 | GPU | Lower LR (3e-4) + 50 epochs | ~25 min |
| v27 | GPU | Large batch (256 vs 128) | ~15 min |
| v15 | CPU | GradientBoosting ensemble | ~20 min |
| v20 | CPU | MuonCNN, seed=123 | ~30 min |
| v21 | CPU | ResNet, seed=456 | ~30 min |

---

## PHASE 2: Ready (Attempts 15-19)

Models will auto-queue when Phase 1 completes:

| Model | Architecture | Exploration |
|-------|---|---|
| v28 | Ensemble (v2+v4+v1) | Equal-weight combination |
| v29 | ResNet + SGD | Optimizer exploration |
| v30 | UNet-style CNN | Skip connections everywhere |
| v31 | ResNet no dropout | Regularization sweep |
| v32 | ResNet dropout=0.5 | Regularization sweep |

---

## PHASE 3-5: Queued (Attempts 20+)

Additional diversity pending Phase 1-2 results:
- Channel ablation (muon-only, electron-only)
- Multi-seed ensemble strategies
- Refinements of best-performing architecture
- Cross-architecture combinations

---

## How to Track Progress

**Watch results appear:**
```bash
tail -f results.tsv
```

**Latest metric:**
```bash
tail -1 results.tsv | awk -F'\t' '{print "Attempt", NR, ":", $2, "-", $NF}'
```

**Check active processes:**
```bash
ps aux | grep train_v | grep python
nvidia-smi
```

---

## Key Design Decisions

✓ **Wide > Deep:** Exploring many approaches rather than optimizing one  
✓ **GPU Sequential:** One training job at a time avoids OOM/contention  
✓ **CPU Parallel:** Multi-seed variants run in parallel (no GPU conflict)  
✓ **Auto-logging:** All results written to results.tsv immediately  
✓ **No polling:** Models run autonomously; check results when ready  

---

## Success Criteria

- **Primary:** Beat v2's 4.67e-04 with any single model
- **Secondary:** Ensemble reaching 3.5e-04 (match composition baseline)
- **Insight:** Understand what architecture choices matter on this data

---

## Quick Links

- **Journal:** journal.md (experiment log)
- **Results:** results.tsv (auto-updated leaderboard)
- **Config:** CLAUDE.md (task specification)
