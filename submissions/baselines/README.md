# DNN Baselines — Built by Claude Opus 4.6 (supervised)

## Summary

An AI agent (Claude Opus 4.6) built DNN baselines for both KASCADE tasks in a single session, supervised by a domain expert (V. Sotnikov). The composition baseline nearly matched years of expert work in 3 training runs. The gamma baseline failed completely at the key metric — an instructive result.

## Composition Task: 49.9% accuracy

| Attempt | Architecture | Test Accuracy | Notes |
|---------|-------------|---------------|-------|
| 1 (v1) | MLP (512×3, ELU+BN, dropout 0.3) | ~49.8% | 15 epochs, 796K params |
| 2 (v3) | Same + input normalization | ~49.8% | 15 epochs, 796K params |
| 3 (final) | MLP (512×2, ELU+BN, dropout 0.15) | **49.9%** | 30 epochs, 532K params |

**Published CNN baseline (JINST 2024): ~51% accuracy** — achieved by a team of 4 physicists over ~2 years of iteration, using a LeNet-5 CNN with 36K params trained on QGSJet-II.04 only, with carefully chosen features (log₁₀Ne, log₁₀Nμ, θ, s).

The agent's MLP uses a brute-force approach: flatten the 16×16×2 matrices into 512 dims, concatenate 5 scalar features, normalize, and classify. No physics-informed feature engineering, no convolutional structure, no domain knowledge beyond what was provided in `challenge.md`. Yet it reaches 49.9% — within 1.1 percentage points of the published result.

### Why the gap is small

The ablation study in the JINST paper (Fig. 8) showed that CNN on deposits-only vs features-only gives similar performance, and combining both gives only a marginal improvement. The spatial structure in 16×16 matrices provides limited additional information beyond what's captured by the scalar features. An MLP that sees all 517 dimensions can learn similar decision boundaries without explicit spatial convolutions.

### Why the gap exists

The remaining ~1% gap likely comes from:
- The published CNN uses domain-informed features (log₁₀Ne, log₁₀Nμ, θ, s) that are more physically meaningful than raw (E, Ze, Az, Ne, Nmu)
- Convolutional structure provides useful inductive bias for spatial patterns
- The published model was trained on QGSJet-II.04 only and tested on QGSJet-II.04 test data; our model trains on mixed QGS+EPOS data
- Our quality cuts differ slightly (Ze<30 vs Ze<18, no Nmu cut)

## Gamma Task: 0.90 survival rate (failure)

| Metric | Value |
|--------|-------|
| Test accuracy | 98.97% |
| Survival @ 50% gamma eff | 1.5×10⁻⁴ |
| Survival @ 90% gamma eff | 2.9×10⁻² |
| Survival @ 95% gamma eff | 3.7×10⁻¹ |
| **Survival @ 99% gamma eff** | **9.0×10⁻¹** |

**Published baseline (ICRC 2021): suppression 10²–10³** (survival ~10⁻²–10⁻³)

The 98.97% accuracy is a trap. The test set is 96% hadrons — predicting "hadron" for everything gives 95.8%. The model learned to identify easy gammas (the 50% efficiency point is excellent at 1.5×10⁻⁴ survival), but cannot push gamma efficiency to 99% without the decision boundary collapsing. At 99% gamma efficiency, the threshold drops to ~0 and 90% of hadrons survive.

### Why it fails

The fundamental issue: gamma rays are rare and their feature distributions overlap heavily with hadrons at the tails. A simple MLP with cross-entropy loss optimizes for average accuracy, not for the extreme tail of the score distribution where the 99% gamma efficiency operating point lives.

To achieve good performance at 99% gamma efficiency, a model likely needs:
- **Regression output** (continuous score) rather than classification, as in the ICRC 2021 RF regressor approach
- **Focal loss or class weighting** to focus on hard examples near the decision boundary
- **CNN architecture** to leverage spatial muon patterns (gammas produce almost no muons — this is a spatial signature)
- **Much more training data** for gammas (~91K gamma vs ~1.8M proton in training set)

### The instructive contrast

The composition task has roughly balanced classes (5 groups, each 15-28%) and a straightforward metric (accuracy). A naive MLP works well.

The gamma task has extreme imbalance (1:23 in test, 1:10⁶ in reality) and a metric that lives in the tail of the score distribution. The same naive approach fails catastrophically. This is exactly the kind of problem where domain expertise matters — the ICRC 2021 team knew to use a regression approach and to optimize the threshold for signal-to-background ratio.

## Reproduction

```bash
# Composition
CUDA_VISIBLE_DEVICES=0 uv run python submissions/baselines/train_composition_dnn.py
uv run python verify.py submissions/baselines/predictions_composition_dnn.npz

# Gamma
CUDA_VISIBLE_DEVICES=0 uv run python submissions/baselines/train_gamma_dnn.py
uv run python verify.py --task gamma submissions/baselines/predictions_gamma_dnn.npz
```

## Agent details

- **Model**: Claude Opus 4.6 (claude-opus-4-6)
- **Interface**: Claude Code CLI
- **Supervision**: Domain expert provided challenge.md, pointed out errors, answered questions
- **Total training runs**: 3 (composition) + 1 (gamma)
- **Wall time**: ~10 minutes training (GPU: available CUDA device)
- **Code iterations**: The agent wrote the training scripts from scratch based on challenge.md and prior baseline code in the repository
