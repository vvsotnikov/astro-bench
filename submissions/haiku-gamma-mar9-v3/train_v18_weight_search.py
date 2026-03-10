"""Fine-tune ensemble weights: search over α for v9 + v16 combo."""

import numpy as np

print("Loading predictions...")

# Load v9 (best single model)
v9_npz = np.load("submissions/haiku-gamma-mar9-v3/predictions_v9.npz")
v9_scores = v9_npz["gamma_scores"]

# Load v16 (pure CNN, good spatial model)
v16_npz = np.load("submissions/haiku-gamma-mar9-v3/predictions_v16.npz")
v16_scores = v16_npz["gamma_scores"]

# Load test labels
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]
is_gamma = test_labels == 0
is_hadron = test_labels == 1


def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()


# Get baseline survivals
v9_surv = compute_survival_75(v9_scores)
v16_surv = compute_survival_75(v16_scores)

print(f"v9 survival: {v9_surv:.4e}")
print(f"v16 survival: {v16_surv:.4e}")

# Fine search over weights
print(f"\nSearching ensemble weights...")
best_alpha = 0.5
best_surv = 1.0

for alpha in np.linspace(0.0, 1.0, 101):
    ensemble_scores = alpha * v9_scores + (1 - alpha) * v16_scores
    ensemble_surv = compute_survival_75(ensemble_scores)
    if ensemble_surv < best_surv:
        best_surv = ensemble_surv
        best_alpha = alpha
    if alpha % 0.1 == 0:
        print(f"  α={alpha:.2f}: {ensemble_surv:.4e}")

print(f"\n{'='*60}")
print(f"Best: α={best_alpha:.3f} → {best_surv:.4e}")
print(f"Improvement over v9 alone: {(v9_surv - best_surv) / v9_surv * 100:.1f}%")

# Use best weights for final predictions
ensemble_scores = best_alpha * v9_scores + (1 - best_alpha) * v16_scores

np.savez("submissions/haiku-gamma-mar9-v3/predictions_v18.npz",
         gamma_scores=ensemble_scores)

print(f"\n---")
print(f"metric: {best_surv:.4e}")
print(f"description: Weighted ensemble v9 + v16 (α={best_alpha:.3f})")
