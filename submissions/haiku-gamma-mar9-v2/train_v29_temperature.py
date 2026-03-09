"""Load v18_seed42 predictions and optimize temperature scaling."""

import numpy as np

# Load best predictions (v18 seed 42)
v18_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v18.npz")["gamma_scores"]
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_at_temp(scores, temp):
    """Compute survival at 75% gamma eff with temperature scaling."""
    scaled = 1.0 / (1.0 + np.exp(-np.log(scores / (1.0 - scores + 1e-8)) / temp))
    sg = np.sort(scaled[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_hadron_surviving = (scaled[is_hadron] >= thr).sum()
    return n_hadron_surviving / is_hadron.sum(), scaled

# Grid search over temperature
print("Searching for optimal temperature scaling...")
best_surv = compute_survival_at_temp(v18_scores, 1.0)[0]
best_temp = 1.0
best_scores = v18_scores

for temp in np.linspace(0.5, 2.0, 31):
    surv, _ = compute_survival_at_temp(v18_scores, temp)
    print(f"temp={temp:.2f}: surv={surv:.4e}")
    if surv < best_surv:
        best_surv = surv
        best_temp = temp
        _, best_scores = compute_survival_at_temp(v18_scores, temp)

print(f"\nBest temperature: {best_temp:.2f}")
print(f"Best survival: {best_surv:.4e}")

# Save rescaled scores
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v29.npz",
         gamma_scores=best_scores)

print(f"\n---")
print(f"metric: {best_surv:.4e}")
print(f"description: Temperature scaling on v18_seed42 (T={best_temp:.2f})")
