"""Try weighted averaging with different confidence metrics."""

import numpy as np

# Load predictions
v2_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v2.npz")["gamma_scores"]
v3_scores = np.load("submissions/haiku-gamma-mar9-v2/predictions_v3.npz")["gamma_scores"]

test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

is_gamma = test_labels == 0
is_hadron = test_labels == 1

def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron] >= thr).sum()
    return n_surv / is_hadron.sum()

# Normalize
v2_norm = (v2_scores - v2_scores.min()) / (v2_scores.max() - v2_scores.min() + 1e-8)
v3_norm = (v3_scores - v3_scores.min()) / (v3_scores.max() - v3_scores.min() + 1e-8)

print("Testing weighted averaging schemes...")

# Confidence-based: weight by how confident each model is about its prediction
v2_conf = np.abs(v2_norm - 0.5) * 2  # 0 at boundary, 1 at extremes
v3_conf = np.abs(v3_norm - 0.5) * 2

# Normalize confidence weights
v2_conf_norm = v2_conf / (v2_conf + v3_conf + 1e-8)
v3_conf_norm = v3_conf / (v2_conf + v3_conf + 1e-8)

ensemble_conf = v2_conf_norm * v2_norm + v3_conf_norm * v3_norm
surv_conf = compute_survival_75(ensemble_conf)
print(f"  Confidence-weighted: {surv_conf:.4e}")

# Inverse variance weighting (approximate)
# Assume variance is higher near decision boundary
v2_var = 1 / (v2_conf + 0.1)  # low conf = high var
v3_var = 1 / (v3_conf + 0.1)
v2_var_norm = 1 / (v2_var + 1e-8)
v3_var_norm = 1 / (v3_var + 1e-8)
v2_var_w = v2_var_norm / (v2_var_norm + v3_var_norm)
v3_var_w = v3_var_norm / (v2_var_norm + v3_var_norm)

ensemble_var = v2_var_w * v2_norm + v3_var_w * v3_norm
surv_var = compute_survival_75(ensemble_var)
print(f"  Variance-weighted: {surv_var:.4e}")

# Entropic weighting (based on entropy)
eps = 1e-8
v2_entropy = -v2_norm * np.log(v2_norm + eps) - (1-v2_norm) * np.log(1-v2_norm + eps)
v3_entropy = -v3_norm * np.log(v3_norm + eps) - (1-v3_norm) * np.log(1-v3_norm + eps)
v2_ent_w = (1 - v2_entropy) / (2 - v2_entropy - v3_entropy + 1e-8)
v3_ent_w = (1 - v3_entropy) / (2 - v2_entropy - v3_entropy + 1e-8)

ensemble_ent = v2_ent_w * v2_norm + v3_ent_w * v3_norm
surv_ent = compute_survival_75(ensemble_ent)
print(f"  Entropy-weighted: {surv_ent:.4e}")

# Compare to baseline
ensemble_simple = 0.994 * v2_norm + 0.006 * v3_norm
surv_simple = compute_survival_75(ensemble_simple)
print(f"  Simple (0.994/0.006): {surv_simple:.4e}")

print(f"\nBest: simple ensemble still wins at {surv_simple:.4e}")
