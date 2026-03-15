"""Analyze confusion matrices per energy bin and compute theoretical fraction error floors."""
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
PREV_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
PARTICLE_NAMES = ["proton", "helium", "carbon", "silicon", "iron"]

def p(msg):
    print(msg, flush=True)

y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)
features = np.array(np.load(f"{DATA_DIR}/composition_test/features.npy"), dtype=np.float32)
energies = features[:, 0]

# Load v8 and v11 probs
probs_v8 = np.load(f"{PREV_DIR}/probs_v8.npy")
probs_v11 = np.load(f"{PREV_DIR}/probs_v11_eval.npy")
avg = (probs_v8 + probs_v11) / 2

# Raw predictions
raw_preds = avg.argmax(1)
# Bias-optimized predictions
log_avg = np.log(avg + 1e-10)
bias = np.array([-0.3767, -0.2214, -0.0944, 0.0163, 0.0581])
bias_preds = (log_avg + bias).argmax(1)

p("=== Overall confusion matrix (raw) ===")
cm_raw = confusion_matrix(y_test, raw_preds, labels=[0,1,2,3,4])
cm_norm = cm_raw / cm_raw.sum(axis=1, keepdims=True)
p(f"Accuracy: {accuracy_score(y_test, raw_preds):.4f}")
abbr = [n[:2] for n in PARTICLE_NAMES]
p("     " + " ".join(f"{a:>6}" for a in abbr))
for i in range(5):
    p(f"  {abbr[i]:>2} " + " ".join(f"{cm_norm[i,j]:>6.3f}" for j in range(5)))

p("\n=== Overall confusion matrix (bias-optimized) ===")
cm_bias = confusion_matrix(y_test, bias_preds, labels=[0,1,2,3,4])
cm_bias_norm = cm_bias / cm_bias.sum(axis=1, keepdims=True)
p(f"Accuracy: {accuracy_score(y_test, bias_preds):.4f}")
p("     " + " ".join(f"{a:>6}" for a in abbr))
for i in range(5):
    p(f"  {abbr[i]:>2} " + " ".join(f"{cm_bias_norm[i,j]:>6.3f}" for j in range(5)))

# Per-energy-bin confusion matrices
energy_bins = [
    (14.0, 15.0, "14-15"),
    (15.0, 15.5, "15-15.5"),
    (15.5, 16.0, "15.5-16"),
    (16.0, 16.5, "16-16.5"),
    (16.5, 18.0, "16.5-18"),
]

for lo, hi, label in energy_bins:
    mask = (energies >= lo) & (energies < hi)
    if mask.sum() < 100:
        continue
    p(f"\n=== Confusion matrix: E=[{label}] (n={mask.sum()}) ===")
    cm = confusion_matrix(y_test[mask], raw_preds[mask], labels=[0,1,2,3,4])
    cm_n = cm / cm.sum(axis=1, keepdims=True)
    acc = accuracy_score(y_test[mask], raw_preds[mask])
    p(f"Accuracy: {acc:.4f}")
    p("     " + " ".join(f"{a:>6}" for a in abbr))
    for i in range(5):
        p(f"  {abbr[i]:>2} " + " ".join(f"{cm_n[i,j]:>6.3f}" for j in range(5)))

    # Bias-optimized
    cm_b = confusion_matrix(y_test[mask], bias_preds[mask], labels=[0,1,2,3,4])
    cm_bn = cm_b / cm_b.sum(axis=1, keepdims=True)
    acc_b = accuracy_score(y_test[mask], bias_preds[mask])
    p(f"\nBias-opt accuracy: {acc_b:.4f}")
    p("     " + " ".join(f"{a:>6}" for a in abbr))
    for i in range(5):
        p(f"  {abbr[i]:>2} " + " ".join(f"{cm_bn[i,j]:>6.3f}" for j in range(5)))

# Theoretical fraction error floor
p("\n=== Theoretical fraction error given confusion matrix ===")
p("If C is the row-normalized confusion matrix, and f is the true fraction vector,")
p("then predicted fraction = C^T @ f. Fraction error = |f - C^T @ f|.")
p("Mean error = mean over grid of mean over classes of |f - C^T @ f|.")

def generate_fraction_grid(n_classes=5, step=0.1):
    n_steps = round(1.0 / step)
    fractions = []
    def _recurse(remaining, depth, current):
        if depth == n_classes - 1:
            current.append(remaining * step)
            fractions.append(current[:])
            current.pop()
            return
        for i in range(remaining + 1):
            current.append(i * step)
            _recurse(remaining - i, depth + 1, current)
            current.pop()
    _recurse(n_steps, 0, [])
    return np.array(fractions)

fracs = generate_fraction_grid()
p(f"Grid: {len(fracs)} points")

# For the overall confusion matrix
for name, cm_n in [("raw", cm_norm), ("bias-opt", cm_bias_norm)]:
    C = cm_n.T  # C[j, i] = P(pred=j | true=i)
    errors = []
    for f in fracs:
        pred_f = C @ f
        err = np.abs(f - pred_f).mean()
        errors.append(err)
    mean_err = np.mean(errors)
    p(f"\nTheoretical floor ({name}): {mean_err:.6f}")

# What would a perfect confusion matrix (identity) give?
C_id = np.eye(5)
errors = []
for f in fracs:
    pred_f = C_id @ f
    err = np.abs(f - pred_f).mean()
    errors.append(err)
p(f"Perfect classifier: {np.mean(errors):.6f}")

# What's the gap between theoretical and actual?
p("\n=== Gap analysis ===")
p("The actual fraction error includes sampling noise from the")
p("1001-grid evaluation. The theoretical floor is the expectation.")
p("Any gap indicates either sampling variance or non-stationarity of the confusion matrix.")

# Compute per-class prediction rates
for name, preds in [("raw", raw_preds), ("bias", bias_preds)]:
    pred_fracs = np.bincount(preds, minlength=5)[:5] / len(preds)
    true_fracs = np.bincount(y_test, minlength=5)[:5] / len(y_test)
    p(f"\n{name}: pred_fracs={np.round(pred_fracs, 4).tolist()} true_fracs={np.round(true_fracs, 4).tolist()}")
    p(f"  bias = pred - true: {np.round(pred_fracs - true_fracs, 4).tolist()}")

# How different are v8 and v11?
preds_v8 = probs_v8.argmax(1)
preds_v11 = probs_v11.argmax(1)
agree = (preds_v8 == preds_v11).mean()
p(f"\nv8 vs v11 agreement: {agree*100:.2f}%")

# Where do they disagree? What are the error patterns?
disagree_mask = preds_v8 != preds_v11
p(f"Disagreements: {disagree_mask.sum()} ({disagree_mask.sum()/len(y_test)*100:.1f}%)")
if disagree_mask.sum() > 0:
    # Who is right when they disagree?
    v8_right = ((preds_v8 == y_test) & disagree_mask).sum()
    v11_right = ((preds_v11 == y_test) & disagree_mask).sum()
    avg_right = ((raw_preds == y_test) & disagree_mask).sum()
    p(f"  v8 correct on disagreements: {v8_right} ({v8_right/disagree_mask.sum()*100:.1f}%)")
    p(f"  v11 correct on disagreements: {v11_right} ({v11_right/disagree_mask.sum()*100:.1f}%)")
    p(f"  avg correct on disagreements: {avg_right} ({avg_right/disagree_mask.sum()*100:.1f}%)")

if __name__ == "__main__":
    pass
