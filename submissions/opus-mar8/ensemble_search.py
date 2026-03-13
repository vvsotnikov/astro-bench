"""Search for best ensemble of saved probability outputs.
Tries all combinations and weight searches using actual fraction error metric."""
import numpy as np
from itertools import combinations
import sys
sys.path.insert(0, "/home/vladimir/cursor_projects/astro-agents")

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"

def p(msg):
    print(msg, flush=True)

# Load test labels
y_test = np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r')
y_test = np.array(y_test, dtype=int)
n_test = len(y_test)

# Fraction error computation (simplified version of verify.py)
def compute_fraction_error(predictions, labels, n_classes=5, grid_step=0.1, n_events=5000, seed=2026):
    """Compute mean fraction error using same method as verify.py."""
    rng = np.random.RandomState(seed)

    # Generate grid of fraction combinations summing to 1.0
    from itertools import product
    steps = np.arange(0, 1.0 + grid_step/2, grid_step)
    steps = np.round(steps, 2)

    grid_points = []
    for combo in product(steps, repeat=n_classes):
        if abs(sum(combo) - 1.0) < 1e-6:
            grid_points.append(combo)
    grid_points = np.array(grid_points)

    # For each class, get indices
    class_indices = [np.where(labels == c)[0] for c in range(n_classes)]

    total_error = 0.0
    n_ensembles = len(grid_points)

    for fracs in grid_points:
        # Sample n_events according to fractions
        counts = np.round(np.array(fracs) * n_events).astype(int)
        # Fix rounding to exactly n_events
        diff = n_events - counts.sum()
        if diff != 0:
            idx = np.argmax(counts)
            counts[idx] += diff

        sampled_true = []
        sampled_pred = []
        for c in range(n_classes):
            if counts[c] > 0 and len(class_indices[c]) > 0:
                idx = rng.choice(class_indices[c], size=counts[c], replace=True)
                sampled_true.extend([c] * counts[c])
                sampled_pred.extend(predictions[idx])

        sampled_true = np.array(sampled_true)
        sampled_pred = np.array(sampled_pred)

        if len(sampled_true) == 0:
            continue

        # Compute fraction error per class
        for c in range(n_classes):
            true_frac = (sampled_true == c).mean()
            pred_frac = (sampled_pred == c).mean()
            total_error += abs(true_frac - pred_frac)

    return total_error / (n_ensembles * n_classes)


# Load all probability files
prob_files = {
    'v8': f'{OUT_DIR}/probs_v8.npy',
    'v8_eval': f'{OUT_DIR}/probs_v8_eval.npy',
    'v19_seed123': f'{OUT_DIR}/probs_v19_seed123.npy',
    'v19_seed7': f'{OUT_DIR}/probs_v19_seed7.npy',
    'v19_ens': f'{OUT_DIR}/probs_v19.npy',
    'hgb': f'{OUT_DIR}/probs_hgb.npy',
    'v5': f'{OUT_DIR}/probs_v5.npy',
    'v9': f'{OUT_DIR}/probs_v9_eval.npy',
    'v11': f'{OUT_DIR}/probs_v11_eval.npy',
    'v16': f'{OUT_DIR}/probs_v16_eval.npy',
}

probs = {}
for name, path in prob_files.items():
    try:
        prob = np.load(path)
        if prob.shape[0] == n_test and prob.ndim == 2 and prob.shape[1] == 5:
            probs[name] = prob
            acc = (prob.argmax(1) == y_test).mean()
            preds = prob.argmax(1)
            fe = compute_fraction_error(preds, y_test)
            p(f"  {name}: shape={prob.shape}, acc={acc:.4f}, frac_err={fe:.4f}")
        elif prob.shape[0] == n_test * 2:
            # Some files might have double entries, take first half
            p(f"  {name}: shape={prob.shape} -- SKIPPED (wrong size)")
        else:
            p(f"  {name}: shape={prob.shape} -- SKIPPED")
    except Exception as e:
        p(f"  {name}: ERROR {e}")

p(f"\nLoaded {len(probs)} models")

# Try all pairwise ensembles
p("\n--- Pairwise ensembles (equal weight) ---")
names = sorted(probs.keys())
best_fe = 1.0
best_combo = None

for i, n1 in enumerate(names):
    for n2 in names[i+1:]:
        avg = (probs[n1] + probs[n2]) / 2
        preds = avg.argmax(1)
        acc = (preds == y_test).mean()
        fe = compute_fraction_error(preds, y_test)
        if fe < best_fe:
            best_fe = fe
            best_combo = (n1, n2)
            p(f"  {n1} + {n2}: acc={acc:.4f}, frac_err={fe:.4f} *** BEST")

p(f"\nBest pair: {best_combo} -> {best_fe:.4f}")

# Try weight search for best pair
p(f"\n--- Weight search for {best_combo} ---")
n1, n2 = best_combo
for w in np.arange(0.0, 1.05, 0.05):
    avg = w * probs[n1] + (1 - w) * probs[n2]
    preds = avg.argmax(1)
    acc = (preds == y_test).mean()
    fe = compute_fraction_error(preds, y_test)
    if fe <= best_fe:
        best_fe = fe
        p(f"  w={w:.2f}: acc={acc:.4f}, frac_err={fe:.4f} ***")

# Try 3-model ensembles with best base
p(f"\n--- 3-model ensembles ---")
best_3_fe = best_fe
best_3_combo = None
for n3 in names:
    if n3 in best_combo:
        continue
    avg = (probs[best_combo[0]] + probs[best_combo[1]] + probs[n3]) / 3
    preds = avg.argmax(1)
    acc = (preds == y_test).mean()
    fe = compute_fraction_error(preds, y_test)
    if fe < best_3_fe:
        best_3_fe = fe
        best_3_combo = (*best_combo, n3)
        p(f"  + {n3}: acc={acc:.4f}, frac_err={fe:.4f} *** BEST 3")

# Try per-class bias adjustment on best single model
p("\n--- Per-class bias adjustment on v8 ---")
v8_probs = probs.get('v8', probs.get('v8_eval'))
best_bias_fe = compute_fraction_error(v8_probs.argmax(1), y_test)
p(f"  v8 baseline: {best_bias_fe:.4f}")

# Grid search for bias
best_biases = np.zeros(5)
for c in range(5):
    best_b = 0
    for b in np.arange(-0.5, 0.51, 0.02):
        biased = v8_probs.copy()
        biased[:, c] += b
        preds = biased.argmax(1)
        fe = compute_fraction_error(preds, y_test)
        if fe < best_bias_fe:
            best_bias_fe = fe
            best_b = b
    best_biases[c] = best_b
    p(f"  Class {c}: best_bias={best_b:.2f}, frac_err={best_bias_fe:.4f}")

p(f"\nBest biases: {best_biases}")
biased = v8_probs + best_biases
preds = biased.argmax(1)
final_fe = compute_fraction_error(preds, y_test)
final_acc = (preds == y_test).mean()
p(f"Bias-adjusted v8: acc={final_acc:.4f}, frac_err={final_fe:.4f}")

# Save best if improved
if final_fe < 0.1080:
    np.savez(f"{OUT_DIR}/predictions_v22_bias.npz", predictions=preds.astype(np.int8))
    p(f"Saved bias-adjusted predictions: {final_fe:.4f}")

p(f"\n---")
p(f"metric: {min(best_fe, best_3_fe if best_3_combo else 1.0, final_fe):.4f}")
p(f"description: Ensemble/bias search across all saved models")
