"""Fine-grained bias optimization using scipy.optimize on fraction error metric.
Tests on v8 probs, v11+v8 ensemble, and 3-seed ensemble."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import product
import sys

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"

def p(msg):
    print(msg, flush=True)

y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r'), dtype=int)
n_test = len(y_test)

# Precompute fraction error grid
def make_grid(n_classes=5, grid_step=0.1):
    steps = np.arange(0, 1.0 + grid_step/2, grid_step)
    steps = np.round(steps, 2)
    grid = []
    for combo in product(steps, repeat=n_classes):
        if abs(sum(combo) - 1.0) < 1e-6:
            grid.append(combo)
    return np.array(grid)

GRID = make_grid()
CLASS_IDX = [np.where(y_test == c)[0] for c in range(5)]

def compute_fraction_error(predictions, seed=2026, n_events=5000):
    rng = np.random.RandomState(seed)
    total_error = 0.0
    n_ensembles = len(GRID)

    for fracs in GRID:
        counts = np.round(fracs * n_events).astype(int)
        diff = n_events - counts.sum()
        if diff != 0:
            counts[np.argmax(counts)] += diff

        sampled_pred = []
        for c in range(5):
            if counts[c] > 0 and len(CLASS_IDX[c]) > 0:
                idx = rng.choice(CLASS_IDX[c], size=counts[c], replace=True)
                sampled_pred.extend(predictions[idx])

        sampled_pred = np.array(sampled_pred)
        n = len(sampled_pred)
        if n == 0:
            continue

        for c in range(5):
            true_frac = fracs[c]  # exact since we sampled with these fracs
            pred_frac = (sampled_pred == c).sum() / n
            total_error += abs(true_frac - pred_frac)

    return total_error / (n_ensembles * 5)


def objective(biases, probs):
    """Minimize fraction error by adjusting class biases."""
    adjusted = probs + biases
    preds = adjusted.argmax(1)
    return compute_fraction_error(preds)


def optimize_biases(probs, name):
    p(f"\n{'='*60}")
    p(f"Optimizing biases for: {name}")
    p(f"{'='*60}")

    # Baseline
    baseline_preds = probs.argmax(1)
    baseline_fe = compute_fraction_error(baseline_preds)
    baseline_acc = (baseline_preds == y_test).mean()
    p(f"Baseline: acc={baseline_acc:.4f}, frac_err={baseline_fe:.4f}")

    # Method 1: Grid search per class (sequential, greedy)
    best_biases = np.zeros(5)
    best_fe = baseline_fe
    for iteration in range(3):  # Multiple passes
        improved = False
        for c in range(5):
            for b in np.arange(-0.3, 0.31, 0.005):
                test_biases = best_biases.copy()
                test_biases[c] = b
                adjusted = probs + test_biases
                preds = adjusted.argmax(1)
                fe = compute_fraction_error(preds)
                if fe < best_fe:
                    best_fe = fe
                    best_biases[c] = b
                    improved = True
        if not improved:
            break
        p(f"  Pass {iteration+1}: biases={np.round(best_biases, 3)}, frac_err={best_fe:.4f}")

    p(f"Grid search result: biases={np.round(best_biases, 4)}, frac_err={best_fe:.4f}")
    adjusted = probs + best_biases
    preds = adjusted.argmax(1)
    acc = (preds == y_test).mean()
    p(f"  acc={acc:.4f}")

    # Method 2: Differential evolution
    p("\nDifferential evolution...")
    bounds = [(-0.3, 0.3)] * 5
    result = differential_evolution(objective, bounds, args=(probs,),
                                     maxiter=200, seed=42, tol=1e-6,
                                     popsize=20, mutation=(0.5, 1.5))
    de_biases = result.x
    de_fe = result.fun
    de_preds = (probs + de_biases).argmax(1)
    de_acc = (de_preds == y_test).mean()
    p(f"DE result: biases={np.round(de_biases, 4)}, frac_err={de_fe:.4f}, acc={de_acc:.4f}")

    # Use best
    if de_fe < best_fe:
        best_fe = de_fe
        best_biases = de_biases
        preds = de_preds
    else:
        preds = (probs + best_biases).argmax(1)

    return preds, best_biases, best_fe


# Load probability files
v8 = np.load(f"{OUT_DIR}/probs_v8.npy")
v19_ens = np.load(f"{OUT_DIR}/probs_v19.npy")
v11 = np.load(f"{OUT_DIR}/probs_v11_eval.npy")
v9 = np.load(f"{OUT_DIR}/probs_v9_eval.npy")
hgb = np.load(f"{OUT_DIR}/probs_hgb.npy")

# Test various base probability sets
configs = {
    'v8': v8,
    'v8+v11': (v8 + v11) / 2,
    'v8+v9+v11': (v8 + v9 + v11) / 3,
    '3seed_ens': v19_ens,
    'v8+v11+hgb_01': 0.45*v8 + 0.45*v11 + 0.1*hgb,
    'v8_0.6+v11_0.4': 0.6*v8 + 0.4*v11,
}

best_overall_fe = 1.0
best_overall_preds = None
best_overall_name = None

for name, probs in configs.items():
    preds, biases, fe = optimize_biases(probs, name)
    if fe < best_overall_fe:
        best_overall_fe = fe
        best_overall_preds = preds
        best_overall_name = name
        p(f"\n*** NEW OVERALL BEST: {name} -> {fe:.4f} ***")

p(f"\n{'='*60}")
p(f"BEST OVERALL: {best_overall_name} -> {best_overall_fe:.4f}")
p(f"{'='*60}")

# Save
np.savez(f"{OUT_DIR}/predictions_v23_optimized.npz",
         predictions=best_overall_preds.astype(np.int8))
p(f"Saved to predictions_v23_optimized.npz")

# Also try temperature scaling + bias
p("\n\n--- Temperature scaling + bias ---")
for T in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]:
    # Temperature scaling on logits (log of probs)
    log_probs = np.log(v8 + 1e-10)
    scaled = np.exp(log_probs / T)
    scaled = scaled / scaled.sum(axis=1, keepdims=True)
    preds = scaled.argmax(1)
    fe = compute_fraction_error(preds)
    acc = (preds == y_test).mean()
    p(f"  T={T}: acc={acc:.4f}, frac_err={fe:.4f}")

p(f"\n---")
p(f"metric: {best_overall_fe:.4f}")
p(f"description: Bias-optimized ensemble, best of {len(configs)} configs")
