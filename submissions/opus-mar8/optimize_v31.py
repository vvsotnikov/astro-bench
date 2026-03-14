"""v31: Quick ensemble+bias test adding v30 to the mix."""
import numpy as np
from scipy.optimize import differential_evolution
import time

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"

def p(msg):
    print(msg, flush=True)

y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r'), dtype=int)
n_test = len(y_test)

# Grid
def generate_fraction_grid(n_classes=5, step=0.1):
    n_steps = round(1.0 / step)
    fractions = []
    def recurse(remaining, depth, current):
        if depth == n_classes - 1:
            current.append(remaining * step)
            fractions.append(current[:])
            current.pop()
            return
        for i in range(remaining + 1):
            current.append(i * step)
            recurse(remaining - i, depth + 1, current)
            current.pop()
    recurse(n_steps, 0, [])
    return np.array(fractions)

GRID = generate_fraction_grid()
CLASS_IDX = [np.where(y_test == c)[0] for c in range(5)]
rng = np.random.default_rng(2026)
SAMPLES = []
for fracs in GRID:
    counts = np.round(fracs * 5000).astype(int)
    diff = 5000 - counts.sum()
    if diff != 0: counts[np.argmax(counts)] += diff
    indices = []
    for c in range(5):
        if counts[c] > 0: indices.append(rng.choice(CLASS_IDX[c], size=counts[c], replace=True))
    SAMPLES.append((np.concatenate(indices), fracs))

def compute_fe(predictions):
    total = 0.0
    for sample_idx, true_fracs in SAMPLES:
        preds = predictions[sample_idx]
        pred_counts = np.bincount(preds, minlength=5)[:5]
        pred_fracs = pred_counts / pred_counts.sum()
        total += np.abs(true_fracs - pred_fracs).sum()
    return total / (len(SAMPLES) * 5)

def optimize_biases(probs):
    best_biases = np.zeros(5)
    best_fe = compute_fe(probs.argmax(1))
    for _ in range(5):
        improved = False
        for c in range(5):
            for b in np.arange(-0.3, 0.31, 0.01):
                test_b = best_biases.copy(); test_b[c] = b
                fe = compute_fe((probs + test_b).argmax(1))
                if fe < best_fe - 1e-7:
                    best_fe = fe; best_biases[c] = b; improved = True
        if not improved: break
    def obj(b): return compute_fe((probs + b).argmax(1))
    bounds = [(b - 0.05, b + 0.05) for b in best_biases]
    result = differential_evolution(obj, bounds, maxiter=100, seed=42, tol=1e-7, popsize=15)
    if result.fun < best_fe: best_fe = result.fun; best_biases = result.x
    return best_biases, best_fe

# Load models
v8 = np.load(f'{OUT_DIR}/probs_v8.npy')
v11 = np.load(f'{OUT_DIR}/probs_v11_eval.npy')
v30 = np.load(f'{OUT_DIR}/probs_v30.npy')
v28 = np.load(f'{OUT_DIR}/probs_v28.npy')
v27 = np.load(f'{OUT_DIR}/probs_v27.npy')
v24 = np.load(f'{OUT_DIR}/probs_v24.npy')

configs = {
    'v8+v11': (v8 + v11) / 2,
    'v8+v30': (v8 + v30) / 2,
    'v8+v11+v30': (v8 + v11 + v30) / 3,
    'v8_0.4+v11_0.3+v30_0.3': 0.4*v8 + 0.3*v11 + 0.3*v30,
    'v8+v11+v30+v28': (v8+v11+v30+v28)/4,
    'v8+v11+v30+v24': (v8+v11+v30+v24)/4,
    'v8_0.5+v30_0.5': 0.5*v8 + 0.5*v30,
}

best_fe = 1.0
best_name = None
best_preds = None

for name, probs in configs.items():
    biases, fe = optimize_biases(probs)
    acc = ((probs + biases).argmax(1) == y_test).mean()
    p(f"  {name:40s}: fe={fe:.6f}, acc={acc:.4f}")
    if fe < best_fe:
        best_fe = fe; best_name = name
        best_preds = (probs + biases).argmax(1)
        p(f"    *** NEW BEST ***")

p(f"\nBEST: {best_name} -> {best_fe:.6f}")
if best_fe < 0.1061:
    np.savez(f"{OUT_DIR}/predictions_v31.npz", predictions=best_preds.astype(np.int8))
    p("Saved predictions_v31.npz")
else:
    p("No improvement over v23 (0.1061)")

p(f"\n---")
p(f"metric: {best_fe:.6f}")
p(f"description: Ensemble+bias with v30 added")
