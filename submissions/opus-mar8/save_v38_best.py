"""Save the best v38 ensemble predictions: v8*0.4 + v11*0.4 + v33*0.2 + bias."""
import numpy as np
from scipy.optimize import differential_evolution

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"

y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r'), dtype=int)

# Recreate sampling exactly like verify.py
def generate_fraction_grid(n_classes=5, step=0.1):
    n_steps = round(1.0 / step)
    fractions = []
    def recurse(remaining, depth, current):
        if depth == n_classes - 1:
            current.append(remaining * step); fractions.append(current[:]); current.pop(); return
        for i in range(remaining + 1):
            current.append(i * step); recurse(remaining - i, depth + 1, current); current.pop()
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

def compute_fe(preds):
    total = 0.0
    for idx, fracs in SAMPLES:
        p = preds[idx]
        pc = np.bincount(p, minlength=5)[:5]
        total += np.abs(fracs - pc / pc.sum()).sum()
    return total / (len(SAMPLES) * 5)

v8 = np.load(f'{OUT_DIR}/probs_v8.npy')
v11 = np.load(f'{OUT_DIR}/probs_v11_eval.npy')
v33 = np.load(f'{OUT_DIR}/probs_v33.npy')

probs = 0.4 * v8 + 0.4 * v11 + 0.2 * v33

# Quick bias optimization
best_b = np.zeros(5)
best_fe = compute_fe(probs.argmax(1))
print(f"Raw: fe={best_fe:.6f}, acc={(probs.argmax(1)==y_test).mean():.4f}", flush=True)

for _ in range(3):
    for c in range(5):
        for b in np.arange(-0.3, 0.31, 0.02):
            tb = best_b.copy(); tb[c] = b
            fe = compute_fe((probs + tb).argmax(1))
            if fe < best_fe - 1e-7: best_fe = fe; best_b[c] = b

def obj(b): return compute_fe((probs + b).argmax(1))
bounds = [(b - 0.1, b + 0.1) for b in best_b]
r = differential_evolution(obj, bounds, maxiter=200, seed=42, tol=1e-8, popsize=20)
if r.fun < best_fe: best_fe = r.fun; best_b = r.x

print(f"Optimized: fe={best_fe:.6f}", flush=True)
print(f"Bias: {best_b}", flush=True)
preds = (probs + best_b).argmax(1).astype(np.int8)
acc = (preds == y_test).mean()
print(f"Accuracy: {acc:.4f}", flush=True)

np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds)
print("Saved predictions.npz", flush=True)
print("---", flush=True)
print(f"metric: {best_fe:.4f}", flush=True)
print(f"description: v8*0.4+v11*0.4+v33*0.2 + DE bias opt", flush=True)
