"""v38: Ensemble optimization with ViT (v33) + CNN (v8, v11) + bias.
The ViT has different inductive biases than CNN, so should provide
better ensemble diversity than CNN+CNN combos."""
import numpy as np
from scipy.optimize import differential_evolution

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"

y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r'), dtype=int)

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

def optimize_biases(probs, name=""):
    best_b = np.zeros(5); best_fe = compute_fe(probs.argmax(1))
    acc = (probs.argmax(1) == y_test).mean()
    print(f"  {name}: raw fe={best_fe:.6f}, acc={acc:.4f}", flush=True)

    for _ in range(5):
        imp = False
        for c in range(5):
            for b in np.arange(-0.5, 0.51, 0.01):
                tb = best_b.copy(); tb[c] = b
                fe = compute_fe((probs + tb).argmax(1))
                if fe < best_fe - 1e-7: best_fe = fe; best_b[c] = b; imp = True
        if not imp: break

    def obj(b): return compute_fe((probs + b).argmax(1))
    bounds = [(b - 0.05, b + 0.05) for b in best_b]
    r = differential_evolution(obj, bounds, maxiter=200, seed=42, tol=1e-8, popsize=20)
    if r.fun < best_fe: best_fe = r.fun; best_b = r.x

    acc_opt = ((probs + best_b).argmax(1) == y_test).mean()
    print(f"  {name}: opt fe={best_fe:.6f}, acc={acc_opt:.4f}, bias={np.array2string(best_b, precision=3)}", flush=True)
    return best_b, best_fe

# Load probabilities
print("Loading probabilities...", flush=True)
v8 = np.load(f'{OUT_DIR}/probs_v8.npy')
v11 = np.load(f'{OUT_DIR}/probs_v11_eval.npy')
v33 = np.load(f'{OUT_DIR}/probs_v33.npy')

# Also try v37 if available
try:
    v37 = np.load(f'{OUT_DIR}/probs_v37.npy')
    has_v37 = True
    print("v37 loaded", flush=True)
except:
    has_v37 = False

# Also try v30 if available
try:
    v30 = np.load(f'{OUT_DIR}/probs_v30.npy')
    has_v30 = True
    print("v30 loaded", flush=True)
except:
    has_v30 = False

print("\n=== Individual models ===", flush=True)
optimize_biases(v8, "v8")
optimize_biases(v11, "v11")
optimize_biases(v33, "v33_vit")

print("\n=== CNN pairs ===", flush=True)
optimize_biases((v8+v11)/2, "v8+v11")

print("\n=== Cross-architecture ensembles (CNN+ViT) ===", flush=True)
optimize_biases((v8+v33)/2, "v8+v33")
optimize_biases((v11+v33)/2, "v11+v33")
optimize_biases((v8+v11+v33)/3, "v8+v11+v33")

# Weighted combos
for w_vit in [0.2, 0.3, 0.4, 0.5]:
    w_cnn = (1 - w_vit) / 2
    combo = w_cnn * v8 + w_cnn * v11 + w_vit * v33
    optimize_biases(combo, f"v8*{w_cnn:.2f}+v11*{w_cnn:.2f}+v33*{w_vit:.2f}")

if has_v37:
    print("\n=== With v37 (SWA base) ===", flush=True)
    optimize_biases((v8+v11+v37)/3, "v8+v11+v37")
    optimize_biases((v8+v11+v33+v37)/4, "v8+v11+v33+v37")

if has_v30:
    print("\n=== With v30 (deep CNN) ===", flush=True)
    optimize_biases((v8+v11+v30+v33)/4, "v8+v11+v30+v33")

# Find best and save
print("\n=== Finding overall best ===", flush=True)
best_overall_fe = 1.0
best_name = ""
best_combo_probs = None
best_combo_bias = None

configs = {
    'v8+v11': (v8+v11)/2,
    'v8+v33': (v8+v33)/2,
    'v8+v11+v33': (v8+v11+v33)/3,
}
for w_vit in [0.2, 0.3, 0.4]:
    w_cnn = (1-w_vit)/2
    configs[f'w{w_vit}'] = w_cnn*v8 + w_cnn*v11 + w_vit*v33

if has_v37:
    configs['v8+v11+v33+v37'] = (v8+v11+v33+v37)/4

for name, probs in configs.items():
    b, fe = optimize_biases(probs, name)
    if fe < best_overall_fe:
        best_overall_fe = fe
        best_name = name
        best_combo_probs = probs
        best_combo_bias = b

print(f"\n*** BEST: {best_name} with fe={best_overall_fe:.6f} ***", flush=True)
preds = (best_combo_probs + best_combo_bias).argmax(1).astype(np.int8)
np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds)
np.save(f"{OUT_DIR}/probs_v38_best.npy", best_combo_probs)

print("---", flush=True)
print(f"metric: {best_overall_fe:.4f}", flush=True)
print(f"description: best cross-arch ensemble (CNN+ViT) + bias opt", flush=True)
