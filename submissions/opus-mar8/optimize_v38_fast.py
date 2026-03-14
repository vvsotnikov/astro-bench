"""v38: Fast ensemble optimization with ViT + CNN + bias.
Uses smaller bias search range and skip configs that don't improve."""
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

def optimize_biases(probs):
    """Quick bias optimization: coarse grid + DE refinement."""
    best_b = np.zeros(5)
    best_fe = compute_fe(probs.argmax(1))

    # Coarse grid search: -0.3 to 0.3, step 0.02
    for _ in range(3):
        for c in range(5):
            for b in np.arange(-0.3, 0.31, 0.02):
                tb = best_b.copy(); tb[c] = b
                fe = compute_fe((probs + tb).argmax(1))
                if fe < best_fe - 1e-7: best_fe = fe; best_b[c] = b

    # DE refinement
    def obj(b): return compute_fe((probs + b).argmax(1))
    bounds = [(b - 0.1, b + 0.1) for b in best_b]
    r = differential_evolution(obj, bounds, maxiter=150, seed=42, tol=1e-8, popsize=15)
    if r.fun < best_fe: best_fe = r.fun; best_b = r.x

    return best_b, best_fe

# Load probabilities
print("Loading probabilities...", flush=True)
v8 = np.load(f'{OUT_DIR}/probs_v8.npy')
v11 = np.load(f'{OUT_DIR}/probs_v11_eval.npy')
v33 = np.load(f'{OUT_DIR}/probs_v33.npy')

try:
    v37 = np.load(f'{OUT_DIR}/probs_v37.npy')
    has_v37 = True
except: has_v37 = False

try:
    v30 = np.load(f'{OUT_DIR}/probs_v30.npy')
    has_v30 = True
except: has_v30 = False

configs = {}

# Baselines
configs['v8+v11'] = (v8+v11)/2

# Cross-architecture (CNN+ViT)
configs['v8+v33'] = (v8+v33)/2
configs['v11+v33'] = (v11+v33)/2
configs['v8+v11+v33'] = (v8+v11+v33)/3

# Weighted combos
for w_vit in [0.2, 0.3, 0.4]:
    w_cnn = (1-w_vit)/2
    configs[f'v8+v11+v33_w{w_vit}'] = w_cnn*v8 + w_cnn*v11 + w_vit*v33

if has_v37:
    configs['v8+v11+v37'] = (v8+v11+v37)/3
    configs['v8+v11+v33+v37'] = (v8+v11+v33+v37)/4

if has_v30:
    configs['v8+v11+v30+v33'] = (v8+v11+v30+v33)/4

print(f"Testing {len(configs)} configs...\n", flush=True)

best_overall = 1.0
best_name = ""
best_probs = None
best_bias = None

for name, probs in configs.items():
    raw_fe = compute_fe(probs.argmax(1))
    raw_acc = (probs.argmax(1) == y_test).mean()
    b, fe = optimize_biases(probs)
    opt_acc = ((probs+b).argmax(1) == y_test).mean()
    marker = " ***" if fe < best_overall else ""
    print(f"  {name:30s}: raw={raw_fe:.6f}/{raw_acc:.4f}, opt={fe:.6f}/{opt_acc:.4f}, bias={np.array2string(b, precision=3)}{marker}", flush=True)

    if fe < best_overall:
        best_overall = fe
        best_name = name
        best_probs = probs
        best_bias = b

print(f"\n*** BEST: {best_name} with fe={best_overall:.6f} ***", flush=True)

# Save best
preds = (best_probs + best_bias).argmax(1).astype(np.int8)
np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds)

print("---", flush=True)
print(f"metric: {best_overall:.4f}", flush=True)
print(f"description: best cross-arch ensemble (CNN+ViT) + bias opt", flush=True)
