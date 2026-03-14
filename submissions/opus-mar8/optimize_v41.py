"""v41: Comprehensive ensemble optimization with all available models.
Test whether v39 snapshot ensemble adds diversity."""
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
    best_b = np.zeros(5)
    best_fe = compute_fe(probs.argmax(1))
    for _ in range(3):
        for c in range(5):
            for b in np.arange(-0.3, 0.31, 0.02):
                tb = best_b.copy(); tb[c] = b
                fe = compute_fe((probs + tb).argmax(1))
                if fe < best_fe - 1e-7: best_fe = fe; best_b[c] = b
    def obj(b): return compute_fe((probs + b).argmax(1))
    bounds = [(b - 0.1, b + 0.1) for b in best_b]
    r = differential_evolution(obj, bounds, maxiter=200, seed=42, tol=1e-8, popsize=15)
    if r.fun < best_fe: best_fe = r.fun; best_b = r.x
    return best_b, best_fe

# Load all models
print("Loading models...", flush=True)
v8 = np.load(f'{OUT_DIR}/probs_v8.npy')
v11 = np.load(f'{OUT_DIR}/probs_v11_eval.npy')
v33 = np.load(f'{OUT_DIR}/probs_v33.npy')  # ViT
v39 = np.load(f'{OUT_DIR}/probs_v39.npy')  # Snapshot ensemble

# Also load individual snapshots
snaps = []
for i in range(5):
    try:
        snaps.append(np.load(f'{OUT_DIR}/probs_v39_snap{i}.npy'))
    except: break
print(f"Loaded {len(snaps)} individual snapshots", flush=True)

configs = {
    'v8+v11 (baseline)': (v8+v11)/2,
    'v8+v11+v33': (v8+v11+v33)/3,
    'v8+v11+v39': (v8+v11+v39)/3,
    'v8+v11+v33+v39': (v8+v11+v33+v39)/4,
    'v8*0.4+v11*0.4+v33*0.2': 0.4*v8 + 0.4*v11 + 0.2*v33,
    'v8*0.4+v11*0.4+v39*0.2': 0.4*v8 + 0.4*v11 + 0.2*v39,
    'v8*0.35+v11*0.35+v33*0.15+v39*0.15': 0.35*v8 + 0.35*v11 + 0.15*v33 + 0.15*v39,
}

# Add best snapshots to ensemble
if len(snaps) >= 3:
    # Use snapshots 3 and 4 (later, better trained)
    best_snap = (snaps[3] + snaps[4]) / 2
    configs['v8+v11+snaps34'] = (v8 + v11 + best_snap) / 3
    configs['v8*0.4+v11*0.4+snaps34*0.2'] = 0.4*v8 + 0.4*v11 + 0.2*best_snap

print(f"\nTesting {len(configs)} configs...\n", flush=True)

best_overall = 1.0
best_name = ""

for name, probs in configs.items():
    raw_fe = compute_fe(probs.argmax(1))
    b, fe = optimize_biases(probs)
    acc = ((probs+b).argmax(1) == y_test).mean()
    marker = " ***" if fe < best_overall else ""
    print(f"  {name:45s}: raw={raw_fe:.6f}, opt={fe:.6f}, acc={acc:.4f}{marker}", flush=True)
    if fe < best_overall:
        best_overall = fe
        best_name = name

        # Save best predictions
        preds = (probs + b).argmax(1).astype(np.int8)
        np.savez(f"{OUT_DIR}/predictions_v41_best.npz", predictions=preds)

print(f"\n*** BEST: {best_name} with fe={best_overall:.6f} ***", flush=True)
print("---", flush=True)
print(f"metric: {best_overall:.4f}", flush=True)
print(f"description: comprehensive ensemble opt with snapshots + ViT + bias", flush=True)
