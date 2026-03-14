"""v43: Fine-grained weight optimization with all models.
Exhaustive grid search over weights for v8, v11, snapshots, v33, v42."""
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
v33 = np.load(f'{OUT_DIR}/probs_v33.npy')
v42 = np.load(f'{OUT_DIR}/probs_v42.npy')

snaps = []
for i in range(5):
    try: snaps.append(np.load(f'{OUT_DIR}/probs_v39_snap{i}.npy'))
    except: break

# Try different snapshot combinations
snap_combos = {
    'snap34': (snaps[3] + snaps[4]) / 2,
    'snap234': (snaps[2] + snaps[3] + snaps[4]) / 3,
    'snap24': (snaps[2] + snaps[4]) / 2,
    'snap4': snaps[4],
    'snap3': snaps[3],
}

print("\n=== Snapshot pair selection (v8*0.4+v11*0.4+snap*0.2) ===", flush=True)
for name, sp in snap_combos.items():
    probs = 0.4*v8 + 0.4*v11 + 0.2*sp
    b, fe = optimize_biases(probs)
    print(f"  {name:12s}: fe={fe:.6f}", flush=True)

# Fine-grained weight grid for v8, v11, snap34
snap34 = (snaps[3] + snaps[4]) / 2
print("\n=== Weight grid for v8+v11+snap34 ===", flush=True)
best_overall = 1.0
best_cfg = ""
for w8 in np.arange(0.3, 0.55, 0.05):
    for w11 in np.arange(0.3, 0.55, 0.05):
        for ws in np.arange(0.1, 0.35, 0.05):
            total = w8 + w11 + ws
            if abs(total - 1.0) > 0.01: continue
            probs = w8*v8 + w11*v11 + ws*snap34
            b, fe = optimize_biases(probs)
            marker = " ***" if fe < best_overall else ""
            if fe < best_overall + 0.0002:  # Only print near-best
                print(f"  v8={w8:.2f} v11={w11:.2f} s34={ws:.2f}: fe={fe:.6f}{marker}", flush=True)
            if fe < best_overall:
                best_overall = fe
                best_cfg = f"v8={w8:.2f} v11={w11:.2f} s34={ws:.2f}"
                best_probs = probs.copy()
                best_bias = b.copy()

# Also try with v33 and v42 at small weights
print("\n=== Adding v33/v42 at small weights ===", flush=True)
for extra_name, extra_probs in [('v33', v33), ('v42', v42)]:
    for we in [0.05, 0.1, 0.15]:
        scale = 1 - we
        probs = scale * best_probs + we * extra_probs
        b, fe = optimize_biases(probs)
        marker = " ***" if fe < best_overall else ""
        print(f"  best*{scale:.2f}+{extra_name}*{we:.2f}: fe={fe:.6f}{marker}", flush=True)
        if fe < best_overall:
            best_overall = fe
            best_cfg += f"+{extra_name}*{we:.2f}"

print(f"\n*** BEST: {best_cfg} with fe={best_overall:.6f} ***", flush=True)

# Save best
preds = (best_probs + best_bias).argmax(1).astype(np.int8)
np.savez(f"{OUT_DIR}/predictions_v43_best.npz", predictions=preds)

print("---", flush=True)
print(f"metric: {best_overall:.4f}", flush=True)
print(f"description: fine-grained weight opt {best_cfg}", flush=True)
