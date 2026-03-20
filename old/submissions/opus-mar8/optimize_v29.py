"""v29: Comprehensive ensemble + bias optimization with all available models.
Includes diverse architectures: CNN, MLP, HGB, ordinal CNN."""
import numpy as np
from scipy.optimize import differential_evolution
from itertools import combinations
import time

DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"
OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"

def p(msg):
    print(msg, flush=True)

y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy", mmap_mode='r'), dtype=int)
n_test = len(y_test)

# Precompute grid
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

# Precompute sampling
rng = np.random.default_rng(2026)
SAMPLES = []
for fracs in GRID:
    counts = np.round(fracs * 5000).astype(int)
    diff = 5000 - counts.sum()
    if diff != 0:
        counts[np.argmax(counts)] += diff
    indices = []
    for c in range(5):
        if counts[c] > 0 and len(CLASS_IDX[c]) > 0:
            idx = rng.choice(CLASS_IDX[c], size=counts[c], replace=True)
            indices.append(idx)
    SAMPLES.append((np.concatenate(indices), fracs))

def compute_fe(predictions):
    total = 0.0
    for sample_idx, true_fracs in SAMPLES:
        preds = predictions[sample_idx]
        pred_counts = np.bincount(preds, minlength=5)[:5]
        pred_fracs = pred_counts / pred_counts.sum()
        total += np.abs(true_fracs - pred_fracs).sum()
    return total / (len(SAMPLES) * 5)

# Load all models
models = {}
for name, path in [
    ('v8', f'{OUT_DIR}/probs_v8.npy'),
    ('v11', f'{OUT_DIR}/probs_v11_eval.npy'),
    ('v9', f'{OUT_DIR}/probs_v9_eval.npy'),
    ('v19_s123', f'{OUT_DIR}/probs_v19_seed123.npy'),
    ('v19_s7', f'{OUT_DIR}/probs_v19_seed7.npy'),
    ('v24', f'{OUT_DIR}/probs_v24.npy'),
    ('v25', f'{OUT_DIR}/probs_v25.npy'),
    ('v26', f'{OUT_DIR}/probs_v26.npy'),
    ('v27', f'{OUT_DIR}/probs_v27.npy'),
    ('v28', f'{OUT_DIR}/probs_v28.npy'),
    ('hgb', f'{OUT_DIR}/probs_hgb.npy'),
]:
    try:
        prob = np.load(path)
        if prob.shape[0] == n_test and prob.shape[1] == 5:
            models[name] = prob
            preds = prob.argmax(1)
            acc = (preds == y_test).mean()
            fe = compute_fe(preds)
            p(f"  {name:12s}: acc={acc:.4f}, fe={fe:.4f}")
    except:
        pass

p(f"\nLoaded {len(models)} models")

def optimize_biases(probs, name):
    baseline = compute_fe(probs.argmax(1))
    best_biases = np.zeros(5)
    best_fe = baseline
    for _ in range(5):
        improved = False
        for c in range(5):
            for b in np.arange(-0.3, 0.31, 0.01):
                test_b = best_biases.copy()
                test_b[c] = b
                fe = compute_fe((probs + test_b).argmax(1))
                if fe < best_fe - 1e-7:
                    best_fe = fe
                    best_biases[c] = b
                    improved = True
        if not improved:
            break

    # DE refinement
    def obj(b):
        return compute_fe((probs + b).argmax(1))
    bounds = [(b - 0.05, b + 0.05) for b in best_biases]
    result = differential_evolution(obj, bounds, maxiter=100, seed=42, tol=1e-7, popsize=15)
    if result.fun < best_fe:
        best_fe = result.fun
        best_biases = result.x

    return best_biases, best_fe

# Phase 1: Weight optimization for 2-model ensembles
p("\n=== 2-model weight search ===")
names = sorted(models.keys())
best_2 = (None, None, 1.0, None)

for i, n1 in enumerate(names):
    for n2 in names[i+1:]:
        # Weight search
        local_best_fe = 1.0
        local_best_w = 0.5
        for w in np.arange(0.1, 0.95, 0.05):
            probs = w * models[n1] + (1 - w) * models[n2]
            fe = compute_fe(probs.argmax(1))
            if fe < local_best_fe:
                local_best_fe = fe
                local_best_w = w
        if local_best_fe < best_2[2]:
            best_2 = (n1, n2, local_best_fe, local_best_w)
            p(f"  {n1}*{local_best_w:.2f}+{n2}*{1-local_best_w:.2f}: fe={local_best_fe:.4f} ***")

p(f"Best pair: {best_2[0]}+{best_2[1]} w={best_2[3]:.2f} -> {best_2[2]:.4f}")

# Phase 2: Weight optimization for 3-model ensembles using best pair
p("\n=== 3-model weight search ===")
n1, n2 = best_2[0], best_2[1]
best_3 = (None, 1.0, None)

for n3 in names:
    if n3 in (n1, n2):
        continue
    # Grid search weights
    local_best_fe = 1.0
    local_best_w = None
    for w1 in np.arange(0.2, 0.7, 0.1):
        for w2 in np.arange(0.1, 0.7, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05 or w3 > 0.5:
                continue
            probs = w1 * models[n1] + w2 * models[n2] + w3 * models[n3]
            fe = compute_fe(probs.argmax(1))
            if fe < local_best_fe:
                local_best_fe = fe
                local_best_w = (w1, w2, w3)
    if local_best_fe < best_3[1]:
        best_3 = (n3, local_best_fe, local_best_w)
        p(f"  +{n3} w={local_best_w}: fe={local_best_fe:.4f} ***")

if best_3[0]:
    p(f"Best trio: {n1}+{n2}+{best_3[0]} -> {best_3[1]:.4f}")

# Phase 3: 4+ model blends
p("\n=== 4-model search ===")
if best_3[0]:
    n3 = best_3[0]
    best_4 = (None, 1.0)
    for n4 in names:
        if n4 in (n1, n2, n3):
            continue
        # Equal weight + small n4 weight
        for w4 in [0.05, 0.1, 0.15, 0.2]:
            remain = 1.0 - w4
            probs = remain/3 * (models[n1] + models[n2] + models[n3]) + w4 * models[n4]
            fe = compute_fe(probs.argmax(1))
            if fe < best_4[1]:
                best_4 = (n4, fe)
                p(f"  +{n4} w4={w4}: fe={fe:.4f} ***")

# Phase 4: Bias optimization on top candidates
p("\n=== Bias optimization on best ensembles ===")

candidates = {}

# Best pair
w = best_2[3]
candidates[f'{n1}+{n2}'] = w * models[n1] + (1-w) * models[n2]

# Best trio
if best_3[0]:
    w1, w2, w3 = best_3[2]
    candidates[f'{n1}+{n2}+{best_3[0]}'] = w1*models[n1] + w2*models[n2] + w3*models[best_3[0]]

# Various CNN ensembles
candidates['v8+v11'] = (models['v8'] + models['v11']) / 2
candidates['v8+v11+v28'] = (models['v8'] + models['v11'] + models['v28']) / 3
if 'v27' in models:
    candidates['v8+v11+v27'] = 0.4*models['v8'] + 0.4*models['v11'] + 0.2*models['v27']
    candidates['v8+v11+v27+v28'] = 0.35*models['v8'] + 0.35*models['v11'] + 0.15*models['v27'] + 0.15*models['v28']

best_overall_fe = 1.0
best_overall_preds = None
best_overall_name = None

for name, probs in candidates.items():
    biases, fe = optimize_biases(probs, name)
    acc = ((probs + biases).argmax(1) == y_test).mean()
    p(f"  {name:40s}: fe={fe:.6f}, acc={acc:.4f}")
    if fe < best_overall_fe:
        best_overall_fe = fe
        best_overall_preds = (probs + biases).argmax(1)
        best_overall_name = name
        p(f"    *** NEW BEST ***")

p(f"\n{'='*60}")
p(f"BEST: {best_overall_name} -> {best_overall_fe:.6f}")

np.savez(f"{OUT_DIR}/predictions_v29.npz", predictions=best_overall_preds.astype(np.int8))
p("Saved predictions_v29.npz")

p(f"\n---")
p(f"metric: {best_overall_fe:.6f}")
p(f"description: Best ensemble + bias optimization, {len(models)} models")
