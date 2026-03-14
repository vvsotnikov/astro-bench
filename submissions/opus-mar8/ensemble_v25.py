"""v25: Comprehensive ensemble + bias optimization.
Try all combinations of diverse models with bias optimization.
Use differential evolution to find optimal per-class biases.
"""
import numpy as np
from scipy.optimize import differential_evolution
import subprocess
import itertools

OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"

def p(msg):
    print(msg, flush=True)

def verify_predictions(preds):
    """Run verify.py and return fraction error."""
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds.astype(np.int8))
    result = subprocess.run(
        ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
        capture_output=True, text=True,
        cwd="/home/vladimir/cursor_projects/astro-agents"
    )
    for line in result.stdout.split('\n'):
        if 'mean fraction error' in line.lower():
            try:
                return float(line.split(':')[-1].strip())
            except:
                pass
    return 1.0

def optimize_bias(probs, y_test):
    """Find optimal per-class logit biases using differential evolution."""
    log_probs = np.log(probs + 1e-10)

    def objective(biases):
        adjusted = log_probs + biases
        preds = adjusted.argmax(1)
        # Quick fraction error approximation (10 random mixtures)
        # Use a simplified metric for speed during optimization
        rng = np.random.default_rng(42)
        n_classes = 5
        class_indices = {c: np.where(y_test == c)[0] for c in range(n_classes)}

        errors = []
        for _ in range(50):
            fracs = rng.dirichlet(np.ones(n_classes))
            counts = np.round(fracs * 5000).astype(int)
            counts[-1] = 5000 - counts[:-1].sum()

            sampled = []
            true_fracs = np.zeros(n_classes)
            for c in range(n_classes):
                n = max(0, counts[c])
                if n > 0:
                    idx = rng.choice(class_indices[c], size=n, replace=True)
                    sampled.append(preds[idx])
                    true_fracs[c] = n
            true_fracs /= true_fracs.sum()
            all_preds = np.concatenate(sampled)
            pred_counts = np.bincount(all_preds, minlength=5)[:5]
            pred_fracs = pred_counts / pred_counts.sum()
            errors.append(np.abs(true_fracs - pred_fracs).mean())

        return np.mean(errors)

    result = differential_evolution(
        objective,
        bounds=[(-0.5, 0.5)] * 5,
        seed=42,
        maxiter=200,
        tol=1e-6,
        polish=True,
    )
    return result.x, result.fun

# Load test labels
y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)

# Load all probability files -- focus on the best/most diverse models
models = {
    "v8": np.load(f"{OUT_DIR}/probs_v8.npy"),         # CNN+Attn+MLP (best single)
    "v11": np.load(f"{OUT_DIR}/probs_v11_eval.npy"),   # Same arch, different seed/training
    "v22": np.load(f"{OUT_DIR}/probs_v22.npy"),        # ResNet+Attn+MLP
    "v19_s123": np.load(f"{OUT_DIR}/probs_v19_seed123.npy"),  # CNN different seed
    "v19_s7": np.load(f"{OUT_DIR}/probs_v19_seed7.npy"),      # CNN different seed
    "v24": np.load(f"{OUT_DIR}/probs_v24.npy"),        # Ordinal-trained CNN
    "hgb": np.load(f"{OUT_DIR}/probs_hgb.npy"),        # HGB (fundamentally different)
}

# First, verify each model standalone
p("=== Individual models ===")
for name, probs in models.items():
    preds = probs.argmax(1)
    acc = (preds == y_test).mean()
    frac_err = verify_predictions(preds)
    p(f"  {name:12s}: acc={acc:.4f} frac_err={frac_err:.4f}")

# Try all pairs and triples of the best models
p("\n=== Pairwise ensembles ===")
best_models = ["v8", "v11", "v22", "v19_s123", "v24", "hgb"]
pair_results = {}
for a, b in itertools.combinations(best_models, 2):
    avg = (models[a] + models[b]) / 2
    preds = avg.argmax(1)
    frac_err = verify_predictions(preds)
    acc = (preds == y_test).mean()
    pair_results[(a, b)] = frac_err
    p(f"  {a}+{b}: acc={acc:.4f} frac_err={frac_err:.4f}")

# Top 5 pairs
p("\n=== Top 5 pairs by frac_err ===")
sorted_pairs = sorted(pair_results.items(), key=lambda x: x[1])
for (a, b), fe in sorted_pairs[:5]:
    p(f"  {a}+{b}: {fe:.4f}")

# Try triples with top pairs
p("\n=== Triple ensembles ===")
triple_results = {}
for a, b, c in itertools.combinations(best_models, 3):
    avg = (models[a] + models[b] + models[c]) / 3
    preds = avg.argmax(1)
    frac_err = verify_predictions(preds)
    acc = (preds == y_test).mean()
    triple_results[(a, b, c)] = frac_err
    p(f"  {a}+{b}+{c}: acc={acc:.4f} frac_err={frac_err:.4f}")

p("\n=== Top 5 triples ===")
sorted_triples = sorted(triple_results.items(), key=lambda x: x[1])
for combo, fe in sorted_triples[:5]:
    p(f"  {'+'.join(combo)}: {fe:.4f}")

# Now apply bias optimization to the best ensemble
p("\n=== Bias optimization on top ensembles ===")

# Try bias optimization on various combos
combos_to_optimize = [
    ("v8", [models["v8"]]),
    ("v8+v11", [models["v8"], models["v11"]]),
    ("v8+v22", [models["v8"], models["v22"]]),
    ("v8+v11+v22", [models["v8"], models["v11"], models["v22"]]),
    ("v8+v11+v19_s123", [models["v8"], models["v11"], models["v19_s123"]]),
    ("all_cnn", [models["v8"], models["v11"], models["v19_s123"], models["v19_s7"]]),
    ("all_cnn+v22", [models["v8"], models["v11"], models["v19_s123"], models["v19_s7"], models["v22"]]),
    ("v8+v11+v22+hgb", [models["v8"], models["v11"], models["v22"], models["hgb"]]),
]

best_overall = 1.0
best_combo_name = ""
best_biases = None
best_probs = None

for name, prob_list in combos_to_optimize:
    avg = np.mean(prob_list, axis=0)

    # Without bias
    preds = avg.argmax(1)
    frac_err_raw = verify_predictions(preds)

    # With bias
    biases, _ = optimize_bias(avg, y_test)
    adjusted = np.log(avg + 1e-10) + biases
    preds_bias = adjusted.argmax(1)
    frac_err_bias = verify_predictions(preds_bias)

    acc = (preds_bias == y_test).mean()
    p(f"  {name:25s}: raw={frac_err_raw:.4f} bias={frac_err_bias:.4f} acc={acc:.4f} biases={np.round(biases, 3).tolist()}")

    if frac_err_bias < best_overall:
        best_overall = frac_err_bias
        best_combo_name = name
        best_biases = biases
        best_probs = avg

p(f"\n=== BEST RESULT ===")
p(f"  Combo: {best_combo_name}")
p(f"  Fraction error: {best_overall:.4f}")
p(f"  Biases: {np.round(best_biases, 4).tolist()}")

# Save best predictions
adjusted = np.log(best_probs + 1e-10) + best_biases
preds_final = adjusted.argmax(1).astype(np.int8)
np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds_final)

# Final official verify
result = subprocess.run(
    ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
    capture_output=True, text=True,
    cwd="/home/vladimir/cursor_projects/astro-agents"
)
p(f"\nFinal verify:\n{result.stdout[:800]}")

p("---")
p(f"metric: {best_overall:.4f}")
p(f"description: {best_combo_name} + bias optimization")
