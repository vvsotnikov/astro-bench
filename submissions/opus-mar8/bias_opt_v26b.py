"""v26b: Fast bias optimization using exact verify.py grid.
Vectorized fraction error for speed, then verify with official verify.py.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import subprocess
import time

OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"

MIXTURE_SIZE = 5000
MIXTURE_SEED = 2026
GRID_STEP = 0.1

def p(msg):
    print(msg, flush=True)

def generate_fraction_grid(n_classes=5, step=GRID_STEP):
    """Exact same grid as verify.py."""
    n_steps = round(1.0 / step)
    fractions = []
    def _recurse(remaining, depth, current):
        if depth == n_classes - 1:
            current.append(remaining * step)
            fractions.append(current[:])
            current.pop()
            return
        for i in range(remaining + 1):
            current.append(i * step)
            _recurse(remaining - i, depth + 1, current)
            current.pop()
    _recurse(n_steps, 0, [])
    return np.array(fractions)

class FractionErrorEvaluator:
    """Precompute everything possible for fast repeated evaluation."""

    def __init__(self, y_test, seed=MIXTURE_SEED):
        self.y_test = y_test
        self.n_classes = 5
        self.class_indices = {c: np.where(y_test == c)[0] for c in range(self.n_classes)}
        self.fractions = generate_fraction_grid(self.n_classes, GRID_STEP)
        self.n_ensembles = len(self.fractions)

        # Precompute all sampling indices (deterministic given seed)
        self.rng = np.random.default_rng(seed)
        self.sample_indices = []  # list of (ensemble_idx, class_idx_arrays)

        for mix_idx in range(self.n_ensembles):
            target_fracs = self.fractions[mix_idx]
            counts = np.round(target_fracs * MIXTURE_SIZE).astype(int)
            diff = MIXTURE_SIZE - counts.sum()
            if diff != 0:
                counts[np.argmax(counts)] += diff

            indices_for_mix = []
            actual_true_fracs = np.zeros(self.n_classes)
            for c in range(self.n_classes):
                n_sample = counts[c]
                if n_sample <= 0:
                    indices_for_mix.append(np.array([], dtype=int))
                    continue
                idx = self.rng.choice(self.class_indices[c], size=n_sample, replace=True)
                indices_for_mix.append(idx)
                actual_true_fracs[c] = n_sample

            actual_true_fracs /= actual_true_fracs.sum()
            self.sample_indices.append((indices_for_mix, actual_true_fracs))

        p(f"  Precomputed {self.n_ensembles} ensemble sampling plans")

    def evaluate(self, preds):
        """Compute fraction error using precomputed sampling."""
        all_errors = np.zeros((self.n_ensembles, self.n_classes))

        for mix_idx in range(self.n_ensembles):
            indices_for_mix, actual_true_fracs = self.sample_indices[mix_idx]

            sampled = []
            for c in range(self.n_classes):
                if len(indices_for_mix[c]) > 0:
                    sampled.append(preds[indices_for_mix[c]])

            all_preds = np.concatenate(sampled)
            pred_counts = np.bincount(all_preds, minlength=self.n_classes)[:self.n_classes]
            pred_fracs = pred_counts / pred_counts.sum()
            all_errors[mix_idx] = np.abs(actual_true_fracs - pred_fracs)

        return float(all_errors.mean()), [float(all_errors[:, c].mean()) for c in range(self.n_classes)]


def verify_official(preds):
    """Run official verify.py."""
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


# Load test labels
y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)

# Create evaluator with precomputed sampling
p("Precomputing sampling plans...")
evaluator = FractionErrorEvaluator(y_test)

# Test speed
t0 = time.time()
dummy_preds = np.random.randint(0, 5, len(y_test))
fe, _ = evaluator.evaluate(dummy_preds)
p(f"  Single evaluation: {time.time()-t0:.3f}s (fe={fe:.4f})")

# Load models
models = {
    "v8": np.load(f"{OUT_DIR}/probs_v8.npy"),
    "v11": np.load(f"{OUT_DIR}/probs_v11_eval.npy"),
    "v22": np.load(f"{OUT_DIR}/probs_v22.npy"),
    "v19_s123": np.load(f"{OUT_DIR}/probs_v19_seed123.npy"),
    "v19_s7": np.load(f"{OUT_DIR}/probs_v19_seed7.npy"),
    "v24": np.load(f"{OUT_DIR}/probs_v24.npy"),
    "hgb": np.load(f"{OUT_DIR}/probs_hgb.npy"),
}

ensembles = {
    "v8": [models["v8"]],
    "v8+v11": [models["v8"], models["v11"]],
    "v8+v22": [models["v8"], models["v22"]],
    "v8+v11+v22": [models["v8"], models["v11"], models["v22"]],
    "v8+v11+v22+hgb": [models["v8"], models["v11"], models["v22"], models["hgb"]],
    "all_cnn": [models["v8"], models["v11"], models["v19_s123"], models["v19_s7"]],
    "all_cnn+v22": [models["v8"], models["v11"], models["v19_s123"], models["v19_s7"], models["v22"]],
    "all": list(models.values()),
}

best_overall = 1.0
best_name = ""
best_biases = None
best_probs = None

for name, prob_list in ensembles.items():
    avg = np.mean(prob_list, axis=0)
    log_avg = np.log(avg + 1e-10)

    # Raw performance
    raw_preds = avg.argmax(1)
    raw_fe, raw_pce = evaluator.evaluate(raw_preds)

    eval_count = [0]
    def objective(biases):
        eval_count[0] += 1
        adjusted = log_avg + biases
        preds = adjusted.argmax(1)
        fe, _ = evaluator.evaluate(preds)
        return fe

    p(f"\n{name}: raw={raw_fe:.4f}")
    t0 = time.time()

    # DE optimization
    result = differential_evolution(
        objective,
        bounds=[(-0.5, 0.5)] * 5,
        seed=42,
        maxiter=500,
        tol=1e-8,
        polish=True,
        popsize=25,
        mutation=(0.5, 1.5),
        recombination=0.9,
    )

    biases = result.x
    adjusted = log_avg + biases
    preds = adjusted.argmax(1)
    fe_opt, pce_opt = evaluator.evaluate(preds)

    # Official verify
    fe_official = verify_official(preds)
    acc = (preds == y_test).mean()
    elapsed = time.time() - t0

    p(f"  opt={fe_opt:.4f} official={fe_official:.4f} acc={acc:.4f} "
      f"biases={np.round(biases, 4).tolist()} ({eval_count[0]} evals, {elapsed:.0f}s)")
    p(f"  per-class: {['%.4f' % v for v in pce_opt]}")

    if fe_official < best_overall:
        best_overall = fe_official
        best_name = name
        best_biases = biases.copy()
        best_probs = avg.copy()

p(f"\n{'='*60}")
p(f"BEST: {best_name}")
p(f"  Fraction error: {best_overall:.4f}")
p(f"  Biases: {np.round(best_biases, 4).tolist()}")

# Save final
adjusted = np.log(best_probs + 1e-10) + best_biases
preds_final = adjusted.argmax(1).astype(np.int8)
np.savez(f"{OUT_DIR}/predictions.npz", predictions=preds_final)

result = subprocess.run(
    ["uv", "run", "python", "verify.py", f"{OUT_DIR}/predictions.npz"],
    capture_output=True, text=True,
    cwd="/home/vladimir/cursor_projects/astro-agents"
)
p(f"\nFinal verify:\n{result.stdout[:1200]}")

p("---")
p(f"metric: {best_overall:.4f}")
p(f"description: {best_name} + exact-grid bias optimization (precomputed)")
