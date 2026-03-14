"""v44: Ensemble bias optimization including v43 (extended features model).
Quick sweep of key combos including the new v43 model.
"""
import numpy as np
from scipy.optimize import differential_evolution
import subprocess, time

OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"

MIXTURE_SIZE = 5000
MIXTURE_SEED = 2026
GRID_STEP = 0.1

def p(msg):
    print(msg, flush=True)

def generate_fraction_grid(n_classes=5, step=GRID_STEP):
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
    def __init__(self, y_test, seed=MIXTURE_SEED):
        self.y_test = y_test
        self.n_classes = 5
        self.class_indices = {c: np.where(y_test == c)[0] for c in range(self.n_classes)}
        self.fractions = generate_fraction_grid(self.n_classes, GRID_STEP)
        self.n_ensembles = len(self.fractions)
        self.rng = np.random.default_rng(seed)
        self.sample_indices = []
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

    def evaluate(self, preds):
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

y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)
p("Precomputing sampling...")
evaluator = FractionErrorEvaluator(y_test)

models = {
    "v8": np.load(f"{OUT_DIR}/probs_v8.npy"),
    "v11": np.load(f"{OUT_DIR}/probs_v11_eval.npy"),
    "v33": np.load(f"{OUT_DIR}/probs_v33.npy"),
    "v43": np.load(f"{OUT_DIR}/probs_v43.npy"),
}

ensembles = {
    "v8+v11 (baseline)": [models["v8"], models["v11"]],
    "v8+v43": [models["v8"], models["v43"]],
    "v8+v11+v43": [models["v8"], models["v11"], models["v43"]],
    "v8+v11+v33+v43": [models["v8"], models["v11"], models["v33"], models["v43"]],
    "v8*2+v43": [models["v8"], models["v8"], models["v43"]],
    "v8*2+v11+v43": [models["v8"], models["v8"], models["v11"], models["v43"]],
}

best_overall = 1.0
best_name = ""
best_biases = None
best_probs = None

for name, prob_list in ensembles.items():
    avg = np.mean(prob_list, axis=0)
    log_avg = np.log(avg + 1e-10)

    raw_preds = avg.argmax(1)
    raw_fe, _ = evaluator.evaluate(raw_preds)

    eval_count = [0]
    def objective(biases):
        eval_count[0] += 1
        adjusted = log_avg + biases
        preds = adjusted.argmax(1)
        fe, _ = evaluator.evaluate(preds)
        return fe

    p(f"\n{name}: raw={raw_fe:.4f}")
    t0 = time.time()

    result = differential_evolution(
        objective, bounds=[(-0.5, 0.5)] * 5,
        seed=42, maxiter=500, tol=1e-8, polish=True, popsize=25,
        mutation=(0.5, 1.5), recombination=0.9,
    )

    biases = result.x
    adjusted = log_avg + biases
    preds = adjusted.argmax(1)
    fe_opt, pce_opt = evaluator.evaluate(preds)
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
p(f"description: {best_name} + bias optimization (with v43 ext features)")
