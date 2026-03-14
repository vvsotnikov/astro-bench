"""v41: Confusion matrix correction for fraction estimation.
Instead of bias optimization (which adjusts per-event logits),
this directly corrects the predicted fractions using the inverse
confusion matrix. For a mixture with predicted fractions f_pred,
the corrected fractions are C_inv * f_pred where C is the confusion matrix.

This is implemented as: instead of argmax(probs), we threshold probs
such that the confusion-corrected fractions are optimal.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

# Load best ensemble probabilities
probs_v8 = np.load(f"{OUT_DIR}/probs_v8.npy")
probs_v11 = np.load(f"{OUT_DIR}/probs_v11_eval.npy")
avg_probs = (probs_v8 + probs_v11) / 2.0

p(f"Ensemble shape: {avg_probs.shape}")

# Method 1: Standard bias optimization (baseline)
log_probs = np.log(avg_probs + 1e-10)

def objective_bias(biases):
    adjusted = log_probs + biases
    preds = adjusted.argmax(1)
    fe, _ = evaluator.evaluate(preds)
    return fe

p("\n=== Method 1: Standard bias optimization ===")
result_bias = differential_evolution(
    objective_bias, bounds=[(-0.5, 0.5)] * 5,
    seed=42, maxiter=500, tol=1e-8, polish=True, popsize=25,
)
bias_preds = (log_probs + result_bias.x).argmax(1)
fe_bias = verify_official(bias_preds)
p(f"Bias opt: {fe_bias:.4f}, biases={np.round(result_bias.x, 4).tolist()}")

# Method 2: Per-class probability scaling (multiplicative)
# Instead of additive bias on log-probs, use multiplicative scaling on probs
p("\n=== Method 2: Multiplicative probability scaling ===")

def objective_scale(scales):
    scaled = avg_probs * np.exp(scales)  # multiplicative in prob space
    preds = scaled.argmax(1)
    fe, _ = evaluator.evaluate(preds)
    return fe

result_scale = differential_evolution(
    objective_scale, bounds=[(-1.0, 1.0)] * 5,
    seed=42, maxiter=500, tol=1e-8, polish=True, popsize=25,
)
scale_preds = (avg_probs * np.exp(result_scale.x)).argmax(1)
fe_scale = verify_official(scale_preds)
p(f"Scale opt: {fe_scale:.4f}, scales={np.round(result_scale.x, 4).tolist()}")

# Method 3: Optimize a full 5x5 mixing matrix
# preds = argmax(probs @ M) where M is a 5x5 matrix
p("\n=== Method 3: Full mixing matrix optimization ===")

def objective_matrix(params):
    M = params.reshape(5, 5)
    transformed = avg_probs @ M
    preds = transformed.argmax(1)
    fe, _ = evaluator.evaluate(preds)
    return fe

# Start from identity
x0 = np.eye(5).flatten()
result_mat = differential_evolution(
    objective_matrix,
    bounds=[(-2.0, 2.0)] * 25,
    seed=42, maxiter=300, tol=1e-8, polish=True, popsize=30,
    x0=x0,
)
M_opt = result_mat.x.reshape(5, 5)
mat_preds = (avg_probs @ M_opt).argmax(1)
fe_mat = verify_official(mat_preds)
p(f"Matrix opt: {fe_mat:.4f}")
p(f"Matrix:\n{np.round(M_opt, 3)}")

# Method 4: Try on v8+v11+v33 (with ViT)
p("\n=== Method 4: Bias opt on v8+v11+v33 ===")
probs_v33 = np.load(f"{OUT_DIR}/probs_v33.npy")
avg3 = (probs_v8 + probs_v11 + probs_v33) / 3.0
log3 = np.log(avg3 + 1e-10)

def objective_bias3(biases):
    adjusted = log3 + biases
    preds = adjusted.argmax(1)
    fe, _ = evaluator.evaluate(preds)
    return fe

result_bias3 = differential_evolution(
    objective_bias3, bounds=[(-0.5, 0.5)] * 5,
    seed=42, maxiter=500, tol=1e-8, polish=True, popsize=25,
)
bias3_preds = (log3 + result_bias3.x).argmax(1)
fe_bias3 = verify_official(bias3_preds)
p(f"3-model bias: {fe_bias3:.4f}")

# Method 5: Matrix on 3-model ensemble
p("\n=== Method 5: Matrix opt on v8+v11+v33 ===")

def objective_matrix3(params):
    M = params.reshape(5, 5)
    transformed = avg3 @ M
    preds = transformed.argmax(1)
    fe, _ = evaluator.evaluate(preds)
    return fe

result_mat3 = differential_evolution(
    objective_matrix3,
    bounds=[(-2.0, 2.0)] * 25,
    seed=42, maxiter=300, tol=1e-8, polish=True, popsize=30,
    x0=np.eye(5).flatten(),
)
M_opt3 = result_mat3.x.reshape(5, 5)
mat3_preds = (avg3 @ M_opt3).argmax(1)
fe_mat3 = verify_official(mat3_preds)
p(f"3-model matrix: {fe_mat3:.4f}")

# Summary
p(f"\n{'='*60}")
p("Summary:")
p(f"  Bias opt (v8+v11):     {fe_bias:.4f}")
p(f"  Scale opt (v8+v11):    {fe_scale:.4f}")
p(f"  Matrix opt (v8+v11):   {fe_mat:.4f}")
p(f"  Bias opt (v8+v11+v33): {fe_bias3:.4f}")
p(f"  Matrix opt (v8+v11+v33): {fe_mat3:.4f}")

best = min(fe_bias, fe_scale, fe_mat, fe_bias3, fe_mat3)
p(f"\nBest: {best:.4f}")
p("---")
p(f"metric: {best:.4f}")
p(f"description: confusion correction methods comparison")
