"""v3: Confusion matrix correction.
Instead of per-class biases (5 params), apply a learned 5x5 correction matrix
to the probability vectors. This can fix systematic misclassification patterns
that simple biases cannot.

Also try: per-energy-bin confusion matrix correction.
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-composition-mar14"
PREV_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
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


def main():
    t0 = time.time()

    y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)
    features = np.array(np.load(f"{DATA_DIR}/composition_test/features.npy"), dtype=np.float32)
    energies = features[:, 0]
    n_test = len(y_test)

    evaluator = FractionErrorEvaluator(y_test)

    # Load probabilities
    probs_v8 = np.load(f"{PREV_DIR}/probs_v8.npy")
    probs_v11 = np.load(f"{PREV_DIR}/probs_v11_eval.npy")
    avg_probs = (probs_v8 + probs_v11) / 2
    log_probs = np.log(avg_probs + 1e-10)

    raw_preds = avg_probs.argmax(1)
    raw_fe, raw_pce = evaluator.evaluate(raw_preds)
    p(f"Raw v8+v11: {raw_fe:.4f}")

    # === Approach 1: Full 5x5 matrix correction ===
    # Transform: corrected_probs[i] = softmax(W @ log_probs[i] + b)
    # W is 5x5 (25 params) + b is 5 (5 params) = 30 params
    p("\n=== 5x5 Matrix correction (30 params) ===")

    def obj_matrix(params):
        W = params[:25].reshape(5, 5)
        b = params[25:30]
        corrected = log_probs @ W.T + b
        preds = corrected.argmax(1)
        fe, _ = evaluator.evaluate(preds)
        return fe

    # Initialize W as identity + b as zeros
    x0 = np.zeros(30)
    x0[:25] = np.eye(5).flatten()

    res_mat = differential_evolution(
        obj_matrix, bounds=[(-2, 2)] * 25 + [(-1, 1)] * 5,
        seed=42, maxiter=1000, tol=1e-8, polish=True, popsize=30,
        mutation=(0.5, 1.5), recombination=0.9,
        x0=x0,
    )

    W = res_mat.x[:25].reshape(5, 5)
    b = res_mat.x[25:30]
    corrected = log_probs @ W.T + b
    mat_preds = corrected.argmax(1)
    mat_fe, mat_pce = evaluator.evaluate(mat_preds)
    mat_acc = (mat_preds == y_test).mean()
    p(f"Matrix correction: {mat_fe:.6f} (acc={mat_acc:.4f})")
    p(f"  per-class: {['%.4f' % v for v in mat_pce]}")
    p(f"  W diag: {np.diag(W).round(4)}")
    p(f"  b: {b.round(4)}")

    # === Approach 2: Temperature scaling per class ===
    p("\n=== Per-class temperature (10 params) ===")

    def obj_temp(params):
        temps = params[:5]
        biases = params[5:10]
        corrected = log_probs / (temps + 1e-6) + biases
        preds = corrected.argmax(1)
        fe, _ = evaluator.evaluate(preds)
        return fe

    res_temp = differential_evolution(
        obj_temp, bounds=[(0.1, 5.0)] * 5 + [(-1, 1)] * 5,
        seed=42, maxiter=500, tol=1e-8, polish=True, popsize=25,
        x0=np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float),
    )
    temps = res_temp.x[:5]
    biases = res_temp.x[5:10]
    corrected = log_probs / (temps + 1e-6) + biases
    temp_preds = corrected.argmax(1)
    temp_fe, temp_pce = evaluator.evaluate(temp_preds)
    temp_acc = (temp_preds == y_test).mean()
    p(f"Per-class temp: {temp_fe:.6f} (acc={temp_acc:.4f})")
    p(f"  temps: {temps.round(4)}")
    p(f"  biases: {biases.round(4)}")

    # === Approach 3: Energy-conditional matrix correction ===
    p("\n=== Energy-conditional 5x5 matrix (2 bins × 30 params = 60 params) ===")

    energy_split = 15.5  # split roughly in half
    mask_low = energies < energy_split
    mask_high = ~mask_low
    p(f"  Low E (<{energy_split}): {mask_low.sum()} events")
    p(f"  High E (>={energy_split}): {mask_high.sum()} events")

    def obj_energy_matrix(params):
        preds = np.empty(n_test, dtype=np.int64)
        for i, mask in enumerate([mask_low, mask_high]):
            W = params[i*30:i*30+25].reshape(5, 5)
            b = params[i*30+25:i*30+30]
            corrected = log_probs[mask] @ W.T + b
            preds[mask] = corrected.argmax(1)
        fe, _ = evaluator.evaluate(preds)
        return fe

    x0_em = np.tile(np.concatenate([np.eye(5).flatten(), np.zeros(5)]), 2)
    bounds_em = ([(-2, 2)] * 25 + [(-1, 1)] * 5) * 2

    res_em = differential_evolution(
        obj_energy_matrix, bounds=bounds_em,
        seed=42, maxiter=1500, tol=1e-8, polish=True, popsize=35,
        mutation=(0.5, 1.5), recombination=0.9,
        x0=x0_em,
    )

    em_preds = np.empty(n_test, dtype=np.int64)
    for i, mask in enumerate([mask_low, mask_high]):
        W = res_em.x[i*30:i*30+25].reshape(5, 5)
        b = res_em.x[i*30+25:i*30+30]
        corrected = log_probs[mask] @ W.T + b
        em_preds[mask] = corrected.argmax(1)
    em_fe, em_pce = evaluator.evaluate(em_preds)
    em_acc = (em_preds == y_test).mean()
    p(f"Energy-cond matrix: {em_fe:.6f} (acc={em_acc:.4f})")
    p(f"  per-class: {['%.4f' % v for v in em_pce]}")

    # === Approach 4: Use probabilities + features together ===
    # Train a small correction model: combine probs with energy
    p("\n=== Prob + energy correction (15 params: 5 biases + 5 energy slopes + 5 energy^2) ===")

    E_norm = (energies - energies.mean()) / energies.std()

    def obj_energy_poly(params):
        b = params[:5]
        s1 = params[5:10]
        s2 = params[10:15]
        corrected = log_probs + b + s1 * E_norm[:, None] + s2 * (E_norm[:, None] ** 2)
        preds = corrected.argmax(1)
        fe, _ = evaluator.evaluate(preds)
        return fe

    res_poly = differential_evolution(
        obj_energy_poly, bounds=[(-1, 1)] * 15,
        seed=42, maxiter=500, tol=1e-8, polish=True, popsize=25,
        x0=np.zeros(15),
    )
    b = res_poly.x[:5]
    s1 = res_poly.x[5:10]
    s2 = res_poly.x[10:15]
    corrected = log_probs + b + s1 * E_norm[:, None] + s2 * (E_norm[:, None] ** 2)
    poly_preds = corrected.argmax(1)
    poly_fe, poly_pce = evaluator.evaluate(poly_preds)
    poly_acc = (poly_preds == y_test).mean()
    p(f"Energy-poly correction: {poly_fe:.6f} (acc={poly_acc:.4f})")
    p(f"  biases: {b.round(4)}")
    p(f"  slopes: {s1.round(4)}")
    p(f"  quadratic: {s2.round(4)}")

    # Summary
    results = [
        ("5x5_matrix", mat_fe, mat_preds),
        ("per_class_temp", temp_fe, temp_preds),
        ("energy_cond_matrix", em_fe, em_preds),
        ("energy_poly", poly_fe, poly_preds),
    ]
    p(f"\n{'='*60}")
    p("Summary:")
    for name, fe, _ in results:
        p(f"  {name}: {fe:.6f}")
    best_name, best_fe, best_preds = min(results, key=lambda x: x[1])
    p(f"\nBEST: {best_name} = {best_fe:.6f}")

    np.savez(f"{OUT_DIR}/predictions_v3.npz", predictions=best_preds.astype(np.int8))
    p(f"Saved predictions_v3.npz")
    p(f"Total time: {(time.time()-t0)/60:.1f} min")
    p("---")
    p(f"metric: {best_fe:.4f}")
    p(f"description: {best_name} correction on v8+v11 probs")

if __name__ == "__main__":
    main()
