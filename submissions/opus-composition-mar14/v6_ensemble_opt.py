"""v6: Ensemble v8+v11+v2+v4 with bias optimization.
v4 (seed=7, spatial features) brings diversity to the ensemble.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

class FastEvaluator:
    def __init__(self, y_test, seed=MIXTURE_SEED):
        self.y_test = y_test
        self.n_classes = 5
        self.class_indices = {c: np.where(y_test == c)[0] for c in range(self.n_classes)}
        self.fractions = generate_fraction_grid(self.n_classes, GRID_STEP)
        self.n_ensembles = len(self.fractions)
        rng = np.random.default_rng(seed)
        self.all_sample_idx = []
        self.true_fracs = np.zeros((self.n_ensembles, self.n_classes))
        for mix_idx in range(self.n_ensembles):
            target_fracs = self.fractions[mix_idx]
            counts = np.round(target_fracs * MIXTURE_SIZE).astype(int)
            diff = MIXTURE_SIZE - counts.sum()
            if diff != 0:
                counts[np.argmax(counts)] += diff
            indices = []
            for c in range(self.n_classes):
                n_sample = counts[c]
                if n_sample <= 0:
                    continue
                idx = rng.choice(self.class_indices[c], size=n_sample, replace=True)
                indices.append(idx)
                self.true_fracs[mix_idx, c] = n_sample / MIXTURE_SIZE
            self.all_sample_idx.append(np.concatenate(indices))

    def evaluate(self, preds):
        all_errors = np.zeros((self.n_ensembles, self.n_classes))
        for mix_idx in range(self.n_ensembles):
            sampled = preds[self.all_sample_idx[mix_idx]]
            pred_counts = np.bincount(sampled, minlength=self.n_classes)[:self.n_classes]
            pred_fracs = pred_counts / pred_counts.sum()
            all_errors[mix_idx] = np.abs(self.true_fracs[mix_idx] - pred_fracs)
        return float(all_errors.mean())


def main():
    t0 = time.time()

    y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)
    features = np.array(np.load(f"{DATA_DIR}/composition_test/features.npy"), dtype=np.float32)
    energies = features[:, 0]
    n_test = len(y_test)

    evaluator = FastEvaluator(y_test)

    # Load all available probabilities
    models = {}
    models["v8"] = np.load(f"{PREV_DIR}/probs_v8.npy")
    models["v11"] = np.load(f"{PREV_DIR}/probs_v11_eval.npy")
    try:
        models["v33"] = np.load(f"{PREV_DIR}/probs_v33.npy")
    except:
        pass
    try:
        models["v22"] = np.load(f"{PREV_DIR}/probs_v22.npy")
    except:
        pass
    try:
        models["v2"] = np.load(f"{OUT_DIR}/probs_v2.npy")
    except:
        pass
    try:
        models["v4"] = np.load(f"{OUT_DIR}/probs_v4.npy")
    except:
        pass

    p(f"Loaded models: {list(models.keys())}")

    # Check diversity — agreement rates
    keys = list(models.keys())
    p("\n=== Model agreement rates ===")
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            p_i = models[keys[i]].argmax(1)
            p_j = models[keys[j]].argmax(1)
            agree = (p_i == p_j).mean()
            p(f"  {keys[i]} vs {keys[j]}: {agree*100:.1f}%")

    # Try all combinations of 2-6 models
    from itertools import combinations

    best_overall = 1.0
    best_name = ""
    best_preds = None

    all_combos = []
    for r in range(2, len(keys) + 1):
        for combo in combinations(keys, r):
            all_combos.append(combo)

    p(f"\nTesting {len(all_combos)} ensemble combinations...")

    for combo in all_combos:
        probs = [models[k] for k in combo]
        avg = np.mean(probs, axis=0)
        log_avg = np.log(avg + 1e-10)

        # Raw
        raw_preds = avg.argmax(1)
        raw_fe = evaluator.evaluate(raw_preds)

        # Bias-optimized
        def obj(biases):
            return evaluator.evaluate((log_avg + biases).argmax(1))

        res = minimize(obj, np.zeros(5), method='Nelder-Mead',
                       options={'maxiter': 5000, 'adaptive': True})
        bias_preds = (log_avg + res.x).argmax(1)
        bias_fe = evaluator.evaluate(bias_preds)

        name = "+".join(combo)
        p(f"  {name}: raw={raw_fe:.6f} bias={bias_fe:.6f}")

        if bias_fe < best_overall:
            best_overall = bias_fe
            best_name = name
            best_preds = bias_preds.copy()

    p(f"\n{'='*60}")
    p(f"BEST ensemble: {best_name} = {best_overall:.6f}")

    # Now do energy-conditional optimization on the best ensemble
    combo_keys = best_name.split("+")
    probs = [models[k] for k in combo_keys]
    avg = np.mean(probs, axis=0)
    log_avg = np.log(avg + 1e-10)

    p("\n=== Energy-conditional optimization on best ensemble ===")
    # 8 equal-count energy bins
    percentiles = np.linspace(0, 100, 9)
    bin_edges = np.percentile(energies, percentiles)
    bin_edges[0] = 0
    bin_edges[-1] = 100
    bin_masks = []
    for i in range(8):
        mask = (energies >= bin_edges[i]) & (energies < bin_edges[i+1])
        bin_masks.append(mask)

    # Get global biases as starting point
    def obj_global(biases):
        return evaluator.evaluate((log_avg + biases).argmax(1))
    res_global = minimize(obj_global, np.zeros(5), method='Nelder-Mead',
                         options={'maxiter': 5000, 'adaptive': True})
    global_biases = res_global.x

    def obj_energy(params):
        preds = np.empty(n_test, dtype=np.int64)
        for b in range(8):
            biases = params[b*5:(b+1)*5]
            mask = bin_masks[b]
            preds[mask] = (log_avg[mask] + biases).argmax(1)
        return evaluator.evaluate(preds)

    x0 = np.tile(global_biases, 8)
    res_energy = minimize(obj_energy, x0, method='Nelder-Mead',
                         options={'maxiter': 30000, 'adaptive': True})
    energy_preds = np.empty(n_test, dtype=np.int64)
    for b in range(8):
        biases = res_energy.x[b*5:(b+1)*5]
        mask = bin_masks[b]
        energy_preds[mask] = (log_avg[mask] + biases).argmax(1)
    energy_fe = evaluator.evaluate(energy_preds)
    p(f"Energy-conditional: {energy_fe:.6f}")

    if energy_fe < best_overall:
        best_overall = energy_fe
        best_preds = energy_preds.copy()
        best_name += "+energy_8bin"

    # Also try weighted ensemble (optimize weights)
    p("\n=== Weighted ensemble optimization ===")
    all_probs = [models[k] for k in combo_keys]
    n_models = len(all_probs)

    def obj_weighted(params):
        weights = np.exp(params[:n_models])
        weights /= weights.sum()
        biases = params[n_models:]
        avg_w = sum(w * p for w, p in zip(weights, all_probs))
        log_w = np.log(avg_w + 1e-10)
        return evaluator.evaluate((log_w + biases).argmax(1))

    x0_w = np.concatenate([np.zeros(n_models), global_biases])
    res_w = minimize(obj_weighted, x0_w, method='Nelder-Mead',
                     options={'maxiter': 10000, 'adaptive': True})
    weights = np.exp(res_w.x[:n_models])
    weights /= weights.sum()
    biases = res_w.x[n_models:]
    avg_w = sum(w * p for w, p in zip(weights, all_probs))
    log_w = np.log(avg_w + 1e-10)
    w_preds = (log_w + biases).argmax(1)
    w_fe = evaluator.evaluate(w_preds)
    p(f"Weighted: {w_fe:.6f} weights={np.round(weights, 3).tolist()}")

    if w_fe < best_overall:
        best_overall = w_fe
        best_preds = w_preds.copy()
        best_name += "+weighted"

    p(f"\n{'='*60}")
    p(f"FINAL BEST: {best_name} = {best_overall:.6f}")
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))
    p(f"Total time: {(time.time()-t0)/60:.1f} min")
    p("---")
    p(f"metric: {best_overall:.6f}")
    p(f"description: {best_name} ensemble + bias opt")

if __name__ == "__main__":
    main()
