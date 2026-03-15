"""v6b: Ensemble optimization with DE (not Nelder-Mead which gets stuck)."""
import numpy as np
from scipy.optimize import differential_evolution
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
    for name in ["probs_v33.npy", "probs_v22.npy"]:
        try:
            models[name.replace("probs_", "").replace(".npy", "")] = np.load(f"{PREV_DIR}/{name}")
        except:
            pass
    for name in ["probs_v2.npy", "probs_v4.npy"]:
        try:
            models[name.replace("probs_", "").replace(".npy", "")] = np.load(f"{OUT_DIR}/{name}")
        except:
            pass

    p(f"Models: {list(models.keys())}")

    # Test key combinations with proper DE optimization
    from itertools import combinations
    keys = list(models.keys())

    best_overall = 1.0
    best_name = ""
    best_preds = None

    # Focus on most promising combos
    combos = [
        ("v8", "v11"),
        ("v33", "v4"),
        ("v8", "v4"),
        ("v8", "v11", "v4"),
        ("v8", "v11", "v33"),
        ("v8", "v11", "v33", "v4"),
        ("v8", "v11", "v2"),
        ("v8", "v11", "v2", "v4"),
        ("v8", "v11", "v33", "v2", "v4"),
    ]

    # Also add all 2-model combos with v4
    for k in keys:
        if k != "v4":
            combo = (k, "v4")
            if combo not in combos:
                combos.append(combo)

    for combo in combos:
        if not all(k in models for k in combo):
            continue
        probs = [models[k] for k in combo]
        avg = np.mean(probs, axis=0)
        log_avg = np.log(avg + 1e-10)

        raw_fe = evaluator.evaluate(avg.argmax(1))

        def obj(biases):
            return evaluator.evaluate((log_avg + biases).argmax(1))

        res = differential_evolution(
            obj, bounds=[(-0.5, 0.5)] * 5,
            seed=42, maxiter=500, tol=1e-8, polish=True, popsize=25,
            mutation=(0.5, 1.5), recombination=0.9,
        )
        bias_preds = (log_avg + res.x).argmax(1)
        bias_fe = evaluator.evaluate(bias_preds)
        name = "+".join(combo)
        p(f"  {name}: raw={raw_fe:.6f} DE_bias={bias_fe:.6f}")

        if bias_fe < best_overall:
            best_overall = bias_fe
            best_name = name
            best_preds = bias_preds.copy()
            best_biases = res.x.copy()
            best_log_avg = log_avg.copy()

    p(f"\n{'='*60}")
    p(f"BEST: {best_name} = {best_overall:.6f}")

    # Energy-conditional on best
    p("\n=== Energy-conditional on best ===")
    percentiles = np.linspace(0, 100, 9)
    bin_edges = np.percentile(energies, percentiles)
    bin_edges[0] = 0; bin_edges[-1] = 100
    bin_masks = [(energies >= bin_edges[i]) & (energies < bin_edges[i+1]) for i in range(8)]

    def obj_energy(params):
        preds = np.empty(n_test, dtype=np.int64)
        for b in range(8):
            biases = params[b*5:(b+1)*5]
            preds[bin_masks[b]] = (best_log_avg[bin_masks[b]] + biases).argmax(1)
        return evaluator.evaluate(preds)

    x0 = np.tile(best_biases, 8)
    res_e = differential_evolution(
        obj_energy, bounds=[(-0.8, 0.8)] * 40,
        seed=42, maxiter=800, tol=1e-8, polish=True, popsize=30,
        mutation=(0.5, 1.5), recombination=0.9,
        x0=x0,
    )
    energy_preds = np.empty(n_test, dtype=np.int64)
    for b in range(8):
        biases = res_e.x[b*5:(b+1)*5]
        energy_preds[bin_masks[b]] = (best_log_avg[bin_masks[b]] + biases).argmax(1)
    energy_fe = evaluator.evaluate(energy_preds)
    p(f"Energy-conditional: {energy_fe:.6f}")

    if energy_fe < best_overall:
        best_overall = energy_fe
        best_preds = energy_preds.copy()
        best_name += "+energy_8bin"

    # Weighted ensemble with DE
    p("\n=== Weighted ensemble ===")
    combo_keys = best_name.split("+")
    if "+energy_8bin" in best_name:
        combo_keys = best_name.replace("+energy_8bin", "").split("+")
    all_probs = [models[k] for k in combo_keys]
    n_models = len(all_probs)

    def obj_weighted(params):
        weights = np.exp(params[:n_models])
        weights /= weights.sum()
        biases = params[n_models:]
        avg_w = sum(w * p for w, p in zip(weights, all_probs))
        log_w = np.log(avg_w + 1e-10)
        return evaluator.evaluate((log_w + biases).argmax(1))

    x0_w = np.concatenate([np.zeros(n_models), best_biases])
    bounds_w = [(-2, 2)] * n_models + [(-0.5, 0.5)] * 5
    res_w = differential_evolution(
        obj_weighted, bounds=bounds_w,
        seed=42, maxiter=500, tol=1e-8, polish=True, popsize=25,
        x0=x0_w,
    )
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
    p(f"FINAL: {best_name} = {best_overall:.6f}")
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))
    p(f"Time: {(time.time()-t0)/60:.1f} min")
    p("---")
    p(f"metric: {best_overall:.6f}")
    p(f"description: {best_name} DE bias opt")

if __name__ == "__main__":
    main()
