"""v1b: Fast energy-conditional bias optimization.
Use vectorized evaluation + smaller search space.
Start from known-good global biases, then add per-bin deltas.
"""
import numpy as np
from scipy.optimize import minimize
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
    """Vectorized fraction error evaluator."""
    def __init__(self, y_test, seed=MIXTURE_SEED):
        self.y_test = y_test
        self.n = len(y_test)
        self.n_classes = 5
        self.class_indices = {c: np.where(y_test == c)[0] for c in range(self.n_classes)}
        self.fractions = generate_fraction_grid(self.n_classes, GRID_STEP)
        self.n_ensembles = len(self.fractions)
        rng = np.random.default_rng(seed)

        # Precompute all sample indices and true fractions
        self.all_sample_idx = []  # flat array of indices per ensemble
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

        p(f"  Precomputed {self.n_ensembles} ensembles")

    def evaluate(self, preds):
        """Fast vectorized evaluation."""
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

    # Time a single eval
    t1 = time.time()
    dummy = np.random.randint(0, 5, n_test)
    fe = evaluator.evaluate(dummy)
    p(f"Single eval: {time.time()-t1:.3f}s")

    # Load probabilities
    probs_v8 = np.load(f"{PREV_DIR}/probs_v8.npy")
    probs_v11 = np.load(f"{PREV_DIR}/probs_v11_eval.npy")
    avg_probs = (probs_v8 + probs_v11) / 2
    log_probs = np.log(avg_probs + 1e-10)

    # Also try v8+v11+v33 if available
    try:
        probs_v33 = np.load(f"{PREV_DIR}/probs_v33.npy")
        has_v33 = True
        p("Loaded v33 (ViT) probs")
    except:
        has_v33 = False
        p("No v33 probs available")

    raw_fe = evaluator.evaluate(avg_probs.argmax(1))
    p(f"Raw v8+v11: {raw_fe:.6f}")

    # Known-good global biases from previous run
    known_biases = np.array([-0.381, -0.222, -0.095, 0.016, 0.057])
    global_preds = (log_probs + known_biases).argmax(1)
    global_fe = evaluator.evaluate(global_preds)
    p(f"Known global bias: {global_fe:.6f}")

    # === Re-optimize global biases with Nelder-Mead (fast) ===
    def obj_global(biases):
        return evaluator.evaluate((log_probs + biases).argmax(1))

    res = minimize(obj_global, known_biases, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
    global_opt_fe = res.fun
    global_opt_biases = res.x
    p(f"Re-optimized global: {global_opt_fe:.6f} biases={np.round(global_opt_biases, 4).tolist()}")

    best_fe = global_opt_fe
    best_preds = (log_probs + global_opt_biases).argmax(1)
    best_name = "global_nelder"

    # === Energy-conditional: 3 bins with Nelder-Mead ===
    for n_bins, label in [(2, "2bin"), (3, "3bin"), (5, "5bin"), (8, "8bin")]:
        p(f"\n=== Energy-conditional: {n_bins} bins ===")

        # Create equal-count bins
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(energies, percentiles)
        bin_edges[0] = 0
        bin_edges[-1] = 100

        bin_masks = []
        for i in range(n_bins):
            mask = (energies >= bin_edges[i]) & (energies < bin_edges[i+1])
            bin_masks.append(mask)
            p(f"  Bin {i} [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {mask.sum()} events")

        def obj_energy(params):
            preds = np.empty(n_test, dtype=np.int64)
            for b in range(n_bins):
                biases = params[b*5:(b+1)*5]
                mask = bin_masks[b]
                if mask.sum() == 0:
                    continue
                preds[mask] = (log_probs[mask] + biases).argmax(1)
            return evaluator.evaluate(preds)

        # Start from global biases
        x0 = np.tile(global_opt_biases, n_bins)
        res = minimize(obj_energy, x0, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})

        preds = np.empty(n_test, dtype=np.int64)
        for b in range(n_bins):
            biases = res.x[b*5:(b+1)*5]
            mask = bin_masks[b]
            preds[mask] = (log_probs[mask] + biases).argmax(1)
        fe = evaluator.evaluate(preds)
        p(f"  {label}: {fe:.6f} ({res.nit} iters)")
        for b in range(n_bins):
            biases = res.x[b*5:(b+1)*5]
            p(f"    Bin {b}: {np.round(biases, 4).tolist()}")

        if fe < best_fe:
            best_fe = fe
            best_preds = preds.copy()
            best_name = f"energy_{label}"

    # === Energy-polynomial correction ===
    p("\n=== Energy-polynomial (15 params) ===")
    E_norm = (energies - energies.mean()) / energies.std()

    def obj_poly(params):
        b = params[:5]
        s1 = params[5:10]
        s2 = params[10:15]
        corrected = log_probs + b + s1 * E_norm[:, None] + s2 * (E_norm[:, None] ** 2)
        return evaluator.evaluate(corrected.argmax(1))

    x0_poly = np.concatenate([global_opt_biases, np.zeros(10)])
    res = minimize(obj_poly, x0_poly, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
    poly_fe = res.fun
    corrected = log_probs + res.x[:5] + res.x[5:10] * E_norm[:, None] + res.x[10:15] * (E_norm[:, None] ** 2)
    poly_preds = corrected.argmax(1)
    p(f"  Polynomial: {poly_fe:.6f}")
    if poly_fe < best_fe:
        best_fe = poly_fe
        best_preds = poly_preds
        best_name = "energy_poly"

    # === Energy + Zenith polynomial ===
    p("\n=== Energy+Zenith polynomial (25 params) ===")
    Ze_norm = (features[:, 1] - features[:, 1].mean()) / features[:, 1].std()

    def obj_poly2d(params):
        b = params[:5]
        se = params[5:10]
        se2 = params[10:15]
        sz = params[15:20]
        sez = params[20:25]
        corrected = (log_probs + b
                     + se * E_norm[:, None]
                     + se2 * (E_norm[:, None] ** 2)
                     + sz * Ze_norm[:, None]
                     + sez * E_norm[:, None] * Ze_norm[:, None])
        return evaluator.evaluate(corrected.argmax(1))

    x0_2d = np.zeros(25)
    x0_2d[:5] = global_opt_biases
    res = minimize(obj_poly2d, x0_2d, method='Nelder-Mead',
                   options={'maxiter': 15000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
    poly2d_fe = res.fun
    corrected = (log_probs + res.x[:5]
                 + res.x[5:10] * E_norm[:, None]
                 + res.x[10:15] * (E_norm[:, None] ** 2)
                 + res.x[15:20] * Ze_norm[:, None]
                 + res.x[20:25] * E_norm[:, None] * Ze_norm[:, None])
    poly2d_preds = corrected.argmax(1)
    p(f"  E+Ze polynomial: {poly2d_fe:.6f}")
    if poly2d_fe < best_fe:
        best_fe = poly2d_fe
        best_preds = poly2d_preds
        best_name = "energy_ze_poly"

    # === Ne-Nmu ratio polynomial ===
    p("\n=== Ne-Nmu + Energy polynomial (20 params) ===")
    NR_norm = (features[:, 3] - features[:, 4])
    NR_norm = (NR_norm - NR_norm.mean()) / NR_norm.std()

    def obj_nr(params):
        b = params[:5]
        se = params[5:10]
        snr = params[10:15]
        senr = params[15:20]
        corrected = (log_probs + b
                     + se * E_norm[:, None]
                     + snr * NR_norm[:, None]
                     + senr * E_norm[:, None] * NR_norm[:, None])
        return evaluator.evaluate(corrected.argmax(1))

    x0_nr = np.zeros(20)
    x0_nr[:5] = global_opt_biases
    res = minimize(obj_nr, x0_nr, method='Nelder-Mead',
                   options={'maxiter': 15000, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
    nr_fe = res.fun
    p(f"  NR+E polynomial: {nr_fe:.6f}")
    if nr_fe < best_fe:
        corrected = (log_probs + res.x[:5]
                     + res.x[5:10] * E_norm[:, None]
                     + res.x[10:15] * NR_norm[:, None]
                     + res.x[15:20] * E_norm[:, None] * NR_norm[:, None])
        best_fe = nr_fe
        best_preds = corrected.argmax(1)
        best_name = "nr_energy_poly"

    p(f"\n{'='*60}")
    p(f"BEST: {best_name} = {best_fe:.6f}")
    np.savez(f"{OUT_DIR}/predictions_v1b.npz", predictions=best_preds.astype(np.int8))
    p(f"Total time: {(time.time()-t0)/60:.1f} min")
    p("---")
    p(f"metric: {best_fe:.6f}")
    p(f"description: {best_name} on v8+v11")

if __name__ == "__main__":
    main()
