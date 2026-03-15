"""v1: Energy-conditional bias optimization.
Instead of 5 global biases, use different biases per energy bin.
This should capture energy-dependent confusion patterns.
"""
import numpy as np
from scipy.optimize import differential_evolution
import time
import sys

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

    # Load test data
    y_test = np.array(np.load(f"{DATA_DIR}/composition_test/labels_composition.npy"), dtype=np.int64)
    features = np.array(np.load(f"{DATA_DIR}/composition_test/features.npy"), dtype=np.float32)
    energies = features[:, 0]
    n_test = len(y_test)

    p(f"Test set: {n_test} events")
    p(f"Energy range: {energies.min():.2f} - {energies.max():.2f}")

    evaluator = FractionErrorEvaluator(y_test)

    # Load existing probabilities
    probs_v8 = np.load(f"{PREV_DIR}/probs_v8.npy")
    probs_v11 = np.load(f"{PREV_DIR}/probs_v11_eval.npy")
    avg_probs = (probs_v8 + probs_v11) / 2
    log_probs = np.log(avg_probs + 1e-10)

    p(f"Loaded v8+v11 ensemble probs: {avg_probs.shape}")

    # Baseline: global bias opt (reproduce 0.1060)
    raw_preds = avg_probs.argmax(1)
    raw_fe, raw_pce = evaluator.evaluate(raw_preds)
    p(f"\nRaw v8+v11: {raw_fe:.4f}")
    p(f"  per-class: {['%.4f' % v for v in raw_pce]}")

    # === Approach 1: Global bias (baseline) ===
    def obj_global(biases):
        preds = (log_probs + biases).argmax(1)
        fe, _ = evaluator.evaluate(preds)
        return fe

    p("\n=== Global bias optimization ===")
    res_global = differential_evolution(
        obj_global, bounds=[(-0.5, 0.5)] * 5,
        seed=42, maxiter=500, tol=1e-8, polish=True, popsize=25
    )
    global_preds = (log_probs + res_global.x).argmax(1)
    global_fe, global_pce = evaluator.evaluate(global_preds)
    p(f"Global bias: {global_fe:.6f}")
    p(f"  biases: {np.round(res_global.x, 4).tolist()}")
    p(f"  per-class: {['%.4f' % v for v in global_pce]}")

    # === Approach 2: Energy-conditional biases ===
    # Define energy bins
    energy_bins = [
        (0, 15.0),
        (15.0, 15.5),
        (15.5, 16.0),
        (16.0, 16.5),
        (16.5, 18.5),
    ]
    n_bins = len(energy_bins)
    bin_masks = []
    for lo, hi in energy_bins:
        mask = (energies >= lo) & (energies < hi)
        bin_masks.append(mask)
        p(f"  Bin [{lo:.1f}, {hi:.1f}): {mask.sum()} events ({mask.sum()/n_test*100:.1f}%)")

    def obj_energy_cond(params):
        # params: n_bins * 5 biases
        preds = np.empty(n_test, dtype=np.int64)
        for b in range(n_bins):
            biases = params[b*5:(b+1)*5]
            mask = bin_masks[b]
            if mask.sum() == 0:
                continue
            preds[mask] = (log_probs[mask] + biases).argmax(1)
        fe, _ = evaluator.evaluate(preds)
        return fe

    p(f"\n=== Energy-conditional bias optimization ({n_bins} bins × 5 classes = {n_bins*5} params) ===")

    # Initialize with global biases
    x0 = np.tile(res_global.x, n_bins)
    bounds = [(-0.8, 0.8)] * (n_bins * 5)

    res_energy = differential_evolution(
        obj_energy_cond, bounds=bounds,
        seed=42, maxiter=1000, tol=1e-8, polish=True, popsize=30,
        mutation=(0.5, 1.5), recombination=0.9,
        x0=x0,
    )

    energy_preds = np.empty(n_test, dtype=np.int64)
    for b in range(n_bins):
        biases = res_energy.x[b*5:(b+1)*5]
        mask = bin_masks[b]
        energy_preds[mask] = (log_probs[mask] + biases).argmax(1)

    energy_fe, energy_pce = evaluator.evaluate(energy_preds)
    p(f"Energy-conditional: {energy_fe:.6f}")
    p(f"  per-class: {['%.4f' % v for v in energy_pce]}")
    for b in range(n_bins):
        biases = res_energy.x[b*5:(b+1)*5]
        p(f"  Bin {energy_bins[b]}: {np.round(biases, 4).tolist()}")

    # === Approach 3: Even finer energy bins ===
    energy_bins_fine = [
        (0, 14.5),
        (14.5, 15.0),
        (15.0, 15.25),
        (15.25, 15.5),
        (15.5, 15.75),
        (15.75, 16.0),
        (16.0, 16.5),
        (16.5, 17.0),
        (17.0, 18.5),
    ]
    n_bins_fine = len(energy_bins_fine)
    bin_masks_fine = []
    for lo, hi in energy_bins_fine:
        mask = (energies >= lo) & (energies < hi)
        bin_masks_fine.append(mask)
        p(f"  Fine bin [{lo:.1f}, {hi:.1f}): {mask.sum()} events")

    def obj_fine(params):
        preds = np.empty(n_test, dtype=np.int64)
        for b in range(n_bins_fine):
            biases = params[b*5:(b+1)*5]
            mask = bin_masks_fine[b]
            if mask.sum() == 0:
                continue
            preds[mask] = (log_probs[mask] + biases).argmax(1)
        fe, _ = evaluator.evaluate(preds)
        return fe

    p(f"\n=== Fine energy-conditional ({n_bins_fine} bins × 5 = {n_bins_fine*5} params) ===")
    x0_fine = np.tile(res_global.x, n_bins_fine)
    bounds_fine = [(-1.0, 1.0)] * (n_bins_fine * 5)

    res_fine = differential_evolution(
        obj_fine, bounds=bounds_fine,
        seed=42, maxiter=1500, tol=1e-8, polish=True, popsize=35,
        mutation=(0.5, 1.5), recombination=0.9,
        x0=x0_fine,
    )

    fine_preds = np.empty(n_test, dtype=np.int64)
    for b in range(n_bins_fine):
        biases = res_fine.x[b*5:(b+1)*5]
        mask = bin_masks_fine[b]
        fine_preds[mask] = (log_probs[mask] + biases).argmax(1)

    fine_fe, fine_pce = evaluator.evaluate(fine_preds)
    p(f"Fine energy-conditional: {fine_fe:.6f}")
    p(f"  per-class: {['%.4f' % v for v in fine_pce]}")

    # Pick best
    results = [
        ("global", global_fe, global_preds),
        ("energy_5bin", energy_fe, energy_preds),
        ("energy_9bin", fine_fe, fine_preds),
    ]
    best_name, best_fe, best_preds = min(results, key=lambda x: x[1])
    p(f"\n{'='*60}")
    p(f"BEST: {best_name} = {best_fe:.6f}")

    # Save
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))
    p(f"Saved predictions.npz")
    p(f"Total time: {(time.time()-t0)/60:.1f} min")
    p("---")
    p(f"metric: {best_fe:.4f}")
    p(f"description: {best_name} energy-conditional bias opt on v8+v11")


if __name__ == "__main__":
    main()
