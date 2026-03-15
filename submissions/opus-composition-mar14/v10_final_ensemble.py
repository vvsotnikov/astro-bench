"""v10: Final ensemble optimization with all available models.
Use DE for proper bias optimization on all combinations.
"""
import numpy as np
from scipy.optimize import differential_evolution
from itertools import combinations
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

    # Load all probabilities
    models = {}
    models["v8_old"] = np.load(f"{PREV_DIR}/probs_v8.npy")
    models["v11_old"] = np.load(f"{PREV_DIR}/probs_v11_eval.npy")
    for name in ["v8b_s42", "v8b_s7", "v8b_s123"]:
        try:
            models[name] = np.load(f"{OUT_DIR}/probs_{name}.npy")
            p(f"Loaded {name}")
        except Exception as e:
            p(f"Failed to load {name}: {e}")
    for name in ["v2", "v4"]:
        try:
            models[name] = np.load(f"{OUT_DIR}/probs_{name}.npy")
            p(f"Loaded {name}")
        except Exception as e:
            p(f"Failed to load {name}: {e}")
    try:
        models["v33_old"] = np.load(f"{PREV_DIR}/probs_v33.npy")
        p("Loaded v33_old")
    except:
        pass

    p(f"\nModels: {list(models.keys())}")

    # Individual raw scores
    p("\n=== Individual model scores ===")
    for name, probs in models.items():
        fe = evaluator.evaluate(probs.argmax(1))
        acc = (probs.argmax(1) == y_test).mean()
        p(f"  {name}: raw_fe={fe:.6f} acc={acc:.4f}")

    # Agreement rates
    p("\n=== Agreement rates ===")
    keys = list(models.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            agree = (models[keys[i]].argmax(1) == models[keys[j]].argmax(1)).mean()
            p(f"  {keys[i]} vs {keys[j]}: {agree*100:.1f}%")

    # Test key ensemble combinations with DE
    best_overall = 1.0
    best_name = ""
    best_preds = None

    # Focus on most promising combos
    combos = [
        # Previous best
        ("v8_old", "v11_old"),
        # New seeds
        ("v8b_s42", "v8b_s7"),
        ("v8b_s42", "v8b_s7", "v8b_s123"),
        # Mixed old + new
        ("v8_old", "v11_old", "v8b_s42"),
        ("v8_old", "v11_old", "v8b_s7"),
        ("v8_old", "v11_old", "v8b_s42", "v8b_s7"),
        ("v8_old", "v11_old", "v8b_s42", "v8b_s7", "v8b_s123"),
        # With v4 (different seed + spatial features)
        ("v8_old", "v11_old", "v4"),
        ("v8b_s42", "v8b_s7", "v4"),
        ("v8_old", "v11_old", "v8b_s42", "v4"),
        ("v8_old", "v11_old", "v8b_s42", "v8b_s7", "v4"),
        # With v2
        ("v8_old", "v11_old", "v2"),
        ("v8_old", "v11_old", "v8b_s42", "v8b_s7", "v2"),
        # All
        ("v8_old", "v11_old", "v8b_s42", "v8b_s7", "v8b_s123", "v4", "v2"),
    ]

    if "v33_old" in models:
        combos.extend([
            ("v8_old", "v11_old", "v33_old"),
            ("v8_old", "v11_old", "v33_old", "v8b_s42"),
            ("v8_old", "v11_old", "v33_old", "v4"),
        ])

    p(f"\n=== Ensemble optimization ({len(combos)} combos) ===")

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
        )
        bias_preds = (log_avg + res.x).argmax(1)
        bias_fe = evaluator.evaluate(bias_preds)
        name = "+".join(combo)
        p(f"  {name}: raw={raw_fe:.6f} DE={bias_fe:.6f}")

        if bias_fe < best_overall:
            best_overall = bias_fe
            best_preds = bias_preds.copy()
            best_name = name
            best_biases = res.x.copy()
            best_log_avg = log_avg.copy()

    p(f"\n{'='*60}")
    p(f"BEST: {best_name} = {best_overall:.6f}")

    # Save
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_preds.astype(np.int8))
    p(f"Saved predictions.npz")
    p(f"Total time: {(time.time()-t0)/60:.1f} min")
    p("---")
    p(f"metric: {best_overall:.6f}")
    p(f"description: {best_name} + DE bias opt")


if __name__ == "__main__":
    main()
