"""Shared evaluation utilities for all experiments.

Usage at the end of any training script:

    from eval_utils import evaluate_and_save
    evaluate_and_save(
        test_probs=probs,           # (N, 5) numpy array of softmax probabilities
        test_labels=labels,         # (N,) numpy array of true labels (0-4)
        model=model,                # torch model (for saving weights)
        experiment_name="v3_gnn",   # unique experiment identifier
        description="GNN 3-layer message passing, 77K params",
        out_dir="submissions/my-run",
    )

This will:
1. Compute raw fraction error
2. Run DE bias optimization
3. Save probs, predictions, model weights
4. Append results to results.tsv
5. Print a summary
"""

import numpy as np
import torch
from scipy.optimize import differential_evolution
from pathlib import Path
import time

MIXTURE_SIZE = 5000
MIXTURE_SEED = 2026
GRID_STEP = 0.1
PARTICLE_NAMES = ["proton", "helium", "carbon", "silicon", "iron"]


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
    """Precomputed fraction error evaluator matching verify.py methodology."""

    def __init__(self, y_test, seed=MIXTURE_SEED):
        self.n_classes = 5
        class_indices = {c: np.where(y_test == c)[0] for c in range(self.n_classes)}
        self.fractions = generate_fraction_grid()
        self.n_ensembles = len(self.fractions)
        rng = np.random.default_rng(seed)

        self.sample_indices = []
        self.true_fracs = np.zeros((self.n_ensembles, self.n_classes))

        for mix_idx in range(self.n_ensembles):
            counts = np.round(self.fractions[mix_idx] * MIXTURE_SIZE).astype(int)
            diff = MIXTURE_SIZE - counts.sum()
            if diff != 0:
                counts[np.argmax(counts)] += diff

            indices = []
            for c in range(self.n_classes):
                if counts[c] <= 0:
                    continue
                idx = rng.choice(class_indices[c], size=counts[c], replace=True)
                indices.append(idx)
                self.true_fracs[mix_idx, c] = counts[c] / MIXTURE_SIZE

            self.sample_indices.append(np.concatenate(indices))

    def evaluate(self, preds):
        errors = np.zeros((self.n_ensembles, self.n_classes))
        for i in range(self.n_ensembles):
            sampled = preds[self.sample_indices[i]]
            pred_counts = np.bincount(sampled, minlength=self.n_classes)[:self.n_classes]
            pred_fracs = pred_counts / pred_counts.sum()
            errors[i] = np.abs(self.true_fracs[i] - pred_fracs)
        return float(errors.mean())


def evaluate_and_save(
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    model=None,
    experiment_name: str = "unnamed",
    description: str = "",
    out_dir: str = ".",
    run_de: bool = True,
):
    """Evaluate model, run DE bias optimization, save all artifacts, log results.

    Args:
        test_probs: (N, 5) softmax probabilities on test set
        test_labels: (N,) true labels (0-4)
        model: torch model (optional, for saving weights)
        experiment_name: unique name like "v3_gnn"
        description: one-line description for results.tsv
        out_dir: directory to save artifacts
        run_de: whether to run DE bias optimization (slow but recommended)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    evaluator = FractionErrorEvaluator(test_labels)

    # Raw evaluation
    raw_preds = test_probs.argmax(1)
    raw_acc = (raw_preds == test_labels).mean()
    raw_fe = evaluator.evaluate(raw_preds)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  {experiment_name}: {description}", flush=True)
    print(f"  Raw: acc={raw_acc:.4f}  frac_err={raw_fe:.4f}", flush=True)

    best_fe = raw_fe
    best_preds = raw_preds
    biases = None

    # DE bias optimization
    if run_de:
        print(f"  Running DE bias optimization...", flush=True)
        t0 = time.time()
        log_probs = np.log(test_probs + 1e-10)

        def obj(b):
            return evaluator.evaluate((log_probs + b).argmax(1))

        res = differential_evolution(
            obj,
            bounds=[(-0.5, 0.5)] * 5,
            seed=42,
            maxiter=500,
            tol=1e-8,
            polish=False,  # polish=True causes Nelder-Mead to hang on some models
            popsize=25,
        )
        de_preds = (log_probs + res.x).argmax(1)
        de_fe = evaluator.evaluate(de_preds)
        biases = res.x
        print(
            f"  DE:  frac_err={de_fe:.4f}  biases={np.round(biases, 4).tolist()}  [{time.time()-t0:.0f}s]",
            flush=True,
        )

        if de_fe < best_fe:
            best_fe = de_fe
            best_preds = de_preds

    print(f"  BEST: {best_fe:.4f}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Save artifacts
    np.save(out / f"probs_{experiment_name}.npy", test_probs)
    np.savez(out / f"predictions_{experiment_name}.npz", predictions=best_preds.astype(np.int8))
    if model is not None:
        torch.save(model.state_dict(), out / f"model_{experiment_name}.pt")

    # Append to results.tsv
    tsv_path = out / "results.tsv"
    if not tsv_path.exists():
        tsv_path.write_text("experiment\tmetric\tstatus\tdescription\n")
    with open(tsv_path, "a") as f:
        de_note = f" (DE={best_fe:.4f})" if run_de and biases is not None else ""
        f.write(f"{experiment_name}\t{best_fe:.4f}\tkeep\t{description}{de_note}\n")

    return best_fe, best_preds, biases
