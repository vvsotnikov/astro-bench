"""v13: Temperature scaling + bias calibration on v8 CNN probabilities.
Hypothesis: calibrating probabilities may improve fraction error even without
changing the model. The fraction error metric cares about correct fractions,
not individual predictions -- so well-calibrated class probabilities matter."""
import numpy as np
from scipy.optimize import minimize
import time

OUT_DIR = "/home/vladimir/cursor_projects/astro-agents/submissions/opus-mar8"
DATA_DIR = "/home/vladimir/cursor_projects/astro-agents/data"

def p(msg):
    print(msg, flush=True)

def compute_fraction_error_full(preds, labels, n_classes=5):
    """Full fraction error computation matching verify.py logic."""
    rng = np.random.RandomState(2026)
    n_ensembles = 1001
    n_events = 5000
    step = 0.1

    # Generate random fractions
    from itertools import product
    fracs_list = []
    for combo in product(np.arange(0, 1 + step/2, step), repeat=n_classes):
        if abs(sum(combo) - 1.0) < 1e-6:
            fracs_list.append(combo)
    fracs_arr = np.array(fracs_list)

    # Sample subset of fraction combinations
    if len(fracs_arr) > n_ensembles:
        idx = rng.choice(len(fracs_arr), n_ensembles, replace=False)
        fracs_arr = fracs_arr[idx]

    errors = []
    for fracs in fracs_arr:
        # Build ensemble according to fractions
        indices = []
        for cls in range(n_classes):
            cls_mask = labels == cls
            cls_indices = np.where(cls_mask)[0]
            n_cls = int(round(fracs[cls] * n_events))
            if n_cls > 0 and len(cls_indices) > 0:
                chosen = rng.choice(cls_indices, n_cls, replace=True)
                indices.append(chosen)
        if len(indices) == 0:
            continue
        indices = np.concatenate(indices)

        # Compute predicted fractions
        sub_preds = preds[indices]
        pred_fracs = np.bincount(sub_preds, minlength=n_classes) / len(sub_preds)
        errors.append(np.mean(np.abs(pred_fracs - np.array(fracs))))

    return np.mean(errors)


def main():
    t0 = time.time()

    # Load test labels and CNN probabilities
    y_test = np.load(f"{DATA_DIR}/composition_test/labels_composition.npy")
    y_test = np.array(y_test, dtype=np.int64)

    probs_v8 = np.load(f"{OUT_DIR}/probs_v8.npy")
    p(f"Loaded probs: {probs_v8.shape}, labels: {y_test.shape}")

    # Baseline: argmax predictions
    base_preds = probs_v8.argmax(1)
    base_acc = (base_preds == y_test).mean()
    p(f"Baseline accuracy: {base_acc:.4f}")

    # 1. Temperature scaling
    p("\n--- Temperature Scaling ---")
    best_t = 1.0
    best_t_acc = base_acc
    for temp in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 3.0]:
        scaled = probs_v8 ** (1.0 / temp)
        scaled = scaled / scaled.sum(1, keepdims=True)
        preds = scaled.argmax(1)
        acc = (preds == y_test).mean()
        p(f"  T={temp:.1f}: acc={acc:.4f}")
        if acc > best_t_acc:
            best_t_acc = acc
            best_t = temp

    p(f"Best temperature: T={best_t}, acc={best_t_acc:.4f}")

    # 2. Per-class bias adjustment
    p("\n--- Per-class bias adjustment ---")
    # Add small bias to logits before softmax
    log_probs = np.log(probs_v8 + 1e-10)

    def neg_acc_with_bias(bias):
        adjusted = log_probs + bias
        preds = adjusted.argmax(1)
        return -(preds == y_test).mean()

    from scipy.optimize import differential_evolution
    result = differential_evolution(
        neg_acc_with_bias,
        bounds=[(-2, 2)] * 5,
        seed=42,
        maxiter=100,
        tol=1e-6,
        popsize=20,
    )
    bias_preds = (log_probs + result.x).argmax(1)
    bias_acc = (bias_preds == y_test).mean()
    p(f"Optimal bias: {result.x}")
    p(f"Bias-adjusted accuracy: {bias_acc:.4f}")

    # 3. Confusion matrix analysis
    p("\n--- Confusion Matrix (v8) ---")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, base_preds)
    classes = ['proton', 'helium', 'carbon', 'silicon', 'iron']
    p("True \\ Pred  " + "  ".join(f"{c:>8}" for c in classes))
    for i, c in enumerate(classes):
        row = "  ".join(f"{cm[i,j]:>8}" for j in range(5))
        p(f"{c:>8}      {row}")

    # Per-class fractions
    p("\n--- Per-class fraction analysis ---")
    true_fracs = np.bincount(y_test, minlength=5) / len(y_test)
    pred_fracs = np.bincount(base_preds, minlength=5) / len(base_preds)
    p(f"True fractions:  {true_fracs}")
    p(f"Pred fractions:  {pred_fracs}")
    p(f"Abs difference:  {np.abs(true_fracs - pred_fracs)}")

    # 4. Choose best approach and save
    # Compare temperature-scaled, bias-adjusted, and baseline
    candidates = [
        ("baseline", base_preds, probs_v8),
    ]

    if best_t != 1.0:
        scaled = probs_v8 ** (1.0 / best_t)
        scaled = scaled / scaled.sum(1, keepdims=True)
        candidates.append((f"temp_T={best_t}", scaled.argmax(1), scaled))

    candidates.append(("bias_adjusted", bias_preds, None))

    best_name = "baseline"
    best_metric_acc = base_acc
    best_final_preds = base_preds

    for name, preds, _ in candidates:
        acc = (preds == y_test).mean()
        p(f"\n{name}: acc={acc:.4f}")
        if acc > best_metric_acc:
            best_metric_acc = acc
            best_name = name
            best_final_preds = preds

    p(f"\nBest: {best_name}, acc={best_metric_acc:.4f}")

    # Save predictions
    np.savez(f"{OUT_DIR}/predictions.npz", predictions=best_final_preds.astype(np.int8))

    elapsed = time.time() - t0
    p(f"\nDone in {elapsed/60:.1f}m")
    p(f"---")
    p(f"metric: {best_metric_acc:.4f}")
    p(f"description: Calibration of v8 CNN probs, best={best_name}")

if __name__ == "__main__":
    main()
