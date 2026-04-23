"""Stacking experiment: can a learned aggregator over agent softmaxes recover oracle headroom?

Loads per-event class probabilities from Opus v34, GPT v47, and Sonnet v44 (extracted via
extract_*_probs.py scripts in each agent run dir). Splits the 115K test set 50/50 stratified
by class, fits a logistic regression on the concatenated per-agent softmaxes from the train
half, evaluates fraction error on the eval half. Compares against:
  - best single agent (Opus) on the eval half
  - hard-label majority vote on the eval half
  - score-averaging (mean softmax) on the eval half
  - per-event oracle on the eval half (upper bound)

Test-set leakage caveat: the agents' best models (v34/v47/v44) were already selected by
Best@50 on the full test set, so their inputs to the stacker carry selection information from
both halves. Splitting the stacker fit/eval doesn't undo this; we report the experiment as a
demonstration of whether a realized aggregator can recover the oracle headroom in principle,
not as a clean held-out generalization test.
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

# Use the verifier from composition/ to score fraction errors consistently
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "composition"))
from verify import _compute_fraction_error, _load_test_data  # noqa: E402

ROOT = Path("/home/vladimir/cursor_projects/astro-bench-experiments")

OPUS_PROBS = ROOT / "composition-opus-2apr" / "opus_v34_probs.npy"
GPT_PROBS = ROOT / "composition-gpt-2apr" / "gpt_v47_probs.npy"
SONNET_LOGITS = ROOT / "composition-sonnet-27mar" / "logits_v44.npy"  # sum across 5 models * 8 TTA = 40 views


def load_probs():
    opus = np.load(OPUS_PROBS).astype(np.float32)
    gpt = np.load(GPT_PROBS).astype(np.float32)
    sonnet_sum = np.load(SONNET_LOGITS).astype(np.float32)
    sonnet = sonnet_sum / sonnet_sum.sum(axis=1, keepdims=True)  # normalize to probs
    return opus, gpt, sonnet


def report(name, preds, labels):
    res = _compute_fraction_error(labels, preds.astype(int))
    fe = res["mean"]
    acc = float((preds == labels).mean())
    print(f"  {name:>32s}  fe={fe:.5f}  acc={acc:.4f}")
    return fe, acc


def majority_vote(stacked):
    n_events = stacked.shape[1]
    out = np.empty(n_events, dtype=int)
    for i in range(n_events):
        vals, counts = np.unique(stacked[:, i], return_counts=True)
        out[i] = int(vals[counts.argmax()]) if counts.max() >= 2 else int(stacked[0, i])
    return out


def oracle(stacked, labels):
    out = stacked[0].copy()
    for i in range(stacked.shape[1]):
        for j in range(stacked.shape[0]):
            if stacked[j, i] == labels[i]:
                out[i] = labels[i]
                break
    return out


def main():
    print("Loading test labels and per-agent softmaxes...", flush=True)
    labels, _ = _load_test_data()
    opus, gpt, sonnet = load_probs()
    n = len(labels)
    print(f"  test events: {n}")
    print(f"  opus: {opus.shape}, gpt: {gpt.shape}, sonnet: {sonnet.shape}")

    opus_pred = opus.argmax(axis=1)
    gpt_pred = gpt.argmax(axis=1)
    sonnet_pred = sonnet.argmax(axis=1)
    stacked = np.stack([opus_pred, gpt_pred, sonnet_pred])

    # Concatenated softmax features
    X = np.concatenate([opus, gpt, sonnet], axis=1)  # (N, 15)
    print(f"  stacker feature matrix: {X.shape}")

    print("\n=== Full-test-set baselines (no split) ===", flush=True)
    report("Opus alone (full test)", opus_pred, labels)
    mv_full = majority_vote(stacked)
    report("Majority vote (full test)", mv_full, labels)
    avg_full = ((opus + gpt + sonnet) / 3.0).argmax(axis=1)
    report("Score average (full test)", avg_full, labels)
    oracle_full = oracle(stacked, labels)
    report("Oracle (full test)", oracle_full, labels)

    print("\n=== Stacker on stratified 50/50 splits (5 random seeds) ===", flush=True)
    print("  (stacker fit on stratified train half, evaluated on eval half;")
    print("   test-set leakage caveat: agent best-model selection used full test set)", flush=True)

    from sklearn.neural_network import MLPClassifier

    seeds = [42, 1, 7, 100, 2026]
    results = {name: [] for name in [
        "Opus alone", "Majority vote", "Score average",
        "Stacker LR", "Stacker LR-balanced",
        "Stacker MLP", "Stacker MLP-balanced",
        "Oracle",
    ]}

    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        train_idx, eval_idx = next(sss.split(X, labels))
        eval_labels = labels[eval_idx]

        # Baselines on eval half
        results["Opus alone"].append(report(f"[{seed}] Opus alone", opus_pred[eval_idx], eval_labels)[0])
        results["Majority vote"].append(report(f"[{seed}] Majority vote", mv_full[eval_idx], eval_labels)[0])
        results["Score average"].append(report(f"[{seed}] Score average", avg_full[eval_idx], eval_labels)[0])

        # Stacker: vanilla LR
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
        clf.fit(X[train_idx], labels[train_idx])
        results["Stacker LR"].append(report(f"[{seed}] Stacker LR", clf.predict(X[eval_idx]), eval_labels)[0])

        # Stacker: class-balanced LR (compensates for fraction-error class-bias penalty)
        clf_bal = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0, class_weight="balanced")
        clf_bal.fit(X[train_idx], labels[train_idx])
        results["Stacker LR-balanced"].append(report(f"[{seed}] Stacker LR-bal", clf_bal.predict(X[eval_idx]), eval_labels)[0])

        # Stacker: small MLP
        mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=seed, early_stopping=True)
        mlp.fit(X[train_idx], labels[train_idx])
        results["Stacker MLP"].append(report(f"[{seed}] Stacker MLP", mlp.predict(X[eval_idx]), eval_labels)[0])

        # Stacker: class-balanced MLP via sample weights (manually)
        cls_counts = np.bincount(labels[train_idx], minlength=5).astype(float)
        sw = (cls_counts.max() / cls_counts)[labels[train_idx]]
        mlp_bal = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=seed, early_stopping=True)
        # MLPClassifier doesn't accept sample_weight; emulate by upsampling
        rng = np.random.default_rng(seed)
        weights_norm = sw / sw.sum()
        n_sample = len(train_idx) * 2  # double-size weighted sample
        sample = rng.choice(len(train_idx), size=n_sample, replace=True, p=weights_norm)
        mlp_bal.fit(X[train_idx][sample], labels[train_idx][sample])
        results["Stacker MLP-balanced"].append(report(f"[{seed}] Stacker MLP-bal", mlp_bal.predict(X[eval_idx]), eval_labels)[0])

        # Stacker: per-agent temperature calibration on train half, then equal-weight soft vote
        from scipy.optimize import minimize_scalar
        from scipy.special import logsumexp
        def fit_temp(probs_train, y_train):
            logits = np.log(np.clip(probs_train, 1e-12, 1.0))
            def nll(T):
                Tlogits = logits / T
                logp = Tlogits - logsumexp(Tlogits, axis=1, keepdims=True)
                return -logp[np.arange(len(y_train)), y_train].mean()
            r = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
            return r.x
        T_opus = fit_temp(opus[train_idx], labels[train_idx])
        T_gpt = fit_temp(gpt[train_idx], labels[train_idx])
        T_sonnet = fit_temp(sonnet[train_idx], labels[train_idx])
        def calib(probs, T):
            logits = np.log(np.clip(probs, 1e-12, 1.0)) / T
            return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        soft_calib = (calib(opus[eval_idx], T_opus)
                       + calib(gpt[eval_idx], T_gpt)
                       + calib(sonnet[eval_idx], T_sonnet)) / 3.0
        results.setdefault("Calibrated soft-vote", []).append(
            report(f"[{seed}] Calibrated soft-vote (T_opus={T_opus:.2f}, T_gpt={T_gpt:.2f}, T_sonnet={T_sonnet:.2f})",
                   soft_calib.argmax(axis=1), eval_labels)[0]
        )

        results["Oracle"].append(report(f"[{seed}] Oracle", oracle_full[eval_idx], eval_labels)[0])
        print()

    print("\n=== Summary across 5 seeds (mean ± std on eval half) ===")
    sigma = 4e-4  # single-recipe noise floor from sec:comp_noise_floor
    opus_mean = np.mean(results["Opus alone"])
    oracle_mean = np.mean(results["Oracle"])
    headroom = opus_mean - oracle_mean
    print(f"  {'method':>22s}  {'fe (mean)':>10s}  {'± std':>9s}  {'Δ vs Opus':>10s}  {'σ-units':>8s}  {'% headroom':>10s}")
    for name, arr in results.items():
        a = np.array(arr)
        delta = opus_mean - a.mean()
        recovered = 100 * delta / headroom if headroom > 0 else 0.0
        print(f"  {name:>22s}  {a.mean():.5f}  {a.std():.5f}  {delta:+.5f}  {delta/sigma:+.2f}σ  {recovered:+.1f}%")
    print(f"\n  Oracle headroom (Opus → Oracle): {headroom:.5f} = {headroom/sigma:.1f}σ_noise")


if __name__ == "__main__":
    main()
