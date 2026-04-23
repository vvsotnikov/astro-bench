"""Estimate the Bayes-error ceiling on the reconstructed KASCADE features.

Trains several convergent-estimator classifiers on just the 6 reconstructed
features (E, Ze, Az, log Ne, log Nmu, Age) --- ignoring the 16x16 detector
images --- and reports their mean-fraction-error on the sim test set.

If these classifiers reach fraction error near 0.105 (the cross-agent
convergence band), it supports the claim that the composition ceiling is
driven by physical overlap in the feature distributions rather than any
architectural limitation.

Classifiers:
  - HistGradientBoostingClassifier (scikit-learn's GBT, strong baseline)
  - RandomForestClassifier (500 trees)
  - KNeighborsClassifier (k=1000, nonparametric Bayes-error proxy)

Uses verify._compute_fraction_error for the metric, consistent with the
benchmark. Does NOT touch composition/results.tsv.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(REPO_ROOT / "composition"))
from load_data import load_train, load_test  # noqa: E402
from verify import _compute_fraction_error  # noqa: E402


def fit_and_score(name, clf, X_train, y_train, X_test, y_test, scale=False):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    if scale:
        sc = StandardScaler().fit(X_train)
        X_train_s = sc.transform(X_train)
        X_test_s = sc.transform(X_test)
    else:
        X_train_s = X_train
        X_test_s = X_test
    clf.fit(X_train_s, y_train)
    t_fit = time.time() - t0
    print(f"  fit: {t_fit:.1f}s")

    t0 = time.time()
    pred = clf.predict(X_test_s).astype(int)
    t_pred = time.time() - t0
    print(f"  predict: {t_pred:.1f}s")

    acc = float((pred == y_test).mean())
    res = _compute_fraction_error(y_test, pred)
    fe = res["mean"]
    print(f"  accuracy = {acc:.4f}")
    print(f"  mean fraction error = {fe:.8f}  (rounded: {fe:.4f})")
    return fe, acc


def main():
    print("Loading data (features only, no matrices)...", flush=True)
    _, f_train, y_train = load_train()
    _, f_test, y_test = load_test()
    X_train = np.asarray(f_train, dtype=np.float32)
    X_test = np.asarray(f_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    print(f"  train: {X_train.shape}  test: {X_test.shape}")
    print(f"  feature order (from CLAUDE.md): "
          f"[E, Ze, Az, log10(Ne), log10(Nmu), Age]")

    results = {}

    # HistGradientBoosting --- usually the strongest tabular classifier out-of-the-box
    results["HGB"] = fit_and_score(
        "HistGradientBoosting (max_iter=500, depth unlim)",
        HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
            l2_regularization=0.1, random_state=42, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=30,
        ),
        X_train, y_train, X_test, y_test,
    )

    # Random Forest --- standard reference, used by Kuznetsov+ for composition
    results["RF"] = fit_and_score(
        "RandomForest (500 trees)",
        RandomForestClassifier(
            n_estimators=500, n_jobs=-1, random_state=42,
            class_weight="balanced", max_features="sqrt",
        ),
        X_train, y_train, X_test, y_test,
    )

    # k-NN with large k --- nonparametric Bayes-error proxy
    # k=1000 is well above the typical sqrt(n) rule and keeps k/n small.
    results["k-NN-1000"] = fit_and_score(
        "k-NN (k=1000, standard-scaled features)",
        KNeighborsClassifier(n_neighbors=1000, n_jobs=-1),
        X_train, y_train, X_test, y_test,
        scale=True,  # k-NN requires scaled features
    )

    print("\n\n=== Summary: features-only fraction error ===")
    print(f"  {'Classifier':>36s}  {'accuracy':>10s}  {'fraction_error':>16s}")
    for name, (fe, acc) in results.items():
        print(f"  {name:>36s}  {acc:>10.4f}  {fe:>16.6f}")
    print()
    best_fe = min(fe for fe, _ in results.values())
    print(f"Best features-only fraction error: {best_fe:.6f}")
    print(f"Cross-agent best (Opus v34 ensemble, with images): 0.10507")
    print(f"Published baseline (LeNet CNN, with images): 0.107")


if __name__ == "__main__":
    main()
