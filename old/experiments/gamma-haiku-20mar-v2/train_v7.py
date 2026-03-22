"""v7: XGBoost on engineered features.

Gradient boosting alternative to Random Forest.
Might capture feature interactions better than RF.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from load_data import load_train, load_test
from verify import evaluate


def engineer_features(matrices, features):
    """Same features as v1."""
    N = len(matrices)
    eng = np.zeros((N, 23), dtype=np.float32)

    eng[:, :5] = features[:, :5]

    e = matrices[:, :, :, 0]
    m = matrices[:, :, :, 1]

    eng[:, 5] = m.sum(axis=(1, 2))
    eng[:, 6] = m.max(axis=(1, 2))
    eng[:, 7] = (m > 0).sum(axis=(1, 2))
    eng[:, 8] = e.sum(axis=(1, 2))
    eng[:, 9] = e.max(axis=(1, 2))
    eng[:, 10] = (e > 0).sum(axis=(1, 2))

    ne = features[:, 3]
    nmu = features[:, 4]
    eng[:, 11] = ne - nmu

    eng[:, 12] = (m > 0).sum(axis=(1, 2)) / ((e > 0).sum(axis=(1, 2)) + 1e-6)
    eng[:, 13] = m.sum(axis=(1, 2)) / (e.sum(axis=(1, 2)) + 1e-6)
    eng[:, 14] = m.var(axis=(1, 2))
    eng[:, 15] = e.var(axis=(1, 2))

    for i in range(N):
        e_i = e[i]
        m_i = m[i]
        if e_i.sum() > 0:
            cy, cx = np.indices(e_i.shape)
            eng[i, 16] = np.average(cy, weights=e_i)
            eng[i, 17] = np.average(cx, weights=e_i)
        if m_i.sum() > 0:
            cy, cx = np.indices(m_i.shape)
            eng[i, 18] = np.average(cy, weights=m_i)
            eng[i, 19] = np.average(cx, weights=m_i)

    eng[:, 20] = matrices[:, :, :, :].sum(axis=(1, 2, 3))
    eng[:, 21] = (matrices[:, :, :, :] > 0).sum(axis=(1, 2, 3))
    eng[:, 22] = np.log1p(eng[:, 20] / (eng[:, 21] + 1))

    return eng


def main():
    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")

    print("Engineering features...")
    feat_train = engineer_features(X_train, f_train)
    feat_test = engineer_features(X_test, f_test)

    # Normalize
    mean = feat_train.mean(axis=0)
    std = feat_train.std(axis=0)
    std[std < 1e-6] = 1.0
    feat_train = (feat_train - mean) / std
    feat_test = (feat_test - mean) / std

    # Class weights
    n_gamma = (y_train == 0).sum()
    n_hadron = (y_train == 1).sum()
    scale_pos_weight = n_hadron / n_gamma

    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(feat_train, y_train)

    # Get gamma probabilities
    test_proba = gb.predict_proba(feat_test)
    test_scores = test_proba[:, 0]
    np.savez("predictions_v7.npz", gamma_scores=test_scores)

    metric = evaluate(test_scores, "v7: Gradient Boosting on engineered features (150 trees, depth=6)")
    return metric


if __name__ == "__main__":
    main()
