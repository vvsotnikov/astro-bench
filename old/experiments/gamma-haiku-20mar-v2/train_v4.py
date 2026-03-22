"""v4: Random Forest on engineered features.

Classical approach: let XGBoost learn the non-linear feature combinations.
Leverage domain features: Ne/Nmu ratio, muon statistics, spatial moments.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from load_data import load_train, load_test
from verify import evaluate


def engineer_features(matrices, features):
    """Physics-informed feature engineering."""
    N = len(matrices)
    eng = np.zeros((N, 15), dtype=np.float32)

    # Original features
    eng[:, 0] = features[:, 0]  # E
    eng[:, 1] = features[:, 1]  # Ze
    eng[:, 2] = features[:, 3] - features[:, 4]  # log(Ne) - log(Nmu)

    # Channel statistics
    e = matrices[:, :, :, 0]
    m = matrices[:, :, :, 1]

    eng[:, 3] = e.sum(axis=(1, 2))
    eng[:, 4] = m.sum(axis=(1, 2))
    eng[:, 5] = e.max(axis=(1, 2))
    eng[:, 6] = m.max(axis=(1, 2))
    eng[:, 7] = (e > 0).sum(axis=(1, 2))
    eng[:, 8] = (m > 0).sum(axis=(1, 2))

    # Ratios
    eng[:, 9] = np.log1p(e.sum(axis=(1, 2)) / (m.sum(axis=(1, 2)) + 1e-6))
    eng[:, 10] = (m > 0).sum(axis=(1, 2)) / ((e > 0).sum(axis=(1, 2)) + 1e-6)
    eng[:, 11] = e.var(axis=(1, 2))
    eng[:, 12] = m.var(axis=(1, 2))
    eng[:, 13] = (matrices[:, :, :, :] > 0).sum(axis=(1, 2, 3))
    eng[:, 14] = matrices[:, :, :, :].sum(axis=(1, 2, 3))

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
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    sample_weights = np.where(y_train == 0, w_gamma, w_hadron)

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=50,
        min_samples_leaf=20,
        n_jobs=8,
        random_state=42,
    )
    rf.fit(feat_train, y_train, sample_weight=sample_weights)

    # Get gamma probabilities
    test_scores = rf.predict_proba(feat_test)[:, 0]
    np.savez("predictions_v4.npz", gamma_scores=test_scores)

    metric = evaluate(test_scores, "v4: Random Forest on engineered features (15 features)")
    return metric


if __name__ == "__main__":
    main()
