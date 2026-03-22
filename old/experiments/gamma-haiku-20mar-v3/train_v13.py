"""v13: Logistic regression on muon-focused features

Ultra-simple baseline to establish lower bound on what works.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from load_data import load_train, load_test
from verify import evaluate


def extract_muon_features(matrices, features):
    """Extract only muon-based features + scalar features."""
    muon_channel = matrices[:, :, :, 1]

    muon_stats = np.stack([
        muon_channel.reshape(len(matrices), -1).mean(axis=1),
        muon_channel.reshape(len(matrices), -1).max(axis=1),
        muon_channel.reshape(len(matrices), -1).sum(axis=1),
        muon_channel.reshape(len(matrices), -1).std(axis=1),
    ], axis=1)

    return np.concatenate([features, muon_stats], axis=1)


def main():
    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()

    print("Extracting muon features...")
    X_train_feat = extract_muon_features(X_train, f_train)
    X_test_feat = extract_muon_features(X_test, f_test)

    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat = scaler.transform(X_test_feat)

    # Class weights
    neg_weight = 1.0
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("Training Logistic Regression...")
    model = LogisticRegression(
        C=0.1,
        class_weight={0: neg_weight, 1: pos_weight},
        max_iter=1000,
        solver='lbfgs',
        n_jobs=-1,
    )
    model.fit(X_train_feat, y_train)
    print("Done")

    test_probs = model.predict_proba(X_test_feat)
    gamma_scores = test_probs[:, 0]

    evaluate(gamma_scores, "v13: Logistic Regression on muon + scalar features")


if __name__ == "__main__":
    main()
