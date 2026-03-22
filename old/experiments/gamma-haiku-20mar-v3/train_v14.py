"""v14: SVM on engineered muon features

Support Vector Machine with RBF kernel - often effective for physics problems.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from load_data import load_train, load_test
from verify import evaluate


def extract_features(matrices, features):
    muon_channel = matrices[:, :, :, 1]
    electron_channel = matrices[:, :, :, 0]

    muon_stats = np.stack([
        muon_channel.reshape(len(matrices), -1).mean(axis=1),
        muon_channel.reshape(len(matrices), -1).max(axis=1),
        muon_channel.reshape(len(matrices), -1).sum(axis=1),
        muon_channel.reshape(len(matrices), -1).std(axis=1),
        np.percentile(muon_channel.reshape(len(matrices), -1), 75, axis=1),
    ], axis=1)

    electron_stats = np.stack([
        electron_channel.reshape(len(matrices), -1).mean(axis=1),
        electron_channel.reshape(len(matrices), -1).max(axis=1),
    ], axis=1)

    return np.concatenate([features, muon_stats, electron_stats], axis=1)


def main():
    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()

    print("Extracting features...")
    X_train_feat = extract_features(X_train, f_train)
    X_test_feat = extract_features(X_test, f_test)

    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat = scaler.transform(X_test_feat)

    print("Training SVM...")
    # Use class_weight for imbalance
    class_weight = {
        0: 1.0,
        1: (y_train == 0).sum() / (y_train == 1).sum(),
    }

    model = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        class_weight=class_weight,
        probability=True,
    )

    # Train on a sample due to SVM scalability
    sample_idx = np.random.default_rng(42).choice(len(y_train), size=min(100000, len(y_train)), replace=False)
    model.fit(X_train_feat[sample_idx], y_train[sample_idx])
    print("Done")

    test_probs = model.predict_proba(X_test_feat)
    gamma_scores = test_probs[:, 0]

    evaluate(gamma_scores, "v14: SVM RBF on 100k sample, engineered features")


if __name__ == "__main__":
    main()
