"""v5: Simple sklearn RandomForest on engineered features

Baseline approach using Random Forest, which often works well on physics problems.
Focus on muon statistics as primary predictors.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from load_data import load_train, load_test
from verify import evaluate


def extract_features(matrices, features):
    """Extract spatial statistics from muon and electron channels."""
    muon_channel = matrices[:, :, :, 1]  # (N, 16, 16)
    electron_channel = matrices[:, :, :, 0]

    # Muon statistics (primary predictors)
    muon_stats = np.stack([
        muon_channel.reshape(len(matrices), -1).mean(axis=1),
        muon_channel.reshape(len(matrices), -1).max(axis=1),
        muon_channel.reshape(len(matrices), -1).sum(axis=1),
        muon_channel.reshape(len(matrices), -1).std(axis=1),
        np.percentile(muon_channel.reshape(len(matrices), -1), 75, axis=1),
        np.percentile(muon_channel.reshape(len(matrices), -1), 90, axis=1),
    ], axis=1)

    # Electron statistics (secondary)
    electron_stats = np.stack([
        electron_channel.reshape(len(matrices), -1).mean(axis=1),
        electron_channel.reshape(len(matrices), -1).max(axis=1),
        electron_channel.reshape(len(matrices), -1).sum(axis=1),
    ], axis=1)

    # Raw features (5) + muon_stats (6) + electron_stats (3) = 14 total
    return np.concatenate([features, muon_stats, electron_stats], axis=1)


def main():
    print("Loading data...")
    X_train, f_train, y_train = load_train()
    X_test, f_test, y_test = load_test()
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")
    print(f"  Train gamma: {(y_train==0).sum()}, hadron: {(y_train==1).sum()}")

    print("Extracting features...")
    X_train_feat = extract_features(X_train, f_train)
    X_test_feat = extract_features(X_test, f_test)
    print(f"  Feature dim: {X_train_feat.shape[1]}")

    # Normalize
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat = scaler.transform(X_test_feat)

    print("Training RandomForest...")
    # Class weights for imbalance
    class_weight = {
        0: len(y_train) / (2 * (y_train == 0).sum()),
        1: len(y_train) / (2 * (y_train == 1).sum()),
    }

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_feat, y_train)
    print("Done training")

    # Get probability of gamma class (class 0)
    test_probs = model.predict_proba(X_test_feat)
    gamma_scores = test_probs[:, 0]

    evaluate(gamma_scores, "v5: RandomForest 500 trees, engineered muon features")


if __name__ == "__main__":
    main()
