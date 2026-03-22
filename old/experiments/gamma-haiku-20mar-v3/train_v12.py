"""v12: XGBoost on engineered features

Gradient boosting is often very effective for tabular features.
"""

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not installed, using GradientBoosting instead")
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

import numpy as np
from sklearn.preprocessing import StandardScaler
from load_data import load_train, load_test
from verify import evaluate


def extract_features(matrices, features):
    muon_channel = matrices[:, :, :, 1]
    electron_channel = matrices[:, :, :, 0]

    muon_stats = np.stack([
        muon_channel.reshape(len(matrices), -1).mean(axis=1),
        muon_channel.reshape(len(matrices), -1).max(axis=1),
        muon_channel.reshape(len(matrices), -1).min(axis=1),
        muon_channel.reshape(len(matrices), -1).sum(axis=1),
        muon_channel.reshape(len(matrices), -1).std(axis=1),
        np.percentile(muon_channel.reshape(len(matrices), -1), 75, axis=1),
        np.percentile(muon_channel.reshape(len(matrices), -1), 90, axis=1),
    ], axis=1)

    electron_stats = np.stack([
        electron_channel.reshape(len(matrices), -1).mean(axis=1),
        electron_channel.reshape(len(matrices), -1).max(axis=1),
        electron_channel.reshape(len(matrices), -1).sum(axis=1),
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

    # Class weights
    class_weight = {
        0: len(y_train) / (2 * (y_train == 0).sum()),
        1: len(y_train) / (2 * (y_train == 1).sum()),
    }

    print("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=class_weight[0] / class_weight[1],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_feat, y_train)
    print("Done")

    test_probs = model.predict_proba(X_test_feat)
    gamma_scores = test_probs[:, 0]

    evaluate(gamma_scores, "v12: XGBoost 500 trees, engineered features")


if __name__ == "__main__":
    main()
