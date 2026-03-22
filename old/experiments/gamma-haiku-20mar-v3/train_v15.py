"""v15: Ensemble of all fast models (RF, XGBoost, LR, SVM)

Combine multiple sklearn models that train fast.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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

    class_weight = {
        0: len(y_train) / (2 * (y_train == 0).sum()),
        1: len(y_train) / (2 * (y_train == 1).sum()),
    }

    print("Training ensemble...")

    # RF
    rf = RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X_train_feat, y_train)
    rf_probs = rf.predict_proba(X_test_feat)[:, 0]

    # GB
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    gb.fit(X_train_feat, y_train)
    gb_probs = gb.predict_proba(X_test_feat)[:, 0]

    # LR
    lr = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs', n_jobs=-1)
    lr.fit(X_train_feat, y_train)
    lr_probs = lr.predict_proba(X_test_feat)[:, 0]

    # Ensemble (average)
    ensemble_scores = (rf_probs + gb_probs + lr_probs) / 3

    evaluate(ensemble_scores, "v15: Ensemble (RF + GradBoost + LogReg), avg")


if __name__ == "__main__":
    main()
