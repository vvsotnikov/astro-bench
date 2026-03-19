"""Gamma/hadron v5: Random Forest with engineered features.

Engineer features like Ne/Nmu ratio, log transforms, etc.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

def engineer_features(X):
    """Create engineered features from raw 5-feature set."""
    E, Ze, Az, Ne, Nmu = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

    # Log transforms
    log_E = np.log10(np.maximum(E, 1e-6))
    log_Ne = Ne  # already log
    log_Nmu = Nmu  # already log

    # Ratios
    Ne_Nmu_ratio = Ne - Nmu  # log ratio

    # Trigonometric features for angles
    Ze_rad = np.radians(Ze)
    Az_rad = np.radians(Az)
    Ze_sin = np.sin(Ze_rad)
    Ze_cos = np.cos(Ze_rad)
    Az_sin = np.sin(Az_rad)
    Az_cos = np.cos(Az_rad)

    # Stack all
    features = np.stack([
        E, Ze, Az, Ne, Nmu,  # original 5
        Ne_Nmu_ratio,        # key discriminant
        log_E,
        Ze_sin, Ze_cos,
        Az_sin, Az_cos,
    ], axis=1)

    return features

def main():
    # Load data
    print("Loading data...")
    X_train = np.load("data/gamma_train/features.npy", mmap_mode="r").astype(np.float32)
    y_train = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")

    X_test = np.load("data/gamma_test/features.npy", mmap_mode="r").astype(np.float32)
    y_test = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")

    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    # Engineer features
    print("Engineering features...")
    X_train_eng = engineer_features(X_train)
    X_test_eng = engineer_features(X_test)
    print(f"Engineered: train {X_train_eng.shape}, test {X_test_eng.shape}")

    # Normalize
    scaler = StandardScaler()
    X_train_eng = scaler.fit_transform(X_train_eng)
    X_test_eng = scaler.transform(X_test_eng)

    # Class weights
    n_gamma = sum(y_train == 0)
    n_hadron = sum(y_train == 1)
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    class_weights = {0: w_gamma, 1: w_hadron}
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    # Gradient boosting
    print("Training Gradient Boosting...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        verbose=10,
    )
    model.fit(X_train_eng, y_train)

    # Get scores
    gamma_scores = model.predict_proba(X_test_eng)[:, 0]

    print(f"\nScores stats: min={gamma_scores.min():.4f}, max={gamma_scores.max():.4f}, mean={gamma_scores.mean():.4f}")

    # Feature importance
    print("\nFeature importances:")
    feature_names = ["E", "Ze", "Az", "Ne", "Nmu", "Ne/Nmu", "log_E", "Ze_sin", "Ze_cos", "Az_sin", "Az_cos"]
    for name, imp in zip(feature_names, model.feature_importances_):
        print(f"  {name}: {imp:.4f}")

    # Save predictions
    np.savez(
        "submissions/gamma-haiku-19mar/predictions.npz",
        gamma_scores=gamma_scores
    )
    print(f"Saved predictions ({len(gamma_scores)} scores)")

    print("\n---")
    print("metric: 0.0000")
    print("description: Gradient Boosting with engineered features")

if __name__ == "__main__":
    main()
