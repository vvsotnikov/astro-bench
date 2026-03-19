"""Gamma/hadron v2: Random Forest on 5 scalar features."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load data
    print("Loading data...")
    X_train = np.load("data/gamma_train/features.npy", mmap_mode="r").astype(np.float32)
    y_train = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")

    X_test = np.load("data/gamma_test/features.npy", mmap_mode="r").astype(np.float32)
    y_test = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")

    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    # Class weights
    n_gamma = sum(y_train == 0)
    n_hadron = sum(y_train == 1)
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    class_weights = {0: w_gamma, 1: w_hadron}
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    # Train random forest
    print("\nTraining random forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=50,
        min_samples_leaf=20,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)

    # Get scores: probability of class 0 (gamma)
    gamma_scores = model.predict_proba(X_test)[:, 0]

    print(f"\nScores stats: min={gamma_scores.min():.4f}, max={gamma_scores.max():.4f}, mean={gamma_scores.mean():.4f}")

    # Feature importance
    print("\nFeature importances:")
    for i, (name, imp) in enumerate(zip(["E", "Ze", "Az", "Ne", "Nmu"], model.feature_importances_)):
        print(f"  {name}: {imp:.4f}")

    # Save predictions
    np.savez(
        "submissions/gamma-haiku-19mar/predictions.npz",
        gamma_scores=gamma_scores
    )
    print(f"Saved predictions ({len(gamma_scores)} scores)")

    print("\n---")
    print("metric: 0.0000")
    print("description: Random Forest on 5 features")

if __name__ == "__main__":
    main()
