"""Gamma/hadron v1: Logistic regression on 5 scalar features.

Simple baseline to understand data separability.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def main():
    # Load data
    print("Loading data...")
    X_train = np.load("data/gamma_train/features.npy", mmap_mode="r").astype(np.float32)
    y_train = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")

    X_test = np.load("data/gamma_test/features.npy", mmap_mode="r").astype(np.float32)
    y_test = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")

    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")
    print(f"Train class distribution: gamma={sum(y_train==0)}, hadron={sum(y_train==1)}")
    print(f"Test class distribution: gamma={sum(y_test==0)}, hadron={sum(y_test==1)}")

    # Normalize features
    print("\nNormalizing...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression with class weights (for imbalanced data)
    print("\nTraining logistic regression...")
    n_gamma = sum(y_train == 0)
    n_hadron = sum(y_train == 1)
    w_gamma = len(y_train) / (2 * n_gamma)
    w_hadron = len(y_train) / (2 * n_hadron)
    class_weights = {0: w_gamma, 1: w_hadron}
    print(f"Class weights: gamma={w_gamma:.2f}, hadron={w_hadron:.2f}")

    model = LogisticRegression(
        max_iter=10000,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)

    # Get scores: probability of class 0 (gamma)
    gamma_scores = model.predict_proba(X_test_scaled)[:, 0]

    print(f"\nScores stats: min={gamma_scores.min():.4f}, max={gamma_scores.max():.4f}, mean={gamma_scores.mean():.4f}")

    # Save predictions
    np.savez(
        "submissions/gamma-haiku-19mar/predictions.npz",
        gamma_scores=gamma_scores
    )
    print(f"Saved predictions ({len(gamma_scores)} scores)")

    print("\n---")
    print("metric: 0.0000")
    print("description: Logistic regression on 5 features")

if __name__ == "__main__":
    main()
