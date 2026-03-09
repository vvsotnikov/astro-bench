#!/usr/bin/env python3
"""Gamma/hadron classifier using Random Forest on engineered features.

Strategy: Follow the ICRC 2021 approach more closely:
1. Use scalar features + basic matrix statistics
2. RandomForest regressor to output continuous score
3. Optimize threshold for 99% gamma efficiency
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

def load_data(split):
    """Load gamma/hadron data."""
    matrices = np.load(f"/home/vladimir/cursor_projects/astro-agents/data/gamma_{split}/matrices.npy", mmap_mode="r")
    features = np.load(f"/home/vladimir/cursor_projects/astro-agents/data/gamma_{split}/features.npy", mmap_mode="r")
    labels = np.load(f"/home/vladimir/cursor_projects/astro-agents/data/gamma_{split}/labels_gamma.npy", mmap_mode="r")

    return np.array(matrices), np.array(features), np.array(labels)

def extract_features(matrices, features):
    """Extract hand-engineered features from matrices + scalar features.

    Input:
    - matrices: (N, 16, 16, 2) — electron/photon and muon density
    - features: (N, 5) — E, Ze, Az, Ne, Nmu

    Output: (N, ~25) — combined feature vector
    """
    N = len(matrices)

    # Scalar features: 5
    X = np.column_stack([features])  # (N, 5)

    # Matrix statistics
    # Channel 0: electron/photon density
    electron_chan = matrices[:, :, :, 0]
    # Channel 1: muon density
    muon_chan = matrices[:, :, :, 1]

    # Electron channel stats
    e_sum = electron_chan.reshape(N, -1).sum(axis=1)
    e_max = electron_chan.reshape(N, -1).max(axis=1)
    e_std = electron_chan.reshape(N, -1).std(axis=1)

    # Muon channel stats (KEY for gamma rays - they have no muons)
    mu_sum = muon_chan.reshape(N, -1).sum(axis=1)
    mu_max = muon_chan.reshape(N, -1).max(axis=1)
    mu_std = muon_chan.reshape(N, -1).std(axis=1)
    mu_nzero = (muon_chan.reshape(N, -1) > 0).sum(axis=1)

    # Ratios (muon suppression is key physics)
    e_mu_ratio = np.divide(e_sum, mu_sum + 1e-6)

    # Total energy in detector
    total_sum = e_sum + mu_sum

    # Fraction of energy in muons
    mu_frac = np.divide(mu_sum, total_sum + 1e-6)

    # Combine all
    engineered = np.column_stack([
        e_sum, e_max, e_std,
        mu_sum, mu_max, mu_std, mu_nzero,
        e_mu_ratio, mu_frac,
        total_sum
    ])

    X = np.column_stack([X, engineered])

    return X

def main():
    print("=" * 80)
    print("GAMMA/HADRON CLASSIFIER - RANDOM FOREST")
    print("=" * 80)

    print("\nLoading training data...")
    X_train_mat, X_train_feat, y_train = load_data("train")
    print(f"  Train: {len(y_train)} samples")

    print("Loading test data...")
    X_test_mat, X_test_feat, y_test = load_data("test")
    print(f"  Test: {len(y_test)} samples")

    print("\nExtracting engineered features...")
    X_train = extract_features(X_train_mat, X_train_feat)
    X_test = extract_features(X_test_mat, X_test_feat)
    print(f"  Feature dimension: {X_train.shape[1]}")

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert labels to binary for regression: 0=gamma, 1=hadron
    # We want gamma to be positive, hadron to be negative
    y_train_binary = 1 - y_train  # 1 for gamma (0->1), 0 for hadron (1->0)

    print("\nTraining Random Forest regressor...")
    print(f"  Gamma samples: {(y_train==0).sum()}")
    print(f"  Hadron samples: {(y_train==1).sum()}")

    # Use sample weights to handle imbalance
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == 0] = 10.45  # Upweight gammas
    sample_weights[y_train == 1] = 0.53

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    rf.fit(X_train, y_train_binary, sample_weight=sample_weights)

    print("\nEvaluating on test set...")
    scores = rf.predict(X_test)

    # Scores should be in ~[0, 1] range (gamma-like)
    print(f"  Score min: {scores.min():.4f}, max: {scores.max():.4f}, mean: {scores.mean():.4f}")

    # Compute survival at 99% gamma efficiency
    is_gamma = y_test == 0
    is_hadron = y_test == 1

    sg = np.sort(scores[is_gamma])
    ng = len(sg)
    idx_99 = max(0, int(np.floor(ng * (1 - 0.99))))
    thr_99 = sg[idx_99]

    n_hadron_surviving = (scores[is_hadron] >= thr_99).sum()
    survival_99 = n_hadron_surviving / is_hadron.sum()

    print(f"\nSurvival at 99% gamma efficiency:")
    print(f"  Threshold: {thr_99:.4f}")
    print(f"  Survival rate: {survival_99:.2e}")
    print(f"  Hadrons surviving: {n_hadron_surviving} / {is_hadron.sum()}")

    # Also check other efficiency points
    for eff in [50, 90, 95]:
        idx = max(0, int(np.floor(ng * (1 - eff/100))))
        thr = sg[idx]
        n_surv = (scores[is_hadron] >= thr).sum()
        surv = n_surv / is_hadron.sum()
        print(f"  Survival at {eff}% gamma eff: {surv:.2e}")

    # Save predictions
    os.makedirs("/home/vladimir/cursor_projects/astro-agents/submissions/haiku-gamma-mar8", exist_ok=True)
    np.savez(
        "/home/vladimir/cursor_projects/astro-agents/submissions/haiku-gamma-mar8/predictions_rf.npz",
        gamma_scores=scores,
    )
    print(f"\nSaved predictions to predictions_rf.npz ({len(scores)} scores)")

if __name__ == "__main__":
    main()
