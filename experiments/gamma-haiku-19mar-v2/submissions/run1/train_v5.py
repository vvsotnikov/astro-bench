#!/usr/bin/env python3
"""Gamma/hadron v5: Random Forest on engineered features.

Strategy: strong baseline using feature engineering.
- Original 5 features
- Log transforms, ratios, energy normalization
- Physics-informed: Ne/Nmu ratio, angle encodings
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

# Load data
print("Loading data...")
X_train_mat = np.load("data/gamma_train/matrices.npy", mmap_mode="r")
X_train_feat = np.load("data/gamma_train/features.npy", mmap_mode="r")
y_train = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")

X_test_mat = np.load("data/gamma_test/matrices.npy", mmap_mode="r")
X_test_feat = np.load("data/gamma_test/features.npy", mmap_mode="r")
y_test = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")

print(f"Train: {len(y_train)}, Test: {len(y_test)}")

def engineer_features(matrices, features):
    """Create engineered features from raw data."""
    # matrices: (N, 16, 16, 2)
    # features: (N, 5) = [E, Ze, Az, Ne, Nmu]
    N = len(features)

    # Start with scalar features
    eng_feat = features.copy()  # (N, 5)

    # Add spatial features from matrices
    # Sum over each channel
    e_sum = matrices[:, :, :, 0].reshape(N, -1).sum(axis=1)  # electron channel sum
    mu_sum = matrices[:, :, :, 1].reshape(N, -1).sum(axis=1)  # muon channel sum

    # Add to features
    eng_feat = np.column_stack([eng_feat, e_sum, mu_sum])  # (N, 7)

    # E, Ze, Az, Ne, Nmu, e_sum, mu_sum
    E = features[:, 0]
    Ze = features[:, 1]
    Az = features[:, 2]
    Ne = features[:, 3]
    Nmu = features[:, 4]

    # Physics-motivated features
    # Ratios
    Ne_Nmu_ratio = Ne - Nmu  # log ratio
    eng_feat = np.column_stack([eng_feat, Ne_Nmu_ratio])  # (N, 8)

    # Energy-normalized quantities
    Ne_per_E = Ne - E
    Nmu_per_E = Nmu - E
    eng_feat = np.column_stack([eng_feat, Ne_per_E, Nmu_per_E])  # (N, 10)

    # Angle encodings
    Ze_norm = Ze / 30.0
    Az_norm = Az / 360.0
    sin_ze = np.sin(np.radians(Ze))
    cos_ze = np.cos(np.radians(Ze))
    sin_az = np.sin(np.radians(Az))
    cos_az = np.cos(np.radians(Az))
    eng_feat = np.column_stack([eng_feat, Ze_norm, Az_norm, sin_ze, cos_ze, sin_az, cos_az])  # (N, 16)

    # Matrix gradient/structure features
    e_max = matrices[:, :, :, 0].reshape(N, -1).max(axis=1)
    mu_max = matrices[:, :, :, 1].reshape(N, -1).max(axis=1)
    e_mean = matrices[:, :, :, 0].reshape(N, -1).mean(axis=1)
    mu_mean = matrices[:, :, :, 1].reshape(N, -1).mean(axis=1)

    eng_feat = np.column_stack([eng_feat, e_max, mu_max, e_mean, mu_mean])  # (N, 20)

    # Sparsity (% zeros)
    e_sparsity = (matrices[:, :, :, 0].reshape(N, -1) == 0).sum(axis=1) / 256.0
    mu_sparsity = (matrices[:, :, :, 1].reshape(N, -1) == 0).sum(axis=1) / 256.0
    eng_feat = np.column_stack([eng_feat, e_sparsity, mu_sparsity])  # (N, 22)

    return eng_feat.astype(np.float32)

print("Engineering features...")
X_train_eng = engineer_features(X_train_mat, X_train_feat)
X_test_eng = engineer_features(X_test_mat, X_test_feat)

print(f"Engineered feature shape: {X_train_eng.shape}")

# Train random forest
print("Training RandomForest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
)
rf.fit(X_train_eng, y_train)

# Predict on test set
print("Predicting...")
y_pred_proba = rf.predict_proba(X_test_eng)
gamma_scores = y_pred_proba[:, 0]  # probability of gamma (class 0)

# Save predictions
os.makedirs("submissions/run1", exist_ok=True)
np.savez("submissions/run1/predictions_v5.npz", gamma_scores=gamma_scores)
np.save("submissions/run1/probs_v5.npy", gamma_scores)

# Compute survival @ 75% gamma efficiency for reference
is_gamma = y_test == 0
is_hadron = y_test == 1
sg = np.sort(gamma_scores[is_gamma])
ng = len(sg)
thr_75 = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
n_hadron_surviving = (gamma_scores[is_hadron] >= thr_75).sum()
survival_75 = n_hadron_surviving / is_hadron.sum() if is_hadron.sum() > 0 else 1.0

print(f"Survival rate @ 75% gamma efficiency: {survival_75:.2e}")
print(f"Saved predictions to submissions/run1/predictions_v5.npz")

# Feature importance
print("\nTop 10 feature importances:")
importance = rf.feature_importances_
feature_names = ["E", "Ze", "Az", "Ne", "Nmu", "e_sum", "mu_sum", "Ne_Nmu_ratio",
                 "Ne_per_E", "Nmu_per_E", "Ze_norm", "Az_norm", "sin_ze", "cos_ze",
                 "sin_az", "cos_az", "e_max", "mu_max", "e_mean", "mu_mean", "e_sparsity", "mu_sparsity"]
for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1])[:10]:
    print(f"  {name}: {imp:.4f}")
