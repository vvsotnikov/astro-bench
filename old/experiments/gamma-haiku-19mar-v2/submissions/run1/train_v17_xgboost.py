#!/usr/bin/env python3
"""Gamma/hadron v17: XGBoost on engineered features + matrix statistics.

Different paradigm: tree-based boosting instead of neural networks.
Better at capturing non-linear relationships in tabular data.
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os

print("Loading data...")
X_train_mat = np.load("data/gamma_train/matrices.npy", mmap_mode="r")
X_train_feat = np.load("data/gamma_train/features.npy", mmap_mode="r")
y_train = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")

X_test_mat = np.load("data/gamma_test/matrices.npy", mmap_mode="r")
X_test_feat = np.load("data/gamma_test/features.npy", mmap_mode="r")
y_test = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")

print(f"Train: {len(y_train)}, Test: {len(y_test)}")

def engineer_features(matrices, features):
    """Create rich feature set from spatial + scalar features."""
    N = len(features)
    eng_feat = []

    # Original features
    eng_feat.append(features)  # E, Ze, Az, Ne, Nmu

    # Spatial statistics
    e_mat = matrices[:, :, :, 0]  # (N, 16, 16)
    mu_mat = matrices[:, :, :, 1]  # (N, 16, 16)

    e_sum = e_mat.reshape(N, -1).sum(axis=1, keepdims=True)
    e_mean = e_mat.reshape(N, -1).mean(axis=1, keepdims=True)
    e_std = e_mat.reshape(N, -1).std(axis=1, keepdims=True)
    e_max = e_mat.reshape(N, -1).max(axis=1, keepdims=True)
    e_min = e_mat.reshape(N, -1).min(axis=1, keepdims=True)

    mu_sum = mu_mat.reshape(N, -1).sum(axis=1, keepdims=True)
    mu_mean = mu_mat.reshape(N, -1).mean(axis=1, keepdims=True)
    mu_std = mu_mat.reshape(N, -1).std(axis=1, keepdims=True)
    mu_max = mu_mat.reshape(N, -1).max(axis=1, keepdims=True)
    mu_min = mu_mat.reshape(N, -1).min(axis=1, keepdims=True)

    eng_feat.append(np.hstack([e_sum, e_mean, e_std, e_max, e_min,
                                mu_sum, mu_mean, mu_std, mu_max, mu_min]))

    # Ratios
    E = features[:, 0:1]
    Ze = features[:, 1:2]
    Az = features[:, 2:3]
    Ne = features[:, 3:4]
    Nmu = features[:, 4:5]

    Ne_Nmu = Ne - Nmu
    Ne_per_E = Ne - E
    Nmu_per_E = Nmu - E

    eng_feat.append(np.hstack([Ne_Nmu, Ne_per_E, Nmu_per_E]))

    # Angle encodings
    Ze_norm = Ze / 30.0
    Az_norm = Az / 360.0
    sin_ze = np.sin(np.radians(Ze))
    cos_ze = np.cos(np.radians(Ze))
    sin_az = np.sin(np.radians(Az))
    cos_az = np.cos(np.radians(Az))

    eng_feat.append(np.hstack([Ze_norm, Az_norm, sin_ze, cos_ze, sin_az, cos_az]))

    # Sparsity
    e_sparsity = (e_mat.reshape(N, -1) == 0).sum(axis=1, keepdims=True) / 256.0
    mu_sparsity = (mu_mat.reshape(N, -1) == 0).sum(axis=1, keepdims=True) / 256.0

    eng_feat.append(np.hstack([e_sparsity, mu_sparsity]))

    # Spatial distribution (center of mass like features)
    e_h_sum = e_mat.sum(axis=2)  # (N, 16)
    e_w_sum = e_mat.sum(axis=1)  # (N, 16)
    mu_h_sum = mu_mat.sum(axis=2)
    mu_w_sum = mu_mat.sum(axis=1)

    e_h_weight = np.arange(16)
    e_w_weight = np.arange(16)
    e_h_center = (e_h_sum * e_h_weight).sum(axis=1, keepdims=True) / (e_h_sum.sum(axis=1, keepdims=True) + 1e-8)
    e_w_center = (e_w_sum * e_w_weight).sum(axis=1, keepdims=True) / (e_w_sum.sum(axis=1, keepdims=True) + 1e-8)
    mu_h_center = (mu_h_sum * e_h_weight).sum(axis=1, keepdims=True) / (mu_h_sum.sum(axis=1, keepdims=True) + 1e-8)
    mu_w_center = (mu_w_sum * e_w_weight).sum(axis=1, keepdims=True) / (mu_w_sum.sum(axis=1, keepdims=True) + 1e-8)

    eng_feat.append(np.hstack([e_h_center, e_w_center, mu_h_center, mu_w_center]))

    return np.hstack(eng_feat).astype(np.float32)

print("Engineering features...")
X_train_eng = engineer_features(X_train_mat, X_train_feat)
X_test_eng = engineer_features(X_test_mat, X_test_feat)

print(f"Engineered feature shape: {X_train_eng.shape}")

# Split train into train/val
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_eng, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Train: {len(X_tr)}, Val: {len(X_val)}")

# Train XGBoost
print("Training XGBoost...")
# Scale class weights
n_gamma = (y_tr == 0).sum()
n_hadron = (y_tr == 1).sum()
w_gamma = len(y_tr) / (2 * n_gamma)
w_hadron = len(y_tr) / (2 * n_hadron)

sample_weight = np.where(y_tr == 0, w_gamma, w_hadron)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    scale_pos_weight=w_hadron / w_gamma,  # Class imbalance
)

xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=10,
)

# Predict on test set
print("Predicting...")
gamma_scores = xgb_model.predict_proba(X_test_eng)[:, 0]  # Probability of gamma

# Save predictions
os.makedirs("submissions/run1", exist_ok=True)
np.savez("submissions/run1/predictions_v17.npz", gamma_scores=gamma_scores)
np.save("submissions/run1/probs_v17.npy", gamma_scores)

# Compute survival @ 75% gamma efficiency
is_gamma = y_test == 0
is_hadron = y_test == 1
sg = np.sort(gamma_scores[is_gamma])
ng = len(sg)
thr_75 = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
n_hadron_surviving = (gamma_scores[is_hadron] >= thr_75).sum()
survival_75 = n_hadron_surviving / is_hadron.sum() if is_hadron.sum() > 0 else 1.0

print(f"Survival rate @ 75% gamma efficiency: {survival_75:.2e}")
print(f"Saved predictions to submissions/run1/predictions_v17.npz")

# Feature importance
print("\nTop 15 feature importances:")
importance = xgb_model.feature_importances_
for i, imp in sorted(enumerate(importance), key=lambda x: -x[1])[:15]:
    print(f"  Feature {i}: {imp:.4f}")
