"""Gradient Boosting on engineered features only (no CNN)."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import torch

print(f"Loading data...")

# Load training data
matrices = np.load("data/gamma_train/matrices.npy", mmap_mode="r")
features = np.load("data/gamma_train/features.npy", mmap_mode="r")
labels = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")[:]

# Engineered features from v9
E = features[:, 0].astype(np.float32)
Ze = features[:, 1].astype(np.float32)
Az = features[:, 2].astype(np.float32)
Ne = features[:, 3].astype(np.float32)
Nmu = features[:, 4].astype(np.float32)

Ne_minus_Nmu = Ne - Nmu
cos_Ze = np.cos(np.deg2rad(Ze))
sin_Ze = np.sin(np.deg2rad(Ze))

X_train = np.column_stack([E, Ze, Az, Ne, Nmu, Ne_minus_Nmu, cos_Ze, sin_Ze])
y_train = labels.astype(np.int32)

print(f"X_train shape: {X_train.shape}")
print(f"y_train distribution: {np.bincount(y_train)}")

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Split into train/val
n_train = int(0.8 * len(X_train))
idx = np.arange(len(X_train))
rng = np.random.default_rng(42)
idx = rng.permutation(idx)

train_idx = idx[:n_train]
val_idx = idx[n_train:]

X_tr = X_train_scaled[train_idx]
X_val = X_train_scaled[val_idx]
y_tr = y_train[train_idx]
y_val = y_train[val_idx]

print(f"Train: {X_tr.shape}, Val: {X_val.shape}")

# Train Gradient Boosting
print(f"\nTraining GradientBoosting classifier...")
scale_pos_weight = (y_tr == 1).sum() / (y_tr == 0).sum()

model = GradientBoostingClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    verbose=1
)

model.fit(X_tr, y_tr)

print(f"\nGenerating test predictions...")

# Load test data
test_features = np.load("data/gamma_test/features.npy", mmap_mode="r")
test_labels = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")[:]

# Engineered features for test
E_test = test_features[:, 0].astype(np.float32)
Ze_test = test_features[:, 1].astype(np.float32)
Az_test = test_features[:, 2].astype(np.float32)
Ne_test = test_features[:, 3].astype(np.float32)
Nmu_test = test_features[:, 4].astype(np.float32)

Ne_minus_Nmu_test = Ne_test - Nmu_test
cos_Ze_test = np.cos(np.deg2rad(Ze_test))
sin_Ze_test = np.sin(np.deg2rad(Ze_test))

X_test = np.column_stack([E_test, Ze_test, Az_test, Ne_test, Nmu_test, Ne_minus_Nmu_test, cos_Ze_test, sin_Ze_test])
X_test_scaled = scaler.transform(X_test)

# Get probability scores (higher = more likely to be gamma)
test_proba = model.predict_proba(X_test_scaled)
gamma_scores = test_proba[:, 0]  # P(gamma)

print(f"gamma_scores shape: {gamma_scores.shape}")
print(f"gamma_scores range: [{gamma_scores.min():.4f}, {gamma_scores.max():.4f}]")

# Compute survival rate @ 75% gamma efficiency
is_gamma = test_labels == 0
is_hadron = test_labels == 1
sg = np.sort(gamma_scores[is_gamma])
ng = len(sg)
thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
n_hadron_surviving = (gamma_scores[is_hadron] >= thr).sum()
surv_75 = n_hadron_surviving / is_hadron.sum()

print(f"Hadronic survival @ 75% gamma eff: {surv_75:.4e}")

# Save predictions
np.savez("submissions/haiku-gamma-mar9-v3/predictions_v24.npz",
         gamma_scores=gamma_scores)

print(f"\n---")
print(f"metric: {surv_75:.4e}")
print(f"description: GradientBoosting on engineered features (500 trees, depth=6)")
