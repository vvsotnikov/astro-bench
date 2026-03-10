"""XGBoost gradient boosting on flattened features."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import subprocess
import sys

# Check if xgboost is available, try to import it
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, will try sklearn GradientBoosting")

print(f"Loading data...")

# Load training data
matrices = np.load("data/gamma_train/matrices.npy", mmap_mode="r")
features = np.load("data/gamma_train/features.npy", mmap_mode="r")
labels = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")[:]

# Flatten matrices and concatenate with features
print(f"Flattening matrices...")
n_samples = len(labels)
flattened = matrices[:].reshape(n_samples, -1).astype(np.float32)
X_train = np.concatenate([flattened, features[:].astype(np.float32)], axis=1)

# Convert labels to binary (0=gamma, 1=hadron)
y_train = labels.astype(np.int32)

print(f"X_train shape: {X_train.shape}")
print(f"y_train distribution: {np.bincount(y_train)}")

# Split into train/val
n_train = int(0.8 * len(X_train))
idx = np.arange(len(X_train))
np.random.seed(42)
np.random.shuffle(idx)

train_idx = idx[:n_train]
val_idx = idx[n_train:]

X_tr, X_val = X_train[train_idx], X_train[val_idx]
y_tr, y_val = y_train[train_idx], y_train[val_idx]

print(f"Train: {X_tr.shape}, Val: {X_val.shape}")

# Train with XGBoost if available, else sklearn
if HAS_XGBOOST:
    print(f"\nTraining XGBoost classifier...")
    # Use scale_pos_weight to handle class imbalance
    scale_pos_weight = (y_tr == 1).sum() / (y_tr == 0).sum()

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        tree_method='gpu_hist',
        device='cuda:0',
        verbosity=2,
        eval_metric='logloss'
    )
else:
    print(f"\nTraining sklearn GradientBoosting classifier...")
    model = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        verbose=1
    )

model.fit(X_tr, y_tr)

print(f"\nGenerating test predictions...")
# Load test data
test_matrices = np.load("data/gamma_test/matrices.npy", mmap_mode="r")
test_features = np.load("data/gamma_test/features.npy", mmap_mode="r")
test_labels = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")[:]

n_test = len(test_labels)
test_flattened = test_matrices[:].reshape(n_test, -1).astype(np.float32)
X_test = np.concatenate([test_flattened, test_features[:].astype(np.float32)], axis=1)

# Get probability scores (higher = more likely to be gamma)
# XGBoost returns P(y=1), so we need P(y=0) = 1 - P(y=1)
test_proba = model.predict_proba(X_test)
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
np.savez("submissions/haiku-gamma-mar9-v3/predictions_v2.npz",
         gamma_scores=gamma_scores)

print(f"\n---")
print(f"metric: {surv_75:.4e}")
print(f"description: XGBoost gradient boosting (500 trees, depth=8)")
