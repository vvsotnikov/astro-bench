"""XGBoost on flattened features."""

import numpy as np
import sys

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost not available, trying to install...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost as xgb

# Load data
print("Loading data...")
train_matrices = np.load("data/gamma_train/matrices.npy", mmap_mode="r")
train_features = np.load("data/gamma_train/features.npy", mmap_mode="r")
train_labels = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")

test_matrices = np.load("data/gamma_test/matrices.npy", mmap_mode="r")
test_features = np.load("data/gamma_test/features.npy", mmap_mode="r")
test_labels = np.load("data/gamma_test/labels_gamma.npy")[:]

# Flatten and concatenate
train_matrices_flat = train_matrices.reshape(len(train_matrices), -1).astype(np.float32)
train_X = np.concatenate([train_matrices_flat, train_features.astype(np.float32)], axis=1)

test_matrices_flat = test_matrices.reshape(len(test_matrices), -1).astype(np.float32)
test_X = np.concatenate([test_matrices_flat, test_features.astype(np.float32)], axis=1)

# Subsample for speed
print(f"Full training size: {len(train_X)}")
rng = np.random.default_rng(42)
subsample_size = min(500_000, len(train_X))
subsample_idx = rng.choice(len(train_X), size=subsample_size, replace=False)

X_train_sub = train_X[subsample_idx]
y_train_sub = (train_labels[subsample_idx] == 0).astype(int)  # 1 for gamma, 0 for hadron

print(f"Subsampled to: {len(X_train_sub)}")
print(f"Gamma fraction: {y_train_sub.sum() / len(y_train_sub):.3%}")

# Normalize
mean = X_train_sub.mean(axis=0)
std = X_train_sub.std(axis=0)
std[std == 0] = 1.0

X_train_norm = (X_train_sub - mean) / (std + 1e-8)
X_test_norm = (test_X - mean) / (std + 1e-8)

# Train XGBoost
print("\nTraining XGBoost...")
dtrain = xgb.DMatrix(X_train_norm, label=y_train_sub)
dtest = xgb.DMatrix(X_test_norm)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'gpu_hist',
    'device': 'cuda'
}

evals = [(dtrain, 'train')]
evals_result = {}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=evals,
    evals_result=evals_result,
    early_stopping_rounds=50,
    verbose_eval=50
)

# Predict
test_probs = model.predict(dtest)

# Evaluate
is_gamma_test = test_labels == 0
is_hadron_test = test_labels == 1

def compute_survival_75(scores):
    sg = np.sort(scores[is_gamma_test])
    ng = len(sg)
    thr = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
    n_surv = (scores[is_hadron_test] >= thr).sum()
    return n_surv / is_hadron_test.sum()

test_survival = compute_survival_75(test_probs)
print(f"\nTest survival @ 75% gamma eff: {test_survival:.4e}")

# Save
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v21.npz",
         gamma_scores=test_probs)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: XGBoost on flattened 517 features, subsampled to 500k")
