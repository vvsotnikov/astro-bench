"""LightGBM for fast gradient boosting."""

import numpy as np
import lightgbm as lgb

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

# Train LightGBM with validation set
print("\nTraining LightGBM...")
# Split subsampled training for early stopping
n_sub = len(X_train_norm)
n_train_sub = int(0.8 * n_sub)
train_idx = np.arange(n_train_sub)
val_idx = np.arange(n_train_sub, n_sub)

X_train_lgb = X_train_norm[train_idx]
y_train_lgb = y_train_sub[train_idx]
X_val_lgb = X_train_norm[val_idx]
y_val_lgb = y_train_sub[val_idx]

train_data = lgb.Dataset(X_train_lgb, label=y_train_lgb)
val_data = lgb.Dataset(X_val_lgb, label=y_val_lgb, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 8,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(-1)
    ]
)

# Predict
print("\nPredicting...")
test_probs = model.predict(X_test_norm)

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
print(f"Test survival @ 75% gamma eff: {test_survival:.4e}")

# Save
np.savez("submissions/haiku-gamma-mar9-v2/predictions_v25.npz",
         gamma_scores=test_probs)

print(f"\n---")
print(f"metric: {test_survival:.4e}")
print(f"description: LightGBM gradient boosting on flattened features")
