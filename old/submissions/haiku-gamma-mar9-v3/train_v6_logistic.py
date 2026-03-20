"""Fast baseline: Logistic Regression on engineered features."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("Loading data...")

# Load training data
matrices = np.load("data/gamma_train/matrices.npy", mmap_mode="r")
features = np.load("data/gamma_train/features.npy", mmap_mode="r")
labels = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")[:]

# Extract and engineer features
E = features[:, 0]
Ze = features[:, 1]
Az = features[:, 2]
Ne = features[:, 3]
Nmu = features[:, 4]

Ne_minus_Nmu = Ne - Nmu
cos_Ze = np.cos(np.radians(Ze))
sin_Ze = np.sin(np.radians(Ze))
cos_Az = np.cos(np.radians(Az))
sin_Az = np.sin(np.radians(Az))
E_norm = (E - E.mean()) / E.std()

# Matrix statistics
n_samples = len(labels)
matrices_flat = matrices[:].reshape(n_samples, -1).astype(np.float32)
mat_sum = matrices_flat.sum(axis=1)
mat_mean = matrices_flat.mean(axis=1)
mat_std = matrices_flat.std(axis=1)
mat_max = matrices_flat.max(axis=1)

# Combine all features
X_train = np.column_stack([
    E, Ze, Az, Ne, Nmu,
    Ne_minus_Nmu,
    cos_Ze, sin_Ze, cos_Az, sin_Az,
    E_norm,
    mat_sum, mat_mean, mat_std, mat_max
])

print(f"X_train shape: {X_train.shape}")
print(f"y_train distribution: {np.bincount(labels)}")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train/val split
n_train = int(0.8 * len(X_train))
idx = np.arange(len(X_train))
np.random.seed(42)
np.random.shuffle(idx)

train_idx = idx[:n_train]
val_idx = idx[n_train:]

X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
y_tr, y_val = labels[train_idx], labels[val_idx]

print(f"Train: {X_tr.shape}, Val: {X_val.shape}")

# Train logistic regression
print(f"Training Logistic Regression...")
model = LogisticRegression(
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced',
    verbose=1
)

model.fit(X_tr, y_tr)

# Validate
val_proba = model.predict_proba(X_val)
gamma_scores_val = val_proba[:, 0]

# Compute survival @ 75% gamma eff on validation
is_gamma_val = y_val == 0
is_hadron_val = y_val == 1
sg_val = np.sort(gamma_scores_val[is_gamma_val])
ng_val = len(sg_val)
thr_val = sg_val[max(0, int(np.floor(ng_val * (1 - 0.75))))]
surv_val = (gamma_scores_val[is_hadron_val] >= thr_val).sum() / is_hadron_val.sum()
print(f"Validation survival @ 75%: {surv_val:.4e}")

print(f"\nGenerating test predictions...")
# Load test data
test_matrices = np.load("data/gamma_test/matrices.npy", mmap_mode="r")
test_features = np.load("data/gamma_test/features.npy", mmap_mode="r")
test_labels = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")[:]

# Engineer test features
E_test = test_features[:, 0]
Ze_test = test_features[:, 1]
Az_test = test_features[:, 2]
Ne_test = test_features[:, 3]
Nmu_test = test_features[:, 4]

Ne_minus_Nmu_test = Ne_test - Nmu_test
cos_Ze_test = np.cos(np.radians(Ze_test))
sin_Ze_test = np.sin(np.radians(Ze_test))
cos_Az_test = np.cos(np.radians(Az_test))
sin_Az_test = np.sin(np.radians(Az_test))
E_norm_test = (E_test - E.mean()) / E.std()

n_test = len(test_labels)
test_matrices_flat = test_matrices[:].reshape(n_test, -1).astype(np.float32)
test_mat_sum = test_matrices_flat.sum(axis=1)
test_mat_mean = test_matrices_flat.mean(axis=1)
test_mat_std = test_matrices_flat.std(axis=1)
test_mat_max = test_matrices_flat.max(axis=1)

X_test = np.column_stack([
    E_test, Ze_test, Az_test, Ne_test, Nmu_test,
    Ne_minus_Nmu_test,
    cos_Ze_test, sin_Ze_test, cos_Az_test, sin_Az_test,
    E_norm_test,
    test_mat_sum, test_mat_mean, test_mat_std, test_mat_max
])

X_test_scaled = scaler.transform(X_test)

# Predict
test_proba = model.predict_proba(X_test_scaled)
gamma_scores = test_proba[:, 0]

print(f"gamma_scores range: [{gamma_scores.min():.4f}, {gamma_scores.max():.4f}]")

# Compute test survival
is_gamma_test = test_labels == 0
is_hadron_test = test_labels == 1
sg_test = np.sort(gamma_scores[is_gamma_test])
ng_test = len(sg_test)
thr_test = sg_test[max(0, int(np.floor(ng_test * (1 - 0.75))))]
surv_test = (gamma_scores[is_hadron_test] >= thr_test).sum() / is_hadron_test.sum()

print(f"Test survival @ 75% gamma eff: {surv_test:.4e}")

# Save predictions
np.savez("submissions/haiku-gamma-mar9-v3/predictions_v6.npz",
         gamma_scores=gamma_scores)

print(f"\n---")
print(f"metric: {surv_test:.4e}")
print(f"description: Logistic Regression with 15 engineered features")
