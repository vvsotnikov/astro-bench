"""Anomaly detection approach: Isolation Forest treating gammas as outliers."""

import numpy as np
from sklearn.ensemble import IsolationForest
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
Ne_ratio = Ne / (Nmu + 1e-6)
cos_Ze = np.cos(np.radians(Ze * 180 / np.pi))
sin_Ze = np.sin(np.radians(Ze * 180 / np.pi))

# Matrix statistics
n_samples = len(labels)
matrices_flat = matrices[:].reshape(n_samples, -1).astype(np.float32)
mat_sum = matrices_flat.sum(axis=1)
mat_mean_val = matrices_flat.mean(axis=1)
mat_std_val = matrices_flat.std(axis=1)
mat_max = matrices_flat.max(axis=1)

# Combine all features
X_train = np.column_stack([
    E, Ze, Az, Ne, Nmu,
    Ne_minus_Nmu,
    Ne_ratio,
    cos_Ze, sin_Ze,
    mat_sum, mat_mean_val, mat_std_val, mat_max
])

print(f"X_train shape: {X_train.shape}")
print(f"y_train distribution: {np.bincount(labels)}")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Isolation Forest
# Gammas are anomalies (1.3% of dataset)
contamination = labels[labels == 0].sum() / len(labels)
print(f"Contamination (gamma fraction): {contamination:.4f}")

print(f"\nTraining Isolation Forest...")
iso_forest = IsolationForest(
    contamination=contamination,
    n_estimators=500,
    max_samples='auto',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

iso_forest.fit(X_train_scaled)

# Get anomaly scores (negative distance to separating hyperplane)
# Higher score = more anomalous (gamma-like)
train_scores = -iso_forest.score_samples(X_train_scaled)
train_scores = (train_scores - train_scores.min()) / (train_scores.max() - train_scores.min())

# Validation on training set
is_gamma_val = labels == 0
is_hadron_val = labels == 1
sg_val = np.sort(train_scores[is_gamma_val])
ng_val = len(sg_val)
thr_val = sg_val[max(0, int(np.floor(ng_val * (1 - 0.75))))]
surv_val = (train_scores[is_hadron_val] >= thr_val).sum() / is_hadron_val.sum()
print(f"Training survival @ 75%: {surv_val:.4e}")

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
Ne_ratio_test = Ne_test / (Nmu_test + 1e-6)
cos_Ze_test = np.cos(np.radians(Ze_test * 180 / np.pi))
sin_Ze_test = np.sin(np.radians(Ze_test * 180 / np.pi))

n_test = len(test_labels)
test_matrices_flat = test_matrices[:].reshape(n_test, -1).astype(np.float32)
test_mat_sum = test_matrices_flat.sum(axis=1)
test_mat_mean = test_matrices_flat.mean(axis=1)
test_mat_std = test_matrices_flat.std(axis=1)
test_mat_max = test_matrices_flat.max(axis=1)

X_test = np.column_stack([
    E_test, Ze_test, Az_test, Ne_test, Nmu_test,
    Ne_minus_Nmu_test,
    Ne_ratio_test,
    cos_Ze_test, sin_Ze_test,
    test_mat_sum, test_mat_mean, test_mat_std, test_mat_max
])

X_test_scaled = scaler.transform(X_test)

# Get test anomaly scores (higher = more gamma-like)
gamma_scores = -iso_forest.score_samples(X_test_scaled)
gamma_scores = (gamma_scores - gamma_scores.min()) / (gamma_scores.max() - gamma_scores.min())

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
np.savez("submissions/haiku-gamma-mar9-v3/predictions_v21.npz",
         gamma_scores=gamma_scores)

print(f"\n---")
print(f"metric: {surv_test:.4e}")
print(f"description: Isolation Forest anomaly detection (gammas as outliers)")
