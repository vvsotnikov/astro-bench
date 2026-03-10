"""SVM with RBF kernel on engineered features + normalized matrices."""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

print("Loading data...")

# Load training data
matrices = np.load("data/gamma_train/matrices.npy", mmap_mode="r")
features = np.load("data/gamma_train/features.npy", mmap_mode="r")
labels = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")[:]

# Extract key features and engineer new ones
# Features: [E, Ze, Az, Ne, Nmu]
E = features[:, 0]  # log10(energy)
Ze = features[:, 1]  # zenith
Az = features[:, 2]  # azimuth
Ne = features[:, 3]  # log10(electrons)
Nmu = features[:, 4]  # log10(muons)

print(f"Feature ranges:")
print(f"  E: [{E.min():.2f}, {E.max():.2f}]")
print(f"  Ze: [{Ze.min():.2f}, {Ze.max():.2f}]")
print(f"  Az: [{Az.min():.2f}, {Az.max():.2f}]")
print(f"  Ne: [{Ne.min():.2f}, {Ne.max():.2f}]")
print(f"  Nmu: [{Nmu.min():.2f}, {Nmu.max():.2f}]")

# Engineer derived features
Ne_minus_Nmu = Ne - Nmu  # Strong discriminant
Ne_div_Nmu = Ne - Nmu  # Log ratio (more stable than division)
cos_Ze = np.cos(np.radians(Ze))
sin_Ze = np.sin(np.radians(Ze))
cos_Az = np.cos(np.radians(Az))
sin_Az = np.sin(np.radians(Az))
E_norm = (E - E.mean()) / E.std()

print(f"\nEngineered features:")
print(f"  Ne-Nmu: [{Ne_minus_Nmu.min():.2f}, {Ne_minus_Nmu.max():.2f}]")

# Flatten matrices
n_samples = len(labels)
flattened = matrices[:].reshape(n_samples, -1).astype(np.float32)

# Combine all features
X_train = np.column_stack([
    flattened,
    E, Ze, Az, Ne, Nmu,
    Ne_minus_Nmu,
    cos_Ze, sin_Ze, cos_Az, sin_Az,
    E_norm
])

print(f"\nX_train shape: {X_train.shape}")
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

# Train SVM with RBF kernel
print(f"\nTraining SVM with RBF kernel...")
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    verbose=2,
    class_weight='balanced'
)

svm_model.fit(X_tr, y_tr)

# Validate
val_proba = svm_model.predict_proba(X_val)
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

# Engineer test features same way
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
test_flattened = test_matrices[:].reshape(n_test, -1).astype(np.float32)

X_test = np.column_stack([
    test_flattened,
    E_test, Ze_test, Az_test, Ne_test, Nmu_test,
    Ne_minus_Nmu_test,
    cos_Ze_test, sin_Ze_test, cos_Az_test, sin_Az_test,
    E_norm_test
])

X_test_scaled = scaler.transform(X_test)

# Predict
test_proba = svm_model.predict_proba(X_test_scaled)
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
np.savez("submissions/haiku-gamma-mar9-v3/predictions_v5.npz",
         gamma_scores=gamma_scores)

print(f"\n---")
print(f"metric: {surv_test:.4e}")
print(f"description: SVM RBF kernel with engineered features (Ne-Nmu, angles, energy)")
