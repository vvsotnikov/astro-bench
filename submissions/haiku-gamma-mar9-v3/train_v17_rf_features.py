"""RandomForest on engineered features - another architecture family."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

print("Loading data...")

# Load training data
features = np.load("data/gamma_train/features.npy", mmap_mode="r")
labels = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")[:]

# Extract and engineer features
E = features[:, 0]
Ze = features[:, 1]
Az = features[:, 2]
Ne = features[:, 3]
Nmu = features[:, 4]

Ne_minus_Nmu = Ne - Nmu
cos_Ze = np.cos(np.radians(Ze * 180 / np.pi))
sin_Ze = np.sin(np.radians(Ze * 180 / np.pi))
cos_Az = np.cos(np.radians(Az * 180 / np.pi))
sin_Az = np.sin(np.radians(Az * 180 / np.pi))

# Combine all features
X_train = np.column_stack([
    E, Ze, Az, Ne, Nmu,
    Ne_minus_Nmu,
    cos_Ze, sin_Ze, cos_Az, sin_Az
])

print(f"X_train shape: {X_train.shape}")
print(f"y_train distribution: {np.bincount(labels)}")

# Train/val split
n_train = int(0.8 * len(X_train))
idx = np.arange(len(X_train))
np.random.seed(42)
np.random.shuffle(idx)

train_idx = idx[:n_train]
val_idx = idx[n_train:]

X_tr, X_val = X_train[train_idx], X_train[val_idx]
y_tr, y_val = labels[train_idx], labels[val_idx]

print(f"Train: {X_tr.shape}, Val: {X_val.shape}")

# Train RandomForest
print(f"\nTraining RandomForest classifier...")
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    n_jobs=-1,
    verbose=2,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_tr, y_tr)

# Validate
val_proba = rf_model.predict_proba(X_val)
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
test_features = np.load("data/gamma_test/features.npy", mmap_mode="r")
test_labels = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")[:]

# Engineer test features same way
E_test = test_features[:, 0]
Ze_test = test_features[:, 1]
Az_test = test_features[:, 2]
Ne_test = test_features[:, 3]
Nmu_test = test_features[:, 4]

Ne_minus_Nmu_test = Ne_test - Nmu_test
cos_Ze_test = np.cos(np.radians(Ze_test * 180 / np.pi))
sin_Ze_test = np.sin(np.radians(Ze_test * 180 / np.pi))
cos_Az_test = np.cos(np.radians(Az_test * 180 / np.pi))
sin_Az_test = np.sin(np.radians(Az_test * 180 / np.pi))

X_test = np.column_stack([
    E_test, Ze_test, Az_test, Ne_test, Nmu_test,
    Ne_minus_Nmu_test,
    cos_Ze_test, sin_Ze_test, cos_Az_test, sin_Az_test
])

# Predict
test_proba = rf_model.predict_proba(X_test)
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
np.savez("submissions/haiku-gamma-mar9-v3/predictions_v17.npz",
         gamma_scores=gamma_scores)

print(f"\n---")
print(f"metric: {surv_test:.4e}")
print(f"description: RandomForest (500 trees) on 10 engineered features")
