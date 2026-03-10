"""Proper stacking: Base learners + meta-learner using cross-validation."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

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

X_train = np.column_stack([
    E, Ze, Az, Ne, Nmu,
    Ne_minus_Nmu,
    cos_Ze, sin_Ze
])

print(f"X_train shape: {X_train.shape}")
print(f"y_train distribution: {np.bincount(labels)}")

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# 5-fold cross-validation for stacking
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Base learners
base_learners = {
    'rf': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    'gb': GradientBoostingClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42),
}

print(f"\nGenerating meta-features via {len(base_learners)} base learners...")

# Generate meta-features
meta_features = np.zeros((len(X_scaled), len(base_learners)))

for base_idx, (name, learner) in enumerate(base_learners.items()):
    print(f"  {name}...")
    fold_preds = np.zeros(len(X_scaled))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr = labels[train_idx]

        learner_clone = type(learner)(**learner.get_params())
        learner_clone.fit(X_tr, y_tr)
        fold_preds[val_idx] = learner_clone.predict_proba(X_val)[:, 0]

    meta_features[:, base_idx] = fold_preds

# Train meta-learner on meta-features
print(f"\nTraining meta-learner...")
meta_learner = LogisticRegression(max_iter=1000, class_weight='balanced')
meta_learner.fit(meta_features, labels)

# Validation on meta-features
val_preds = meta_learner.predict_proba(meta_features)[:, 0]

is_gamma_val = labels == 0
is_hadron_val = labels == 1
sg_val = np.sort(val_preds[is_gamma_val])
ng_val = len(sg_val)
thr_val = sg_val[max(0, int(np.floor(ng_val * (1 - 0.75))))]
surv_val = (val_preds[is_hadron_val] >= thr_val).sum() / is_hadron_val.sum()
print(f"Training survival @ 75%: {surv_val:.4e}")

# Generate test meta-features
print(f"\nGenerating test predictions...")
test_features = np.load("data/gamma_test/features.npy", mmap_mode="r")
test_labels = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")[:]

E_test = test_features[:, 0]
Ze_test = test_features[:, 1]
Az_test = test_features[:, 2]
Ne_test = test_features[:, 3]
Nmu_test = test_features[:, 4]

Ne_minus_Nmu_test = Ne_test - Nmu_test
cos_Ze_test = np.cos(np.radians(Ze_test * 180 / np.pi))
sin_Ze_test = np.sin(np.radians(Ze_test * 180 / np.pi))

X_test = np.column_stack([
    E_test, Ze_test, Az_test, Ne_test, Nmu_test,
    Ne_minus_Nmu_test,
    cos_Ze_test, sin_Ze_test
])

X_test_scaled = scaler.transform(X_test)

# Generate test meta-features
test_meta = np.zeros((len(X_test_scaled), len(base_learners)))

for base_idx, (name, learner) in enumerate(base_learners.items()):
    learner.fit(X_scaled, labels)
    test_meta[:, base_idx] = learner.predict_proba(X_test_scaled)[:, 0]

# Final predictions via meta-learner
gamma_scores = meta_learner.predict_proba(test_meta)[:, 0]

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
np.savez("submissions/haiku-gamma-mar9-v3/predictions_v22.npz",
         gamma_scores=gamma_scores)

print(f"\n---")
print(f"metric: {surv_test:.4e}")
print(f"description: Stacking with RF + GB base learners and logistic regression meta-learner")
