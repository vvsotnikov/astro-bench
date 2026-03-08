"""Baseline: Random Forest on 5 scalar features only (no matrices)."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load data
print("Loading data...")
f_train = np.load("data/composition_train/features.npy", mmap_mode="r")
y_train = np.load("data/composition_train/labels_composition.npy", mmap_mode="r")
f_test = np.load("data/composition_test/features.npy", mmap_mode="r")

# Convert to float32 for sklearn
X_train = np.array(f_train, dtype=np.float32)
y_train = np.array(y_train)
X_test = np.array(f_test, dtype=np.float32)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {np.unique(y_train)}")

# Train
print("Training RandomForest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=30, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Predict
print("Predicting...")
predictions = rf.predict(X_test)

# Save
np.savez("submissions/baselines/predictions.npz", predictions=predictions)
print("Saved predictions.npz")

# Feature importances
for name, imp in zip(["E", "Ze", "Az", "Ne", "Nmu"], rf.feature_importances_):
    print(f"  {name}: {imp:.4f}")
