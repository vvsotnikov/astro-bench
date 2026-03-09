"""Baseline: RandomForest on 5 scalar features."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load data
f_train = np.load('data/composition_train/features.npy', mmap_mode='r')
y_train = np.load('data/composition_train/labels_composition.npy', mmap_mode='r')
f_test = np.load('data/composition_test/features.npy', mmap_mode='r')

# Convert to arrays
f_train = np.array(f_train, dtype=np.float32)
y_train = np.array(y_train)
f_test = np.array(f_test, dtype=np.float32)

print(f"Train: {f_train.shape}, Test: {f_test.shape}")

# Train RF
print("Training RandomForest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
rf.fit(f_train, y_train)

# Predict
preds = rf.predict(f_test)
print(f"Predictions shape: {preds.shape}")
print(f"Prediction distribution: {dict(zip(*np.unique(preds, return_counts=True)))}")

# Save
np.savez('submissions/haiku-mar8/predictions.npz', predictions=preds.astype(int))
print("Saved predictions.npz")
