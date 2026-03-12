"""v6: RandomForest on engineered features.

Completely different approach - tree-based instead of neural networks.
Use scikit-learn RandomForest with comprehensive feature engineering.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


print("Loading data...")
train_matrices = np.load("data/composition_train/matrices.npy", mmap_mode="r")
train_features = np.load("data/composition_train/features.npy", mmap_mode="r")[:]
train_labels = np.load("data/composition_train/labels_composition.npy")[:]

test_matrices = np.load("data/composition_test/matrices.npy", mmap_mode="r")
test_features = np.load("data/composition_test/features.npy", mmap_mode="r")[:]
test_labels = np.load("data/composition_test/labels_composition.npy")[:]

print(f"Train: {len(train_labels)} samples")
print(f"Test: {len(test_labels)} samples")

# Engineer features from raw data
def engineer_features(matrices, features):
    """Create rich feature set from matrices and raw features."""
    E, Ze, Az, Ne, Nmu = features[:, 0], features[:, 1], features[:, 2], features[:, 3], features[:, 4]

    # Extract spatial statistics from matrices
    mat_ch0 = matrices[:, :, :, 0]  # electron channel
    mat_ch1 = matrices[:, :, :, 1]  # muon channel

    X = []
    for i in range(len(features)):
        feats = [
            E[i],
            Ze[i],
            Az[i],
            Ne[i],
            Nmu[i],
            Ne[i] - Nmu[i],  # key discriminant
            np.cos(np.radians(Ze[i])),
            np.sin(np.radians(Az[i])),
            np.cos(np.radians(Az[i])),
            np.log1p(np.sum(mat_ch0[i])),  # total electron signal
            np.log1p(np.sum(mat_ch1[i])),  # total muon signal
            np.log1p(np.sum(mat_ch0[i]) + np.sum(mat_ch1[i])),  # total signal
            np.std(mat_ch0[i][mat_ch0[i] > 0]) if np.any(mat_ch0[i] > 0) else 0,  # electron variance
            np.std(mat_ch1[i][mat_ch1[i] > 0]) if np.any(mat_ch1[i] > 0) else 0,  # muon variance
        ]
        X.append(feats)
    return np.array(X, dtype=np.float32)

print("Engineering features...")
X_train = engineer_features(train_matrices, train_features)
X_test = engineer_features(test_matrices, test_features)

print(f"Feature shape: {X_train.shape}")

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train RandomForest
print("Training RandomForest (500 trees, max_depth=20, n_jobs=8)...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=8,
    verbose=1
)

rf.fit(X_train, train_labels)

# Evaluate
train_acc = rf.score(X_train, train_labels)
test_acc = rf.score(X_test, test_labels)

print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Save predictions
test_preds = rf.predict(X_test)
np.savez("submissions/haiku-composition-mar11/predictions_v6.npz",
         predictions=test_preds)

print(f"\n---")
print(f"metric: {test_acc:.4f}")
print(f"description: RandomForest (500 trees) on 14 engineered features from matrices + raw features")
