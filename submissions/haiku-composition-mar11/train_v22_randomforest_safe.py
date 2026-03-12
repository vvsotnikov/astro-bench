"""v22: RandomForest with proper preprocessing (Phase 4: B4a safe variant).

RandomForest on engineered features with safety guarantees:
- Extract 7 engineered features (same as haiku-mar8)
- Apply log1p to matrices, then compute spatial statistics (mean, std, max per channel)
- Total: 7 scalar features + 6 spatial features = 13D input
- RandomForest with 200 trees, depth=15
- Proper nan/inf handling
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

print("Loading data...")
X_train = np.load('data/composition_train/matrices.npy', mmap_mode='r')
f_train = np.load('data/composition_train/features.npy', mmap_mode='r')
y_train = np.load('data/composition_train/labels_composition.npy', mmap_mode='r')[:]

X_test = np.load('data/composition_test/matrices.npy', mmap_mode='r')
f_test = np.load('data/composition_test/features.npy', mmap_mode='r')
y_test = np.load('data/composition_test/labels_composition.npy', mmap_mode='r')[:]

print(f"Train: {len(y_train)}, Test: {len(y_test)}")

# Extract spatial features from log1p matrices
def extract_spatial_features(matrices):
    """Extract summary statistics from each matrix."""
    n = len(matrices)
    feats = []
    for i in range(n):
        mat = np.log1p(np.array(matrices[i], dtype=np.float32))
        # Per-channel stats
        ch0_mean = mat[:, :, 0].mean()
        ch0_max = mat[:, :, 0].max()
        ch1_mean = mat[:, :, 1].mean()
        ch1_max = mat[:, :, 1].max()
        feats.append([ch0_mean, ch0_max, ch1_mean, ch1_max])
        if i % 100000 == 0:
            print(f"  {i}/{n}")
    return np.array(feats, dtype=np.float32)

print("Extracting spatial features from train...")
spatial_train = extract_spatial_features(X_train)
spatial_train = np.nan_to_num(spatial_train, nan=0.0, posinf=0.0, neginf=0.0)

print("Extracting spatial features from test...")
spatial_test = extract_spatial_features(X_test)
spatial_test = np.nan_to_num(spatial_test, nan=0.0, posinf=0.0, neginf=0.0)

# Extract engineered scalar features (haiku-mar8 set)
def extract_scalar_features(features):
    """Extract 7 engineered features."""
    feats = []
    for feat in features:
        E, Ze, Az, Ne, Nmu = feat
        eng = np.array([E, np.cos(np.radians(Ze)),
                       np.sin(np.radians(Az)), np.cos(np.radians(Az)),
                       Ne, Nmu, Ne - Nmu], dtype=np.float32)
        feats.append(eng)
    return np.array(feats, dtype=np.float32)

print("Extracting scalar features from train...")
scalar_train = extract_scalar_features(f_train)
scalar_train = np.nan_to_num(scalar_train, nan=0.0, posinf=0.0, neginf=0.0)

print("Extracting scalar features from test...")
scalar_test = extract_scalar_features(f_test)
scalar_test = np.nan_to_num(scalar_test, nan=0.0, posinf=0.0, neginf=0.0)

# Combine features
X_train_combined = np.concatenate([spatial_train, scalar_train], axis=1)  # 11D
X_test_combined = np.concatenate([spatial_test, scalar_test], axis=1)

print(f"Feature shape: train={X_train_combined.shape}, test={X_test_combined.shape}")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_test_scaled = scaler.transform(X_test_combined)

# Ensure no nan/inf after scaling
X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Train RandomForest
print("Training RandomForest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42, verbose=1)
rf.fit(X_train_scaled, y_train)

# Evaluate
print("Evaluating...")
train_acc = rf.score(X_train_scaled, y_train)
test_acc = rf.score(X_test_scaled, y_test)

print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Predictions
test_preds = rf.predict(X_test_scaled)
np.savez("submissions/haiku-composition-mar11/predictions_v22.npz",
         predictions=test_preds)

print(f"\n---")
print(f"metric: {test_acc:.4f}")
print(f"description: RandomForest on spatial+scalar features with safe preprocessing")
