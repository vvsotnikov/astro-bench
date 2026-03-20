#!/usr/bin/env python3
"""Gamma/hadron v18: Smart ensemble of v6 CNN + MLP on scalar features.

Combine learned spatial patterns (v6) with traditional ML on hand-crafted features.
Different architectures should have complementary strengths.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import os

print("Loading data...")
X_train_mat = np.load("data/gamma_train/matrices.npy", mmap_mode="r")
X_train_feat = np.load("data/gamma_train/features.npy", mmap_mode="r")
y_train = np.load("data/gamma_train/labels_gamma.npy", mmap_mode="r")

X_test_mat = np.load("data/gamma_test/matrices.npy", mmap_mode="r")
X_test_feat = np.load("data/gamma_test/features.npy", mmap_mode="r")
y_test = np.load("data/gamma_test/labels_gamma.npy", mmap_mode="r")

print(f"Train: {len(y_train)}, Test: {len(y_test)}")

# Engineer simple features
E = X_train_feat[:, 0:1]
Ne = X_train_feat[:, 3:4]
Nmu = X_train_feat[:, 4:5]
Ne_Nmu = Ne - Nmu

X_train_simple = np.hstack([Ne_Nmu])
X_test_simple = np.hstack([X_test_feat[:, 3:4] - X_test_feat[:, 4:5]])

print("Training lightweight GBM on Ne/Nmu ratio...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_simple)
X_test_scaled = scaler.transform(X_test_simple)

gbm = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    verbose=0,
)
gbm.fit(X_train_scaled, y_train)
gbm_scores = gbm.predict_proba(X_test_scaled)[:, 0]

print(f"GBM result @ 75% gamma eff: computing...")

# Load v6 CNN predictions
print("Loading v6 CNN predictions...")
v6_scores = np.load("submissions/run1/probs_v6.npy")

# Ensemble: weight CNN (which is better) more heavily
alpha_cnn = 0.7
alpha_gbm = 0.3
ensemble_scores = alpha_cnn * v6_scores + alpha_gbm * gbm_scores

# Save predictions
os.makedirs("submissions/run1", exist_ok=True)
np.savez("submissions/run1/predictions_v18.npz", gamma_scores=ensemble_scores)
np.save("submissions/run1/probs_v18.npy", ensemble_scores)

# Compute survival @ 75% gamma efficiency
is_gamma = y_test == 0
is_hadron = y_test == 1
sg = np.sort(ensemble_scores[is_gamma])
ng = len(sg)
thr_75 = sg[max(0, int(np.floor(ng * (1 - 0.75))))]
n_hadron_surviving = (ensemble_scores[is_hadron] >= thr_75).sum()
survival_75 = n_hadron_surviving / is_hadron.sum() if is_hadron.sum() > 0 else 1.0

print(f"Ensemble survival rate @ 75% gamma efficiency: {survival_75:.2e}")
print(f"  v6 CNN: 3.50e-04")
print(f"  GBM simple: check individually")
print(f"  Ensemble (0.7*CNN + 0.3*GBM): {survival_75:.2e}")
print(f"Saved predictions to submissions/run1/predictions_v18.npz")
