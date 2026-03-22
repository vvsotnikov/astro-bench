#!/usr/bin/env python3
"""v15: GradientBoosting ensemble (CPU-only, no GPU)"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from load_data import load_train, load_test
from sklearn.ensemble import GradientBoostingClassifier
from verify import evaluate

matrices, features, labels = load_train()
X_test, f_test, y_test = load_test()

# Flatten matrices and concatenate with features
X_train = np.concatenate([
    matrices.reshape(matrices.shape[0], -1),
    features
], axis=1)
X_test_full = np.concatenate([X_test.reshape(X_test.shape[0], -1), f_test], axis=1)

print(f"Training on {X_train.shape[0]} samples...")
clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
clf.fit(X_train, labels)
probs = clf.predict_proba(X_test_full)
gamma_scores = probs[:, 0]
np.savez_compressed('predictions_v15.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v15: GradientBoosting 200 trees, max_depth=5")
print(f"Metric: {metric}")
