#!/usr/bin/env python3
"""v46: Logistic Regression on raw 16x16x2 grid + 5 features (simple baseline)"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from load_data import load_train, load_test
from verify import evaluate

matrices, features, labels = load_train()
X_test, f_test, y_test = load_test()

# Flatten and concatenate
X_train = np.concatenate([
    matrices.reshape(matrices.shape[0], -1),
    features
], axis=1)
X_test_full = np.concatenate([X_test.reshape(X_test.shape[0], -1), f_test], axis=1)

print(f"Training logistic regression on {X_train.shape[0]} x {X_train.shape[1]} features")
clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1)
clf.fit(X_train, labels)
probs = clf.predict_proba(X_test_full)
gamma_scores = probs[:, 0]

np.savez_compressed('predictions_v46.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v46: Logistic Regression (baseline)")
print(f"Metric: {metric}")
