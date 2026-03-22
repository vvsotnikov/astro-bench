#!/usr/bin/env python3
"""v47: Random Forest (1000 trees, max_depth=15)"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from load_data import load_train, load_test
from verify import evaluate

matrices, features, labels = load_train()
X_test, f_test, y_test = load_test()

X_train = np.concatenate([matrices.reshape(matrices.shape[0], -1), features], axis=1)
X_test_full = np.concatenate([X_test.reshape(X_test.shape[0], -1), f_test], axis=1)

print(f"Training RandomForest on {X_train.shape[0]} samples")
clf = RandomForestClassifier(n_estimators=1000, max_depth=15, random_state=42, n_jobs=-1)
clf.fit(X_train, labels)
probs = clf.predict_proba(X_test_full)
gamma_scores = probs[:, 0]

np.savez_compressed('predictions_v47.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v47: RandomForest 1000 trees, max_depth=15")
print(f"Metric: {metric}")
