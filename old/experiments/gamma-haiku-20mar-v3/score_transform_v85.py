#!/usr/bin/env python3
"""v85: v36 with log transformation (log(scores+eps)) - emphasize low scores"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate

preds_v36 = np.load('predictions_v36.npz')['gamma_scores']

# Log transformation
eps = 1e-10
gamma_scores = np.log(preds_v36 + eps)
# Normalize to [0, 1]
gamma_scores = (gamma_scores - gamma_scores.min()) / (gamma_scores.max() - gamma_scores.min() + eps)

np.savez_compressed('predictions_v85.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v85: v36 with log transformation (normalized)")
print(f"Metric: {metric}")
