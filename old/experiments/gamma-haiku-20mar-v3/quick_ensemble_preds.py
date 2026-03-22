#!/usr/bin/env python3
"""v76 (retry): Ensemble pre-computed predictions v36 + v31 with weights 0.9/0.1"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate

# Load pre-computed predictions
preds_v36 = np.load('predictions_v36.npz')['gamma_scores']
preds_v31 = np.load('predictions_v31.npz')['gamma_scores']

# Weighted ensemble: v36 (0.9) + v31 (0.1)
gamma_scores = 0.9 * preds_v36 + 0.1 * preds_v31

np.savez_compressed('predictions_v76.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v76: Ensemble predictions v36 (0.9) + v31 (0.1)")
print(f"Metric: {metric}")
