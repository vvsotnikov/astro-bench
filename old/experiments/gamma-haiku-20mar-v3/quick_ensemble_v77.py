#!/usr/bin/env python3
"""v77: Ensemble v36 (0.7) + v31 (0.3)"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate

preds_v36 = np.load('predictions_v36.npz')['gamma_scores']
preds_v31 = np.load('predictions_v31.npz')['gamma_scores']

gamma_scores = 0.7 * preds_v36 + 0.3 * preds_v31

np.savez_compressed('predictions_v77.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v77: Ensemble v36 (0.7) + v31 (0.3)")
print(f"Metric: {metric}")
