#!/usr/bin/env python3
"""v83: v36 with root transformation (scores^0.7) - reduce confidence separation"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate

preds_v36 = np.load('predictions_v36.npz')['gamma_scores']

# Root transformation to soften confidence
gamma_scores = np.power(preds_v36, 0.7)

np.savez_compressed('predictions_v83.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v83: v36 with root transformation (^0.7)")
print(f"Metric: {metric}")
