#!/usr/bin/env python3
"""v82: v36 with power transformation (scores^1.5) - boost confidence for medium-confidence preds"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate

preds_v36 = np.load('predictions_v36.npz')['gamma_scores']

# Power transformation to boost confidence
# scores in [0,1], ^1.5 will increase separation
gamma_scores = np.power(preds_v36, 1.5)

np.savez_compressed('predictions_v82.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v82: v36 with power transformation (^1.5)")
print(f"Metric: {metric}")
