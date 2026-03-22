#!/usr/bin/env python3
"""v49: Ensemble of v2 (best) + v39 (muon-only) + v48 (flatten) - diverse architectures"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate
import time

# Wait for all three models
models_to_load = {'v2': False, 'v39': False, 'v48': False}

for attempt in range(120):
    for model in list(models_to_load.keys()):
        try:
            np.load(f'predictions_{model}.npz')
            models_to_load[model] = True
        except:
            pass
    
    if all(models_to_load.values()):
        print("✓ All models loaded")
        break
    
    if attempt % 12 == 0:
        loaded = sum(models_to_load.values())
        print(f"Waiting for v2, v39, v48... {loaded}/3 ready ({attempt*5}s)")
    time.sleep(5)

s2 = np.load('predictions_v2.npz')['gamma_scores']
s39 = np.load('predictions_v39.npz')['gamma_scores']
s48 = np.load('predictions_v48.npz')['gamma_scores']

# Weight best model heavily (v2), then v48, then v39
gamma_scores = 0.6*s2 + 0.25*s48 + 0.15*s39

np.savez_compressed('predictions_v49.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v49: Ensemble (v2:0.6 + v48:0.25 + v39:0.15) - diverse architectures")
print(f"Metric: {metric}")
