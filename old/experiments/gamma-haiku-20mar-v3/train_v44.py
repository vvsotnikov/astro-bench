#!/usr/bin/env python3
"""v44: Ensemble (average) of v30, v31, v32"""
import numpy as np
import sys
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')
from verify import evaluate

# Wait for v30, v31, v32 to complete
import time
for attempt in range(120):
    v30 = v31 = v32 = False
    try:
        np.load('predictions_v30.npz')
        v30 = True
    except:
        pass
    try:
        np.load('predictions_v31.npz')
        v31 = True
    except:
        pass
    try:
        np.load('predictions_v32.npz')
        v32 = True
    except:
        pass
    
    if v30 and v31 and v32:
        print(f"✓ All models loaded")
        break
    
    if attempt % 12 == 0:
        print(f"Waiting for v30,v31,v32... ({attempt*5}s)")
    time.sleep(5)

s30 = np.load('predictions_v30.npz')['gamma_scores']
s31 = np.load('predictions_v31.npz')['gamma_scores']
s32 = np.load('predictions_v32.npz')['gamma_scores']

gamma_scores = (s30 + s31 + s32) / 3.0

np.savez_compressed('predictions_v44.npz', gamma_scores=gamma_scores)
metric = evaluate(gamma_scores, "v44: Simple ensemble (v30 + v31 + v32)")
print(f"Metric: {metric}")
