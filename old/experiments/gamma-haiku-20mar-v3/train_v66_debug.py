#!/usr/bin/env python3
"""v66_debug: Debug version with print statements"""
import sys
print("START", flush=True)
sys.path.insert(0, '/home/vladimir/cursor_projects/astro-agents/experiments/gamma-haiku-20mar-v3')

print("Importing torch...", flush=True)
import torch
import torch.nn as nn

print("Importing data loaders...", flush=True)
from load_data import load_train, load_test

print("Loading train data...", flush=True)
matrices, features, labels = load_train()
print(f"Loaded: matrices {matrices.shape}, features {features.shape}, labels {labels.shape}", flush=True)

print("Loading test data...", flush=True)
X_test, f_test, y_test = load_test()
print(f"Loaded test: {X_test.shape}, {f_test.shape}, {y_test.shape}", flush=True)

print("Test complete")
