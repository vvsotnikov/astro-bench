#!/bin/bash
while [ ! -f predictions_v3.npz ] || [ ! -f predictions_v1.npz ]; do
  sleep 5
done
echo "Predictions available, running v13..."
uv run python train_v13.py
