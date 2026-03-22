#!/bin/bash
# Phase A: Run architecture variations sequentially

echo "=== PHASE A: ARCHITECTURE VARIATIONS ==="
echo "Starting v29 monitoring and follow-on variants..."
echo ""

# Wait for v29 to complete
while [ ! -f predictions_v29.npz ]; do
  echo "[$(date '+%H:%M')] Waiting for v29..."
  sleep 30
done

echo "✓ v29 complete. Running v30..."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python train_v30.py

echo ""
echo "✓ v30 complete. Results so far:"
tail -3 results.tsv
