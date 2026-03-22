#!/bin/bash
# Continue Phase A - run v31+ after v30

echo "Waiting for v30 to complete..."
while [ ! -f predictions_v30.npz ]; do
  sleep 15
done
echo "✓ v30 complete"
echo ""

# Show current results
echo "=== Results after v30 ==="
tail -3 results.tsv
echo ""

echo "Starting v31 (multi-head attention, 4 heads)..."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python train_v31.py

echo ""
echo "=== All Phase A variants complete ==="
tail -2 results.tsv
