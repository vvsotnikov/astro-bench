#!/bin/bash
echo "Waiting for phase 2 to complete..."
while [ ! -f results.tsv ] || [ $(wc -l < results.tsv) -lt 15 ]; do
  sleep 30
done

echo "Phase 2 complete. Starting phase 3: ablation + longer training..."
for v in 23 24; do
  echo "Running v$v..."
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python train_v${v}.py > submissions/v1/v${v}.log 2>&1
  echo "v$v done"
done

echo "=== ALL EXPERIMENTS COMPLETE ==="
tail -20 results.tsv
