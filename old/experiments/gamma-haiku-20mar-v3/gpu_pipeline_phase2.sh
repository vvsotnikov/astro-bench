#!/bin/bash
# GPU pipeline for v30-v35 - run sequentially with proper CUDA setup

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

for model in v30 v31 v32 v33 v34 v35; do
  echo "[$(date '+%H:%M')] Starting $model..."
  timeout 4800 uv run python train_${model}.py > ${model}.log 2>&1
  
  if [ -f "predictions_${model}.npz" ]; then
    echo "[$(date '+%H:%M')] ✓ $model COMPLETE"
    grep "${model}:" results.tsv | tail -1
  else
    echo "[$(date '+%H:%M')] ✗ $model FAILED (no predictions)"
  fi
  
  sleep 5
done

echo "[$(date '+%H:%M')] Phase 2 GPU pipeline complete"
