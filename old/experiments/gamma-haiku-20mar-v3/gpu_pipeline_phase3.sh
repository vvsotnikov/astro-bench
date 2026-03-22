#!/bin/bash
# GPU pipeline for v36-v40 - run sequentially with proper CUDA setup

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

for model in v36 v37 v38 v39 v40; do
  echo "[$(date '+%H:%M')] Starting $model..."
  
  if [ "$model" = "v40" ]; then
    # v40 is an ensemble that waits for v39
    timeout 600 uv run python train_${model}.py > ${model}.log 2>&1
  else
    timeout 4800 uv run python train_${model}.py > ${model}.log 2>&1
  fi
  
  if [ -f "predictions_${model}.npz" ]; then
    echo "[$(date '+%H:%M')] ✓ $model COMPLETE"
    grep "${model}:" results.tsv | tail -1
  else
    echo "[$(date '+%H:%M')] ✗ $model FAILED (no predictions)"
  fi
  
  sleep 5
done

echo "[$(date '+%H:%M')] Phase 3 GPU pipeline complete"
