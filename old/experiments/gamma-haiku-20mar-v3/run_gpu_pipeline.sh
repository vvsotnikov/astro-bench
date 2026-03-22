#!/bin/bash
set -e

models=("v23" "v24" "v25" "v26" "v27")

for model in "${models[@]}"; do
  echo ""
  echo "========================================="
  echo "Starting $model on GPU 1"
  echo "========================================="
  
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 timeout 3600 uv run python "train_${model}.py" > "${model}.log" 2>&1
  
  if [ -f "predictions_${model}.npz" ]; then
    echo "✓ $model COMPLETE"
    tail -1 results.tsv | awk -F'\t' '{print "  Metric:", $2}'
  else
    echo "✗ $model FAILED"
  fi
  
  sleep 2
done
