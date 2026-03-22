#!/bin/bash
set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

echo "[$(date +%H:%M)] Starting final exploration pipeline (v54-v56)..."

for v in 54 55 56; do
    echo "[$(date +%H:%M)] Starting v$v..."
    uv run python train_v${v}.py >> v${v}.log 2>&1
    if [ -f "predictions_v${v}.npz" ]; then
        echo "[$(date +%H:%M)] ✓ v$v COMPLETE"
    else
        echo "[$(date +%H:%M)] ✗ v$v FAILED"
    fi
done

echo "[$(date +%H:%M)] Pipeline complete"
