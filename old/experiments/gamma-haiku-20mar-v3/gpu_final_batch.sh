#!/bin/bash
set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

echo "[$(date +%H:%M)] Waiting for v57 to finish..."
while [ ! -f predictions_v57.npz ]; do
    sleep 10
done
echo "[$(date +%H:%M)] v57 complete. Running v58-v60..."

for v in 58 59 60; do
    echo "[$(date +%H:%M)] Starting v$v..."
    uv run python train_v${v}.py >> v${v}.log 2>&1
    if [ -f "predictions_v${v}.npz" ]; then
        echo "[$(date +%H:%M)] ✓ v$v COMPLETE"
    else
        echo "[$(date +%H:%M)] ✗ v$v FAILED"
    fi
done

echo "[$(date +%H:%M)] Final batch complete"
