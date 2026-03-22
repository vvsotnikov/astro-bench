#!/bin/bash
set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

echo "[$(date +%H:%M)] Starting Phase 5 GPU pipeline (v46-v50)..."

for v in 46 47 48 49 50; do
    echo "[$(date +%H:%M)] Starting v$v..."
    start_time=$(date +%s)

    uv run python train_v${v}.py >> v${v}.log 2>&1

    end_time=$(date +%s)
    duration=$((end_time - start_time))
    duration_min=$((duration / 60))

    if [ -f "predictions_v${v}.npz" ]; then
        echo "[$(date +%H:%M)] ✓ v$v COMPLETE (${duration_min}m)"
    else
        echo "[$(date +%H:%M)] ✗ v$v FAILED (no predictions)"
    fi
done

echo "[$(date +%H:%M)] Phase 5 GPU pipeline complete"
