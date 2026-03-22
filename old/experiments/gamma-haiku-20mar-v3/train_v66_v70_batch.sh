#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1

for v in 66 67 68 69 70; do
    if [ -f "train_v${v}.py" ]; then
        echo "[$(date +%H:%M)] Running v$v..."
        timeout 600 uv run python train_v${v}.py >> v${v}.log 2>&1
        [ -f "predictions_v${v}.npz" ] && echo "✓ v$v done" || echo "✗ v$v failed"
    fi
done
