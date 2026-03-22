#!/bin/bash
# Run v2 and v3 sequentially after v1 finishes

# Wait for v1
echo "Waiting for v1..."
while pgrep -f "train_v1.py" > /dev/null; do
  sleep 5
done

echo "v1 complete. Starting v2..."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python train_v2.py > submissions/v1/v2.log 2>&1

echo "v2 complete. Starting v3..."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python train_v3.py > submissions/v1/v3.log 2>&1

echo "v3 complete. Starting v5..."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python train_v5.py > submissions/v1/v5.log 2>&1

echo "All done"
