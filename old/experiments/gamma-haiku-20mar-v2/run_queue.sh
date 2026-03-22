#!/bin/bash
# Run v11 and v12 sequentially after v10 finishes

echo "Waiting for v10..."
while pgrep -f "train_v10" > /dev/null; do sleep 10; done

echo "v10 done, launching v11..."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python train_v11.py > submissions/v1/v11.log 2>&1

echo "v11 done, launching v12..."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python train_v12.py > submissions/v1/v12.log 2>&1

echo "Done"
