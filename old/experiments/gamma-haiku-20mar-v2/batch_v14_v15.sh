#!/bin/bash
while pgrep -f "train_v12" > /dev/null; do sleep 10; done
echo "v12 done, starting v14 and v15..."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python train_v14.py > submissions/v1/v14.log 2>&1 &
wait
echo "v14 done"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python train_v15.py > submissions/v1/v15.log 2>&1
echo "v15 done"
