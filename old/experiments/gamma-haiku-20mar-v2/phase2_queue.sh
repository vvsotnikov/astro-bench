#!/bin/bash
# Wait for seed variants to complete, then run hyperparameter variations

echo "Waiting for seed variants (v16-v19) to complete..."
while [ $(ls -1 predictions_v1[6789].npz 2>/dev/null | wc -l) -lt 4 ]; do
  echo "Waiting... $(ls -1 predictions_v1[6789].npz 2>/dev/null | wc -l)/4 seed predictions ready"
  sleep 30
done

echo "Seed variants complete! Starting hyperparameter variations..."

# Run v21 and v22
for v in 21 22; do
  echo "Running v$v..."
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python train_v${v}.py > submissions/v1/v${v}.log 2>&1
  echo "v$v complete"
  sleep 2
done

echo "=== PHASE 2 COMPLETE ==="
echo "Now have seeds (v3,v16,v17,v18,v19) + variations (v21,v22)"
cat results.tsv | tail -5
