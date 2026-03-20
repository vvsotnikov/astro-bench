#!/bin/bash
# Phase 2-5 Systematic Exploration

set -e

cd /home/vladimir/cursor_projects/astro-agents

log_result() {
    local version=$1
    local log_file=$2

    if [ -f "$log_file" ]; then
        metric=$(grep "^metric:" "$log_file" | awk '{print $2}')
        description=$(grep "^description:" "$log_file" | cut -d: -f2-)
        if [ -n "$metric" ]; then
            echo -e "${version}\t${metric}\tkeep\t${description}" >> submissions/haiku-composition-mar11/results.tsv
            echo "v${version}: ${metric}"
        fi
    fi
}

echo "=== Phase 2: CNN Architecture Search (v17-v19) ==="

echo "Starting v17 (exact haiku-mar8 replica)..."
timeout 3600 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python submissions/haiku-composition-mar11/train_v17_haiku_mar8_exact.py > submissions/haiku-composition-mar11/train_v17.log 2>&1
log_result "v17" "submissions/haiku-composition-mar11/train_v17.log"

echo "Starting v18 (deeper CNN, 5 blocks)..."
timeout 3600 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python submissions/haiku-composition-mar11/train_v18_deeper_cnn.py > submissions/haiku-composition-mar11/train_v18.log 2>&1
log_result "v18" "submissions/haiku-composition-mar11/train_v18.log"

echo "Starting v19 (wider CNN)..."
timeout 3600 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python submissions/haiku-composition-mar11/train_v19_wider_cnn.py > submissions/haiku-composition-mar11/train_v19.log 2>&1
log_result "v19" "submissions/haiku-composition-mar11/train_v19.log"

echo ""
echo "=== Phase 3: Non-CNN Architectures (v20-v21) ==="

echo "Starting v20 (Vision Transformer 2x2)..."
timeout 3600 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python submissions/haiku-composition-mar11/train_v20_vit_2x2.py > submissions/haiku-composition-mar11/train_v20.log 2>&1
log_result "v20" "submissions/haiku-composition-mar11/train_v20.log"

echo "Starting v21 (MLP flattened)..."
timeout 3600 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python submissions/haiku-composition-mar11/train_v21_mlp_flattened.py > submissions/haiku-composition-mar11/train_v21.log 2>&1
log_result "v21" "submissions/haiku-composition-mar11/train_v21.log"

echo ""
echo "=== Phase 4: Tree Models (v22) ==="

echo "Starting v22 (RandomForest safe)..."
timeout 1800 CUDA_VISIBLE_DEVICES=1 uv run python submissions/haiku-composition-mar11/train_v22_randomforest_safe.py > submissions/haiku-composition-mar11/train_v22.log 2>&1
log_result "v22" "submissions/haiku-composition-mar11/train_v22.log"

echo ""
echo "=== Phase 5: Loss Functions (v23) ==="

echo "Starting v23 (Focal loss)..."
timeout 3600 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 uv run python submissions/haiku-composition-mar11/train_v23_focal_loss.py > submissions/haiku-composition-mar11/train_v23.log 2>&1
log_result "v23" "submissions/haiku-composition-mar11/train_v23.log"

echo ""
echo "=== All systematic exploration complete ==="
