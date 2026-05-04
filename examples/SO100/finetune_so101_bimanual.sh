#!/bin/bash
# Fine-tuning script for SO-101 Bimanual with GR00T N1.6
# 
# Prerequisites:
# 1. Convert V3 datasets to V2 format:
#    python scripts/lerobot_conversion/convert_v3_to_v2.py --repo-id Build22/folding_laundry
#    python scripts/lerobot_conversion/convert_v3_to_v2.py --repo-id Build22/clear_table_clutter_cleaned
#
# 2. Copy the converted datasets to a local directory
#
# 3. Add modality.json to each dataset's meta/ folder:
#    cp examples/SO100/so101_tricam__modality.json <DATASET_PATH>/meta/modality.json
#
# Usage:
#    bash examples/SO100/finetune_so101_bimanual.sh

set -e

# Configure these paths
DATASET_PATH=${DATASET_PATH:-"./data/folding_laundry"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/so101_bimanual_finetuned"}
NUM_GPUS=${NUM_GPUS:-1}
MAX_STEPS=${MAX_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-32}

echo "=== GR00T N1.6 Fine-tuning for SO-101 Bimanual ==="
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Max steps: $MAX_STEPS"
echo "Batch size: $BATCH_SIZE"

CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so101_bimanual_config.py \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$OUTPUT_DIR" \
    --save-total-limit 5 \
    --save-steps 1000 \
    --max-steps "$MAX_STEPS" \
    --use-wandb \
    --global-batch-size "$BATCH_SIZE" \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4

echo "=== Fine-tuning complete! ==="
echo "Checkpoint saved to: $OUTPUT_DIR"
