#!/usr/bin/env bash
set -euo pipefail

###################################
# WOMD Data Preprocessing Script
#
# Converts WOMD TFRecord scenarios to
# agent-centric NPZ files compatible
# with NuPlanDataset.
###################################

# Path to the Flow-Planner repo
export PROJECT_ROOT=/home/shiyinan/code/grid_planner

# Path to raw WOMD scenario TFRecords
export WOMD_ROOT=/home/datasets_2/scenario

# Output directory for NPZ files
export NPZ_ROOT=/home/datasets/womd_npz

# Number of parallel workers per split
NUM_WORKERS=64

cd "${PROJECT_ROOT}"

echo "========================================="
echo "  WOMD Preprocessing — training split"
echo "========================================="
python -m flow_planner.data.data_process.womd_data_processor \
    --input_dir "${WOMD_ROOT}/training" \
    --output_dir "${NPZ_ROOT}/training" \
    --num_workers ${NUM_WORKERS} \
    --split training

echo "========================================="
echo "  WOMD Preprocessing — validation split"
echo "========================================="
python -m flow_planner.data.data_process.womd_data_processor \
    --input_dir "${WOMD_ROOT}/validation" \
    --output_dir "${NPZ_ROOT}/validation" \
    --num_workers ${NUM_WORKERS} \
    --split validation

echo "========================================="
echo "  WOMD Preprocessing — testing split"
echo "========================================="
python -m flow_planner.data.data_process.womd_data_processor \
    --input_dir "${WOMD_ROOT}/testing" \
    --output_dir "${NPZ_ROOT}/testing" \
    --num_workers ${NUM_WORKERS} \
    --split testing

echo "========================================="
echo "  Done. NPZ files saved to ${NPZ_ROOT}"
echo "========================================="
