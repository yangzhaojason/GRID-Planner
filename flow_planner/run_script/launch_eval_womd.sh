#!/usr/bin/env bash
set -euo pipefail

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################

export PROJECT_ROOT=/home/shiyinan/code/grid_planner

# Trained WOMD checkpoint
# CONFIG_PATH=/home/results/shiyinan/grid_planner_womd/outputs/FlowPlannerTraining/flow_planner_womd/2026-02-22_01-42-37/.hydra/config.yaml
# CKPT_PATH=/home/results/shiyinan/grid_planner_womd/outputs/FlowPlannerTraining/flow_planner_womd/2026-02-22_01-42-37/latest.pth

CONFIG_PATH=/home/results/shiyinan/grid_planner_womd/outputs/FlowPlannerTraining/flow_planner_womd/2026-02-23_10-21-37/.hydra/config.yaml
CKPT_PATH=/home/results/shiyinan/grid_planner_womd/outputs/FlowPlannerTraining/flow_planner_womd/2026-02-23_10-21-37/model_epoch_5_trainloss_0.0694.pth

# Preprocessed WOMD validation data
DATA_DIR=/home/datasets/womd_npz/validation
DATA_LIST=/home/datasets/womd_npz/validation/data_list_validation.json

# Output
OUTPUT_PATH=/home/results/shiyinan/grid_planner_womd/eval_results.json

###################################
# Run evaluation
###################################

cd "${PROJECT_ROOT}"

python -m flow_planner.evaluation.womd_eval \
    --config_path "${CONFIG_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --data_dir "${DATA_DIR}" \
    --data_list "${DATA_LIST}" \
    --batch_size 64 \
    --num_workers 16 \
    --num_samples 64 \
    --num_modes 6 \
    --max_samples 2000 \
    --output_path "${OUTPUT_PATH}"
