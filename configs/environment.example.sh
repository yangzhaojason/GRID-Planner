#!/usr/bin/env bash
# Copy to configs/environment.local.sh, edit paths, then source it before a launcher.
# Defaults in common.sh already resolve relative to the checkout location.
GRID_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${GRID_ENV_DIR}/.." && pwd)"
export SAVE_DIR="${PROJECT_ROOT}/results"
export TENSORBOARD_LOG_PATH="${SAVE_DIR}/tensorboard"
export TRAINING_DATA="${PROJECT_ROOT}/data/processed"
export TRAINING_JSON="${TRAINING_DATA}/flow_planner_training.json"
export NUPLAN_DEVKIT_ROOT="${PROJECT_ROOT}/nuplan-devkit"
export NUPLAN_DATA_ROOT="${PROJECT_ROOT}/data/nuplan"
export NUPLAN_MAPS_ROOT="${NUPLAN_DATA_ROOT}/maps"
export NUPLAN_EXP_ROOT="${SAVE_DIR}"
export RAY_TMPDIR="/tmp/grid-ray-${UID}"
export PYTHON_BIN=python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8

# Fine-tuning requires an existing checkpoint; evaluation requires a selected run.
# export PRETRAINED_CKPT="/absolute/path/to/training-run/latest.pth"
# export RUN_DIR="/absolute/path/to/training-run"
# Alternatively set both CKPT_PATH and CONFIG_FILE for single-checkpoint evaluation.
# export CKPT_PATH="/absolute/path/to/model.pth"
# export CONFIG_FILE="/absolute/path/to/training-run/.hydra/config.yaml"

# Configure optional tracking yourself; launchers do not overwrite credentials.
# export WANDB_MODE=offline
