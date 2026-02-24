#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=null
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################

export PROJECT_ROOT=/home/shiyinan/code/grid_planner

export SAVE_DIR=/home/results/shiyinan/grid_planner_womd
export TENSORBOARD_LOG_PATH=/home/results/shiyinan/grid_planner_womd/tensorboard

# Preprocessed WOMD NPZ data (output of launch_womd_preprocess.sh)
export TRAINING_DATA=/home/datasets/womd_npz/training
export TRAINING_JSON=/home/datasets/womd_npz/training/data_list_training.json

export TORCH_LOGS="dynamic,recompiles"

###################################
# Single-stage training with GAIW
###################################

cd "${PROJECT_ROOT}"

python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone -m flow_planner.trainer \
  --config-name flow_planner_womd \
  num_workers=16 \
  train.batch_size=512 \
  train.epoch=30 \
  train.save_utd=5 \
  optimizer.lr=4e-4 \
  core.graph_weighting.enabled=false
