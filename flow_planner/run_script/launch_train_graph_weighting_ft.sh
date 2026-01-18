#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=null
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################

# Path to the Flow-Planner repo on the training machine
export PROJECT_ROOT=/home/code/grid_planner

# Path to save training results and tensorboard logs
export SAVE_DIR=/home/results/yangzhao/grid_planner
export TENSORBOARD_LOG_PATH=/home/results/yangzhao/grid_planner/tensorboard

# Path to the processed nuplan training data and index json
export TRAINING_DATA=/home/datasets_2/nuplan/processed_flowplanner
export TRAINING_JSON=/home/code/grid_planner/flow_planner_training.json

# Base checkpoint for graph-weighting fine-tuning
# NOTE: adjust this path to the .pth you want to fine-tune from.
export PRETRAINED_CKPT=/home/results/flow_planner/base_checkpoint/latest.pth

export TORCH_LOGS="dynamic,recompiles"

###################################
# Graph-weighting fine-tuning run
###################################

# Always run from project root so that Python can find the package and configs
cd "${PROJECT_ROOT}"

python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone -m flow_planner.trainer \
  --config-name flow_planner_standard \
  pretrained_checkpoint=${PRETRAINED_CKPT} \
  train.epoch=10 \
  train.save_utd=1 \
  core.graph_weighting.enabled=true \
  optimizer.lr=5e-5


