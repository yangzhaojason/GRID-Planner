#!/usr/bin/env bash

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
: "${NPROC_PER_NODE:=8}"
export NPROC_PER_NODE
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

# Base checkpoint for fine-tuning
# NOTE: adjust this path to the .pth you want to fine-tune from.
# export PRETRAINED_CKPT=/home/results/flow_planner/base_checkpoint/latest.pth
: "${PRETRAINED_CKPT:=/home/results/yangzhao/grid_planner/outputs/FlowPlannerTraining/flow_planner_standard/2026-01-19_14-14-53/latest.pth}"
export PRETRAINED_CKPT
export TORCH_LOGS="dynamic,recompiles"

###################################
# Graph-weighting + RRS fine-tuning run
###################################

# Always run from project root so that Python can find the package and configs
cd "${PROJECT_ROOT}"

# Optional ablation save root (overrides hydra save_dir directly)
# Example:
#   ABLATION_ROOT=/home/results/yangzhao/grid_planner/outputs/FlowPlannerTraining/flow_planner_standard/ablation_rrs \
#   EXP_TAG=runA bash flow_planner/run_script/launch_train_graph_weighting_rrs_ft.sh
: "${ABLATION_ROOT:=/home/results/yangzhao/grid_planner/outputs/FlowPlannerTraining/flow_planner_standard}"
: "${EXP_TAG:=rrs_ft}"
: "${RUN_STAMP:=$(date +%Y-%m-%d_%H-%M-%S)}"
TARGET_SAVE_DIR="${ABLATION_ROOT}/${EXP_TAG}/${RUN_STAMP}"
mkdir -p "${TARGET_SAVE_DIR}"
echo "Training outputs will be saved to: ${TARGET_SAVE_DIR}"

# Paper-aligned unified objective:
#   L = L_flow_matching + nu * E[ w_GAIW(tau) * RRS_penalty(tau_hat) ]
# where RRS_reward = -RRS_penalty (logged as rrs_reward_mean)

python -m torch.distributed.run --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" --standalone -m flow_planner.trainer \
  --config-name flow_planner_standard \
  pretrained_checkpoint=${PRETRAINED_CKPT} \
  save_dir=${TARGET_SAVE_DIR} \
  job_name=flow_planner_${EXP_TAG} \
  train.epoch=10 \
  train.save_utd=1 \
  scheduler.warm_up_epoch=5 \
  core.graph_weighting.enabled=true \
  core.rrs.enabled=true \
  core.rrs.nu=0.0008 \
  core.rrs.nu_warmup_epochs=5 \
  core.rrs.use_gaiw=true \
  core.rrs.lam_col=0.20 \
  core.rrs.lam_d=0.20 \
  core.rrs.lam_ttc=0.20 \
  core.rrs.lam_lane=0 \
  core.rrs.lam_red=0 \
  core.rrs.lam_spd=0 \
  core.rrs.lam_dir=0 \
  core.rrs.lam_comfort=0 \
  core.rrs.lam_progress=0 \
  core.rrs.lam_low_speed=0 \
  optimizer.lr=5e-5 \
  "$@"

