#!/usr/bin/env bash

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################

# Set environment variables
# TODO: 根据你自己机器的实际路径修改下面几个变量
export PROJECT_ROOT=/home/code/grid_planner                                  # Flow-Planner 仓库根目录
export NUPLAN_DEVKIT_ROOT=/home/code/grid_planner/nuplan-devkit             # nuplan-devkit 绝对路径
export NUPLAN_DATA_ROOT=/home/datasets_2/nuplan/                             # nuplan 原始数据路径
export NUPLAN_MAPS_ROOT=/home/datasets_2/nuplan/nuPlan-v1.1/maps
: "${NUPLAN_EXP_ROOT:=/home/results/yangzhao/grid_planner}"
export NUPLAN_EXP_ROOT

# Dataset split to use
# Options:
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
# Allow overriding from command line, e.g.:
#   SPLIT=test14-hard bash flow_planner/run_script/launch_sim_nuplan_graph_weighting_rrs_ft.sh
: "${SPLIT:=val14}"
export SPLIT

# Challenge type
# Options:
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
# Allow overriding from command line, e.g.:
#   CHALLENGE=closed_loop_nonreactive_agents bash flow_planner/run_script/launch_sim_nuplan_graph_weighting_rrs_ft.sh
: "${CHALLENGE:=closed_loop_reactive_agents}"
export CHALLENGE

###################################

# 确保 Python 能 import 到 flow_planner（给 Hydra 的 pkg:// 路径使用）
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Put Ray temp/object store on the large disk to avoid "Local disk is full"
export RAY_TMPDIR="${RAY_TMPDIR:-/home/results/ray}"
mkdir -p "${NUPLAN_EXP_ROOT}"
mkdir -p "${RAY_TMPDIR}"

# Ray: disable dashboard/metrics to avoid exporter-agent init retries slowing down startup
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_INCLUDE_DASHBOARD=false
export RAY_ENABLE_METRICS_AGENT=false
export RAY_METRICS_EXPORT_PORT=0

# 实验输出根目录（和训练脚本里的 SAVE_DIR 一致）
# Allow overriding from command line, e.g.:
#   EXP_SAVE_ROOT=/home/results/.../ablation_rrs/runA bash ...
: "${EXP_SAVE_ROOT:=/home/results/yangzhao/grid_planner/outputs/FlowPlannerTraining/flow_planner_standard}"

# 方式一：手动指定时间戳（可选）
#   export RUN_TIMESTAMP=2026-02-10_05-47-01
# 方式二：不设置 RUN_TIMESTAMP，则自动选择 EXP_SAVE_ROOT 下最新的目录
if [ -n "${RUN_TIMESTAMP}" ]; then
  RUN_DIR="${EXP_SAVE_ROOT}/${RUN_TIMESTAMP}"
else
  # 按修改时间倒序取第一个子目录
  LATEST_DIR=$(ls -td "${EXP_SAVE_ROOT}"/*/ 2>/dev/null | head -n 1)
  if [ -z "${LATEST_DIR}" ]; then
    echo "No run directory found under ${EXP_SAVE_ROOT}. Please check SAVE_DIR or set RUN_TIMESTAMP manually."
    exit 1
  fi
  # 去掉末尾的斜杠
  RUN_DIR="${LATEST_DIR%/}"
fi

BRANCH_NAME=flow_planner_graph_weighting_rrs_ft
CONFIG_FILE=${RUN_DIR}/.hydra/config.yaml        # path of .hydra/config in ckpt folder
# Optional: choose a specific checkpoint file in RUN_DIR
#   CKPT_NAME=model_epoch_1_trainloss_1.4114.pth bash ...
: "${CKPT_NAME:=latest.pth}"
CKPT_FILE=${RUN_DIR}/${CKPT_NAME}

if [ ! -f "${CKPT_FILE}" ]; then
  echo "Checkpoint not found: ${CKPT_FILE}"
  echo "RUN_DIR=${RUN_DIR}"
  exit 1
fi

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi

echo "Evaluating RUN_DIR: $RUN_DIR"
echo "Evaluating config:  $CONFIG_FILE"
echo "Evaluating ckpt:    $CKPT_FILE"
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=flow_planner

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.flow_planner.config_path=$CONFIG_FILE \
    planner.flow_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=64 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"

