#!/usr/bin/env bash
# Source this file after setting any machine-specific environment overrides.
# It deliberately does not change directories or create output directories.

GRID_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PROJECT_ROOT:=$(cd "${GRID_SCRIPT_DIR}/../.." && pwd)}"
: "${SAVE_DIR:=${PROJECT_ROOT}/results}"
: "${TENSORBOARD_LOG_PATH:=${SAVE_DIR}/tensorboard}"
: "${TRAINING_DATA:=${PROJECT_ROOT}/data/processed}"
: "${TRAINING_JSON:=${TRAINING_DATA}/flow_planner_training.json}"
: "${NUPLAN_DEVKIT_ROOT:=${PROJECT_ROOT}/nuplan-devkit}"
: "${NUPLAN_DATA_ROOT:=${PROJECT_ROOT}/data/nuplan}"
: "${NUPLAN_MAPS_ROOT:=${NUPLAN_DATA_ROOT}/maps}"
: "${NUPLAN_EXP_ROOT:=${SAVE_DIR}}"
# Keep this short: Ray uses Unix-domain sockets with a limited path length.
: "${RAY_TMPDIR:=/tmp/grid-ray-${UID}}"
: "${PYTHON_BIN:=python}"
# '=' preserves an explicitly empty CUDA_VISIBLE_DEVICES value.
: "${CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7}"
: "${NPROC_PER_NODE:=8}"
: "${HYDRA_FULL_ERROR:=1}"

export PROJECT_ROOT SAVE_DIR TENSORBOARD_LOG_PATH TRAINING_DATA TRAINING_JSON
export NUPLAN_DEVKIT_ROOT NUPLAN_DATA_ROOT NUPLAN_MAPS_ROOT NUPLAN_EXP_ROOT
export RAY_TMPDIR PYTHON_BIN CUDA_VISIBLE_DEVICES NPROC_PER_NODE HYDRA_FULL_ERROR
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

grid_require_file() {
  if [ -z "$2" ] || [ ! -f "$2" ]; then
    printf '%s must point to an existing file: %s\n' "$1" "$2" >&2
    return 1
  fi
}

# Single-checkpoint evaluation always uses explicit inputs. A timestamp is
# accepted for compatibility, but newest-directory guessing is intentionally absent.
grid_run_single_simulation() {
  : "${RUN_DIR:=}"
  if [ -z "${RUN_DIR}" ] && [ -n "${RUN_TIMESTAMP:-}" ]; then
    : "${EXP_SAVE_ROOT:=${SAVE_DIR}/outputs/FlowPlannerTraining/flow_planner_standard}"
    RUN_DIR="${EXP_SAVE_ROOT}/${RUN_TIMESTAMP}"
  fi
  CKPT_FILE="${CKPT_PATH:-${CKPT_FILE:-}}"
  if [ -z "${CKPT_FILE}" ] && [ -n "${RUN_DIR}" ]; then
    CKPT_FILE="${RUN_DIR}/${CKPT_NAME:-latest.pth}"
  fi
  if [ -z "${CONFIG_FILE:-}" ] && [ -n "${RUN_DIR}" ]; then
    CONFIG_FILE="${RUN_DIR}/.hydra/config.yaml"
  fi
  if [ -z "${CKPT_FILE}" ] || [ -z "${CONFIG_FILE:-}" ]; then
    echo "Set RUN_DIR, or both CKPT_PATH and CONFIG_FILE, to select the experiment explicitly." >&2
    return 1
  fi
  grid_require_file CKPT_PATH "${CKPT_FILE}" || return 1
  grid_require_file CONFIG_FILE "${CONFIG_FILE}" || return 1

  : "${SPLIT:=val14}"
  : "${CHALLENGE:=closed_loop_reactive_agents}"
  : "${PLANNER:=flow_planner}"
  : "${WORKER_THREADS_PER_NODE:=64}"
  : "${GPU_PER_SIM:=0.15}"
  local scenario_builder="nuplan_challenge"
  [ "${SPLIT}" != "val14" ] || scenario_builder="nuplan"
  local filename="${CKPT_FILE##*/}"
  local checkpoint_tag="${filename%.*}"
  mkdir -p "${NUPLAN_EXP_ROOT}" "${RAY_TMPDIR}"
  export RAY_DISABLE_DASHBOARD=1 RAY_USAGE_STATS_ENABLED=0
  export RAY_INCLUDE_DASHBOARD=false RAY_ENABLE_METRICS_AGENT=false RAY_METRICS_EXPORT_PORT=0
  echo "Evaluating checkpoint: ${CKPT_FILE}"
  "${PYTHON_BIN}" "${NUPLAN_DEVKIT_ROOT}/nuplan/planning/script/run_simulation.py" \
    "+simulation=${CHALLENGE}" \
    "planner=${PLANNER}" \
    "planner.flow_planner.config_path=${CONFIG_FILE}" \
    "planner.flow_planner.ckpt_path=${CKPT_FILE}" \
    "scenario_builder=${scenario_builder}" \
    "scenario_filter=${SPLIT}" \
    "experiment_uid=${PLANNER}/${SPLIT}/${BRANCH_NAME}/${checkpoint_tag}_$(date +%Y-%m-%d-%H-%M-%S)" \
    verbose=true \
    worker=ray_distributed \
    "worker.threads_per_node=${WORKER_THREADS_PER_NODE}" \
    distributed_mode=SINGLE_NODE \
    "number_of_gpus_allocated_per_simulation=${GPU_PER_SIM}" \
    enable_simulation_progress_bar=true \
    'hydra.searchpath=[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]' \
    "$@"
}
