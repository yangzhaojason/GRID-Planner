#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/flow_planner/run_script/common.sh"

: "${NUPLAN_DATA_PATH:=${NUPLAN_DATA_ROOT}/trainval}"
: "${NUPLAN_MAP_PATH:=${NUPLAN_MAPS_ROOT}}"
: "${TRAIN_SET_PATH:=${TRAINING_DATA}}"
: "${TOTAL_SCENARIOS:=1000000}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/data_process.py" \
  --data_path "${NUPLAN_DATA_PATH}" \
  --map_path "${NUPLAN_MAP_PATH}" \
  --save_path "${TRAIN_SET_PATH}" \
  --output_json "${TRAINING_JSON}" \
  --total_scenarios "${TOTAL_SCENARIOS}" \
  "$@"
