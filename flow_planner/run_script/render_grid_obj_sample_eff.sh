#!/usr/bin/env bash
set -euo pipefail

INPUT_ARG="${1:-}"
SUITE_ROOT_INPUT="${SUITE_ROOT:-}"
: "${GRID_OBJ_DATA_JSON:=}"

if [ -n "${INPUT_ARG}" ]; then
  if [ -f "${INPUT_ARG}" ] && [ "${INPUT_ARG##*.}" = "json" ]; then
    GRID_OBJ_DATA_JSON="${INPUT_ARG}"
  else
    SUITE_ROOT_INPUT="${INPUT_ARG}"
  fi
fi

if [ -z "${SUITE_ROOT_INPUT}" ] && [ -z "${GRID_OBJ_DATA_JSON}" ]; then
  echo "Usage:"
  echo "  SUITE_ROOT=/path/to/grid_obj_ablation_suite bash flow_planner/run_script/render_grid_obj_sample_eff.sh"
  echo "or"
  echo "  GRID_OBJ_DATA_JSON=/path/to/grid_obj_sample_eff_data.json bash flow_planner/run_script/render_grid_obj_sample_eff.sh"
  echo "or"
  echo "  bash flow_planner/run_script/render_grid_obj_sample_eff.sh /path/to/grid_obj_ablation_suite"
  echo "or"
  echo "  bash flow_planner/run_script/render_grid_obj_sample_eff.sh /path/to/grid_obj_sample_eff_data.json"
  exit 1
fi

: "${PROJECT_ROOT:=/home/code/grid_planner}"
: "${PRIMARY_SPLIT:=val14}"
: "${PLOT_VARIANTS:=unified_objective,no_advantage_reweighting,no_opsd_term}"
: "${OUTPUT_DATA_JSON:=}"
: "${OUTPUT_DATA_CSV:=}"

if [ -z "${FIG_OUT_DIR:-}" ]; then
  if [ -n "${SUITE_ROOT_INPUT}" ]; then
    FIG_OUT_DIR="${SUITE_ROOT_INPUT%/}/figures"
  else
    FIG_OUT_DIR="$(cd "$(dirname "${GRID_OBJ_DATA_JSON}")" && pwd)"
  fi
fi

mkdir -p "${FIG_OUT_DIR}"

cmd=(
  python "${PROJECT_ROOT}/flow_planner/onpolicy/grid_obj_sample_eff_plot.py"
  --primary-split "${PRIMARY_SPLIT}"
  --variants "${PLOT_VARIANTS}"
  --output-png "${FIG_OUT_DIR}/grid_obj_sample_eff.png"
  --output-svg "${FIG_OUT_DIR}/grid_obj_sample_eff.svg"
)

if [ -n "${SUITE_ROOT_INPUT}" ]; then
  cmd+=(--suite-root "${SUITE_ROOT_INPUT%/}")
  if [ -z "${OUTPUT_DATA_JSON}" ]; then
    OUTPUT_DATA_JSON="${FIG_OUT_DIR}/grid_obj_sample_eff_data.json"
  fi
  if [ -z "${OUTPUT_DATA_CSV}" ]; then
    OUTPUT_DATA_CSV="${FIG_OUT_DIR}/grid_obj_sample_eff_data.csv"
  fi
else
  cmd+=(--input-json "${GRID_OBJ_DATA_JSON}")
fi

if [ -n "${OUTPUT_DATA_JSON}" ]; then
  cmd+=(--output-data-json "${OUTPUT_DATA_JSON}")
fi
if [ -n "${OUTPUT_DATA_CSV}" ]; then
  cmd+=(--output-data-csv "${OUTPUT_DATA_CSV}")
fi

"${cmd[@]}"
