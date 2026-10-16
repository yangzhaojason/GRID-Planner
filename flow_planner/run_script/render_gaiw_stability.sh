#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

INPUT_ARG="${1:-}"
SUITE_ROOT_INPUT="${SUITE_ROOT:-}"
: "${STABILITY_DATA_JSON:=}"
: "${STABILITY_DATA_CSV:=}"

if [ -n "${INPUT_ARG}" ]; then
  if [ -f "${INPUT_ARG}" ] && [ "${INPUT_ARG##*.}" = "json" ]; then
    STABILITY_DATA_JSON="${INPUT_ARG}"
  elif [ -f "${INPUT_ARG}" ] && [ "${INPUT_ARG##*.}" = "csv" ]; then
    STABILITY_DATA_CSV="${INPUT_ARG}"
  else
    SUITE_ROOT_INPUT="${INPUT_ARG}"
  fi
fi

if [ -z "${SUITE_ROOT_INPUT}" ] && [ -z "${STABILITY_DATA_JSON}" ] && [ -z "${STABILITY_DATA_CSV}" ]; then
  echo "Usage:"
  echo "  SUITE_ROOT=/path/to/gaiw_ablation_suite bash flow_planner/run_script/render_gaiw_stability.sh"
  echo "or"
  echo "  STABILITY_DATA_JSON=/path/to/gaiw_stability_data.json bash flow_planner/run_script/render_gaiw_stability.sh"
  echo "or"
  echo "  STABILITY_DATA_CSV=/path/to/gaiw_stability_data.csv bash flow_planner/run_script/render_gaiw_stability.sh"
  echo "or"
  echo "  bash flow_planner/run_script/render_gaiw_stability.sh /path/to/gaiw_ablation_suite"
  echo "or"
  echo "  bash flow_planner/run_script/render_gaiw_stability.sh /path/to/gaiw_stability_data.json"
  echo "or"
  echo "  bash flow_planner/run_script/render_gaiw_stability.sh /path/to/gaiw_stability_data.csv"
  exit 1
fi

: "${PLOT_MODE:=paper}"
: "${OUTPUT_DATA_JSON:=}"
: "${OUTPUT_DATA_CSV:=}"

if [ -z "${FIG_OUT_DIR:-}" ]; then
  if [ -n "${SUITE_ROOT_INPUT}" ]; then
    FIG_OUT_DIR="${SUITE_ROOT_INPUT%/}/figures"
  elif [ -n "${STABILITY_DATA_JSON}" ]; then
    FIG_OUT_DIR="$(cd "$(dirname "${STABILITY_DATA_JSON}")" && pwd)"
  else
    FIG_OUT_DIR="$(cd "$(dirname "${STABILITY_DATA_CSV}")" && pwd)"
  fi
fi

mkdir -p "${FIG_OUT_DIR}"

cmd=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/flow_planner/onpolicy/gaiw_stability_plot.py"
  --plot-mode "${PLOT_MODE}"
  --output-png "${FIG_OUT_DIR}/gaiw_stability.png"
  --output-svg "${FIG_OUT_DIR}/gaiw_stability.svg"
)

if [ -n "${SUITE_ROOT_INPUT}" ]; then
  cmd+=(--suite-root "${SUITE_ROOT_INPUT%/}")
  if [ -z "${OUTPUT_DATA_JSON}" ]; then
    OUTPUT_DATA_JSON="${FIG_OUT_DIR}/gaiw_stability_data.json"
  fi
  if [ -z "${OUTPUT_DATA_CSV}" ]; then
    OUTPUT_DATA_CSV="${FIG_OUT_DIR}/gaiw_stability_data.csv"
  fi
else
  if [ -n "${STABILITY_DATA_JSON}" ]; then
    cmd+=(--input-json "${STABILITY_DATA_JSON}")
  else
    cmd+=(--input-csv "${STABILITY_DATA_CSV}")
  fi
fi

if [ -n "${OUTPUT_DATA_JSON}" ]; then
  cmd+=(--output-data-json "${OUTPUT_DATA_JSON}")
fi
if [ -n "${OUTPUT_DATA_CSV}" ]; then
  cmd+=(--output-data-csv "${OUTPUT_DATA_CSV}")
fi

"${cmd[@]}"
