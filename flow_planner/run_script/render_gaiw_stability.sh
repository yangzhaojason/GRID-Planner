#!/usr/bin/env bash
set -euo pipefail

SUITE_ROOT_INPUT="${1:-${SUITE_ROOT:-}}"
if [ -z "${SUITE_ROOT_INPUT}" ]; then
  echo "Usage:"
  echo "  SUITE_ROOT=/path/to/gaiw_ablation_suite bash flow_planner/run_script/render_gaiw_stability.sh"
  echo "or"
  echo "  bash flow_planner/run_script/render_gaiw_stability.sh /path/to/gaiw_ablation_suite"
  exit 1
fi

: "${PROJECT_ROOT:=/home/code/grid_planner}"
: "${FIG_OUT_DIR:=${SUITE_ROOT_INPUT%/}/figures}"

mkdir -p "${FIG_OUT_DIR}"

python "${PROJECT_ROOT}/flow_planner/onpolicy/gaiw_stability_plot.py" \
  --suite-root "${SUITE_ROOT_INPUT%/}" \
  --output-png "${FIG_OUT_DIR}/gaiw_stability.png" \
  --output-svg "${FIG_OUT_DIR}/gaiw_stability.svg"
