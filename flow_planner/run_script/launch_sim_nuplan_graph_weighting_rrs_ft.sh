#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

: "${BRANCH_NAME:=flow_planner_graph_weighting_rrs_ft}"
grid_run_single_simulation "$@"
