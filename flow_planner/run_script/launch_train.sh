#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

export TORCH_LOGS="${TORCH_LOGS:-dynamic,recompiles}"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" --standalone -m flow_planner.trainer \
  --config-name flow_planner_standard \
  "$@"
