#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

grid_require_file PRETRAINED_CKPT "${PRETRAINED_CKPT:-}"
export TORCH_LOGS="${TORCH_LOGS:-dynamic,recompiles}"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" --standalone -m flow_planner.trainer \
  --config-name flow_planner_standard \
  "pretrained_checkpoint=${PRETRAINED_CKPT}" \
  train.epoch=10 \
  train.save_utd=1 \
  core.graph_weighting.enabled=true \
  optimizer.lr=5e-5 \
  "$@"
