#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

grid_require_file PRETRAINED_CKPT "${PRETRAINED_CKPT:-}"
export TORCH_LOGS="${TORCH_LOGS:-dynamic,recompiles}"
cd "${PROJECT_ROOT}"

# Optional ablation save root (overrides hydra save_dir directly)
# Example:
#   ABLATION_ROOT=${SAVE_DIR}/outputs/FlowPlannerTraining/flow_planner_standard/ablation_rrs \
#   EXP_TAG=runA bash flow_planner/run_script/launch_train_graph_weighting_rrs_ft.sh
: "${ABLATION_ROOT:=${SAVE_DIR}/outputs/FlowPlannerTraining/flow_planner_standard}"
: "${EXP_TAG:=rrs_ft}"
: "${RUN_STAMP:=$(date +%Y-%m-%d_%H-%M-%S)}"
TARGET_SAVE_DIR="${ABLATION_ROOT}/${EXP_TAG}/${RUN_STAMP}"
mkdir -p "${TARGET_SAVE_DIR}"
echo "Training outputs will be saved to: ${TARGET_SAVE_DIR}"

# Safety-only RRS regularization preset (see docs/reproducibility.md):
#   L = L_flow_matching + nu * E[ w_GAIW(tau) * RRS_penalty(tau_hat) ]
# where RRS_reward = -RRS_penalty (logged as rrs_reward_mean)

"${PYTHON_BIN}" -m torch.distributed.run --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" --standalone -m flow_planner.trainer \
  --config-name flow_planner_standard \
  "pretrained_checkpoint=${PRETRAINED_CKPT}" \
  "save_dir=${TARGET_SAVE_DIR}" \
  "job_name=flow_planner_${EXP_TAG}" \
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

