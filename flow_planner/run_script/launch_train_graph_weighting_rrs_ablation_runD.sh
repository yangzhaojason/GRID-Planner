#!/usr/bin/env bash

# Run D: rule only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${ABLATION_ROOT:=/home/results/yangzhao/grid_planner/outputs/FlowPlannerTraining/flow_planner_standard/ablation_rrs}"
: "${EXP_TAG:=runD_rule_only}"

ABLATION_ROOT="${ABLATION_ROOT}" EXP_TAG="${EXP_TAG}" \
bash "${SCRIPT_DIR}/launch_train_graph_weighting_rrs_ft.sh" \
  train.epoch=3 \
  scheduler.warm_up_epoch=1 \
  core.rrs.nu=0.001 \
  core.rrs.nu_warmup_epochs=3 \
  core.rrs.lam_col=0 core.rrs.lam_d=0 core.rrs.lam_ttc=0 \
  core.rrs.lam_lane=0.2 core.rrs.lam_red=0.1 core.rrs.lam_spd=0.1 \
  core.rrs.lam_comfort=0 core.rrs.lam_progress=0 core.rrs.lam_low_speed=0

