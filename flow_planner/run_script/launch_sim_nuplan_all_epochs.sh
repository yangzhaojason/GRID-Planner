#!/usr/bin/env bash
set -euo pipefail

# Batch-evaluate all saved epoch checkpoints under a training run directory.
# Usage:
#   RUN_DIR=/home/results/.../2026-03-16_01-49-59 \
#   bash flow_planner/run_script/launch_sim_nuplan_all_epochs.sh
#
# Or:
#   bash flow_planner/run_script/launch_sim_nuplan_all_epochs.sh /home/results/.../2026-03-16_01-49-59

: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES
export HYDRA_FULL_ERROR=1

: "${PROJECT_ROOT:=/home/code/grid_planner}"
: "${NUPLAN_DEVKIT_ROOT:=/home/code/grid_planner/nuplan-devkit}"
: "${NUPLAN_DATA_ROOT:=/home/datasets_2/nuplan/}"
: "${NUPLAN_MAPS_ROOT:=/home/datasets_2/nuplan/nuPlan-v1.1/maps}"
: "${NUPLAN_EXP_ROOT:=/home/datasets_2/nuplan/nuPlan-v1.1/exp}"

: "${SPLIT:=val14}"
: "${CHALLENGE:=closed_loop_reactive_agents}"
: "${PLANNER:=flow_planner}"
: "${WORKER_THREADS_PER_NODE:=64}"
: "${GPU_PER_SIM:=0.15}"
: "${CKPT_GLOB:=model_epoch_*.pth}"
: "${INCLUDE_LATEST:=false}"

RUN_DIR_INPUT="${1:-${RUN_DIR:-}}"
if [ -z "${RUN_DIR_INPUT}" ]; then
  echo "Usage:"
  echo "  RUN_DIR=/path/to/run_dir bash flow_planner/run_script/launch_sim_nuplan_all_epochs.sh"
  echo "or"
  echo "  bash flow_planner/run_script/launch_sim_nuplan_all_epochs.sh /path/to/run_dir"
  exit 1
fi

RUN_DIR="${RUN_DIR_INPUT%/}"
if [ -z "${BRANCH_NAME:-}" ]; then
  BRANCH_NAME="$(basename "${RUN_DIR}")"
fi

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "PROJECT_ROOT not found: ${PROJECT_ROOT}"
  exit 1
fi
if [ ! -d "${NUPLAN_DEVKIT_ROOT}" ]; then
  echo "NUPLAN_DEVKIT_ROOT not found: ${NUPLAN_DEVKIT_ROOT}"
  exit 1
fi
if [ ! -d "${NUPLAN_DATA_ROOT}" ]; then
  echo "NUPLAN_DATA_ROOT not found: ${NUPLAN_DATA_ROOT}"
  exit 1
fi
if [ ! -d "${NUPLAN_MAPS_ROOT}" ]; then
  echo "NUPLAN_MAPS_ROOT not found: ${NUPLAN_MAPS_ROOT}"
  exit 1
fi
if [ ! -d "${NUPLAN_EXP_ROOT}" ]; then
  echo "NUPLAN_EXP_ROOT not found: ${NUPLAN_EXP_ROOT}"
  exit 1
fi
if [ ! -d "${RUN_DIR}" ]; then
  echo "RUN_DIR not found: ${RUN_DIR}"
  exit 1
fi

CONFIG_FILE="${RUN_DIR}/.hydra/config.yaml"
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config not found: ${CONFIG_FILE}"
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export RAY_TMPDIR="${RAY_TMPDIR:-/home/datasets_2/ray_tmp}"
mkdir -p "${RAY_TMPDIR}"
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_INCLUDE_DASHBOARD=false
export RAY_ENABLE_METRICS_AGENT=false
export RAY_METRICS_EXPORT_PORT=0

if [ "${SPLIT}" = "val14" ]; then
  SCENARIO_BUILDER="nuplan"
else
  SCENARIO_BUILDER="nuplan_challenge"
fi

cd "${PROJECT_ROOT}"

mapfile -t CKPT_NAMES < <(
  python - "${RUN_DIR}" "${CKPT_GLOB}" "${INCLUDE_LATEST}" <<'PY'
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
pattern = sys.argv[2]
include_latest = sys.argv[3].lower() == "true"

def sort_key(path: Path):
    name = path.name
    if name.startswith("model_epoch_"):
        try:
            return (0, int(name.split("_")[2]), name)
        except Exception:
            return (1, name)
    return (1, name)

paths = sorted([p for p in run_dir.glob(pattern) if p.is_file()], key=sort_key)
names = [p.name for p in paths]

latest = run_dir / "latest.pth"
if include_latest and latest.is_file() and latest.name not in names:
    names.append(latest.name)

for name in names:
    print(name)
PY
)

if [ "${#CKPT_NAMES[@]}" -eq 0 ]; then
  echo "No checkpoints matched under ${RUN_DIR}"
  echo "CKPT_GLOB=${CKPT_GLOB}"
  exit 1
fi

EVAL_ARTIFACT_DIR="${RUN_DIR}/eval_${SPLIT}_${CHALLENGE}_$(date +%Y-%m-%d_%H-%M-%S)"
EVAL_LOG_DIR="${EVAL_ARTIFACT_DIR}/logs"
SUMMARY_CSV="${EVAL_ARTIFACT_DIR}/final_scores.csv"
mkdir -p "${EVAL_LOG_DIR}"
echo "checkpoint,score,status,result_dir" > "${SUMMARY_CSV}"

echo "Batch evaluation settings:"
echo "  RUN_DIR=${RUN_DIR}"
echo "  CONFIG_FILE=${CONFIG_FILE}"
echo "  SPLIT=${SPLIT}"
echo "  CHALLENGE=${CHALLENGE}"
echo "  SCENARIO_BUILDER=${SCENARIO_BUILDER}"
echo "  BRANCH_NAME=${BRANCH_NAME}"
echo "  CKPT_GLOB=${CKPT_GLOB}"
echo "  INCLUDE_LATEST=${INCLUDE_LATEST}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  Summary CSV will be saved to: ${SUMMARY_CSV}"

OVERALL_STATUS=0

for CKPT_NAME in "${CKPT_NAMES[@]}"; do
  CKPT_FILE="${RUN_DIR}/${CKPT_NAME}"
  CKPT_TAG="${CKPT_NAME%.pth}"
  RUN_TS="$(date +%Y-%m-%d-%H-%M-%S)"
  EXP_UID="${PLANNER}/${SPLIT}/${BRANCH_NAME}/${CKPT_TAG}_${RUN_TS}"
  RESULT_DIR="${NUPLAN_EXP_ROOT}/exp/simulation/${CHALLENGE}/${EXP_UID}"
  LOG_FILE="${EVAL_LOG_DIR}/${CKPT_TAG}.log"

  echo ""
  echo "=================================================="
  echo "Evaluating checkpoint: ${CKPT_FILE}"
  echo "Result dir: ${RESULT_DIR}"
  echo "Log file: ${LOG_FILE}"

  if python "${NUPLAN_DEVKIT_ROOT}/nuplan/planning/script/run_simulation.py" \
      +simulation="${CHALLENGE}" \
      planner="${PLANNER}" \
      planner.flow_planner.config_path="${CONFIG_FILE}" \
      planner.flow_planner.ckpt_path="${CKPT_FILE}" \
      scenario_builder="${SCENARIO_BUILDER}" \
      scenario_filter="${SPLIT}" \
      experiment_uid="${EXP_UID}" \
      verbose=true \
      worker=ray_distributed \
      worker.threads_per_node="${WORKER_THREADS_PER_NODE}" \
      distributed_mode='SINGLE_NODE' \
      number_of_gpus_allocated_per_simulation="${GPU_PER_SIM}" \
      enable_simulation_progress_bar=true \
      hydra.searchpath="[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
      2>&1 | tee "${LOG_FILE}"; then

    if SCORE="$(python - "${RESULT_DIR}/runner_report.parquet" <<'PY'
import glob
import os
import sys
import pandas as pd

runner_report = sys.argv[1]
if not os.path.exists(runner_report):
    raise FileNotFoundError(f"runner_report not found: {runner_report}")

aggregator_dir = os.path.join(os.path.dirname(runner_report), "aggregator_metric")
candidates = sorted(
    glob.glob(os.path.join(aggregator_dir, "*weighted_average_metrics*.parquet")),
    key=os.path.getmtime,
    reverse=True,
)
metrics_path = candidates[0] if candidates else runner_report

df = pd.read_parquet(metrics_path)
final_score = df[df["scenario"] == "final_score"]
if final_score.empty:
    raise RuntimeError(f"Missing final_score row in: {metrics_path}")

print(f"{float(final_score.iloc[0]['score']):.10f}")
PY
    )"; then
      echo "[OK] ${CKPT_NAME}: score=${SCORE}"
      echo "${CKPT_NAME},${SCORE},OK,${RESULT_DIR}" >> "${SUMMARY_CSV}"
    else
      echo "[PARSE_FAIL] ${CKPT_NAME}"
      echo "${CKPT_NAME},,PARSE_FAIL,${RESULT_DIR}" >> "${SUMMARY_CSV}"
      OVERALL_STATUS=1
    fi
  else
    echo "[SIM_FAIL] ${CKPT_NAME}"
    echo "${CKPT_NAME},,SIM_FAIL,${RESULT_DIR}" >> "${SUMMARY_CSV}"
    OVERALL_STATUS=1
  fi
done

echo ""
python - "${SUMMARY_CSV}" <<'PY'
import csv
import sys

summary_csv = sys.argv[1]
rows = list(csv.DictReader(open(summary_csv, "r", newline="")))

print("========== FINAL SCORE SUMMARY ==========")
best_ckpt = None
best_score = None

for row in rows:
    ckpt = row["checkpoint"]
    score = row["score"].strip()
    status = row["status"].strip()
    if score:
        score_f = float(score)
        print(f"{ckpt}: {score_f:.10f}")
        if best_score is None or score_f > best_score:
            best_score = score_f
            best_ckpt = ckpt
    else:
        print(f"{ckpt}: {status}")

if best_ckpt is not None:
    print("-----------------------------------------")
    print(f"BEST_CHECKPOINT={best_ckpt}")
    print(f"BEST_SCORE={best_score:.10f}")
PY

echo "Summary saved to: ${SUMMARY_CSV}"

exit "${OVERALL_STATUS}"
