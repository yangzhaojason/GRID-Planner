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
: "${NUPLAN_EXP_ROOT:=/home/results/yangzhao/grid_planner}"

: "${SPLIT:=val14}"
: "${CHALLENGE:=closed_loop_reactive_agents}"
: "${PLANNER:=flow_planner}"
: "${WORKER_MODE:=ray_distributed}"
: "${WORKER_THREADS_PER_NODE:=64}"
: "${GPU_PER_SIM:=0.15}"
: "${RAY_STOP_BEFORE_EACH_RUN:=false}"
: "${RAY_STOP_WAIT_SEC:=3}"
: "${ISOLATE_RAY_TMPDIR_PER_CKPT:=true}"
: "${CLEAN_RAY_TMPDIR_AFTER_RUN:=true}"
: "${DELETE_RESULT_DIR_AFTER_PARSE:=false}"
: "${RAY_TMPDIR_ROOT:=${RAY_TMPDIR:-/home/results/ray}}"
: "${RAY_TMPDIR_ALIAS_ROOT:=/tmp}"
: "${CKPT_GLOB:=model_epoch_*.pth}"
: "${INCLUDE_LATEST:=false}"
: "${PRINT_TASK_SUBMIT_LOG:=true}"
: "${RESUME_EVAL:=false}"
: "${RESUME_SKIP_STATUSES:=OK}"
: "${EVAL_ARTIFACT_DIR:=}"

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
if [ ! -d "${RUN_DIR}" ]; then
  echo "RUN_DIR not found: ${RUN_DIR}"
  exit 1
fi

CONFIG_FILE="${RUN_DIR}/.hydra/config.yaml"
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config not found: ${CONFIG_FILE}"
  exit 1
fi

export PROJECT_ROOT
export NUPLAN_DEVKIT_ROOT
export NUPLAN_DATA_ROOT
export NUPLAN_MAPS_ROOT
export NUPLAN_EXP_ROOT
export SPLIT
export CHALLENGE
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${NUPLAN_EXP_ROOT}"
export RAY_TMPDIR="${RAY_TMPDIR_ROOT}"
mkdir -p "${RAY_TMPDIR_ROOT}"
mkdir -p "${RAY_TMPDIR_ALIAS_ROOT}"
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_INCLUDE_DASHBOARD=false
export RAY_ENABLE_METRICS_AGENT=false
export RAY_METRICS_EXPORT_PORT=0
# In containers /dev/shm is often only 64MB. Let Ray spill to disk-backed
# storage under RAY_TMPDIR instead of failing object-store startup outright.
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE="${RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE:-1}"
if [ -d /dev/shm ]; then
  SHM_AVAIL_BYTES="$(python - <<'PY'
import os
st = os.statvfs("/dev/shm")
print(st.f_bavail * st.f_frsize)
PY
)"
  if [ "${SHM_AVAIL_BYTES}" -lt 78643200 ]; then
    echo "Warning: /dev/shm is only ${SHM_AVAIL_BYTES} bytes; Ray may spill object-store data to disk via RAY_TMPDIR=${RAY_TMPDIR}."
    echo "For best stability/performance, restart the container with a larger --shm-size (for example 8g or 16g)."
  fi
fi

if [ "${SPLIT}" = "val14" ]; then
  SCENARIO_BUILDER="nuplan"
else
  SCENARIO_BUILDER="nuplan_challenge"
fi

cd "${PROJECT_ROOT}"

stop_ray_if_requested() {
  if [ "${WORKER_MODE}" != "ray_distributed" ]; then
    return 0
  fi
  if [ "${RAY_STOP_BEFORE_EACH_RUN}" != "true" ]; then
    return 0
  fi
  echo "Stopping any existing Ray runtime before evaluation..."
  ray stop --force >/dev/null 2>&1 || true
  sleep "${RAY_STOP_WAIT_SEC}"
  df -h /dev/shm | sed -n '1,2p'
}

make_private_ray_tmpdir_pair() {
  python - "${RAY_TMPDIR_ROOT}" "${RAY_TMPDIR_ALIAS_ROOT}" "$1" <<'PY'
from pathlib import Path
import hashlib
import sys
import uuid

root = Path(sys.argv[1])
alias_root = Path(sys.argv[2])
tag = sys.argv[3]
digest = hashlib.sha1(f"{tag}-{uuid.uuid4().hex}".encode("utf-8")).hexdigest()[:10]
real_dir = root / f"rt_{digest}"
alias_dir = alias_root / f"rt_{digest}"
real_dir.mkdir(parents=True, exist_ok=False)
alias_dir.symlink_to(real_dir, target_is_directory=True)
print(f"{alias_dir}|{real_dir}")
PY
}

assert_ray_socket_path_safe() {
  python - "$1" <<'PY'
import sys

base = sys.argv[1].rstrip("/")
suffix = "/ray/session_2099-12-31_23-59-59_123456_123456/sockets/plasma_store"
candidate = base + suffix
limit = 107
if len(candidate) > limit:
    print(
        f"RAY_TMPDIR path too long for AF_UNIX sockets ({len(candidate)} > {limit}): {candidate}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

cleanup_private_ray_tmpdir() {
  local alias_dir="${1:-}"
  local real_dir="${2:-}"
  if [ "${CLEAN_RAY_TMPDIR_AFTER_RUN}" != "true" ]; then
    return 0
  fi
  if [ -n "${alias_dir}" ] && [ -L "${alias_dir}" ]; then
    rm -f "${alias_dir}" || true
  fi
  if [ -n "${real_dir}" ] && [ -d "${real_dir}" ]; then
    rm -rf "${real_dir}" || true
  fi
}

delete_result_dir_if_requested() {
  local result_dir="${1:-}"
  if [ "${DELETE_RESULT_DIR_AFTER_PARSE}" != "true" ]; then
    return 0
  fi
  if [ -z "${result_dir}" ] || [ ! -d "${result_dir}" ]; then
    return 0
  fi
  case "${result_dir}" in
    "${NUPLAN_EXP_ROOT}"/exp/simulation/*)
      rm -rf "${result_dir}" || true
      ;;
  esac
}

print_task_submit_log() {
  local log_file="$1"
  local matched

  if [ "${PRINT_TASK_SUBMIT_LOG}" != "true" ]; then
    return 0
  fi
  if [ ! -f "${log_file}" ]; then
    echo "Worker submission log: log file not found: ${log_file}"
    return 0
  fi

  matched="$(python - "${log_file}" <<'PY'
import sys

path = sys.argv[1]
matches = []
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.rstrip()
        if "Submitting " in s and " tasks" in s:
            matches.append(s)

print("\n".join(matches[-5:]))
PY
)"

  if [ -n "${matched}" ]; then
    echo "Worker submission log(s):"
    printf '%s\n' "${matched}"
  else
    echo "Worker submission log: not found in ${log_file}"
  fi
}

choose_eval_artifact_dir() {
  if [ -n "${EVAL_ARTIFACT_DIR}" ]; then
    echo "${EVAL_ARTIFACT_DIR%/}"
    return 0
  fi

  if [ "${RESUME_EVAL}" = "true" ]; then
    python - "${RUN_DIR}" "${SPLIT}" "${CHALLENGE}" <<'PY'
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
split = sys.argv[2]
challenge = sys.argv[3]
pattern = f"eval_{split}_{challenge}_*"
candidates = sorted(
    [p for p in run_dir.glob(pattern) if p.is_dir()],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if candidates:
    print(str(candidates[0]))
PY
    return 0
  fi

  return 1
}

upsert_summary() {
  local checkpoint="$1"
  local score="$2"
  local status="$3"
  local result_dir="$4"

  python - "${SUMMARY_CSV}" "${SUMMARY_TXT}" "${checkpoint}" "${score}" "${status}" "${result_dir}" <<'PY'
import csv
import os
import sys

summary_csv, summary_txt, checkpoint, score, status, result_dir = sys.argv[1:]
fieldnames = ["checkpoint", "score", "status", "result_dir"]
rows = {}
order = []

if os.path.exists(summary_csv):
    with open(summary_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            ckpt = row.get("checkpoint", "").strip()
            if not ckpt:
                continue
            if ckpt not in rows:
                order.append(ckpt)
            rows[ckpt] = {
                "checkpoint": ckpt,
                "score": row.get("score", "").strip(),
                "status": row.get("status", "").strip(),
                "result_dir": row.get("result_dir", "").strip(),
            }

if checkpoint not in rows:
    order.append(checkpoint)
rows[checkpoint] = {
    "checkpoint": checkpoint,
    "score": score.strip(),
    "status": status.strip(),
    "result_dir": result_dir.strip(),
}

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for ckpt in order:
        writer.writerow(rows[ckpt])

best_ckpt = None
best_score = None
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("========== PARTIAL FINAL SCORE SUMMARY ==========\n")
    for ckpt in order:
        row = rows[ckpt]
        if row["score"]:
            score_f = float(row["score"])
            f.write(f"{ckpt}: {score_f:.10f}\n")
            if best_score is None or score_f > best_score:
                best_score = score_f
                best_ckpt = ckpt
        else:
            f.write(f"{ckpt}: {row['status']}\n")
    if best_ckpt is not None:
        f.write("-----------------------------------------\n")
        f.write(f"BEST_CHECKPOINT={best_ckpt}\n")
        f.write(f"BEST_SCORE={best_score:.10f}\n")
PY
}

get_checkpoint_status() {
  local checkpoint="$1"
  python - "${SUMMARY_CSV}" "${checkpoint}" <<'PY'
import csv
import os
import sys

summary_csv, checkpoint = sys.argv[1:]
if not os.path.exists(summary_csv):
    raise SystemExit(1)

with open(summary_csv, "r", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("checkpoint", "").strip() == checkpoint:
            status = row.get("status", "").strip()
            if status:
                print(status)
                raise SystemExit(0)
            raise SystemExit(1)

raise SystemExit(1)
PY
}

should_skip_checkpoint() {
  local checkpoint="$1"
  local status="${2:-}"
  python - "${RESUME_SKIP_STATUSES}" "${status}" <<'PY'
import sys

skip_statuses = {s.strip() for s in sys.argv[1].split(",") if s.strip()}
status = sys.argv[2].strip()
raise SystemExit(0 if status in skip_statuses else 1)
PY
}

print_resume_hint() {
  echo "Partial summary CSV: ${SUMMARY_CSV}"
  echo "Partial summary TXT: ${SUMMARY_TXT}"
  echo "Resume with:"
  echo "  RUN_DIR=\"${RUN_DIR}\" RESUME_EVAL=true EVAL_ARTIFACT_DIR=\"${EVAL_ARTIFACT_DIR}\" bash flow_planner/run_script/launch_sim_nuplan_all_epochs.sh"
}

handle_interrupt() {
  echo ""
  echo "Evaluation interrupted. Completed checkpoint scores were already saved."
  print_resume_hint
  exit 130
}

trap handle_interrupt INT TERM

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

EVAL_ARTIFACT_DIR="$(choose_eval_artifact_dir || true)"
if [ -z "${EVAL_ARTIFACT_DIR}" ]; then
  EVAL_ARTIFACT_DIR="${RUN_DIR}/eval_${SPLIT}_${CHALLENGE}_$(date +%Y-%m-%d_%H-%M-%S)"
fi
EVAL_LOG_DIR="${EVAL_ARTIFACT_DIR}/logs"
SUMMARY_CSV="${EVAL_ARTIFACT_DIR}/final_scores.csv"
SUMMARY_TXT="${EVAL_ARTIFACT_DIR}/final_scores.txt"
mkdir -p "${EVAL_LOG_DIR}"
if [ ! -f "${SUMMARY_CSV}" ]; then
  echo "checkpoint,score,status,result_dir" > "${SUMMARY_CSV}"
fi
if [ ! -f "${SUMMARY_TXT}" ]; then
  : > "${SUMMARY_TXT}"
fi

echo "Batch evaluation settings:"
echo "  RUN_DIR=${RUN_DIR}"
echo "  CONFIG_FILE=${CONFIG_FILE}"
echo "  SPLIT=${SPLIT}"
echo "  CHALLENGE=${CHALLENGE}"
echo "  SCENARIO_BUILDER=${SCENARIO_BUILDER}"
echo "  BRANCH_NAME=${BRANCH_NAME}"
echo "  WORKER_MODE=${WORKER_MODE}"
echo "  WORKER_THREADS_PER_NODE=${WORKER_THREADS_PER_NODE}"
echo "  GPU_PER_SIM=${GPU_PER_SIM}"
echo "  RAY_STOP_BEFORE_EACH_RUN=${RAY_STOP_BEFORE_EACH_RUN}"
echo "  ISOLATE_RAY_TMPDIR_PER_CKPT=${ISOLATE_RAY_TMPDIR_PER_CKPT}"
echo "  CLEAN_RAY_TMPDIR_AFTER_RUN=${CLEAN_RAY_TMPDIR_AFTER_RUN}"
echo "  DELETE_RESULT_DIR_AFTER_PARSE=${DELETE_RESULT_DIR_AFTER_PARSE}"
echo "  CKPT_GLOB=${CKPT_GLOB}"
echo "  INCLUDE_LATEST=${INCLUDE_LATEST}"
echo "  RESUME_EVAL=${RESUME_EVAL}"
echo "  RESUME_SKIP_STATUSES=${RESUME_SKIP_STATUSES}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  Summary CSV will be saved to: ${SUMMARY_CSV}"
echo "  Summary TXT will be saved to: ${SUMMARY_TXT}"

OVERALL_STATUS=0

for CKPT_NAME in "${CKPT_NAMES[@]}"; do
  CKPT_FILE="${RUN_DIR}/${CKPT_NAME}"
  CKPT_TAG="${CKPT_NAME%.pth}"

  if EXISTING_STATUS="$(get_checkpoint_status "${CKPT_NAME}" 2>/dev/null)"; then
    if should_skip_checkpoint "${CKPT_NAME}" "${EXISTING_STATUS}"; then
      echo ""
      echo "=================================================="
      echo "[SKIP] ${CKPT_NAME}: already recorded with status=${EXISTING_STATUS}"
      continue
    fi
  fi

  RUN_TS="$(date +%Y-%m-%d-%H-%M-%S)"
  EXP_UID="${PLANNER}/${SPLIT}/${BRANCH_NAME}/${CKPT_TAG}_${RUN_TS}"
  RESULT_DIR="${NUPLAN_EXP_ROOT}/exp/simulation/${CHALLENGE}/${EXP_UID}"
  LOG_FILE="${EVAL_LOG_DIR}/${CKPT_TAG}.log"
  RAY_TMPDIR_FOR_RUN="${RAY_TMPDIR}"
  PRIVATE_RAY_ALIAS=""
  PRIVATE_RAY_REAL=""

  echo ""
  echo "=================================================="
  echo "Evaluating checkpoint: ${CKPT_FILE}"
  echo "Result dir: ${RESULT_DIR}"
  echo "Log file: ${LOG_FILE}"
  if [ "${WORKER_MODE}" = "ray_distributed" ] && [ "${ISOLATE_RAY_TMPDIR_PER_CKPT}" = "true" ]; then
    RAY_PAIR="$(make_private_ray_tmpdir_pair "${BRANCH_NAME}_${CKPT_TAG}_${RUN_TS}")"
    PRIVATE_RAY_ALIAS="${RAY_PAIR%%|*}"
    PRIVATE_RAY_REAL="${RAY_PAIR#*|}"
    RAY_TMPDIR_FOR_RUN="${PRIVATE_RAY_ALIAS}"
  fi
  if [ "${WORKER_MODE}" = "ray_distributed" ]; then
    assert_ray_socket_path_safe "${RAY_TMPDIR_FOR_RUN}"
    echo "Ray tmpdir: ${RAY_TMPDIR_FOR_RUN}"
  fi

  stop_ray_if_requested

  SIM_ARGS=(
    +simulation="${CHALLENGE}"
    planner="${PLANNER}"
    planner.flow_planner.config_path="${CONFIG_FILE}"
    planner.flow_planner.ckpt_path="${CKPT_FILE}"
    scenario_builder="${SCENARIO_BUILDER}"
    scenario_filter="${SPLIT}"
    experiment_uid="${EXP_UID}"
    verbose=true
    worker="${WORKER_MODE}"
    enable_simulation_progress_bar=true
    hydra.searchpath="[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
  )
  if [ "${WORKER_MODE}" = "ray_distributed" ]; then
    SIM_ARGS+=(
      worker.threads_per_node="${WORKER_THREADS_PER_NODE}"
      distributed_mode='SINGLE_NODE'
      number_of_gpus_allocated_per_simulation="${GPU_PER_SIM}"
    )
  fi

  if RAY_TMPDIR="${RAY_TMPDIR_FOR_RUN}" PYTHONUNBUFFERED=1 python -u "${NUPLAN_DEVKIT_ROOT}/nuplan/planning/script/run_simulation.py" \
      "${SIM_ARGS[@]}" \
      2>&1 | tee "${LOG_FILE}"; then
    print_task_submit_log "${LOG_FILE}"

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
      upsert_summary "${CKPT_NAME}" "${SCORE}" "OK" "${RESULT_DIR}"
      delete_result_dir_if_requested "${RESULT_DIR}"
    else
      echo "[PARSE_FAIL] ${CKPT_NAME}"
      upsert_summary "${CKPT_NAME}" "" "PARSE_FAIL" "${RESULT_DIR}"
      OVERALL_STATUS=1
    fi
  else
    print_task_submit_log "${LOG_FILE}"
    echo "[SIM_FAIL] ${CKPT_NAME}"
    upsert_summary "${CKPT_NAME}" "" "SIM_FAIL" "${RESULT_DIR}"
    OVERALL_STATUS=1
  fi
  cleanup_private_ray_tmpdir "${PRIVATE_RAY_ALIAS}" "${PRIVATE_RAY_REAL}"
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
echo "Readable summary saved to: ${SUMMARY_TXT}"

exit "${OVERALL_STATUS}"
