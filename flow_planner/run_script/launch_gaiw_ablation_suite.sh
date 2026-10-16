#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# End-to-end GAIW ablation pipeline:
#   1) train each variant
#   2) evaluate every saved checkpoint on nuPlan
#   3) export structured benchmark rows
#   4) aggregate a paper-ready table
#   5) render the GAIW stability figure

: "${PRETRAINED_CKPT:=}"

: "${TRAIN_EPOCHS:=10}"
: "${TRAIN_SAVE_UTD:=1}"
: "${LR:=5e-5}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=}"
: "${NCCL_SOCKET_IFNAME:=lo}"
: "${GLOO_SOCKET_IFNAME:=lo}"
: "${NCCL_IB_DISABLE:=1}"
: "${NCCL_DEBUG:=WARN}"
: "${TORCH_NCCL_ASYNC_ERROR_HANDLING:=1}"

: "${SPLIT:=val14}"
: "${CHALLENGE:=closed_loop_reactive_agents}"
: "${WORKER_MODE:=ray_distributed}"
: "${PLANNER:=flow_planner}"
: "${WORKER_THREADS_PER_NODE:=64}"
: "${GPU_PER_SIM:=0.15}"
: "${RAY_STOP_BEFORE_EACH_RUN:=false}"
: "${RAY_STOP_WAIT_SEC:=3}"
: "${ISOLATE_RAY_TMPDIR_PER_CKPT:=true}"
: "${CLEAN_RAY_TMPDIR_AFTER_RUN:=true}"
: "${DELETE_RESULT_DIR_AFTER_PARSE:=false}"
: "${RAY_TMPDIR_ROOT:=${RAY_TMPDIR}}"
: "${RAY_TMPDIR_ALIAS_ROOT:=/tmp}"
: "${LATENCY_MODE:=compute_trajectory_only}"

: "${RUN_STAMP:=$(date +%Y-%m-%d_%H-%M-%S)}"
: "${SUITE_ROOT:=${SAVE_DIR}/gaiw_ablation_suite/${RUN_STAMP}}"
: "${VARIANTS:=no_weighting,centrality_only,risk_only,full,full_no_clipping,full_no_batch_norm}"

: "${RUN_TRAIN:=true}"
: "${RUN_EVAL:=true}"
: "${RUN_SUMMARY:=true}"
: "${RUN_PLOT:=true}"

SUITE_RUNS_DIR="${SUITE_ROOT}/runs"
SUITE_EVAL_DIR="${SUITE_ROOT}/eval"
SUITE_SUMMARY_DIR="${SUITE_ROOT}/summary"
SUITE_FIG_DIR="${SUITE_ROOT}/figures"

mkdir -p "${SUITE_RUNS_DIR}" "${SUITE_EVAL_DIR}" "${SUITE_SUMMARY_DIR}" "${SUITE_FIG_DIR}"

if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "PROJECT_ROOT not found: ${PROJECT_ROOT}"
  exit 1
fi
if [ ! -d "${NUPLAN_DEVKIT_ROOT}" ]; then
  echo "NUPLAN_DEVKIT_ROOT not found: ${NUPLAN_DEVKIT_ROOT}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES
export HYDRA_FULL_ERROR
export PROJECT_ROOT
export SAVE_DIR
export TRAINING_DATA
export TRAINING_JSON
export NUPLAN_DATA_ROOT
export NUPLAN_MAPS_ROOT
export NUPLAN_EXP_ROOT
export MASTER_ADDR
export NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME
export NCCL_IB_DISABLE
export NCCL_DEBUG
export TORCH_NCCL_ASYNC_ERROR_HANDLING
export TORCH_LOGS="${TORCH_LOGS:-dynamic,recompiles}"
mkdir -p "${NUPLAN_EXP_ROOT}"
export RAY_TMPDIR="${RAY_TMPDIR_ROOT}"
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_INCLUDE_DASHBOARD=false
export RAY_ENABLE_METRICS_AGENT=false
export RAY_METRICS_EXPORT_PORT=0
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE="${RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE:-1}"
if [ -d /dev/shm ]; then
  SHM_AVAIL_BYTES="$("${PYTHON_BIN}" - <<'PY'
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
mkdir -p "${RAY_TMPDIR_ROOT}"
mkdir -p "${RAY_TMPDIR_ALIAS_ROOT}"

if [ "${SPLIT}" = "val14" ]; then
  SCENARIO_BUILDER="nuplan"
else
  SCENARIO_BUILDER="nuplan_challenge"
fi

pick_free_port() {
  "${PYTHON_BIN}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
}

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
}

make_private_ray_tmpdir_pair() {
  "${PYTHON_BIN}" - "${RAY_TMPDIR_ROOT}" "${RAY_TMPDIR_ALIAS_ROOT}" "$1" <<'PY'
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
  "${PYTHON_BIN}" - "$1" <<'PY'
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

variant_overrides() {
  case "$1" in
    no_weighting)
      cat <<'EOF'
core.graph_weighting.enabled=false
EOF
      ;;
    centrality_only)
      cat <<'EOF'
core.graph_weighting.enabled=true
core.graph_weighting.alpha1=0.1
core.graph_weighting.alpha2=0.0
core.graph_weighting.enable_clipping=true
core.graph_weighting.normalize_by_mean=true
EOF
      ;;
    risk_only)
      cat <<'EOF'
core.graph_weighting.enabled=true
core.graph_weighting.alpha1=0.0
core.graph_weighting.alpha2=1.0
core.graph_weighting.enable_clipping=true
core.graph_weighting.normalize_by_mean=true
EOF
      ;;
    full)
      cat <<'EOF'
core.graph_weighting.enabled=true
core.graph_weighting.alpha1=0.1
core.graph_weighting.alpha2=1.0
core.graph_weighting.enable_clipping=true
core.graph_weighting.normalize_by_mean=true
EOF
      ;;
    full_no_clipping)
      cat <<'EOF'
core.graph_weighting.enabled=true
core.graph_weighting.alpha1=0.1
core.graph_weighting.alpha2=1.0
core.graph_weighting.enable_clipping=false
core.graph_weighting.normalize_by_mean=true
EOF
      ;;
    full_no_batch_norm)
      cat <<'EOF'
core.graph_weighting.enabled=true
core.graph_weighting.alpha1=0.1
core.graph_weighting.alpha2=1.0
core.graph_weighting.enable_clipping=true
core.graph_weighting.normalize_by_mean=false
EOF
      ;;
    *)
      echo "Unknown variant: $1" >&2
      return 1
      ;;
  esac
}

collect_ckpt_names() {
  "${PYTHON_BIN}" - "$1" <<'PY'
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])

def sort_key(path: Path):
    name = path.name
    if name.startswith("model_epoch_"):
        try:
            return (0, int(name.split("_")[2]), name)
        except Exception:
            return (1, name)
    return (1, name)

for path in sorted(run_dir.glob("model_epoch_*.pth"), key=sort_key):
    if path.is_file():
        print(path.name)
PY
}

train_variant() {
  grid_require_file PRETRAINED_CKPT "${PRETRAINED_CKPT}"
  local variant="$1"
  local run_dir="${SUITE_RUNS_DIR}/${variant}"
  local tb_dir="${run_dir}/tensorboard"

  mkdir -p "${run_dir}" "${tb_dir}"
  export TENSORBOARD_LOG_PATH="${tb_dir}"
  local train_master_port="${MASTER_PORT}"
  if [ -z "${train_master_port}" ]; then
    train_master_port="$(pick_free_port)"
  fi

  local -a train_cmd=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nnodes 1
    --nproc-per-node "${NPROC_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${train_master_port}"
    -m flow_planner.trainer
    --config-name flow_planner_standard
    "pretrained_checkpoint=${PRETRAINED_CKPT}"
    "save_dir=${run_dir}"
    "job_name=flow_planner_gaiw_${variant}"
    "train.epoch=${TRAIN_EPOCHS}"
    "train.save_utd=${TRAIN_SAVE_UTD}"
    "optimizer.lr=${LR}"
  )

  while IFS= read -r line; do
    [ -n "${line}" ] && train_cmd+=("${line}")
  done < <(variant_overrides "${variant}")

  echo "[TRAIN] ${variant}"
  echo "  run_dir=${run_dir}"
  echo "  tensorboard=${tb_dir}"
  echo "  master_addr=${MASTER_ADDR}"
  echo "  master_port=${train_master_port}"
  (
    export MASTER_PORT="${train_master_port}"
    cd "${PROJECT_ROOT}"
    "${train_cmd[@]}"
  ) 2>&1 | tee "${run_dir}/train.stdout.log"
}

append_eval_row() {
  "${PYTHON_BIN}" - "$@" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

(
    variant,
    checkpoint,
    result_dir,
    json_path,
    jsonl_path,
    csv_path,
    status,
) = sys.argv[1:]

epoch = None
m = re.search(r"model_epoch_(\d+)", checkpoint)
if m:
    epoch = int(m.group(1))

payload = {
    "variant_id": variant,
    "checkpoint": checkpoint,
    "epoch": epoch,
    "result_dir": result_dir,
    "status": status,
}

if status == "OK":
    with open(json_path, "r") as f:
        payload.update(json.load(f))
    payload["json_path"] = json_path
else:
    payload["json_path"] = None

with open(jsonl_path, "a") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

fieldnames = [
    "variant_id",
    "checkpoint",
    "epoch",
    "status",
    "score_up",
    "collision_score_loss",
    "coll_down",
    "red_light_down",
    "driving_compliance_loss",
    "lane_dev_down",
    "comfort_failure_rate",
    "jerk_down",
    "latency_down_ms",
    "result_dir",
    "report_path",
    "json_path",
]
row = {k: payload.get(k) for k in fieldnames}
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerow(row)
PY
}

evaluate_variant() {
  local variant="$1"
  local run_dir="${SUITE_RUNS_DIR}/${variant}"
  local eval_dir="${SUITE_EVAL_DIR}/${variant}"
  local config_file="${run_dir}/.hydra/config.yaml"
  local jsonl_path="${eval_dir}/benchmark_per_epoch.jsonl"
  local csv_path="${eval_dir}/benchmark_per_epoch.csv"

  mkdir -p "${eval_dir}/logs" "${eval_dir}/json"

  if [ ! -f "${config_file}" ]; then
    echo "Config not found for variant ${variant}: ${config_file}"
    exit 1
  fi

  : > "${jsonl_path}"
  "${PYTHON_BIN}" - "${csv_path}" <<'PY'
import csv
import sys

fieldnames = [
    "variant_id",
    "checkpoint",
    "epoch",
    "status",
    "score_up",
    "collision_score_loss",
    "coll_down",
    "red_light_down",
    "driving_compliance_loss",
    "lane_dev_down",
    "comfort_failure_rate",
    "jerk_down",
    "latency_down_ms",
    "result_dir",
    "report_path",
    "json_path",
]
with open(sys.argv[1], "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
PY

  ckpt_names=()
  while IFS= read -r ckpt_name; do
    ckpt_names+=("${ckpt_name}")
  done < <(collect_ckpt_names "${run_dir}")
  if [ "${#ckpt_names[@]}" -eq 0 ]; then
    echo "No checkpoints found for variant ${variant} under ${run_dir}"
    exit 1
  fi

  echo "[EVAL] ${variant}"
  echo "  worker_mode=${WORKER_MODE}"
  echo "  ray_stop_before_each_run=${RAY_STOP_BEFORE_EACH_RUN}"
  echo "  isolate_ray_tmpdir_per_ckpt=${ISOLATE_RAY_TMPDIR_PER_CKPT}"
  echo "  clean_ray_tmpdir_after_run=${CLEAN_RAY_TMPDIR_AFTER_RUN}"
  echo "  delete_result_dir_after_parse=${DELETE_RESULT_DIR_AFTER_PARSE}"
  for ckpt_name in "${ckpt_names[@]}"; do
    local ckpt_file="${run_dir}/${ckpt_name}"
    local ckpt_tag="${ckpt_name%.pth}"
    local run_ts
    run_ts="$(date +%Y-%m-%d-%H-%M-%S)"
    local exp_uid="${PLANNER}/${SPLIT}/gaiw_${variant}/${ckpt_tag}_${run_ts}"
    local result_dir="${NUPLAN_EXP_ROOT}/exp/simulation/${CHALLENGE}/${exp_uid}"
    local log_file="${eval_dir}/logs/${ckpt_tag}.log"
    local json_path="${eval_dir}/json/${ckpt_tag}.json"
    local ray_tmpdir_for_run="${RAY_TMPDIR}"
    local private_ray_alias=""
    local private_ray_real=""

    echo "  checkpoint=${ckpt_name}"
    if [ "${WORKER_MODE}" = "ray_distributed" ] && [ "${ISOLATE_RAY_TMPDIR_PER_CKPT}" = "true" ]; then
      local ray_pair
      ray_pair="$(make_private_ray_tmpdir_pair "${variant}_${ckpt_tag}_${run_ts}")"
      private_ray_alias="${ray_pair%%|*}"
      private_ray_real="${ray_pair#*|}"
      ray_tmpdir_for_run="${private_ray_alias}"
    fi
    if [ "${WORKER_MODE}" = "ray_distributed" ]; then
      assert_ray_socket_path_safe "${ray_tmpdir_for_run}"
      echo "    ray_tmpdir=${ray_tmpdir_for_run}"
    fi
    stop_ray_if_requested

    SIM_ARGS=(
      "+simulation=${CHALLENGE}"
      "planner=${PLANNER}"
      "planner.flow_planner.config_path=${config_file}"
      "planner.flow_planner.ckpt_path=${ckpt_file}"
      "scenario_builder=${SCENARIO_BUILDER}"
      "scenario_filter=${SPLIT}"
      "experiment_uid=${exp_uid}"
      "verbose=true"
      "worker=${WORKER_MODE}"
      "enable_simulation_progress_bar=true"
      'hydra.searchpath=[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]'
    )
    if [ "${WORKER_MODE}" = "ray_distributed" ]; then
      SIM_ARGS+=(
        "worker.threads_per_node=${WORKER_THREADS_PER_NODE}"
        "distributed_mode=SINGLE_NODE"
        "number_of_gpus_allocated_per_simulation=${GPU_PER_SIM}"
      )
    fi

    if RAY_TMPDIR="${ray_tmpdir_for_run}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${NUPLAN_DEVKIT_ROOT}/nuplan/planning/script/run_simulation.py" \
      "${SIM_ARGS[@]}" \
      2>&1 | tee "${log_file}"; then
      if "${PYTHON_BIN}" "${PROJECT_ROOT}/flow_planner/onpolicy/benchmark_table.py" \
        --report "${result_dir}" \
        --variant "${variant}" \
        --latency-mode "${LATENCY_MODE}" \
        --json-only \
        --output-json "${json_path}" >/dev/null; then
        append_eval_row "${variant}" "${ckpt_name}" "${result_dir}" "${json_path}" "${jsonl_path}" "${csv_path}" "OK"
        delete_result_dir_if_requested "${result_dir}"
      else
        append_eval_row "${variant}" "${ckpt_name}" "${result_dir}" "${json_path}" "${jsonl_path}" "${csv_path}" "PARSE_FAIL"
      fi
    else
      append_eval_row "${variant}" "${ckpt_name}" "${result_dir}" "${json_path}" "${jsonl_path}" "${csv_path}" "SIM_FAIL"
    fi
    cleanup_private_ray_tmpdir "${private_ray_alias}" "${private_ray_real}"
  done
}

IFS=',' read -r -a selected_variants <<< "${VARIANTS}"

for variant in "${selected_variants[@]}"; do
  variant="${variant// /}"
  [ -z "${variant}" ] && continue
  variant_overrides "${variant}" >/dev/null
  if [ "${RUN_TRAIN}" = "true" ]; then
    train_variant "${variant}"
  fi
  if [ "${RUN_EVAL}" = "true" ]; then
    evaluate_variant "${variant}"
  fi
done

if [ "${RUN_SUMMARY}" = "true" ]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/flow_planner/onpolicy/gaiw_ablation_collect.py" \
    --suite-root "${SUITE_ROOT}" \
    --output-dir "${SUITE_SUMMARY_DIR}"
fi

if [ "${RUN_PLOT}" = "true" ]; then
  bash "${PROJECT_ROOT}/flow_planner/run_script/render_gaiw_stability.sh" "${SUITE_ROOT}"
fi

echo "GAIW ablation suite artifacts saved to: ${SUITE_ROOT}"
