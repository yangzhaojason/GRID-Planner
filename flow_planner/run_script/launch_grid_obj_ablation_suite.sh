#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# End-to-end unified objective ablation pipeline:
#   1) resolve teacher checkpoints from a fixed teacher pool
#   2) train each variant
#   3) evaluate every saved checkpoint on PRIMARY_SPLIT
#   4) pick the best PRIMARY_SPLIT checkpoint and evaluate it on REPORT_SPLITS
#   5) aggregate the paper table and render the sample-efficiency figure

: "${BASE_RUN_DIR:=}"
: "${TEACHER_POOL_RUN_DIR:=}"
: "${PRETRAINED_CKPT:=}"

: "${BEST_TEACHER_CKPT:=}"
: "${RANDOM_TEACHER_CKPT:=}"
: "${TEACHER_CANDIDATE_MODE:=all}"         # all | latest | recent_k
: "${TEACHER_RECENT_K:=3}"
: "${TEACHER_RANDOM_SEED:=3402}"
: "${RANDOM_TEACHER_EXCLUDE_BEST:=true}"
: "${TEACHER_SIM_OUTPUT_ROOT:=${NUPLAN_EXP_ROOT}}"

: "${TRAIN_EPOCHS:=10}"
: "${TRAIN_SAVE_UTD:=1}"
: "${TRAIN_BATCH_SIZE:=2048}"
: "${SCHEDULER_WARMUP_EPOCHS:=1}"
: "${LR:=5e-5}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=}"
: "${NCCL_SOCKET_IFNAME:=lo}"
: "${GLOO_SOCKET_IFNAME:=lo}"
: "${NCCL_IB_DISABLE:=1}"
: "${NCCL_DEBUG:=WARN}"
: "${TORCH_NCCL_ASYNC_ERROR_HANDLING:=1}"

: "${PRIMARY_SPLIT:=val14}"
: "${REPORT_SPLITS:=val14,test14-hard,test14-random}"
: "${TEACHER_SELECT_SPLIT:=val14}"
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

: "${GRAPH_ALPHA1:=0.1}"
: "${GRAPH_ALPHA2:=1.0}"
: "${RRS_NU:=0.0}"
: "${RRS_USE_GAIW:=true}"
: "${RRS_DT:=0.1}"
: "${RRS_LAM_COL:=0.20}"
: "${RRS_LAM_D:=0.20}"
: "${RRS_LAM_TTC:=0.20}"
: "${RRS_LAM_LANE:=0.0}"
: "${RRS_LAM_RED:=0.0}"
: "${RRS_LAM_SPD:=0.0}"
: "${RRS_LAM_DIR:=0.0}"
: "${RRS_LAM_COMFORT:=0.0}"
: "${RRS_LAM_PROGRESS:=0.0}"
: "${RRS_LAM_LOW_SPEED:=0.0}"
: "${RL_WEIGHT:=1.0}"
: "${RL_CRITIC_WEIGHT:=1.0}"
: "${OPSD_DISTILL_WEIGHT:=1.0}"

: "${RUN_STAMP:=$(date +%Y-%m-%d_%H-%M-%S)}"
: "${SUITE_ROOT:=${SAVE_DIR}/grid_obj_ablation_suite/${RUN_STAMP}}"
: "${VARIANTS:=offline_diffusion_only,unified_objective,no_critic_baseline,no_advantage_reweighting,no_opsd_term,no_advantage_gate,random_teacher}"

: "${RUN_TRAIN:=true}"
: "${RUN_PRIMARY_EVAL:=true}"
: "${RUN_REPORT_EVAL:=true}"
: "${RUN_SUMMARY:=true}"
: "${RUN_PLOT:=true}"

if [ -z "${TEACHER_POOL_RUN_DIR}" ]; then
  TEACHER_POOL_RUN_DIR="${BASE_RUN_DIR}"
fi
if [ -z "${PRETRAINED_CKPT}" ] && [ -n "${BASE_RUN_DIR}" ]; then
  PRETRAINED_CKPT="${BASE_RUN_DIR%/}/latest.pth"
fi

if [ -z "${PRETRAINED_CKPT}" ]; then
  echo "PRETRAINED_CKPT is required (or set BASE_RUN_DIR so latest.pth can be inferred)."
  exit 1
fi
if [ ! -f "${PRETRAINED_CKPT}" ]; then
  echo "PRETRAINED_CKPT not found: ${PRETRAINED_CKPT}"
  exit 1
fi
if [ -z "${TEACHER_POOL_RUN_DIR}" ]; then
  echo "TEACHER_POOL_RUN_DIR is required for distillation variants (or set BASE_RUN_DIR)."
  exit 1
fi
if [ ! -d "${TEACHER_POOL_RUN_DIR}" ]; then
  echo "TEACHER_POOL_RUN_DIR not found: ${TEACHER_POOL_RUN_DIR}"
  exit 1
fi
if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "PROJECT_ROOT not found: ${PROJECT_ROOT}"
  exit 1
fi
if [ ! -d "${NUPLAN_DEVKIT_ROOT}" ]; then
  echo "NUPLAN_DEVKIT_ROOT not found: ${NUPLAN_DEVKIT_ROOT}"
  exit 1
fi
if [ ! -f "${TRAINING_JSON}" ]; then
  echo "TRAINING_JSON not found: ${TRAINING_JSON}"
  exit 1
fi

SUITE_RUNS_DIR="${SUITE_ROOT}/runs"
SUITE_EVAL_DIR="${SUITE_ROOT}/eval"
SUITE_REPORT_DIR="${SUITE_ROOT}/reports"
SUITE_SUMMARY_DIR="${SUITE_ROOT}/summary"
SUITE_FIG_DIR="${SUITE_ROOT}/figures"
SUITE_META_DIR="${SUITE_ROOT}/meta"
SUITE_TEACHER_DIR="${SUITE_ROOT}/teachers"

mkdir -p \
  "${SUITE_RUNS_DIR}" \
  "${SUITE_EVAL_DIR}" \
  "${SUITE_REPORT_DIR}" \
  "${SUITE_SUMMARY_DIR}" \
  "${SUITE_FIG_DIR}" \
  "${SUITE_META_DIR}" \
  "${SUITE_TEACHER_DIR}"

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
export RAY_TMPDIR="${RAY_TMPDIR_ROOT}"
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_INCLUDE_DASHBOARD=false
export RAY_ENABLE_METRICS_AGENT=false
export RAY_METRICS_EXPORT_PORT=0
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE="${RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE:-1}"
mkdir -p "${NUPLAN_EXP_ROOT}" "${RAY_TMPDIR_ROOT}" "${RAY_TMPDIR_ALIAS_ROOT}"

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

read -r TRAINSET_SIZE TRAIN_STEPS_PER_EPOCH <<EOF
$("${PYTHON_BIN}" - "${TRAINING_JSON}" "${TRAIN_BATCH_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
batch_size = max(1, int(sys.argv[2]))
data = json.loads(json_path.read_text())
trainset_size = len(data)
steps_per_epoch = max(1, trainset_size // batch_size)
print(trainset_size, steps_per_epoch)
PY
)
EOF
TRAIN_SAMPLES_PER_EPOCH=$(( TRAIN_STEPS_PER_EPOCH * TRAIN_BATCH_SIZE ))

pick_free_port() {
  "${PYTHON_BIN}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
}

scenario_builder_for_split() {
  if [ "$1" = "val14" ]; then
    echo "nuplan"
  else
    echo "nuplan_challenge"
  fi
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

list_teacher_candidates_json() {
  "${PYTHON_BIN}" - "${TEACHER_POOL_RUN_DIR}" "${TEACHER_CANDIDATE_MODE}" "${TEACHER_RECENT_K}" <<'PY'
from pathlib import Path
import json
import sys

run_dir = Path(sys.argv[1])
mode = sys.argv[2]
recent_k = int(sys.argv[3])

def epoch_key(path: Path) -> int:
    name = path.name
    if name.startswith("model_epoch_"):
        try:
            return int(name.split("_")[2])
        except Exception:
            return -1
    return -1

latest = run_dir / "latest.pth"
epoch_ckpts = sorted(
    [p for p in run_dir.glob("model_epoch_*.pth") if p.is_file()],
    key=epoch_key,
    reverse=True,
)

if mode == "latest":
    candidates = [latest] if latest.is_file() else []
elif mode == "recent_k":
    candidates = []
    if latest.is_file():
        candidates.append(latest)
    candidates.extend(epoch_ckpts[:recent_k])
else:
    candidates = []
    if latest.is_file():
        candidates.append(latest)
    candidates.extend(sorted(epoch_ckpts, key=epoch_key))

seen = set()
out = []
for cand in candidates:
    resolved = str(cand.resolve())
    if resolved in seen:
        continue
    seen.add(resolved)
    out.append(resolved)

print(json.dumps(out))
PY
}

persist_static_teacher_json() {
  local path="$1"
  local mode="$2"
  local teacher_ckpt="$3"
  "${PYTHON_BIN}" - "${path}" "${mode}" "${teacher_ckpt}" "${TEACHER_POOL_RUN_DIR}" "${TEACHER_SELECT_SPLIT}" "${CHALLENGE}" <<'PY'
import json
import sys
from pathlib import Path

path, mode, teacher_ckpt, run_dir, split, challenge = sys.argv[1:]
payload = {
    "mode": mode,
    "teacher_ckpt": teacher_ckpt,
    "teacher_pool_run_dir": run_dir,
    "split": split,
    "challenge": challenge,
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n")
print(teacher_ckpt)
PY
}

resolve_best_teacher_checkpoint() {
  local cache_json="${SUITE_TEACHER_DIR}/best_teacher.json"
  if [ -n "${BEST_TEACHER_CKPT}" ]; then
    persist_static_teacher_json "${cache_json}" "best_override" "${BEST_TEACHER_CKPT}" >/dev/null
    echo "${BEST_TEACHER_CKPT}"
    return 0
  fi
  if [ -f "${cache_json}" ]; then
    "${PYTHON_BIN}" - "${cache_json}" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], "r", encoding="utf-8").read())["teacher_ckpt"])
PY
    return 0
  fi

  local candidates_json
  candidates_json="$(list_teacher_candidates_json)"
  "${PYTHON_BIN}" - \
    "${PROJECT_ROOT}" \
    "${NUPLAN_DEVKIT_ROOT}" \
    "${TEACHER_POOL_RUN_DIR}" \
    "${TEACHER_SELECT_SPLIT}" \
    "${CHALLENGE}" \
    "${TEACHER_SIM_OUTPUT_ROOT}" \
    "${cache_json}" \
    "${candidates_json}" <<'PY'
import json
import sys
from pathlib import Path

project_root, nuplan_devkit_root, run_dir, split, challenge, sim_output_root, cache_json, candidates_json = sys.argv[1:]
sys.path.insert(0, project_root)

from flow_planner.onpolicy.teacher_select import select_teacher_checkpoint_by_sim_score
from flow_planner.onpolicy.rollout_runner import NuPlanSimConfig, NuPlanSimPaths

candidates = json.loads(candidates_json)
if not candidates:
    raise FileNotFoundError(f"No teacher candidates found under {run_dir}")

res = select_teacher_checkpoint_by_sim_score(
    paths=NuPlanSimPaths(project_root=project_root, nuplan_devkit_root=nuplan_devkit_root),
    run_dir=run_dir,
    candidates=candidates,
    sim_cfg=NuPlanSimConfig(split=split, challenge=challenge),
    sim_output_root=sim_output_root,
    persist_json=False,
)

payload = {
    "mode": "best",
    "teacher_ckpt": res.teacher_ckpt_path,
    "best_score": res.best_score,
    "ckpt_scores": res.ckpt_scores,
    "candidates": candidates,
    "teacher_pool_run_dir": run_dir,
    "split": split,
    "challenge": challenge,
}
Path(cache_json).write_text(json.dumps(payload, indent=2) + "\n")
print(res.teacher_ckpt_path)
PY
}

resolve_random_teacher_checkpoint() {
  local cache_json="${SUITE_TEACHER_DIR}/random_teacher.json"
  if [ -n "${RANDOM_TEACHER_CKPT}" ]; then
    persist_static_teacher_json "${cache_json}" "random_override" "${RANDOM_TEACHER_CKPT}" >/dev/null
    echo "${RANDOM_TEACHER_CKPT}"
    return 0
  fi
  if [ -f "${cache_json}" ]; then
    "${PYTHON_BIN}" - "${cache_json}" <<'PY'
import json
import sys
print(json.loads(open(sys.argv[1], "r", encoding="utf-8").read())["teacher_ckpt"])
PY
    return 0
  fi

  local candidates_json
  candidates_json="$(list_teacher_candidates_json)"
  local best_teacher=""
  if [ "${RANDOM_TEACHER_EXCLUDE_BEST}" = "true" ]; then
    best_teacher="$(resolve_best_teacher_checkpoint)"
  fi

  "${PYTHON_BIN}" - \
    "${cache_json}" \
    "${candidates_json}" \
    "${TEACHER_RANDOM_SEED}" \
    "${best_teacher}" \
    "${TEACHER_POOL_RUN_DIR}" \
    "${TEACHER_SELECT_SPLIT}" \
    "${CHALLENGE}" <<'PY'
import json
import random
import sys
from pathlib import Path

cache_json, candidates_json, seed, best_teacher, run_dir, split, challenge = sys.argv[1:]
candidates = json.loads(candidates_json)
if best_teacher:
    filtered = [cand for cand in candidates if cand != best_teacher]
    if filtered:
        candidates = filtered
if not candidates:
    raise FileNotFoundError(f"No teacher candidates available for random selection under {run_dir}")

rng = random.Random(int(seed))
teacher_ckpt = rng.choice(candidates)
payload = {
    "mode": "random",
    "teacher_ckpt": teacher_ckpt,
    "seed": int(seed),
    "best_teacher_excluded": bool(best_teacher),
    "best_teacher": best_teacher or None,
    "candidates": candidates,
    "teacher_pool_run_dir": run_dir,
    "split": split,
    "challenge": challenge,
}
Path(cache_json).write_text(json.dumps(payload, indent=2) + "\n")
print(teacher_ckpt)
PY
}

variant_train_mode() {
  case "$1" in
    offline_diffusion_only) echo "offline" ;;
    *) echo "distill" ;;
  esac
}

variant_teacher_mode() {
  case "$1" in
    offline_diffusion_only) echo "none" ;;
    random_teacher) echo "random" ;;
    *) echo "best" ;;
  esac
}

variant_rollout_multiplier() {
  case "$1" in
    offline_diffusion_only) echo "0" ;;
    *) echo "2" ;;
  esac
}

common_distill_overrides() {
  cat <<EOF
core.graph_weighting.enabled=true
core.graph_weighting.alpha1=${GRAPH_ALPHA1}
core.graph_weighting.alpha2=${GRAPH_ALPHA2}
core.rrs.enabled=true
core.rrs.nu=${RRS_NU}
core.rrs.use_gaiw=${RRS_USE_GAIW}
core.rrs.dt=${RRS_DT}
core.rrs.lam_col=${RRS_LAM_COL}
core.rrs.lam_d=${RRS_LAM_D}
core.rrs.lam_ttc=${RRS_LAM_TTC}
core.rrs.lam_lane=${RRS_LAM_LANE}
core.rrs.lam_red=${RRS_LAM_RED}
core.rrs.lam_spd=${RRS_LAM_SPD}
core.rrs.lam_dir=${RRS_LAM_DIR}
core.rrs.lam_comfort=${RRS_LAM_COMFORT}
core.rrs.lam_progress=${RRS_LAM_PROGRESS}
core.rrs.lam_low_speed=${RRS_LAM_LOW_SPEED}
core.opsd.enabled=true
core.opsd.distill_weight=${OPSD_DISTILL_WEIGHT}
core.opsd.unified_with_rl=true
core.opsd.paper_aligned_objective=true
core.opsd.keep_consistency_reg=true
core.opsd.dynamic_gate_with_adv=true
core.opsd.use_advantage_reweighting=true
core.opsd.use_critic_baseline=true
core.opsd.include_distill_term=true
core.opsd.teacher_selection_mode=single
core.opsd.teacher_selection_source=teacher
core.opsd.teacher_selection_k=1
core.rl.enabled=true
core.rl.algorithm=actor_critic
core.rl.use_gaiw=true
core.rl.weight=${RL_WEIGHT}
core.rl.critic_weight=${RL_CRITIC_WEIGHT}
core.rl.warmup_epochs=1
core.rl.critic_warmup_epochs=1
EOF
}

variant_overrides() {
  case "$1" in
    offline_diffusion_only)
      cat <<'EOF'
core=flow_matching
core.graph_weighting.enabled=false
core.rrs.enabled=false
core.rl.enabled=false
EOF
      ;;
    unified_objective)
      common_distill_overrides
      ;;
    no_critic_baseline)
      common_distill_overrides
      cat <<'EOF'
core.opsd.use_critic_baseline=false
core.rl.critic_weight=0.0
EOF
      ;;
    no_advantage_reweighting)
      common_distill_overrides
      cat <<'EOF'
core.opsd.use_advantage_reweighting=false
EOF
      ;;
    no_opsd_term)
      common_distill_overrides
      cat <<'EOF'
core.opsd.include_distill_term=false
EOF
      ;;
    no_advantage_gate)
      common_distill_overrides
      cat <<'EOF'
core.opsd.dynamic_gate_with_adv=false
EOF
      ;;
    random_teacher)
      common_distill_overrides
      ;;
    *)
      echo "Unknown variant: $1" >&2
      return 1
      ;;
  esac
}

write_variant_meta() {
  local variant="$1"
  local run_dir="$2"
  local train_mode="$3"
  local teacher_mode="$4"
  local teacher_ckpt="$5"
  local rollout_multiplier="$6"
  local teacher_meta_path="$7"
  local meta_path="${SUITE_META_DIR}/${variant}.json"
  local rollouts_per_epoch=$(( TRAIN_SAMPLES_PER_EPOCH * rollout_multiplier ))

  "${PYTHON_BIN}" - \
    "${meta_path}" \
    "${variant}" \
    "${run_dir}" \
    "${train_mode}" \
    "${teacher_mode}" \
    "${teacher_ckpt}" \
    "${teacher_meta_path}" \
    "${PRETRAINED_CKPT}" \
    "${TRAIN_EPOCHS}" \
    "${TRAIN_BATCH_SIZE}" \
    "${TRAINSET_SIZE}" \
    "${TRAIN_STEPS_PER_EPOCH}" \
    "${TRAIN_SAMPLES_PER_EPOCH}" \
    "${rollout_multiplier}" \
    "${rollouts_per_epoch}" \
    "${PRIMARY_SPLIT}" \
    "${REPORT_SPLITS}" <<'PY'
import json
import sys
from pathlib import Path

(
    meta_path,
    variant,
    run_dir,
    train_mode,
    teacher_mode,
    teacher_ckpt,
    teacher_meta_path,
    pretrained_ckpt,
    train_epochs,
    train_batch_size,
    trainset_size,
    steps_per_epoch,
    samples_per_epoch,
    rollout_multiplier,
    rollouts_per_epoch,
    primary_split,
    report_splits_csv,
) = sys.argv[1:]

payload = {
    "variant_id": variant,
    "run_dir": run_dir,
    "train_mode": train_mode,
    "teacher_mode": teacher_mode,
    "teacher_ckpt": teacher_ckpt if teacher_ckpt else None,
    "teacher_meta_path": teacher_meta_path if teacher_meta_path else None,
    "pretrained_ckpt": pretrained_ckpt,
    "train_epochs": int(train_epochs),
    "train_batch_size": int(train_batch_size),
    "trainset_size": int(trainset_size),
    "steps_per_epoch": int(steps_per_epoch),
    "samples_per_epoch": int(samples_per_epoch),
    "rollout_multiplier_per_sample": int(rollout_multiplier),
    "rollouts_per_epoch": int(rollouts_per_epoch),
    "primary_split": primary_split,
    "report_splits": [s.strip() for s in report_splits_csv.split(",") if s.strip()],
}
Path(meta_path).write_text(json.dumps(payload, indent=2) + "\n")
PY
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
  local variant="$1"
  local run_dir="${SUITE_RUNS_DIR}/${variant}"
  local tb_dir="${run_dir}/tensorboard"
  local train_mode
  train_mode="$(variant_train_mode "${variant}")"
  local teacher_mode
  teacher_mode="$(variant_teacher_mode "${variant}")"
  local teacher_ckpt=""
  local teacher_meta_path=""
  local rollout_multiplier
  rollout_multiplier="$(variant_rollout_multiplier "${variant}")"

  mkdir -p "${run_dir}" "${tb_dir}"
  export TENSORBOARD_LOG_PATH="${tb_dir}"

  if [ "${train_mode}" = "distill" ]; then
    case "${teacher_mode}" in
      best)
        teacher_ckpt="$(resolve_best_teacher_checkpoint)"
        teacher_meta_path="${SUITE_TEACHER_DIR}/best_teacher.json"
        ;;
      random)
        teacher_ckpt="$(resolve_random_teacher_checkpoint)"
        teacher_meta_path="${SUITE_TEACHER_DIR}/random_teacher.json"
        ;;
      *)
        echo "Unsupported teacher mode: ${teacher_mode}" >&2
        exit 1
        ;;
    esac
  fi

  write_variant_meta "${variant}" "${run_dir}" "${train_mode}" "${teacher_mode}" "${teacher_ckpt}" "${rollout_multiplier}" "${teacher_meta_path}"

  if [ "${RUN_TRAIN}" != "true" ]; then
    return 0
  fi

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
  )

  if [ "${train_mode}" = "offline" ]; then
    train_cmd+=(
      -m flow_planner.trainer
      --config-name flow_planner_standard
      "pretrained_checkpoint=${PRETRAINED_CKPT}"
      "save_dir=${run_dir}"
      "job_name=grid_obj_${variant}"
      "train.epoch=${TRAIN_EPOCHS}"
      "train.save_utd=${TRAIN_SAVE_UTD}"
      "train.batch_size=${TRAIN_BATCH_SIZE}"
      "scheduler.warm_up_epoch=${SCHEDULER_WARMUP_EPOCHS}"
      "optimizer.lr=${LR}"
    )
  else
    export TEACHER_CKPT="${teacher_ckpt}"
    train_cmd+=(
      -m flow_planner.onpolicy.distill_trainer
      --config-name flow_planner_standard
      "save_dir=${run_dir}"
      "job_name=grid_obj_${variant}"
      "pretrained_checkpoint=${PRETRAINED_CKPT}"
      "core=opds_distill"
      "core.opsd.teacher_ckpt_path=${teacher_ckpt}"
      "train.epoch=${TRAIN_EPOCHS}"
      "train.save_utd=${TRAIN_SAVE_UTD}"
      "train.batch_size=${TRAIN_BATCH_SIZE}"
      "scheduler.warm_up_epoch=${SCHEDULER_WARMUP_EPOCHS}"
      "optimizer.lr=${LR}"
    )
  fi

  while IFS= read -r line; do
    [ -n "${line}" ] && train_cmd+=("${line}")
  done < <(variant_overrides "${variant}")

  echo "[TRAIN] ${variant}"
  echo "  run_dir=${run_dir}"
  echo "  train_mode=${train_mode}"
  echo "  teacher_mode=${teacher_mode}"
  if [ -n "${teacher_ckpt}" ]; then
    echo "  teacher_ckpt=${teacher_ckpt}"
  fi
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

(
    variant,
    split,
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
    "split": split,
    "checkpoint": checkpoint,
    "epoch": epoch,
    "result_dir": result_dir,
    "status": status,
}

if status == "OK":
    with open(json_path, "r", encoding="utf-8") as f:
        payload.update(json.load(f))
    payload["json_path"] = json_path
else:
    payload["json_path"] = None

with open(jsonl_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

fieldnames = [
    "variant_id",
    "split",
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
with open(csv_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerow(row)
PY
}

LAST_EVAL_STATUS=""
LAST_EVAL_RESULT_DIR=""

run_single_benchmark_eval() {
  local split="$1"
  local variant="$2"
  local ckpt_file="$3"
  local ckpt_name="$4"
  local exp_prefix="$5"
  local log_file="$6"
  local json_path="$7"
  local scenario_builder
  scenario_builder="$(scenario_builder_for_split "${split}")"
  local run_ts
  run_ts="$(date +%Y-%m-%d-%H-%M-%S)"
  local exp_uid="${PLANNER}/${split}/grid_obj_${variant}/${exp_prefix}_${run_ts}"
  local result_dir="${NUPLAN_EXP_ROOT}/exp/simulation/${CHALLENGE}/${exp_uid}"
  local ray_tmpdir_for_run="${RAY_TMPDIR}"
  local private_ray_alias=""
  local private_ray_real=""

  echo "  checkpoint=${ckpt_name}"
  echo "  split=${split}"

  if [ "${WORKER_MODE}" = "ray_distributed" ] && [ "${ISOLATE_RAY_TMPDIR_PER_CKPT}" = "true" ]; then
    local ray_pair
    ray_pair="$(make_private_ray_tmpdir_pair "${variant}_${split}_${exp_prefix}_${run_ts}")"
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
    "planner.flow_planner.config_path=${SUITE_RUNS_DIR}/${variant}/.hydra/config.yaml"
    "planner.flow_planner.ckpt_path=${ckpt_file}"
    "scenario_builder=${scenario_builder}"
    "scenario_filter=${split}"
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
      LAST_EVAL_STATUS="OK"
      delete_result_dir_if_requested "${result_dir}"
    else
      LAST_EVAL_STATUS="PARSE_FAIL"
    fi
  else
    LAST_EVAL_STATUS="SIM_FAIL"
  fi

  LAST_EVAL_RESULT_DIR="${result_dir}"
  cleanup_private_ray_tmpdir "${private_ray_alias}" "${private_ray_real}"
}

evaluate_variant_all_epochs() {
  local variant="$1"
  local run_dir="${SUITE_RUNS_DIR}/${variant}"
  local eval_dir="${SUITE_EVAL_DIR}/${PRIMARY_SPLIT}/${variant}"
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
    "split",
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
with open(sys.argv[1], "w", newline="", encoding="utf-8") as f:
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

  echo "[EVAL][${PRIMARY_SPLIT}] ${variant}"
  for ckpt_name in "${ckpt_names[@]}"; do
    local ckpt_file="${run_dir}/${ckpt_name}"
    local ckpt_tag="${ckpt_name%.pth}"
    local log_file="${eval_dir}/logs/${ckpt_tag}.log"
    local json_path="${eval_dir}/json/${ckpt_tag}.json"
    run_single_benchmark_eval "${PRIMARY_SPLIT}" "${variant}" "${ckpt_file}" "${ckpt_name}" "${ckpt_tag}" "${log_file}" "${json_path}"
    append_eval_row \
      "${variant}" \
      "${PRIMARY_SPLIT}" \
      "${ckpt_name}" \
      "${LAST_EVAL_RESULT_DIR}" \
      "${json_path}" \
      "${jsonl_path}" \
      "${csv_path}" \
      "${LAST_EVAL_STATUS}"
  done
}

persist_best_primary_selection() {
  local variant="$1"
  local jsonl_path="${SUITE_EVAL_DIR}/${PRIMARY_SPLIT}/${variant}/benchmark_per_epoch.jsonl"
  local out_json="${SUITE_EVAL_DIR}/${PRIMARY_SPLIT}/${variant}/best_checkpoint.json"
  "${PYTHON_BIN}" - "${jsonl_path}" "${out_json}" "${variant}" "${PRIMARY_SPLIT}" <<'PY'
import json
import math
import sys
from pathlib import Path

jsonl_path, out_json, variant, split = sys.argv[1:]
best = None
for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("status") != "OK" or row.get("score_up") is None:
        continue
    try:
        score = float(row["score_up"])
    except (TypeError, ValueError):
        continue
    if not math.isfinite(score):
        continue
    if best is None or score > float(best["score_up"]):
        best = row

if best is None:
    raise SystemExit(f"No valid PRIMARY_SPLIT rows found in {jsonl_path}")

best["variant_id"] = variant
best["split"] = split
Path(out_json).write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(best["checkpoint"])
PY
}

materialize_primary_report_from_best() {
  local variant="$1"
  local best_json="${SUITE_EVAL_DIR}/${PRIMARY_SPLIT}/${variant}/best_checkpoint.json"
  local report_dir="${SUITE_REPORT_DIR}/${PRIMARY_SPLIT}/${variant}"
  mkdir -p "${report_dir}"
  cp "${best_json}" "${report_dir}/selected_checkpoint.json"
}

write_selected_report() {
  "${PYTHON_BIN}" - "$@" <<'PY'
import json
import sys
from pathlib import Path

variant, split, checkpoint, result_dir, raw_json_path, out_json_path, status = sys.argv[1:]
payload = {
    "variant_id": variant,
    "split": split,
    "checkpoint": checkpoint,
    "result_dir": result_dir,
    "status": status,
}
if status == "OK":
    with open(raw_json_path, "r", encoding="utf-8") as f:
        payload.update(json.load(f))
    payload["json_path"] = raw_json_path
else:
    payload["json_path"] = None

Path(out_json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

evaluate_selected_checkpoint_on_split() {
  local variant="$1"
  local split="$2"
  local ckpt_name="$3"
  local run_dir="${SUITE_RUNS_DIR}/${variant}"
  local report_dir="${SUITE_REPORT_DIR}/${split}/${variant}"
  local ckpt_file="${run_dir}/${ckpt_name}"
  local ckpt_tag="${ckpt_name%.pth}"

  mkdir -p "${report_dir}"

  if [ "${split}" = "${PRIMARY_SPLIT}" ]; then
    materialize_primary_report_from_best "${variant}"
    return 0
  fi

  local log_file="${report_dir}/selected_checkpoint.log"
  local raw_json_path="${report_dir}/selected_checkpoint_raw.json"
  local out_json_path="${report_dir}/selected_checkpoint.json"
  run_single_benchmark_eval "${split}" "${variant}" "${ckpt_file}" "${ckpt_name}" "selected_${ckpt_tag}" "${log_file}" "${raw_json_path}"
  write_selected_report \
    "${variant}" \
    "${split}" \
    "${ckpt_name}" \
    "${LAST_EVAL_RESULT_DIR}" \
    "${raw_json_path}" \
    "${out_json_path}" \
    "${LAST_EVAL_STATUS}"
}

evaluate_variant_report_splits() {
  local variant="$1"
  local selected_ckpt="$2"
  IFS=',' read -r -a report_splits <<< "${REPORT_SPLITS}"
  echo "[REPORT] ${variant}"
  echo "  selected_checkpoint=${selected_ckpt}"
  for split in "${report_splits[@]}"; do
    split="${split// /}"
    [ -z "${split}" ] && continue
    evaluate_selected_checkpoint_on_split "${variant}" "${split}" "${selected_ckpt}"
  done
}

IFS=',' read -r -a selected_variants <<< "${VARIANTS}"

for variant in "${selected_variants[@]}"; do
  variant="${variant// /}"
  [ -z "${variant}" ] && continue
  variant_overrides "${variant}" >/dev/null
  train_variant "${variant}"
  if [ "${RUN_PRIMARY_EVAL}" = "true" ]; then
    evaluate_variant_all_epochs "${variant}"
  fi
  if [ "${RUN_PRIMARY_EVAL}" = "true" ] || [ "${RUN_REPORT_EVAL}" = "true" ]; then
    selected_ckpt="$(persist_best_primary_selection "${variant}")"
  fi
  if [ "${RUN_REPORT_EVAL}" = "true" ]; then
    evaluate_variant_report_splits "${variant}" "${selected_ckpt}"
  fi
done

if [ "${RUN_SUMMARY}" = "true" ]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/flow_planner/onpolicy/grid_obj_ablation_collect.py" \
    --suite-root "${SUITE_ROOT}" \
    --output-dir "${SUITE_SUMMARY_DIR}" \
    --primary-split "${PRIMARY_SPLIT}" \
    --metric-split "test14-random"
fi

if [ "${RUN_PLOT}" = "true" ]; then
  bash "${PROJECT_ROOT}/flow_planner/run_script/render_grid_obj_sample_eff.sh" "${SUITE_ROOT}"
fi

echo "GRID objective ablation suite artifacts saved to: ${SUITE_ROOT}"
