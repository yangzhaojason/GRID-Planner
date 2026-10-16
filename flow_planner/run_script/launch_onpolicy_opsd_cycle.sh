#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# On-policy self-distillation outer loop:
#   For each cycle k:
#     1) Evaluate current student checkpoints on nuPlan -> select teacher_ckpt(k)
#     2) Distill student(k) from teacher_ckpt(k) for a few epochs
#     3) Set student(k+1) = distilled checkpoint of this cycle
#
# Usage:
#   RUN_DIR=/path/to/training-run \
#   SPLIT=val14 \
#   bash flow_planner/run_script/launch_onpolicy_opsd_cycle.sh
#
# Assumes the same env vars as existing launch scripts (PROJECT_ROOT, NUPLAN_DEVKIT_ROOT, etc.).

: "${RUN_DIR:=}"
: "${SPLIT:=val14}"
: "${CHALLENGE:=closed_loop_reactive_agents}"
: "${TEACHER_SIM_OUTPUT_ROOT:=${NUPLAN_EXP_ROOT}}"
# self-distillation loop controls
: "${NUM_CYCLES:=3}"
: "${START_FROM_STEP:=select}"   # select | distill (distill only applies to first cycle)
: "${TEACHER_CANDIDATE_MODE:=all}"  # all | latest | recent_k
: "${TEACHER_RECENT_K:=3}"          # used when TEACHER_CANDIDATE_MODE=recent_k
: "${DISTILL_EPOCHS_PER_CYCLE:=2}"
: "${DISTILL_LR:=5e-5}"
: "${DISTILL_SAVE_UTD:=1}"
: "${DISTILL_WARMUP_EPOCH:=1}"
# default objective toggles (can be overridden via env or "$@" CLI overrides)
: "${USE_UNIFIED_WITH_RL:=true}"
: "${USE_PAPER_ALIGNED_OBJECTIVE:=true}"
: "${USE_DYNAMIC_GATE_WITH_ADV:=true}"
: "${ENABLE_RL:=true}"
: "${ENABLE_RRS:=true}"
: "${RRS_NU:=0.0}"
: "${RRS_USE_GAIW:=true}"
: "${RL_USE_GAIW:=true}"
: "${RL_WEIGHT:=1.0}"
: "${RL_CRITIC_WEIGHT:=1.0}"
# distributed / NCCL stability knobs
: "${NCCL_IB_DISABLE:=1}"
: "${NCCL_P2P_DISABLE:=0}"
: "${TORCH_NCCL_ASYNC_ERROR_HANDLING:=1}"
: "${NCCL_DEBUG:=WARN}"

if [ "${START_FROM_STEP}" != "select" ] && [ "${START_FROM_STEP}" != "distill" ]; then
  echo "Invalid START_FROM_STEP='${START_FROM_STEP}'. Expected 'select' or 'distill'."
  exit 1
fi
if [ "${TEACHER_CANDIDATE_MODE}" != "all" ] && [ "${TEACHER_CANDIDATE_MODE}" != "latest" ] && [ "${TEACHER_CANDIDATE_MODE}" != "recent_k" ]; then
  echo "Invalid TEACHER_CANDIDATE_MODE='${TEACHER_CANDIDATE_MODE}'. Expected one of: all, latest, recent_k."
  exit 1
fi
if [ "${TEACHER_CANDIDATE_MODE}" = "recent_k" ] && ! [[ "${TEACHER_RECENT_K}" =~ ^[0-9]+$ ]]; then
  echo "TEACHER_RECENT_K must be a non-negative integer, got '${TEACHER_RECENT_K}'."
  exit 1
fi

if [ -z "${RUN_DIR}" ]; then
  echo "RUN_DIR is required (path containing .hydra/ and checkpoints)."
  exit 1
fi
if [ ! -d "${PROJECT_ROOT}" ]; then
  echo "PROJECT_ROOT not found: ${PROJECT_ROOT}"
  exit 1
fi
if [ ! -d "${TRAINING_DATA}" ]; then
  echo "TRAINING_DATA not found: ${TRAINING_DATA}"
  exit 1
fi
if [ ! -f "${TRAINING_JSON}" ]; then
  echo "TRAINING_JSON not found: ${TRAINING_JSON}"
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

cd "${PROJECT_ROOT}"
export SAVE_DIR
export TENSORBOARD_LOG_PATH
export TRAINING_DATA
export TRAINING_JSON
export NUPLAN_DEVKIT_ROOT
export NUPLAN_DATA_ROOT
export NUPLAN_MAPS_ROOT
export NCCL_IB_DISABLE
export NCCL_P2P_DISABLE
export TORCH_NCCL_ASYNC_ERROR_HANDLING
export NCCL_DEBUG

CURRENT_STUDENT_RUN_DIR="${RUN_DIR}"
SELF_DISTILL_ROOT="${RUN_DIR}/onpolicy_self_distill"
mkdir -p "${SELF_DISTILL_ROOT}"

echo "On-policy self-distillation settings:"
echo "  RUN_DIR=${RUN_DIR}"
echo "  NUM_CYCLES=${NUM_CYCLES}"
echo "  START_FROM_STEP=${START_FROM_STEP}"
echo "  TEACHER_CANDIDATE_MODE=${TEACHER_CANDIDATE_MODE} (recent_k=${TEACHER_RECENT_K})"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  NCCL_IB_DISABLE=${NCCL_IB_DISABLE}, NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}, TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}, NCCL_DEBUG=${NCCL_DEBUG}"
echo "  DISTILL_EPOCHS_PER_CYCLE=${DISTILL_EPOCHS_PER_CYCLE}"
echo "  SPLIT=${SPLIT}, CHALLENGE=${CHALLENGE}"
echo "  unified_with_rl=${USE_UNIFIED_WITH_RL}, paper_aligned=${USE_PAPER_ALIGNED_OBJECTIVE}, rl=${ENABLE_RL}, rrs=${ENABLE_RRS}"

for (( CYCLE=1; CYCLE<=NUM_CYCLES; CYCLE++ )); do
  echo ""
  echo "==================== Cycle ${CYCLE}/${NUM_CYCLES} ===================="
  echo "[Cycle ${CYCLE}] Current student run_dir: ${CURRENT_STUDENT_RUN_DIR}"

  if [ ! -f "${CURRENT_STUDENT_RUN_DIR}/latest.pth" ]; then
    echo "Missing student checkpoint: ${CURRENT_STUDENT_RUN_DIR}/latest.pth"
    exit 1
  fi

  TEACHER_CKPT_SELECTED=""

  if [ "${START_FROM_STEP}" = "distill" ] && [ "${CYCLE}" -eq 1 ]; then
    echo "[Cycle ${CYCLE}] [1/2] START_FROM_STEP=distill -> skip teacher selection."
    if [ -n "${TEACHER_CKPT:-}" ]; then
      TEACHER_CKPT_SELECTED="${TEACHER_CKPT}"
      echo "[Cycle ${CYCLE}] Using TEACHER_CKPT from env: ${TEACHER_CKPT_SELECTED}"
    else
      LOG_PATH_C1="${CURRENT_STUDENT_RUN_DIR}/teacher_select_cycle${CYCLE}.log"
      LOG_PATH_LEGACY="${CURRENT_STUDENT_RUN_DIR}/teacher_select.log"
      TEACHER_LOG_PATH=""
      if [ -f "${LOG_PATH_C1}" ]; then
        TEACHER_LOG_PATH="${LOG_PATH_C1}"
      elif [ -f "${LOG_PATH_LEGACY}" ]; then
        TEACHER_LOG_PATH="${LOG_PATH_LEGACY}"
      fi
      if [ -z "${TEACHER_LOG_PATH}" ]; then
        echo "START_FROM_STEP=distill but no teacher selection log found under:"
        echo "  - ${LOG_PATH_C1}"
        echo "  - ${LOG_PATH_LEGACY}"
        echo "Please set TEACHER_CKPT=/path/to/ckpt and rerun."
        exit 1
      fi
      TEACHER_CKPT_SELECTED=$(grep "^TEACHER_CKPT=" "${TEACHER_LOG_PATH}" | tail -n 1 | cut -d'=' -f2- || true)
    fi
  else
    echo "[Cycle ${CYCLE}] [1/2] Selecting teacher checkpoint by nuPlan score..."
    "${PYTHON_BIN}" - "${PROJECT_ROOT}" "${NUPLAN_DEVKIT_ROOT}" \
      "${CURRENT_STUDENT_RUN_DIR}" "${TEACHER_CANDIDATE_MODE}" "${TEACHER_RECENT_K}" \
      "${SPLIT}" "${CHALLENGE}" "${TEACHER_SIM_OUTPUT_ROOT}" <<'PY' | tee "${CURRENT_STUDENT_RUN_DIR}/teacher_select_cycle${CYCLE}.log"
import sys
from pathlib import Path
from flow_planner.onpolicy.teacher_select import select_teacher_checkpoint_by_sim_score
from flow_planner.onpolicy.rollout_runner import NuPlanSimPaths, NuPlanSimConfig

project_root, devkit_root, run_dir, mode, recent_k, split, challenge, output_root = sys.argv[1:]
candidates = None
if mode == "latest":
    candidates = [str(Path(run_dir) / "latest.pth")]
elif mode == "recent_k":
    run_path = Path(run_dir)
    epochs = sorted(run_path.glob("model_epoch_*.pth"), key=lambda path: int(path.name.split("_")[2]), reverse=True)
    candidates = [str((run_path / "latest.pth").resolve())] + [str(path.resolve()) for path in epochs[:int(recent_k)]]
res = select_teacher_checkpoint_by_sim_score(
    paths=NuPlanSimPaths(project_root=project_root, nuplan_devkit_root=devkit_root),
    run_dir=run_dir,
    candidates=candidates,
    sim_cfg=NuPlanSimConfig(split=split, challenge=challenge),
    sim_output_root=output_root,
    persist_json=True,
)
print("TEACHER_CKPT=" + res.teacher_ckpt_path)
print("BEST_SCORE=" + str(res.best_score))
if mode == "recent_k":
    print("TEACHER_CANDIDATES=" + ",".join(candidates))
PY
    TEACHER_CKPT_SELECTED=$(grep "^TEACHER_CKPT=" "${CURRENT_STUDENT_RUN_DIR}/teacher_select_cycle${CYCLE}.log" | tail -n 1 | cut -d'=' -f2- || true)
  fi

  if [ -z "${TEACHER_CKPT_SELECTED}" ]; then
    echo "Failed to resolve TEACHER_CKPT for cycle ${CYCLE}."
    echo "If resuming, set START_FROM_STEP=distill and export TEACHER_CKPT=/path/to/teacher.pth"
    exit 1
  fi

  echo "[Cycle ${CYCLE}] [2/2] Distilling student from selected teacher..."
  DISTILL_DIR="${SELF_DISTILL_ROOT}/cycle_${CYCLE}_$(date +%Y-%m-%d_%H-%M-%S)"
  mkdir -p "${DISTILL_DIR}"

  export TEACHER_CKPT="${TEACHER_CKPT_SELECTED}"
  "${PYTHON_BIN}" -m torch.distributed.run --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" --standalone -m flow_planner.onpolicy.distill_trainer \
    --config-name flow_planner_standard \
    save_dir="${DISTILL_DIR}" \
    job_name="opds_distill_cycle${CYCLE}" \
    pretrained_checkpoint="${CURRENT_STUDENT_RUN_DIR}/latest.pth" \
    core=opds_distill \
    core.opsd.teacher_ckpt_path="${TEACHER_CKPT}" \
    core.opsd.unified_with_rl="${USE_UNIFIED_WITH_RL}" \
    core.opsd.paper_aligned_objective="${USE_PAPER_ALIGNED_OBJECTIVE}" \
    core.opsd.dynamic_gate_with_adv="${USE_DYNAMIC_GATE_WITH_ADV}" \
    core.rrs.enabled="${ENABLE_RRS}" \
    core.rrs.nu="${RRS_NU}" \
    core.rrs.use_gaiw="${RRS_USE_GAIW}" \
    core.rl.enabled="${ENABLE_RL}" \
    core.rl.use_gaiw="${RL_USE_GAIW}" \
    core.rl.weight="${RL_WEIGHT}" \
    core.rl.critic_weight="${RL_CRITIC_WEIGHT}" \
    train.epoch="${DISTILL_EPOCHS_PER_CYCLE}" \
    train.save_utd="${DISTILL_SAVE_UTD}" \
    scheduler.warm_up_epoch="${DISTILL_WARMUP_EPOCH}" \
    optimizer.lr="${DISTILL_LR}" \
    "$@"

  echo "[Cycle ${CYCLE}] Done. Distilled run_dir: ${DISTILL_DIR}"
  CURRENT_STUDENT_RUN_DIR="${DISTILL_DIR}"
done

echo ""
echo "All cycles completed."
echo "Final distilled run_dir: ${CURRENT_STUDENT_RUN_DIR}"

