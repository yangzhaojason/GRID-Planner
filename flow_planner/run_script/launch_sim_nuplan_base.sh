export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
# Set environment variables
# Flow-Planner repo root on the eval machine
export PROJECT_ROOT=/home/code/grid_planner

# nuplan-devkit absolute path (inside the repo)
export NUPLAN_DEVKIT_ROOT=/home/code/grid_planner/nuplan-devkit

# nuplan dataset / maps / exp roots
export NUPLAN_DATA_ROOT=/home/datasets_2/nuplan/             # e.g. "/data"
export NUPLAN_MAPS_ROOT=/home/datasets_2/nuplan/nuPlan-v1.1/maps
: "${NUPLAN_EXP_ROOT:=/home/results/yangzhao/grid_planner}"
export NUPLAN_EXP_ROOT

# Ensure Python can import flow_planner for Hydra's pkg:// paths
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Put Ray temp/object store on the large disk to avoid "Local disk is full"
export RAY_TMPDIR="${RAY_TMPDIR:-/home/results/ray}"
mkdir -p "${NUPLAN_EXP_ROOT}"
mkdir -p "${RAY_TMPDIR}"

# Ray: disable dashboard/metrics to avoid exporter-agent init retries slowing down startup
export RAY_DISABLE_DASHBOARD=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_INCLUDE_DASHBOARD=false
export RAY_ENABLE_METRICS_AGENT=false
export RAY_METRICS_EXPORT_PORT=0

# Dataset split to use (can be overridden by env var SPLIT)
# Options:
#   - "test14-random"  (paper table "Test14")
#   - "test14-hard"
#   - "val14"
: "${SPLIT:=val14}"
export SPLIT

# Challenge type
# Options:
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
: "${CHALLENGE:=closed_loop_reactive_agents}"
export CHALLENGE
###################################


# Base model checkpoint and config
BRANCH_NAME=flow_planner_base
CONFIG_FILE=/home/results/flow_planner/base_checkpoint/.hydra/config.yaml   # path of .hydra/config in ckpt folder
CKPT_FILE=/home/results/flow_planner/base_checkpoint/latest.pth            # path to the .pth of checkpoint

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=flow_planner

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.flow_planner.config_path=$CONFIG_FILE \
    planner.flow_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=64 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"