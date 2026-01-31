export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT=/home/qianzezhong/Flow-Planner-RL/nuplan-devkit # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
export NUPLAN_DATA_ROOT=/home/datasets_2/nuplan/ # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_MAPS_ROOT=/home/datasets_2/nuplan/nuPlan-v1.1/maps # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
export NUPLAN_EXP_ROOT=/home/datasets_2/nuplan/nuPlan-v1.1/exp # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

# Dataset split to use
# Options: 
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
export SPLIT="val14" # e.g., "val14"

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
export CHALLENGE="closed_loop_reactive_agents" # e.g., "closed_loop_nonreactive_agents"
###################################


BRANCH_NAME=flow_planner_release
CONFIG_FILE=/home/qianzezhong/Flow-Planner/results/outputs/FlowPlannerTraining/flow_planner_standard/2025-11-30_02-26-51/.hydra/config.yaml # path of .hydra/config in ckpt folder
CKPT_FILE=/home/qianzezhong/Flow-Planner-RL/results/rl_training/checkpoint_step_30.pth

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