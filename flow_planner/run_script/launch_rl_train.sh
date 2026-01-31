export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1
# 单卡下仍建议限制线程，避免 CPU 过度并发
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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

BRANCH_NAME=flow_planner_rl
CONFIG_FILE=/home/qianzezhong/Flow-Planner/results/outputs/FlowPlannerTraining/flow_planner_standard/2025-11-30_02-26-51/.hydra/config.yaml # path of .hydra/config in ckpt folder
CKPT_FILE=/home/qianzezhong/Flow-Planner_old/results/outputs/FlowPlannerTraining/flow_planner_standard/2025-11-30_02-26-51/latest.pth # path to the .pth of checkpoint

# RL Training Configuration
RL_OUTPUT_DIR=/home/qianzezhong/Flow-Planner-RL/results/rl_training # RL training output directory
RL_BATCH_SIZE=64 # batch size for simulation
RL_MICRO_BATCH_SIZE=512 # micro batch size for training
# Learnable noise_std range (deprecated: RL_NOISE_STD)
RL_NOISE_STD_MIN=0.08
RL_NOISE_STD_MAX=0.16
RL_MAX_INFLIGHT_RUNNERS=64 # chunk size for Ray execution (reduce spill)
RL_KEEP_ROLLOUT_FILES=false # keep per-runner rollout files under RL_OUTPUT_DIR/rl_rollouts
RL_CLIP_RATIO=0.1 # PPO clip ratio
RL_SAVE_INTERVAL=10 # save checkpoint every N steps
RL_LEARNING_RATE=1e-6 # learning rate for optimizer
RL_JOINT_LOGPROB=true # joint logprob for PPO ratio
# Gradient accumulation steps (per optimizer.step), default 4
RL_GRADIENT_ACCUMULATE_STEP=16
if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=flow_planner

torchrun --standalone --nproc_per_node=8 $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.flow_planner.config_path=$CONFIG_FILE \
    planner.flow_planner.ckpt_path=$CKPT_FILE \
    +planner.flow_planner.use_cfg=false \
    planner.flow_planner.rl.enabled=true \
    planner.flow_planner.rl.batch_size=$RL_BATCH_SIZE \
    +planner.flow_planner.rl.rollouts_per_scenario=64 \
    planner.flow_planner.rl.micro_batch_size=$RL_MICRO_BATCH_SIZE \
    +planner.flow_planner.rl.noise_std_min=$RL_NOISE_STD_MIN \
    +planner.flow_planner.rl.noise_std_max=$RL_NOISE_STD_MAX \
    +planner.flow_planner.rl.max_inflight_runners=$RL_MAX_INFLIGHT_RUNNERS \
    +planner.flow_planner.rl.keep_rollout_files=$RL_KEEP_ROLLOUT_FILES \
    planner.flow_planner.rl.clip_ratio=$RL_CLIP_RATIO \
    planner.flow_planner.rl.save_interval=$RL_SAVE_INTERVAL \
    planner.flow_planner.rl.output_dir=$RL_OUTPUT_DIR \
    planner.flow_planner.rl.learning_rate=$RL_LEARNING_RATE \
    +planner.flow_planner.rl.gradient_accumulate_step=$RL_GRADIENT_ACCUMULATE_STEP \
    planner.flow_planner.rl.joint_logprob=$RL_JOINT_LOGPROB \
    +planner.flow_planner.rl.entropy_coef=1e-4 \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/rl/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=64 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.125 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
