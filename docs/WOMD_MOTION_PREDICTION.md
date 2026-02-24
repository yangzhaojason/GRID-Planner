# GRID-Planner on WOMD Motion Prediction

本文档描述如何将 GRID-Planner 适配到 Waymo Open Motion Dataset (WOMD) 的 Marginal Motion Prediction 任务。

## 概述

核心思路：**数据层适配，模型层复用**。

- 将 WOMD scenario protobuf 转换为 NuPlanDataSample 兼容的 NPZ 格式
- 模型 Encoder / Decoder / FlowODE 全部复用，仅修改 `state_dim: 4 → 2`（只预测 x, y，不预测 heading）
- 推理阶段通过过采样 + NMS 生成 K=6 多模态轨迹（借鉴 MotionDiffuser 思路）
- 损失函数保持不变：Flow Matching MSE + Consistency Loss + GAIW 加权

### 与 nuPlan 任务的关键差异

| 维度 | nuPlan (Ego Planning) | WOMD (Motion Prediction) |
|------|----------------------|--------------------------|
| 预测目标 | 自车 (ego) | 场景中每个 target agent |
| 坐标系 | ego-centric | agent-centric (per target) |
| 输出维度 | (x, y, cos_h, sin_h) 4D | (x, y) 2D |
| 多模态 | 单条轨迹 | K=6 条 + 置信度 |
| Route 信息 | 有 (25 条 route lanes) | 无 (全零填充) |
| Static objects | 有 (5 个) | 无 (全零填充) |
| 历史帧数 | 21 帧 (2.0s) | 11 帧 (1.0s, padding 到 21) |
| 数据增强 | StatePerturbation | 禁用 |

---

## 文件结构

### 新增文件

```
flow_planner/
├── data/
│   ├── augmentation/
│   │   └── identity_augmentation.py        # No-op augmentation
│   └── data_process/
│       ├── womd_utils.py                   # 坐标变换、polyline 重采样、TL 映射等
│       ├── womd_data_processor.py          # WOMD scenario → NPZ 转换器
│       └── compute_womd_norm_stats.py      # 归一化统计量计算
├── evaluation/
│   ├── __init__.py
│   ├── womd_metrics.py                     # minADE/FDE/MissRate/mAP
│   └── womd_eval.py                        # 评估推理主函数
├── run_script/
│   ├── launch_womd_preprocess.sh           # 数据预处理脚本
│   ├── launch_train_womd.sh                # 训练脚本
│   └── launch_eval_womd.sh                 # 评估脚本
└── script/
    ├── flow_planner_womd.yaml              # WOMD 主配置
    ├── core/
    │   └── flow_matching_womd.yaml         # WOMD 训练核心配置
    ├── data/dataset/
    │   └── womd_data.yaml                  # WOMD 数据集配置
    ├── model/
    │   └── flow_planner_womd.yaml          # WOMD 模型配置 (state_dim=2)
    └── normalization_stats/
        └── womd_norm_stats.yaml            # 归一化统计量
```

### 修改文件

```
flow_planner/model/model_utils/input_preprocess.py   # 新增 xy_only kinematic 分支
flow_planner/model/flow_planner_model/flow_planner.py # 新增 forward_inference_multimodal + FDE NMS
```

---

## Step-by-Step 运行指南

### Step 0: 环境准备

```bash
conda activate flow_planner

# 额外依赖（用于 protobuf 编译和 TFRecord 读取）
pip install grpcio-tools tensorflow
# 如不需要 tensorflow，womd_data_processor.py 包含纯 Python TFRecord reader 作为 fallback
```

### Step 1: 数据预处理 — WOMD → NPZ

#### 1.1 配置路径

编辑 `flow_planner/run_script/launch_womd_preprocess.sh`，修改以下变量：

```bash
export PROJECT_ROOT=/home/code/grid_planner       # 项目根目录
export WOMD_ROOT=/home/datasets_2/scenario         # WOMD TFRecord 目录
export NPZ_ROOT=/home/datasets_2/womd_npz          # NPZ 输出目录
NUM_WORKERS=16                                      # 并行 worker 数
```

WOMD TFRecord 目录结构应为：

```
scenario/
├── training/
│   ├── training.tfrecord-00000-of-01000
│   ├── ...
├── validation/
│   ├── validation.tfrecord-00000-of-00150
│   ├── ...
└── testing/
    ├── testing.tfrecord-00000-of-00150
    ├── ...
```

#### 1.2 运行预处理

```bash
cd $PROJECT_ROOT
bash flow_planner/run_script/launch_womd_preprocess.sh
```

每个 scenario 中的每个 target agent 会生成一个 NPZ 文件，命名为 `{scenario_id}_{track_index}.npz`。同时生成 `data_list_{split}.json` 索引文件。

预计输出量：training ~1M-4M 样本，validation ~40k-200k 样本。

#### 1.3 数据验证（可选）

随机加载几个 NPZ 文件检查格式：

```python
import numpy as np

data = np.load("/path/to/womd_npz/training/some_file.npz")
print("Keys:", list(data.keys()))
print("ego_agent_past:", data['ego_agent_past'].shape)        # (21, 14)
print("ego_current_state:", data['ego_current_state'].shape)  # (16,)
print("ego_agent_future:", data['ego_agent_future'].shape)    # (80, 3)
print("neighbor_agents_past:", data['neighbor_agents_past'].shape)  # (32, 21, 11)
print("lanes:", data['lanes'].shape)                          # (70, 20, 12)
print("route_lanes:", data['route_lanes'].shape)              # (25, 20, 12) 全零
print("static_objects:", data['static_objects'].shape)        # (5, 10) 全零
```

### Step 2: 计算归一化统计量

默认的 `womd_norm_stats.yaml` 使用与 nuPlan 相同的经验值。建议从实际 WOMD 数据中计算：

```bash
cd $PROJECT_ROOT

python -m flow_planner.data.data_process.compute_womd_norm_stats \
    --data_dir /path/to/womd_npz/training \
    --data_list /path/to/womd_npz/training/data_list_training.json \
    --output_path flow_planner/script/normalization_stats/womd_norm_stats.yaml \
    --max_samples 50000
```

该脚本会：
1. 随机采样 50000 个 NPZ 文件
2. 对每个字段（ego_past, neighbor_past, lanes, ego_future 等）计算非零元素的均值和标准差
3. 输出 YAML 格式的统计量，直接替换默认值

如果数据量较大，可以减小 `--max_samples`。统计量主要影响训练收敛速度，不影响最终效果的上界。

### Step 3: 训练

#### 3.1 配置路径

编辑 `flow_planner/run_script/launch_train_womd.sh`：

```bash
export PROJECT_ROOT=/home/code/grid_planner
export SAVE_DIR=/home/results/grid_planner_womd
export TENSORBOARD_LOG_PATH=/home/results/grid_planner_womd/tensorboard
export TRAINING_DATA=/home/datasets_2/womd_npz/training
export TRAINING_JSON=/home/datasets_2/womd_npz/training/data_list_training.json
```

#### 3.2 启动训练

```bash
cd $PROJECT_ROOT
bash flow_planner/run_script/launch_train_womd.sh
```

默认配置：
- 8 GPU DDP，batch_size=64/GPU
- 30 epochs，lr=1e-4，AdamW + warmup
- 单阶段训练，GAIW 从头启用
- Loss = Flow Matching MSE + Consistency Loss (权重 1.0 + 0.5)

训练日志和 checkpoint 保存在 `$SAVE_DIR/outputs/` 下。

#### 3.3 关键训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `state_dim` | 2 | 只预测 (x, y) |
| `kinematic` | xy_only | 使用 xy_only 分支处理 GT |
| `action_len` | 20 | 每个 action token 20 帧 |
| `action_overlap` | 10 | 相邻 token 重叠 10 帧 |
| `cfg_prob` | 0.3 | Classifier-Free Guidance 训练概率 |
| `sample_steps` | 4 | ODE 求解步数 |

### Step 4: 评估

#### 4.1 配置路径

编辑 `flow_planner/run_script/launch_eval_womd.sh`：

```bash
CONFIG_PATH=/path/to/.hydra/config.yaml   # 训练时自动生成
CKPT_PATH=/path/to/latest.pth
DATA_DIR=/home/datasets_2/womd_npz/validation
DATA_LIST=/home/datasets_2/womd_npz/validation/data_list_validation.json
OUTPUT_PATH=/home/results/grid_planner_womd/eval_results.json
```

#### 4.2 运行评估

```bash
cd $PROJECT_ROOT
bash flow_planner/run_script/launch_eval_womd.sh
```

推理流程：
1. 加载 checkpoint 和配置
2. 遍历 validation NPZ
3. 每个 target agent 过采样 N=64 条轨迹（不同 noise seed，共享 encoder）
4. FDE-based greedy NMS 选出 K=6 条代表性轨迹
5. 计算指标：minADE, minFDE, Miss Rate, mAP, Soft mAP

输出格式：

```json
{
  "ALL": {"minADE": 1.23, "minFDE": 2.45, "MissRate": 0.15, "mAP": 0.32, "soft_mAP": 0.28, "count": 44920},
  "VEHICLE": {"minADE": 1.15, "minFDE": 2.30, ...},
  "PEDESTRIAN": {"minADE": 0.85, "minFDE": 1.60, ...},
  "CYCLIST": {"minADE": 1.45, "minFDE": 2.80, ...}
}
```

### Step 5: (可选) Leaderboard 提交

WOMD 官方提交需要 `motion_submission.proto` 格式。当前实现输出 JSON metrics。
如需提交到 Leaderboard，需额外实现格式转换（计划后续添加）。

---

## 设计决策说明

### 为什么 route_lanes 和 static_objects 填零而不是改模型？

- `RouteEncoder` 对全零输入自动跳过计算（`valid_indices` 为空），输出全零向量作为加性条件 = 等效于无 route
- `StaticFusionEncoder` 全零行被 attention mask 完全屏蔽，对其他 token 输出无影响
- 保持 `token_num = 32 + 5 + 70 = 107` 不变，`JointAttention.gen_taus` 维度不变
- 论文叙事需要："同一架构，不同任务"

### 为什么不用 WTA loss？

借鉴 MotionDiffuser 的经验：简单 L2 denoising loss + 推理时过采样 + 聚类/NMS 即可覆盖多模态，
无需训练时显式引入 Winner-Takes-All loss。这简化了训练流程，同时保持多模态覆盖能力。

### 为什么禁用数据增强？

- MTR 和 MotionDiffuser 均不对 WOMD 做数据增强
- 现有 `StatePerturbation` 依赖 nuPlan 专属的 Pacifica 车辆动力学模型
- WOMD target agent 不一定是车辆，Pacifica 模型不适用

### 为什么 history padding 到 21 帧而不是改 past_time_len=11？

- `AgentFusionEncoder` 的 `token_pre_project = Mlp(in_features=past_time_len, ...)` 是固定输入维度
- 改为 11 需要修改模型架构，无法复用 nuPlan 权重
- 复制最早有效帧填充到 21 帧是安全的：模型通过 validity mask 学会忽略重复帧

---

## 数据格式对照

### NPZ 字段 → NuPlanDataSample 属性 → Encoder 输入

```
NPZ key                    NuPlanDataSample attr    Encoder 使用？
─────────────────────────   ─────────────────────    ────────────
ego_agent_past        →    ego_past                 否 (仅 augmentation)
ego_current_state     →    ego_current              部分 ([:4] + [4:6])
ego_agent_future      →    ego_future               GT (训练) / 不用 (推理)
neighbor_agents_past  →    neighbor_past             是 (8 geom + 3 type)
neighbor_agents_future→    neighbor_future_observed  GAIW 用 [:2]
lanes                 →    lanes                     是 (8 geom + 4 TL)
lanes_speed_limit     →    lanes_speedlimit          是
lanes_has_speed_limit →    lanes_has_speedlimit      是
route_lanes           →    routes                    全零 → RouteEncoder 输出零
static_objects        →    map_objects               全零 → StaticEncoder 被 mask
```
