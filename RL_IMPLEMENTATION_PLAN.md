# 在线强化学习模块实现技术文档

## 概述
本文档详细描述了在 Flow-Planner-RL 中实现在线强化学习（Online RL）模块的技术方案，包括所有需要修改的文件、新增的文件、配置变更和具体实现逻辑。

---

## 1. 配置系统修改

### 1.1 新增配置文件
**文件路径**: `flow_planner/script/rl/rl_config.yaml`

**内容**:
```yaml
rl:
  enabled: false  # RL 模式总开关
  batch_size: 64  # 每次 simulate 的场景数量
  micro_batch_size: 256  # 训练时的 micro batch 大小
  noise_std: 0.05  # 加噪标准差
  clip_ratio: 0.01  # PPO clip ratio
  save_interval: 100  # 每 N 个 step 保存一次权重
  output_dir: null  # RL 训练输出目录
```

### 1.2 修改主配置文件
**文件路径**: `flow_planner/nuplan_simulation/planner/flow_planner.yaml`

**修改位置**: 在现有配置中添加 RL 相关参数
```yaml
flow_planner:
  # ... 现有配置 ...
  rl:
    enabled: false
    batch_size: 64
    micro_batch_size: 256
    noise_std: 0.05
    clip_ratio: 0.01
    save_interval: 100
```

---

## 2. 核心代码修改

### 2.1 修改 `build_simulations` 函数
**文件路径**: `nuplan-devkit/nuplan/planning/script/builders/simulation_builder.py`

**修改位置**: 第 70-134 行

**修改逻辑**:
1. 在函数开始处检查 `cfg.rl.enabled` 标志
2. 如果 RL 模式关闭，保持原有逻辑（第 83 行的 `for i, scenario in enumerate(scenarios)` 不变）
3. 如果 RL 模式开启：
   - 在外层添加 `while True:` 无限循环
   - 在循环内，将 `scenarios` 按 `batch_size=cfg.rl.batch_size` 分组
   - 使用 `itertools` 的 `grouper` 或手动实现分组逻辑
   - 最后一组如果不足 `batch_size` 则舍弃
   - 每次循环只处理一个 batch 的场景
   - 返回的 `simulations` 列表只包含当前 batch 的场景

**伪代码**:
```python
def build_simulations(...):
    # ... 现有代码到第 70 行 ...
    scenarios = scenario_filter.get_scenarios()
    
    # RL 模式检查
    rl_enabled = cfg.get('rl', {}).get('enabled', False)
    
    if not rl_enabled:
        # 原有逻辑：遍历所有场景
        for i, scenario in enumerate(scenarios):
            # ... 现有代码 ...
    else:
        # RL 模式：无限循环 + batch 处理
        rl_batch_size = cfg.rl.batch_size
        while True:
            # 按 batch_size 分组
            scenario_batches = [scenarios[i:i+rl_batch_size] 
                              for i in range(0, len(scenarios), rl_batch_size)]
            # 舍弃最后一组（如果不足 batch_size）
            if len(scenario_batches[-1]) < rl_batch_size:
                scenario_batches = scenario_batches[:-1]
            
            for batch_scenarios in scenario_batches:
                simulations = []
                for scenario in batch_scenarios:
                    # ... 构建 simulation 的代码 ...
                yield simulations  # 或者通过其他方式返回
```

**注意**: 由于 `build_simulations` 返回 `List[SimulationRunner]`，需要修改调用方以支持迭代器模式，或者修改为返回生成器。

---

### 2.2 修改 `run_simulation` 函数
**文件路径**: `nuplan-devkit/nuplan/planning/script/run_simulation.py`

**修改位置**: 第 37-80 行

**修改逻辑**:
1. 检查 `cfg.rl.enabled`
2. 如果 RL 模式开启：
   - 修改 `build_simulations` 的调用，支持迭代器/生成器模式
   - 添加外层循环处理每个 batch
   - 在每个 batch 完成后，调用 RL 训练逻辑
3. 如果 RL 模式关闭，保持原有逻辑

**伪代码**:
```python
def run_simulation(cfg, planners=None):
    # ... 现有初始化代码 ...
    
    rl_enabled = cfg.get('rl', {}).get('enabled', False)
    
    if not rl_enabled:
        # 原有逻辑
        runners = build_simulations(...)
        run_runners(runners=runners, ...)
    else:
        # RL 模式：循环处理 batch
        step_count = 0
        while True:
            # 获取当前 batch 的 runners
            runners = build_simulations(...)  # 需要修改为支持迭代器
            
            # 运行 simulation（不带梯度）
            reports = run_runners(runners=runners, ...)
            
            # 收集 RL 数据并训练
            rl_trainer.train_step(reports, step_count)
            
            step_count += 1
            if step_count % cfg.rl.save_interval == 0:
                rl_trainer.save_checkpoint(step_count)
```

---

### 2.3 新增 RL Trainer 类
**文件路径**: `flow_planner/rl/rl_trainer.py` (新建文件)

**类结构**:
```python
class RLTrainer:
    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = cfg.device
        self.micro_batch_size = cfg.rl.micro_batch_size
        self.clip_ratio = cfg.rl.clip_ratio
        
    def collect_data_from_reports(self, reports):
        """从 RunnerReport 中收集 RL 数据"""
        pass
        
    def train_step(self, reports, step_count):
        """执行一个训练 step"""
        pass
        
    def save_checkpoint(self, step_count):
        """保存检查点"""
        pass
```

---

### 2.4 修改 FlowPlanner 类 - 添加 RL 模式支持
**文件路径**: `flow_planner/planner.py`

**修改位置**:
1. `__init__` 方法（第 31-70 行）:
   - 添加 `rl_enabled` 参数
   - 添加 `rl_noise_std` 参数
   - 存储这些参数到实例变量

2. `compute_planner_trajectory` 方法（第 139-151 行）:
   - 检查 `rl_enabled` 标志
   - 如果开启，调用新的 `compute_planner_trajectory_rl` 方法（**use_cfg=False**）
   - 如果关闭，保持原有逻辑

**新增方法**:
```python
def compute_planner_trajectory_rl(self, current_input: PlannerInput) -> AbstractTrajectory:
    """
    RL 模式下的轨迹计算，收集 RL 数据
    
    重要：RL 模式下 use_cfg 必须为 False
    """
    # 1. 准备输入（同原有逻辑）
    # 2. 调用带数据收集的 inference（use_cfg=False）
    # 3. 返回轨迹
    pass
```

---

### 2.5 修改 FlowMatchingCore - 添加 RL 推理模式
**文件路径**: `flow_planner/core/flow_matching_core.py`

**修改位置**:
1. `__init__` 方法（第 11-22 行）:
   - 添加 `rl_enabled` 和 `rl_noise_std` 参数

2. `inference` 方法（第 34-40 行）:
   - 检查 RL 模式
   - 如果开启，调用新的 `inference_rl` 方法（**use_cfg=False**）
   - 如果关闭，保持原有逻辑

**新增方法**:
```python
def inference_rl(self, model, data, use_cfg=False, cfg_weight=None):
    """
    RL 模式下的推理，收集去噪步骤数据
    
    重要：RL 模式下 use_cfg 必须为 False，避免 classifier free guidance
    """
    # 1. 设置模型为 eval 模式（但不关闭梯度，因为后续训练需要）
    # 2. 调用模型的 forward_inference_rl 方法（use_cfg=False）
    # 3. 返回预测结果和收集的数据
    pass
```

---

### 2.6 修改 FlowPlanner 模型类 - 添加 RL 推理方法
**文件路径**: `flow_planner/model/flow_planner_model/flow_planner.py`

**修改位置**:
1. `__init__` 方法（第 16-54 行）:
   - 添加 `rl_enabled` 和 `rl_noise_std` 参数

2. `forward_inference` 方法（第 169-190 行）:
   - 保持原有逻辑不变（用于非 RL 模式）

**新增方法**:
```python
def forward_inference_rl(self, data: NuPlanDataSample, use_cfg=False, cfg_weight=None):
    """
    RL 模式下的推理，手动实现 ODE solver 并收集数据
    
    重要：RL 模式下 use_cfg 应该为 False，避免 classifier free guidance
    影响 log_prob 的准确性
    """
    # 1. 准备输入（同 forward_inference，但 use_cfg=False）
    # 2. 获取 encoder 输出（condition）
    # 3. 初始化 x_init
    # 4. 手动实现 ODE solver（与原有实现一致，但带数据收集）
    # 5. 在每一步：
    #    - 计算 x_next
    #    - 加噪声：x_noised = x_next + N(0, noise_std^2)
    #    - 保存：latent, condition, timestep, x_noised, log_prob
    # 6. 返回 sample 和收集的数据列表
    pass
```

**关键修改**:
- **RL 模式下 use_cfg 必须为 False**，避免 classifier free guidance 影响 log_prob 计算

---

### 2.7 新增手动 ODE Solver 实现
**文件路径**: `flow_planner/model/flow_planner_model/flow_utils/manual_ode_solver.py` (新建文件)

**类结构**:
```python
class ManualODESolver:
    """手动实现的 ODE Solver，与原有 ODESolver 逻辑一致，但支持梯度"""
    
    def __init__(self, velocity_model, path, step_size, method):
        self.velocity_model = velocity_model
        self.path = path
        self.step_size = step_size
        self.method = method
        
    def step(self, x, t, **model_extra):
        """
        执行一步 ODE 求解
        
        重要：t 可能是一个列表/张量，每个样本的 timestep 可能不同
        x: (B, P, T, D) - 当前状态
        t: (B,) 或 (B, 1) - 每个样本的当前 timestep（可能不同）
        """
        # 实现与原有 ODESolver 一致的逻辑
        # 支持 'midpoint' 等方法
        # 注意：需要处理每个样本 timestep 不同的情况
        pass
        
    def sample_with_data_collection(self, x_init, num_steps, noise_std, **model_extra):
        """
        带数据收集的采样（仅用于 simulation 阶段，会加噪）
        
        x_init: (B, P, T, D)
        num_steps: int - ODE 求解步数
        noise_std: float - 加噪标准差
        """
        # 1. 初始化 x = x_init
        # 2. 初始化 timesteps = [1.0, 1.0, ..., 1.0] (B个，都是1.0)
        # 3. 对于每一步：
        #    - 计算 velocity（注意 t 是列表，每个样本可能不同）
        #    - 更新 x_next（使用 midpoint 等方法）
        #    - 更新 timesteps = timesteps - step_size
        #    - 加噪声：x_noised = x_next + N(0, noise_std^2)
        #    - 计算 log_prob（x_noised 相对于 x_next 的高斯分布）
        #    - 保存数据：(x, condition, timesteps, x_noised, log_prob)
        #    - x = x_noised（作为下一步的起始）
        # 4. 返回最终 sample 和收集的数据
        pass
        
    def sample_without_noise(self, x_init, timesteps_init, num_steps, **model_extra):
        """
        不带加噪的采样（用于训练阶段，计算 log_prob_new）
        
        x_init: (B, P, T, D) - 从保存的数据中恢复的起始 latent
        timesteps_init: (B,) - 从保存的数据中恢复的起始 timesteps
        num_steps: int - ODE 求解步数
        """
        # 1. 初始化 x = x_init
        # 2. 初始化 timesteps = timesteps_init（从保存的数据恢复）
        # 3. 对于每一步：
        #    - 计算 velocity（使用当前的 timesteps，每个样本可能不同）
        #    - 更新 x_next（使用 midpoint 等方法）
        #    - 更新 timesteps = timesteps - step_size
        #    - 不加噪声！
        # 4. 返回最终的 x_next（用于计算 log_prob_new）
        pass
```

**关键实现细节**:
- **必须支持 timestep 是列表的情况**：每个样本的 timestep 可能不同
- 必须与 `flow_matching.solver.ode_solver.ODESolver` 的实现完全一致
- 支持 `midpoint` 方法（根据配置中的 `sample_method`）
- 每一步的数学公式必须一致
- **训练阶段（sample_without_noise）绝对不加噪**，直接用 ODE 求解的结果

---

### 2.8 修改 FlowODE 类 - 添加 RL 模式支持
**文件路径**: `flow_planner/model/flow_planner_model/flow_utils/flow_ode.py`

**修改位置**:
1. `__init__` 方法（第 11-27 行）:
   - 添加 `rl_enabled` 和 `rl_noise_std` 参数

2. `generate` 方法（第 44-60 行）:
   - 检查 RL 模式
   - 如果开启，调用新的 `generate_rl` 方法
   - 如果关闭，保持原有逻辑

**新增方法**:
```python
def generate_rl(self, x_init, model_fn, model_pred_type, use_cfg, noise_std, **model_extra):
    """RL 模式下的生成，手动实现 ODE 并收集数据"""
    # 1. 创建 ManualODESolver
    # 2. 调用 sample_with_data_collection
    # 3. 返回 sample 和收集的数据
    pass
```

---

### 2.9 新增 RL 数据收集器
**文件路径**: `flow_planner/rl/data_collector.py` (新建文件)

**类结构**:
```python
class RLDataCollector:
    """收集和管理 RL 训练数据"""
    
    def __init__(self):
        self.batch_data = []  # 存储当前 batch 的数据
        
    def collect_step_data(self, latent, condition, timestep, x_noised, log_prob):
        """收集单步数据"""
        pass
        
    def finalize_simulation(self, score):
        """完成一个 simulation，添加 score 到所有数据"""
        pass
        
    def get_batch_data(self):
        """获取当前 batch 的所有数据"""
        pass
        
    def clear(self):
        """清空数据"""
        pass
```

**数据结构**:
每个数据样本包含 6 个字段：
1. `latent`: torch.Tensor - 去噪起始的 latent
2. `condition`: Dict - 去噪时使用的 condition（encoder 输出等）
3. `timestep`: float - 去噪的 timestep
4. `x_noised`: torch.Tensor - 加噪后的结果
5. `log_prob`: float - log_prob（标量）
6. `score`: float - 该 simulation 的 score（广播后）

---

### 2.10 新增 RL 训练逻辑
**文件路径**: `flow_planner/rl/training.py` (新建文件)

**类结构**:
```python
class RLTrainer:
    """RL 训练器"""
    
    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        self.data_collector = RLDataCollector()
        
    def compute_advantage(self, scores, log_probs_old, log_probs_new):
        """计算 GRPO advantage"""
        # 1. 计算 R_mean 和 R_std（从 64 个 simulation 的 scores）
        # 2. 标准化 scores: (score - R_mean) / R_std
        # 3. 计算 advantage = standardized_score - (log_prob_new - log_prob_old)
        pass
        
    def compute_ppo_loss(self, log_probs_old, log_probs_new, advantages):
        """计算 PPO-styled loss"""
        # 1. 计算 ratio = exp(log_prob_new - log_prob_old)
        # 2. 计算 clipped_ratio = clip(ratio, 1-clip_ratio, 1+clip_ratio)
        # 3. loss = -min(ratio * advantage, clipped_ratio * advantage)
        # 4. 统计 ratio 的 mean, max, min 和被 clip 的比例
        pass
        
    def train_step(self, batch_data, step_count):
        """
        执行一个训练 step
        
        重要：训练阶段不使用随机性，不加噪，直接使用 ODE 求解结果
        """
        # 1. 展平数据：list(T, C) -> (N,)
        # 2. Shuffle 数据（仅用于数据顺序，不影响确定性）
        # 3. 按 micro_batch_size 划分
        # 4. 对于每个 micro batch：
        #    - 从保存的数据中恢复：latent, condition, timesteps, x_noised, log_prob_old, score
        #    - 构建模型输入（使用保存的 latent 和 condition）
        #    - 前向传播（带梯度，use_cfg=False）
        #    - 使用 ManualODESolver.sample_without_noise 计算 x_next_new
        #      注意：使用保存的 timesteps 作为起始，确保与 simulation 阶段一致
        #    - 计算 log_prob_new：使用 x_noised（保存的）和 x_next_new（新计算的）
        #      公式：log_prob_new = log_prob(x_noised | x_next_new, noise_std)
        #    - 计算 advantage（使用保存的 score 和 log_prob_old）
        #    - 计算 loss
        #    - 反向传播
        # 5. 打印统计信息
        pass
```

**关键实现细节**:
- **训练阶段绝对不加噪**：使用 `sample_without_noise` 方法
- **使用保存的 timesteps**：确保与 simulation 阶段的 timestep 序列一致
- **log_prob_new 计算**：使用保存的 `x_noised` 和新计算的 `x_next_new`
- **无随机性**：除了数据 shuffle（不影响确定性），所有计算都是确定性的

---

### 2.11 修改 SimulationRunner - 收集 RL 数据
**文件路径**: `nuplan-devkit/nuplan/planning/simulation/runner/simulations_runner.py`

**修改位置**:
1. `__init__` 方法（第 31-38 行）:
   - 添加 `rl_data_collector` 参数（可选）

2. `run` 方法（第 74-133 行）:
   - 在 `compute_trajectory` 调用后，如果有 `rl_data_collector`，收集数据
   - 在 simulation 结束后，获取 score 并调用 `finalize_simulation`

**注意**: 需要从 `RunnerReport` 中获取 score。可能需要修改 `RunnerReport` 或通过其他方式获取。

---

### 2.12 修改 RunnerReport - 添加 Score 字段
**文件路径**: `nuplan-devkit/nuplan/planning/simulation/runner/runner_report.py`

**修改位置**: 第 12-26 行

**修改内容**:
```python
@dataclass
class RunnerReport:
    # ... 现有字段 ...
    score: Optional[float] = None  # 新增：该 simulation 的最终 score
```

### 2.13 修改 SimulationRunner - 获取并存储 Score
**文件路径**: `nuplan-devkit/nuplan/planning/simulation/runner/simulations_runner.py`

**修改位置**: `run` 方法末尾（第 127-133 行之后）

**修改逻辑**:
```python
# 在 simulation 结束后，获取 score
if hasattr(self.simulation.callback, 'get_final_score'):
    report.score = self.simulation.callback.get_final_score()
elif hasattr(self._planner, 'get_simulation_score'):
    report.score = self._planner.get_simulation_score()
```

**或者更直接的方式**:
- 在 `SimulationRunner.run` 中，simulation 结束后，从 `metric_engine` 直接获取
- 需要确保 `MetricCallback` 的 `metric_engine` 可以被访问

---

## 3. 数据流和梯度管理

### 3.1 Simulation 阶段（无梯度）
- 所有操作在 `torch.no_grad()` 上下文中执行
- 模型设置为 `eval()` 模式
- 数据收集但不计算梯度

### 3.2 Training 阶段（有梯度）
- 模型设置为 `train()` 模式
- 在 `torch.enable_grad()` 上下文中执行
- 重新构建输入，进行前向传播
- 计算 loss 并反向传播

---

## 4. 多卡训练支持

### 4.1 修改 RL Trainer
**文件路径**: `flow_planner/rl/rl_trainer.py`

**修改**:
- 使用 `torch.nn.DataParallel` 或 `torch.nn.parallel.DistributedDataParallel`
- 在数据收集阶段，确保数据在正确的设备上
- 在训练阶段，确保模型和数据在同一设备上

---

## 5. 日志和监控

### 5.1 添加日志输出
**位置**: `flow_planner/rl/training.py`

**输出内容**:
- 每个 step 的 loss
- 成功率（成功 simulation 的比例）
- 平均 score
- Ratio 的 mean, max, min
- 被 clip 的比例

---

## 6. 检查点保存

### 6.1 实现检查点保存
**位置**: `flow_planner/rl/training.py`

**保存内容**:
- 模型权重
- Optimizer 状态
- Step count
- 配置信息

**保存路径**: `cfg.rl.output_dir / f"checkpoint_step_{step_count}.pth"`

---

## 7. 关键实现细节

### 7.1 ODE Solver 一致性（**关键**）
- **必须**确保手动实现的 ODE Solver 与原有的 `ODESolver` 完全一致
- 需要仔细阅读 `flow_matching.solver.ode_solver` 的源码
- 特别是 `midpoint` 方法的实现

**实现步骤**:
1. 找到 `flow_matching.solver.ode_solver.ODESolver` 的源码位置
2. 仔细阅读 `sample` 方法的实现
3. 对于 `midpoint` 方法，确保以下步骤一致：
   - 时间步长的计算方式
   - 中间点的计算方式（k1, k2）
   - 更新公式
4. **强烈建议**: 先实现一个测试函数，对比手动实现和原有实现的输出
5. 使用相同的随机种子，确保输出完全一致（在无梯度模式下）

**关键点**:
- 时间步的索引和计算方式
- Velocity 的计算和调用方式
- 数值积分的具体公式
- 边界条件的处理

### 7.2 Log Prob 计算
- **Simulation 阶段（收集数据）**:
  - 使用高斯分布公式：`log_prob_old = -0.5 * (x_noised - x_next_old)^2 / noise_std^2 - log(noise_std * sqrt(2*pi))`
  - 对所有维度取平均，得到标量
  - 保存：`x_noised`（加噪后的结果）和 `log_prob_old`

- **Training 阶段（计算新策略）**:
  - **重要**：使用保存的 `x_noised` 和新计算的 `x_next_new`（不加噪）
  - 公式：`log_prob_new = -0.5 * (x_noised - x_next_new)^2 / noise_std^2 - log(noise_std * sqrt(2*pi))`
  - 这样计算的 ratio 才是准确的，因为都基于相同的 `x_noised`

### 7.3 Score 获取
- **方案1（推荐）**: 修改 `RunnerReport` 添加 `score` 字段
  - 在 `SimulationRunner.run` 中，simulation 结束后，从 `metric_engine` 获取最终 score
  - 通过 `metric_engine.final_metric_score` 或从 metric aggregator 获取
  - 存储到 `RunnerReport.score` 字段
  
- **方案2**: 在 `MetricCallback.on_simulation_end` 中计算并存储 score
  - 通过 callback 机制传递 score 到 runner
  
- **方案3**: 在 RL Trainer 中，从保存的 metric parquet 文件读取 score
  - 需要知道 scenario 和 planner 的对应关系
  
**推荐使用方案1**，因为最直接且不需要额外的文件 I/O

### 7.4 数据展平
- `list(T, C)` 中每个元素是 `(T_i, C)` 的 tensor
- 展平为 `(N,)` 其中 `N = sum(T_i) * C`
- 需要正确处理不同长度的 simulation

---

## 8. 文件清单

### 8.1 需要修改的文件
1. `nuplan-devkit/nuplan/planning/script/builders/simulation_builder.py`
2. `nuplan-devkit/nuplan/planning/script/run_simulation.py`
3. `flow_planner/planner.py`
4. `flow_planner/core/flow_matching_core.py`
5. `flow_planner/model/flow_planner_model/flow_planner.py`
6. `flow_planner/model/flow_planner_model/flow_utils/flow_ode.py`
7. `nuplan-devkit/nuplan/planning/simulation/runner/simulations_runner.py`
8. `nuplan-devkit/nuplan/planning/simulation/runner/runner_report.py`
9. `flow_planner/nuplan_simulation/planner/flow_planner.yaml`

### 8.2 需要新建的文件
1. `flow_planner/rl/rl_trainer.py`
2. `flow_planner/rl/data_collector.py`
3. `flow_planner/rl/training.py`
4. `flow_planner/model/flow_planner_model/flow_utils/manual_ode_solver.py`
5. `flow_planner/script/rl/rl_config.yaml`
6. `flow_planner/rl/__init__.py`

---

## 9. 测试和调试建议

### 9.1 单元测试
- 测试手动 ODE Solver 与原有实现的一致性
- 测试数据收集的正确性
- 测试 advantage 和 loss 的计算

### 9.2 集成测试
- 测试 RL 模式开关是否正常工作
- 测试梯度是否正确计算
- 测试多卡训练是否正常

### 9.3 调试输出
- 在关键位置添加 print 语句
- 输出 tensor 的 shape 和数值范围
- 验证数据流的正确性

---

## 10. 注意事项

1. **最大程度保留原代码**: 所有修改都应该通过 `if rl_enabled` 分支实现，非 RL 模式完全不受影响

2. **RL 模式下不使用 Classifier Free Guidance**:
   - 所有 RL 相关的推理调用都必须设置 `use_cfg=False`
   - 这包括：`inference_rl`, `forward_inference_rl`, `compute_planner_trajectory_rl`
   - 原因：避免 CFG 影响 log_prob 的准确性

3. **Timestep 列表支持**:
   - 手动实现的 ODE Solver 必须支持 timestep 是列表的情况
   - 每个样本的 timestep 可能不同，需要正确处理
   - 在 simulation 阶段，初始 timestep 都是 1.0，但后续可能因为不同的更新而不同

4. **训练阶段无随机性**:
   - 训练阶段（第二个阶段）绝对不加噪
   - 使用保存的 `x_noised` 和新计算的 `x_next_new` 计算 `log_prob_new`
   - 使用保存的 `timesteps` 作为 ODE 求解的起始，确保与 simulation 阶段一致
   - 除了数据 shuffle（不影响确定性），所有计算都是确定性的

5. **梯度管理**: 确保 simulation 阶段无梯度，training 阶段有梯度

6. **设备管理**: 确保所有 tensor 在正确的设备上

7. **内存管理**: 注意及时清理不需要的数据，避免内存泄漏

8. **数值稳定性**: 注意 log_prob 计算中的数值稳定性

9. **一致性**: ODE Solver 实现必须与原有实现完全一致

---

## 11. 实现顺序建议

1. **第一阶段：基础设施**
   - 实现配置系统和开关
   - 修改 `RunnerReport` 添加 score 字段
   - 修改 `SimulationRunner` 获取并存储 score

2. **第二阶段：ODE Solver**
   - 仔细阅读 `flow_matching.solver.ode_solver.ODESolver` 源码
   - 实现 `ManualODESolver`，确保与原有一致
   - 编写单元测试验证一致性

3. **第三阶段：数据收集**
   - 实现 `RLDataCollector`
   - 实现 `forward_inference_rl` 方法
   - 实现手动 ODE 求解和数据收集逻辑
   - 测试数据收集的正确性

4. **第四阶段：训练逻辑**
   - 实现 advantage 计算（GRPO）
   - 实现 PPO-styled loss
   - 实现 micro batch 训练循环
   - 添加统计信息输出

5. **第五阶段：集成**
   - 修改 `build_simulations` 支持 batch 循环
   - 修改 `run_simulation` 集成 RL 训练
   - 实现检查点保存
   - 添加多卡训练支持

6. **第六阶段：测试和优化**
   - 端到端测试
   - 性能优化
   - 内存优化
   - 数值稳定性检查

## 12. 关键代码片段参考

### 12.1 Log Prob 计算
```python
def compute_log_prob(x_noised, x_next, noise_std):
    """
    计算高斯分布的 log_prob
    x_noised: (B, P, T, D) - 加噪后的值
    x_next: (B, P, T, D) - 原始值
    noise_std: float - 噪声标准差
    """
    diff = x_noised - x_next
    log_prob = -0.5 * (diff ** 2).sum(dim=-1) / (noise_std ** 2)
    log_prob = log_prob - 0.5 * np.log(2 * np.pi) - np.log(noise_std)
    log_prob = log_prob.mean()  # 对所有维度取平均，得到标量
    return log_prob
```

### 12.2 GRPO Advantage 计算
```python
def compute_advantage(scores, log_probs_old, log_probs_new):
    """
    scores: List[float] - 64 个 simulation 的 scores
    log_probs_old: torch.Tensor - 旧策略的 log_prob
    log_probs_new: torch.Tensor - 新策略的 log_prob
    """
    R_mean = np.mean(scores)
    R_std = np.std(scores) + 1e-8  # 避免除零
    
    # 标准化 scores（广播到每个数据点）
    standardized_scores = (scores - R_mean) / R_std
    
    # 计算 advantage
    advantages = standardized_scores - (log_probs_new - log_probs_old)
    return advantages
```

### 12.3 PPO Loss 计算
```python
def compute_ppo_loss(log_probs_old, log_probs_new, advantages, clip_ratio):
    """
    log_probs_old: torch.Tensor - 从保存的数据中获取
    log_probs_new: torch.Tensor - 使用新模型计算，基于保存的 x_noised 和新计算的 x_next_new
    advantages: torch.Tensor - 从 GRPO 公式计算
    clip_ratio: float
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    # 统计信息
    ratio_mean = ratio.mean().item()
    ratio_max = ratio.max().item()
    ratio_min = ratio.min().item()
    clip_fraction = ((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)).float().mean().item()
    
    return loss, {
        'ratio_mean': ratio_mean,
        'ratio_max': ratio_max,
        'ratio_min': ratio_min,
        'clip_fraction': clip_fraction
    }
```

### 12.4 训练阶段的 Log Prob 计算（关键）
```python
def compute_log_prob_new(x_noised, x_next_new, noise_std):
    """
    训练阶段计算 log_prob_new
    
    重要：
    - x_noised: 从保存的数据中获取（simulation 阶段加噪后的结果）
    - x_next_new: 使用新模型和 ODE solver 计算（不加噪）
    - 使用相同的 noise_std 计算 log_prob
    
    这样计算的 ratio 才是准确的，因为：
    - log_prob_old = log_prob(x_noised | x_next_old, noise_std)
    - log_prob_new = log_prob(x_noised | x_next_new, noise_std)
    - ratio = exp(log_prob_new - log_prob_old)
    """
    diff = x_noised - x_next_new
    log_prob = -0.5 * (diff ** 2).sum(dim=-1) / (noise_std ** 2)
    log_prob = log_prob - 0.5 * np.log(2 * np.pi) - np.log(noise_std)
    log_prob = log_prob.mean()  # 对所有维度取平均，得到标量
    return log_prob
```

### 12.5 Timestep 列表处理示例
```python
def step_with_variable_timesteps(x, timesteps, velocity_model, step_size, method='midpoint'):
    """
    处理每个样本 timestep 不同的情况
    
    x: (B, P, T, D) - 当前状态
    timesteps: (B,) - 每个样本的当前 timestep（可能不同）
    step_size: float - ODE 步长
    """
    B = x.shape[0]
    
    if method == 'midpoint':
        # 计算 k1
        v1 = velocity_model(x, timesteps)  # (B, P, T, D)
        
        # 计算中间点
        t_mid = timesteps - step_size / 2  # (B,)
        x_mid = x + step_size / 2 * v1
        
        # 计算 k2
        v2 = velocity_model(x_mid, t_mid)  # (B, P, T, D)
        
        # 更新
        x_next = x + step_size * v2
        timesteps_next = timesteps - step_size  # (B,)
        
        return x_next, timesteps_next
```
