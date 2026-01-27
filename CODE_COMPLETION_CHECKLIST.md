# 代码完成检查清单

## ✅ 已完成

### 新建文件
- [x] `flow_planner/script/rl/rl_config.yaml`
- [x] `flow_planner/rl/__init__.py`
- [x] `flow_planner/rl/data_collector.py`
- [x] `flow_planner/rl/rl_trainer.py`
- [x] `flow_planner/rl/integration.py`
- [x] `flow_planner/model/flow_planner_model/flow_utils/manual_ode_solver.py`

### 修改的文件（Flow-Planner-RL）
- [x] `flow_planner/nuplan_simulation/planner/flow_planner.yaml`
- [x] `flow_planner/planner.py`
- [x] `flow_planner/core/flow_matching_core.py`
- [x] `flow_planner/model/flow_planner_model/flow_planner.py`
- [x] `flow_planner/model/flow_planner_model/flow_utils/flow_ode.py`
- [x] `flow_planner/model/flow_planner_model/flow_utils/velocity_model.py`

## ⚠️ 需要手动修改（NuPlan-Devkit）

由于 nuplan-devkit 目录可能没有写权限，需要手动修改以下文件。详细说明见 `NUPLAN_DEVKIT_MODIFICATIONS.md`：

- [ ] `nuplan-devkit/nuplan/planning/simulation/runner/runner_report.py`
  - 添加 `score: Optional[float] = None` 字段

- [ ] `nuplan-devkit/nuplan/planning/simulation/runner/simulations_runner.py`
  - 在 `run` 方法中：
    1. 获取 score 并存储到 report
    2. 在 simulation 结束后调用 `planner.finalize_rl_simulation(score)`

- [ ] `nuplan-devkit/nuplan/planning/script/builders/simulation_builder.py`
  - 添加 RL 模式的 batch 分组逻辑
  - 支持按 batch_size 分组 scenarios

- [ ] `nuplan-devkit/nuplan/planning/script/run_simulation.py`
  - 添加 RL 模式检查
  - 集成 RL 训练循环

## 🔍 需要验证的关键点

### 1. ODE Solver 一致性
- [ ] 验证 `ManualODESolver` 与原有 `ODESolver` 的输出一致性
- [ ] 测试 midpoint 方法的正确性
- [ ] 验证 timestep 列表处理的正确性

### 2. 数据收集
- [ ] 验证数据收集的完整性（6 个字段）
- [ ] 验证 condition 的序列化和反序列化
- [ ] 验证 score 的正确广播

### 3. 训练逻辑
- [ ] 验证 log_prob_new 的计算正确性
- [ ] 验证 advantage 的计算（GRPO 公式）
- [ ] 验证 PPO loss 的计算
- [ ] 验证梯度是否正确计算和反向传播

### 4. 集成
- [ ] 验证 RL 模式开关是否正常工作
- [ ] 验证非 RL 模式不受影响
- [ ] 验证检查点保存和加载

## 📝 使用前检查

1. **配置文件**：确保 `flow_planner.yaml` 中的 RL 配置正确
2. **NuPlan-Devkit 修改**：确保所有必要的文件都已修改
3. **Score 获取**：确保 metric engine 能够正确计算并返回 score
4. **设备配置**：确保 CUDA 设备配置正确

## 🐛 常见问题排查

### 问题 1: Score 为 None
- 检查 `RunnerReport.score` 是否正确设置
- 检查 metric engine 是否正确计算 score
- 检查 `finalize_rl_simulation` 是否被调用

### 问题 2: 数据收集为空
- 检查 `data_collector` 是否正确初始化
- 检查 `collect_step_data` 是否被调用
- 检查 RL 模式是否已启用

### 问题 3: 梯度为 None
- 检查模型是否设置为 `train()` 模式
- 检查 `torch.enable_grad()` 上下文是否正确
- 检查 condition 的 tensor 是否设置了 `requires_grad=True`

### 问题 4: ODE Solver 输出不一致
- 检查 `ManualODESolver` 的实现是否与原有实现一致
- 检查 timestep 的处理是否正确
- 检查 step_size 的计算是否正确

## 📚 相关文档

- `RL_IMPLEMENTATION_PLAN.md` - 详细技术文档
- `NUPLAN_DEVKIT_MODIFICATIONS.md` - NuPlan-Devkit 修改说明
- `IMPLEMENTATION_SUMMARY.md` - 实现总结
