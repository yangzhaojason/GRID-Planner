# 最终检查清单

## ✅ 已完成的修改

### 1. 修复 Dataclass 字段顺序问题
- ✅ `runner_report.py`: 将 `score` 字段移到所有必需字段之后
- ✅ `simulations_runner.py`: 更新 RunnerReport 创建，添加 `score=None`
- ✅ `executor.py`: 更新 RunnerReport 创建，添加 `score=None`
- ✅ `metric_runner.py`: 更新 RunnerReport 创建，添加 `score=None`

### 2. 创建 RL 训练启动脚本
- ✅ `launch_rl_train.sh`: 创建了完整的 RL 训练启动脚本
  - 包含所有 RL 配置参数
  - 使用用户提供的 CONFIG_FILE 和 CKPT_FILE 路径
  - 支持 resume 训练

### 3. NuPlan-Devkit 文件修改
- ✅ `runner_report.py`: 添加 score 字段
- ✅ `simulations_runner.py`: 添加获取 score 和调用 finalize_rl_simulation 的逻辑
- ✅ `simulation_builder.py`: 支持传入 scenarios 参数（用于 batch 模式）
- ✅ `run_simulation.py`: 添加 RL 模式检查和训练循环

### 4. Flow-Planner-RL 文件修改
- ✅ 所有核心文件已修改以支持 RL 模式
- ✅ Condition 保存和恢复逻辑已完善
- ✅ Encoder outputs 的 tuple 类型处理已修复

## 🔍 需要验证的关键点

### 1. Score 获取
- [ ] 验证 metric engine 能够正确计算并返回 score
- [ ] 验证 score 能够正确传递到 finalize_rl_simulation
- [ ] 验证 score 能够正确广播到所有数据点

### 2. Condition 序列化
- [ ] 验证 encoder_outputs 中的 tuple（encodings, masks）能够正确序列化和反序列化
- [ ] 验证 decoder_model_extra 中的所有字段能够正确恢复

### 3. 训练逻辑
- [ ] 验证 log_prob_new 的计算正确性
- [ ] 验证 advantage 的计算（GRPO 公式）
- [ ] 验证梯度是否正确计算和反向传播
- [ ] 验证 cfg_flags 在训练时正确设置为全 1

### 4. ODE Solver
- [ ] 验证 ManualODESolver 与原有 ODESolver 的一致性
- [ ] 验证 timestep 列表处理的正确性

## 📝 使用说明

### 启动 RL 训练

```bash
cd /home/qianzezhong/Flow-Planner-RL
bash flow_planner/run_script/launch_rl_train.sh
```

### 配置说明

在 `launch_rl_train.sh` 中可以修改以下参数：
- `RL_OUTPUT_DIR`: RL 训练输出目录
- `RL_BATCH_SIZE`: 每次 simulate 的场景数量（默认 64）
- `RL_MICRO_BATCH_SIZE`: 训练时的 micro batch 大小（默认 256）
- `RL_NOISE_STD`: 加噪标准差（默认 0.05）
- `RL_CLIP_RATIO`: PPO clip ratio（默认 0.01）
- `RL_SAVE_INTERVAL`: 每 N 个 step 保存一次权重（默认 100）
- `RL_LEARNING_RATE`: 学习率（默认 1e-4）

### Resume 训练

如果需要 resume 训练，需要：
1. 修改 `launch_rl_train.sh` 中的 `CONFIG_FILE` 和 `CKPT_FILE` 路径
2. 确保 `RL_OUTPUT_DIR` 指向之前的输出目录（如果需要继续保存到同一目录）

## ⚠️ 注意事项

1. **确保 metric 计算完成**: Score 的获取依赖于 metric engine 的计算，可能需要等待异步任务完成
2. **内存管理**: 注意及时清理不需要的数据，避免内存泄漏
3. **数值稳定性**: 注意 log_prob 计算中的数值稳定性
4. **设备管理**: 确保所有 tensor 在正确的设备上

## 🐛 已知问题和限制

1. **Score 获取**: 当前实现从 metric_statistics 中提取 score，如果 metric 计算失败或返回 None，score 将为 None
2. **Condition 序列化**: 复杂的嵌套结构（tuple, dict）需要确保能够正确序列化和反序列化
3. **Timestep 处理**: VelocityModel 中的 timestep 处理可能需要进一步优化

## 📚 相关文档

- `RL_IMPLEMENTATION_PLAN.md` - 详细技术文档
- `IMPLEMENTATION_SUMMARY.md` - 实现总结
- `CODE_COMPLETION_CHECKLIST.md` - 完成检查清单
