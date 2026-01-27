# NuPlan-Devkit 文件修改说明

由于 nuplan-devkit 目录可能没有写权限，需要手动修改以下文件：

## 1. 修改 `runner_report.py`

**文件路径**: `nuplan-devkit/nuplan/planning/simulation/runner/runner_report.py`

**修改位置**: 第 12-26 行

**修改内容**:
```python
@dataclass
class RunnerReport:
    """Report for a runner."""

    succeeded: bool  # True if simulation was successful
    error_message: Optional[str]  # None if simulation succeeded, traceback if it failed
    start_time: float  # Time simulation.run() was called
    end_time: Optional[float]  # Time simulation.run() returned, when the error was logged, or None temporarily
    planner_report: Optional[PlannerReport]  # Planner report containing stats about planner runtime, None if the
    # runner didn't run a planner (eg. MetricRunner), or when a run fails
    score: Optional[float] = None  # 新增：Final score of the simulation (for RL training)

    # Metadata about the simulations
    scenario_name: str
    planner_name: str
    log_name: str
```

## 2. 修改 `simulations_runner.py`

**文件路径**: `nuplan-devkit/nuplan/planning/simulation/runner/simulations_runner.py`

**修改位置**: `run` 方法末尾（第 127-133 行之后）

**修改内容**:
在 `return report` 之前添加：

```python
        # 获取 score（如果可用）
        if hasattr(self.simulation.callback, 'get_final_score'):
            report.score = self.simulation.callback.get_final_score()
        elif hasattr(self.planner, 'get_simulation_score'):
            report.score = self.planner.get_simulation_score()
        else:
            # 尝试从 metric engine 获取
            for callback in self.simulation.callback.callbacks:
                if hasattr(callback, 'metric_engine') and callback.metric_engine is not None:
                    # 从 metric engine 获取最终 score
                    # 这里需要根据实际的 metric engine 接口来实现
                    try:
                        if hasattr(callback.metric_engine, 'final_metric_score'):
                            report.score = callback.metric_engine.final_metric_score
                    except:
                        pass
                    break
```

## 3. 修改 `simulation_builder.py`

**文件路径**: `nuplan-devkit/nuplan/planning/script/builders/simulation_builder.py`

**修改位置**: 第 70-134 行

**修改内容**:
在 `scenarios = scenario_filter.get_scenarios()` 之后添加：

```python
    # RL 模式检查
    rl_enabled = cfg.get('rl', {}).get('enabled', False)
    
    if rl_enabled:
        # RL 模式：按 batch_size 分组
        rl_batch_size = cfg.rl.batch_size
        scenario_batches = []
        
        for i in range(0, len(scenarios), rl_batch_size):
            batch = scenarios[i:i+rl_batch_size]
            if len(batch) == rl_batch_size:  # 只保留完整的 batch
                scenario_batches.append(batch)
        
        # 如果最后一个 batch 不完整，舍弃
        # 这里返回第一个 batch，后续 batch 在循环中处理
        if len(scenario_batches) > 0:
            scenarios = scenario_batches[0]  # 使用第一个 batch
        else:
            scenarios = []  # 没有完整的 batch
```

**注意**: 完整的 RL 循环逻辑需要在 `run_simulation.py` 中实现。

## 4. 修改 `run_simulation.py`

**文件路径**: `nuplan-devkit/nuplan/planning/script/run_simulation.py`

**修改位置**: `run_simulation` 函数（第 37-80 行）

**修改内容**:
在函数开始处添加 RL 模式检查：

```python
def run_simulation(cfg: DictConfig, planners: Optional[Union[AbstractPlanner, List[AbstractPlanner]]] = None) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Helper function for main to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    :param planners: Pre-built planner(s) to run in simulation. Can either be a single planner or list of planners.
    """
    # 检查 RL 模式
    rl_enabled = cfg.get('rl', {}).get('enabled', False)
    
    if rl_enabled:
        # RL 模式：使用 RL 训练循环
        from flow_planner.rl.integration import run_rl_simulation_loop
        
        # 获取模型和优化器
        # 这里需要根据实际配置来获取
        # 假设从 planners 中获取模型
        if planners is None or len(planners) == 0:
            raise ValueError("RL mode requires planners to be provided")
        
        planner = planners[0] if isinstance(planners, list) else planners
        model = planner._planner  # 获取底层模型
        
        # 创建优化器（需要根据配置）
        import torch.optim as optim
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # 默认学习率，应该从配置读取
        
        # 运行 RL 训练循环
        run_rl_simulation_loop(
            cfg=cfg,
            build_simulations_fn=build_simulations,
            run_runners_fn=lambda runners, cfg: run_runners(runners=runners, common_builder=common_builder, cfg=cfg, profiler_name='running_simulation'),
            model=model,
            optimizer=optimizer,
            device=planner._device
        )
        return
    
    # 原有逻辑（非 RL 模式）
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)
    
    # ... 其余原有代码保持不变 ...
```

**注意**: 这个修改需要根据实际的代码结构来调整，特别是 `common_builder` 的创建位置。

## 5. 修改 `metric_callback.py`（可选）

**文件路径**: `nuplan-devkit/nuplan/planning/simulation/callback/metric_callback.py`

如果需要从 metric callback 获取 score，可以添加方法：

```python
def get_final_score(self) -> Optional[float]:
    """获取最终 score（如果可用）"""
    # 这里需要根据实际的 metric engine 接口来实现
    # 可能需要等待 metric 计算完成
    return None  # 占位符
```

## 注意事项

1. 这些修改需要根据实际的 nuplan-devkit 版本和接口来调整
2. Score 的获取方式可能需要根据实际的 metric aggregator 实现来修改
3. RL 循环的实现可能需要进一步优化，特别是 scenarios 的循环处理
4. 确保所有修改都通过 `rl_enabled` 标志来控制，不影响非 RL 模式
