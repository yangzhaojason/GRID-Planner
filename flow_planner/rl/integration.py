"""
RL 集成模块：处理 RL 模式下的 simulation 和训练循环
"""
import logging
from typing import List, Optional
from omegaconf import DictConfig
import torch

from nuplan.planning.simulation.runner.runner_report import RunnerReport
from flow_planner.rl.rl_trainer import RLTrainer

logger = logging.getLogger(__name__)


def run_rl_simulation_loop(
    cfg: DictConfig,
    build_simulations_fn,
    run_runners_fn,
    model,
    optimizer,
    device='cuda',
    all_scenarios=None
):
    """
    RL 模式下的 simulation 和训练循环
    
    Args:
        cfg: 配置对象
        build_simulations_fn: 构建 simulations 的函数（需要支持 batch 模式）
        run_runners_fn: 运行 runners 的函数
        model: FlowPlanner 模型
        optimizer: 优化器
        device: 设备
        all_scenarios: 所有 scenarios 列表（如果 build_simulations_fn 不支持 batch 模式）
    """
    rl_cfg = cfg.rl
    batch_size = rl_cfg.batch_size
    
    # 创建 RL Trainer
    rl_trainer = RLTrainer(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        device=device
    )
    
    step_count = 0
    
    # 如果提供了所有 scenarios，手动处理 batch
    if all_scenarios is not None:
        scenario_batches = []
        for i in range(0, len(all_scenarios), batch_size):
            batch = all_scenarios[i:i+batch_size]
            if len(batch) == batch_size:  # 只保留完整的 batch
                scenario_batches.append(batch)
        
        # 无限循环（重复使用 scenarios）
        batch_idx = 0
        while True:
            logger.info(f"RL Training Step {step_count}, Batch {batch_idx % len(scenario_batches)}")
            
            # 获取当前 batch 的 scenarios
            current_batch_scenarios = scenario_batches[batch_idx % len(scenario_batches)]
            batch_idx += 1
            
            # 构建当前 batch 的 simulations
            # 这里需要修改 build_simulations 以接受 scenarios 参数
            # 或者创建一个新的函数来处理 batch
            try:
                runners = build_simulations_fn(cfg, scenarios=current_batch_scenarios)
            except TypeError:
                # 如果 build_simulations 不支持 scenarios 参数，需要其他方式
                logger.error("build_simulations_fn does not support scenarios parameter. Need to modify it.")
                break
            
            if len(runners) == 0:
                logger.warning("No runners created")
                step_count += 1
                continue
            
            # 运行 simulation（不带梯度）
            with torch.no_grad():
                reports = run_runners_fn(runners=runners, cfg=cfg)
            
            # 收集 scores 和数据
            scores = []
            batch_data = []
            
            for report in reports:
                if report.succeeded and report.score is not None:
                    scores.append(report.score)
                    
                    # 从 planner 获取收集的数据
                    # 需要从对应的 runner 获取 planner
                    # 这里需要根据实际的 reports 结构来调整
                    pass
            
            # 从所有 runners 的 planners 获取数据
            for runner in runners:
                if hasattr(runner, 'planner') and hasattr(runner.planner, 'get_rl_data_collector'):
                    data_collector = runner.planner.get_rl_data_collector()
                    if data_collector is not None:
                        batch_data.extend(data_collector.get_batch_data())
                        data_collector.clear()
            
            if len(scores) == 0 or len(batch_data) == 0:
                logger.warning(f"Step {step_count}: No valid data collected (scores: {len(scores)}, data: {len(batch_data)})")
                step_count += 1
                continue
            
            # 训练
            rl_trainer.train_step(batch_data, scores, step_count)
            
            step_count += 1
            
            # 检查是否达到最大步数（可选）
            if hasattr(rl_cfg, 'max_steps') and step_count >= rl_cfg.max_steps:
                logger.info(f"Reached max steps {rl_cfg.max_steps}")
                break
    else:
        # 如果 build_simulations_fn 支持 batch 模式
        logger.info("Using build_simulations_fn with batch mode support")
        # 这里需要 build_simulations_fn 支持迭代器或生成器模式
        # 具体实现需要根据修改后的 build_simulations 来调整
        pass
