import logging
import os
import time
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Union

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.script.builders.simulation_builder import build_simulations
from nuplan.planning.script.builders.simulation_callback_builder import (
    build_callbacks_worker,
    build_simulation_callbacks,
)
from nuplan.planning.script.utils import run_runners, set_default_path, set_up_common_builder
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.executor import execute_runners

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', 'config/simulation')

if os.environ.get('NUPLAN_HYDRA_CONFIG_PATH') is not None:
    CONFIG_PATH = os.path.join('../../../../', CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != 'simulation':
    CONFIG_PATH = os.path.join(CONFIG_PATH, 'simulation')
CONFIG_NAME = 'default_simulation'


def run_simulation(cfg: DictConfig, planners: Optional[Union[AbstractPlanner, List[AbstractPlanner]]] = None) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Helper function for main to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    :param planners: Pre-built planner(s) to run in simulation. Can either be a single planner or list of planners.
    """
    # 检查 RL 模式
    # RL 配置在 cfg.planner.flow_planner.rl.enabled（从启动脚本传递）
    rl_enabled = False
    if 'planner' in cfg and hasattr(cfg.planner, 'flow_planner') and hasattr(cfg.planner.flow_planner, 'rl'):
        rl_enabled = cfg.planner.flow_planner.rl.get('enabled', False)
    elif 'rl' in cfg:
        # 兼容顶层配置（如果存在）
        rl_enabled = cfg.rl.get('enabled', False)
    
    if rl_enabled:
        # RL 模式：使用 RL 训练循环
        logger.info("RL mode enabled, starting RL training loop...")
        _run_rl_simulation(cfg=cfg, planners=planners)
        return
    # 原有逻辑（非 RL 模式）
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    profiler_name = 'building_simulation'
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

    # Build simulation callbacks
    callbacks_worker_pool = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg=cfg, output_dir=common_builder.output_dir, worker=callbacks_worker_pool)

    # Remove planner from config to make sure run_simulation does not receive multiple planner specifications.
    if planners and 'planner' in cfg.keys():
        logger.info('Using pre-instantiated planner. Ignoring planner in config')
        OmegaConf.set_struct(cfg, False)
        cfg.pop('planner')
        OmegaConf.set_struct(cfg, True)

    # Construct simulations
    if isinstance(planners, AbstractPlanner):
        planners = [planners]

    runners = build_simulations(
        cfg=cfg,
        callbacks=callbacks,
        worker=common_builder.worker,
        pre_built_planners=planners,
        callbacks_worker=callbacks_worker_pool,
    )

    if common_builder.profiler:
        # Stop simulation construction profiling
        common_builder.profiler.save_profiler(profiler_name)

    logger.info('Running simulation...')
    run_runners(runners=runners, common_builder=common_builder, cfg=cfg, profiler_name='running_simulation')
    logger.info('Finished running simulation!')


def _run_rl_simulation(cfg: DictConfig, planners: Optional[Union[AbstractPlanner, List[AbstractPlanner]]] = None) -> None:
    """
    RL 模式下的 simulation 和训练循环
    
    :param cfg: Configuration that is used to run the experiment.
    :param planners: Pre-built planner(s) to run in simulation.
    """
    import torch
    import torch.optim as optim
    from flow_planner.rl.rl_trainer import RLTrainer
    # 单卡模式：每个 scenario 重复 rollouts_per_scenario 次（不同 seed），用于 GRPO 计算 per-scenario mean/std。
    from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.script.builders.utils.utils_type import is_target_type

    def _pct(part: float, total: float) -> str:
        if total <= 0:
            return "0.0%"
        return f"{(100.0 * part / total):.1f}%"

    def _pctl(sorted_vals: List[float], q: float) -> float:
        """Return quantile value from sorted list, using nearest-rank on [0,1]."""
        assert 0.0 <= q <= 1.0
        n = len(sorted_vals)
        if n == 0:
            return 0.0
        idx = int(round(q * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return float(sorted_vals[idx])

    def _build_aligned_score_cfg(cfg: DictConfig) -> dict:
        """
        Build a plain-python score config matching metric aggregator (per-scenario score).
        This aligns RL rollout score computation with evaluation aggregator logic.
        """
        if not hasattr(cfg, "metric_aggregator"):
            raise ValueError("cfg.metric_aggregator not found: cannot align RL score with evaluator aggregator.")

        metric_agg_container = OmegaConf.to_container(cfg.metric_aggregator, resolve=True)
        if not isinstance(metric_agg_container, dict):
            raise TypeError(f"cfg.metric_aggregator must be a dict-like config, got {type(metric_agg_container)}")
        if len(metric_agg_container) != 1:
            raise ValueError(
                f"Expected exactly 1 metric aggregator config for alignment, got {list(metric_agg_container.keys())}"
            )

        agg_cfg = next(iter(metric_agg_container.values()))
        if not isinstance(agg_cfg, dict):
            raise TypeError(f"metric aggregator config must be a dict, got {type(agg_cfg)}")

        if "metric_weights" not in agg_cfg:
            raise KeyError("metric_aggregator.<name>.metric_weights missing in config")
        metric_weights = agg_cfg["metric_weights"]
        if not isinstance(metric_weights, dict):
            raise TypeError(f"metric_weights must be dict, got {type(metric_weights)}")

        multiple_metrics = []
        if "multiple_metrics" in agg_cfg:
            mm = agg_cfg["multiple_metrics"]
            if not isinstance(mm, list):
                raise TypeError(f"multiple_metrics must be a list, got {type(mm)}")
            multiple_metrics = mm

        # Basic validation: weights numeric, names str
        for k, v in metric_weights.items():
            if not isinstance(k, str):
                raise TypeError(f"metric_weights key must be str, got {type(k)}")
            if not isinstance(v, (int, float)):
                raise TypeError(f"metric_weights['{k}'] must be number, got {type(v)}")
        for m in multiple_metrics:
            if not isinstance(m, str):
                raise TypeError(f"multiple_metrics items must be str, got {type(m)}")

        return {
            "metric_weights": metric_weights,
            "multiple_metrics": multiple_metrics,
        }
    
    # 获取 RL 配置（从 cfg.planner.flow_planner.rl）
    if 'planner' in cfg and hasattr(cfg.planner, 'flow_planner') and hasattr(cfg.planner.flow_planner, 'rl'):
        rl_cfg = cfg.planner.flow_planner.rl
    elif 'rl' in cfg:
        # 兼容顶层配置（如果存在）
        rl_cfg = cfg.rl
    else:
        raise ValueError("RL configuration not found in cfg.planner.flow_planner.rl")
    
    batch_size = rl_cfg.batch_size
    rollouts_per_scenario = int(rl_cfg.get('rollouts_per_scenario', 1))
    assert rollouts_per_scenario > 1, f"rollouts_per_scenario must be > 1, got {rollouts_per_scenario}"
    
    # Fix random seed（每次 rollout 的种子会在 executor.run_simulation 内单独设置）
    pl.seed_everything(int(cfg.seed), workers=True)
    
    # 获取所有 scenarios（用于 batch 分组）
    profiler_name = 'building_simulation'
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)
    
    # 获取所有 scenarios
    if not int(os.environ.get("NUPLAN_SIMULATION_ALLOW_ANY_BUILDER", "0")) and not is_target_type(
        cfg.scenario_builder, NuPlanScenarioBuilder
    ):
        raise ValueError(f"Simulation framework only runs with NuPlanScenarioBuilder. Got {cfg.scenario_builder}")
    
    scenario_filter = DistributedScenarioFilter(
        cfg=cfg,
        worker=common_builder.worker,
        node_rank=int(os.environ.get("NODE_RANK", 0)),
        num_nodes=int(os.environ.get("NUM_NODES", 1)),
        synchronization_path=cfg.output_dir,
        timeout_seconds=cfg.distributed_timeout_seconds,
        distributed_mode=DistributedMode[cfg.distributed_mode],
    )
    all_scenarios = scenario_filter.get_scenarios()
    
    # 按 batch_size 分组，舍弃最后一组（如果不足 batch_size）
    scenario_batches = []
    for i in range(0, len(all_scenarios), batch_size):
        batch = all_scenarios[i:i+batch_size]
        if len(batch) == batch_size:  # 只保留完整的 batch
            scenario_batches.append(batch)
    
    if len(scenario_batches) == 0:
        logger.error("No complete batches found for RL training")
        return
    
    logger.info(f"RL mode: Found {len(scenario_batches)} complete batches (batch_size={batch_size})")
    
    # 获取模型和优化器
    # 如果 planners 未提供，需要从配置构建
    if planners is None or (isinstance(planners, list) and len(planners) == 0):
        # 从配置构建 planner（使用第一个 scenario 作为示例）
        if len(all_scenarios) == 0:
            raise ValueError("No scenarios available to build planner")
        
        from nuplan.planning.script.builders.planner_builder import build_planners
        if 'planner' not in cfg.keys():
            raise KeyError('Planner not specified in config. Please specify a planner using "planner" field.')
        
        # RL 配置已经在 cfg.planner.flow_planner.rl 中（从启动脚本传递）
        
        # 使用第一个 scenario 构建 planner（所有 scenario 使用相同的 planner）
        planners = build_planners(cfg.planner, all_scenarios[0])
        if len(planners) == 0:
            raise ValueError("Failed to build planners from config")
    
    planner = planners[0] if isinstance(planners, list) else planners
    
    # 检查 planner 是否有 _planner 属性（FlowPlanner 的结构）
    if hasattr(planner, '_planner'):
        model = planner._planner  # 获取底层模型
    else:
        # 如果 planner 本身就是模型
        model = planner

    # 关键：在 Ray 分布式执行时，worker 进程会对 planner 调用 initialize() 从 ckpt 加载权重；
    # 但主进程这里构建的 planner 通常尚未 initialize，因此 model 可能仍是随机初始化权重。
    # 这会导致：保存的 x_next（来自 worker 已加载 ckpt 的策略）与主进程重算的 x_next_new（随机权重策略）严重不一致。
    # 为保证 old/new 可比，必须在主进程创建 RLTrainer 之前显式加载同一份 ckpt 权重。
    if hasattr(planner, '_ckpt_path') and planner._ckpt_path is not None and hasattr(planner, '_ema_enabled'):
        state_dict = torch.load(planner._ckpt_path, weights_only=True, map_location=planner._device)
        if planner._ema_enabled:
            state_dict = state_dict['ema_state_dict']
        else:
            if "model" in state_dict.keys():
                state_dict = state_dict['model']
        # use for ddp
        model_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
        model.load_state_dict(model_state_dict)
        model.eval()
        model.to(planner._device)
    
    # 创建优化器（从配置读取或使用默认值）
    lr = rl_cfg.get('learning_rate', 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 创建 RL Trainer（传递 rl_cfg 而不是整个 cfg）
    rl_trainer = RLTrainer(
        model=model,
        optimizer=optimizer,
        rl_cfg=rl_cfg,
        device=planner._device
    )
    
    # Build simulation callbacks（在循环外构建一次，可以复用）
    callbacks_worker_pool = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg=cfg, output_dir=common_builder.output_dir, worker=callbacks_worker_pool)
    
    # 确保 planners 是列表格式
    if isinstance(planners, AbstractPlanner):
        planners = [planners]
    
    # 注意：在 RL 模式下，我们不从 config 中移除 planner，因为每个 batch 都需要使用相同的 planner
    # 但是 build_simulations 会检查 pre_built_planners，所以这里应该没问题
    
    step_count = 0
    aligned_score_cfg = _build_aligned_score_cfg(cfg)
    
    # 无限循环
    while True:
        t_step0 = time.perf_counter()
        logger.info(f"RL Training Step {step_count}")
        # 获取当前 batch 的 scenarios（循环使用所有 batches）
        batch_idx = step_count % len(scenario_batches)
        current_batch_scenarios = scenario_batches[batch_idx]

        # -------- Clean eval (no RL noise, with CFG) --------
        # 每轮开头：取当前 batch 前 若干 个 scenario，用“无噪声 + CFG”的方式跑一次快速评测并打印 score。
        # 说明：
        # - 这里通过把 planner.flow_planner.rl.enabled 置为 False 来关闭 RL（从而不会走 generate_rl，也不会加噪）
        # - FlowPlanner 默认 use_cfg=True（若配置里没写），这里显式写入确保开启 CFG
        # - 使用当前训练策略 model 的权重同步到 eval planners，保证评测反映最新策略
        # try:
        #     eval_scenarios = list(current_batch_scenarios)
        #     if len(eval_scenarios) > 0:
        #         # 深拷贝 cfg，避免影响 RL 主循环配置
        #         eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        #         OmegaConf.set_struct(eval_cfg, False)
        #         # 关闭 RL（无噪声）
        #         if hasattr(eval_cfg, "planner") and hasattr(eval_cfg.planner, "flow_planner") and hasattr(eval_cfg.planner.flow_planner, "rl"):
        #             eval_cfg.planner.flow_planner.rl.enabled = False
        #         # 开启 CFG（若 key 不存在则新增）
        #         if not hasattr(eval_cfg.planner.flow_planner, "use_cfg"):
        #             eval_cfg.planner.flow_planner.use_cfg = True
        #         else:
        #             eval_cfg.planner.flow_planner.use_cfg = True
        #         if not hasattr(eval_cfg.planner.flow_planner, "cfg_weight"):
        #             eval_cfg.planner.flow_planner.cfg_weight = 1.0
        #         OmegaConf.set_struct(eval_cfg, True)

        #         from nuplan.planning.script.builders.planner_builder import build_planners
        #         eval_planners = build_planners(eval_cfg.planner, eval_scenarios[0])
        #         if isinstance(eval_planners, AbstractPlanner):
        #             eval_planners = [eval_planners]
        #         # 同步权重到 eval planners，并避免 worker 从旧 ckpt 覆盖
        #         # DDP 下取裸模型 state_dict 给 planner 使用
        #         model_for_sync = model.module if hasattr(model, "module") else model
        #         for p in eval_planners:
        #             if hasattr(p, "_ckpt_path"):
        #                 p._ckpt_path = None
        #             if hasattr(p, "_planner"):
        #                 p._planner.load_state_dict(model_for_sync.state_dict())
        #                 p._planner.eval()
        #                 p._planner = p._planner.to(p._device)
        #             if hasattr(p, "_weights_synced_from_main_process"):
        #                 p._weights_synced_from_main_process = True

        #         eval_runners = build_simulations(
        #             cfg=eval_cfg,
        #             callbacks=callbacks,
        #             worker=common_builder.worker,
        #             pre_built_planners=eval_planners,
        #             callbacks_worker=callbacks_worker_pool,
        #             scenarios=eval_scenarios,
        #         )
        #         with torch.no_grad():
        #             eval_reports = execute_runners(
        #                 runners=eval_runners,
        #                 worker=common_builder.worker,
        #                 num_gpus=cfg.number_of_gpus_allocated_per_simulation,
        #                 num_cpus=cfg.number_of_cpus_allocated_per_simulation,
        #                 exit_on_failure=cfg.exit_on_failure,
        #                 verbose=cfg.verbose,
        #             )
        #         eval_scores = [r.score for r in eval_reports if r.succeeded and r.score is not None]
        #         if len(eval_scores) > 0:
        #             eval_score_mean = float(sum(eval_scores) / len(eval_scores))
        #             # 写入 TensorBoard（与 RLTrainer 共用同一个 writer / logdir）
        #             if getattr(rl_trainer, "tb_writer", None) is not None:
        #                 rl_trainer.tb_writer.add_scalar("eval_score", eval_score_mean, step_count)
        #                 rl_trainer.tb_writer.flush()
        #             logger.info(
        #                 f"Clean-eval (no noise, CFG) step={step_count}: mean_score={eval_score_mean:.4f} "
        #                 f"(n={len(eval_scores)})"
        #             )
        #         else:
        #             logger.warning(f"Clean-eval step={step_count}: no valid scores collected")
        # except Exception as e:
        #     logger.warning(f"Clean-eval step={step_count} failed: {e}")
        
        # logger.info(f"Processing batch {batch_idx + 1}/{len(scenario_batches)} with {len(current_batch_scenarios)} scenarios")
        
        # 构建当前 batch 的 simulations
        # 重要：每轮 RL 训练后，主进程的 model 权重会更新，但 Ray worker 里的 planner 不会自动同步。
        # 因此每次构建 runners 前，都用当前训练策略的 state_dict 覆盖“即将发送到 worker”的 planner 权重。
        # 同时把这些 planner 的 _ckpt_path 置空，避免 worker initialize() 再次从旧 ckpt 覆盖权重。
        t_sync_planners0 = time.perf_counter()
        from nuplan.planning.script.builders.planner_builder import build_planners
        synced_planners = build_planners(cfg.planner, current_batch_scenarios[0])
        if isinstance(synced_planners, AbstractPlanner):
            synced_planners = [synced_planners]
        model_for_sync = model.module if hasattr(model, "module") else model
        for p in synced_planners:
            if hasattr(p, '_ckpt_path'):
                p._ckpt_path = None
            if hasattr(p, '_planner'):
                p._planner.load_state_dict(model_for_sync.state_dict())
                p._planner.eval()
                p._planner = p._planner.to(p._device)
            if hasattr(p, '_weights_synced_from_main_process'):
                p._weights_synced_from_main_process = True
        t_sync_planners = time.perf_counter() - t_sync_planners0
        
        # 单卡多次 rollout：把每个 scenario 复制 rollouts_per_scenario 份，并给每份设置不同 seed
        t_expand0 = time.perf_counter()
        expanded_scenarios = []
        expanded_scenario_indices = []
        expanded_rollout_ids = []
        expanded_rollout_seeds = []
        seed_base = int(cfg.seed) + int(step_count) * 100000
        for scenario_idx, scenario in enumerate(current_batch_scenarios):
            for rollout_id in range(rollouts_per_scenario):
                expanded_scenarios.append(scenario)
                expanded_scenario_indices.append(scenario_idx)
                expanded_rollout_ids.append(rollout_id)
                expanded_rollout_seeds.append(seed_base + scenario_idx * 1000 + rollout_id)
        t_expand = time.perf_counter() - t_expand0

        t_build_runners0 = time.perf_counter()
        runners = build_simulations(
            cfg=cfg,
            callbacks=callbacks,
            worker=common_builder.worker,
            pre_built_planners=synced_planners,
            callbacks_worker=callbacks_worker_pool,
            scenarios=expanded_scenarios,
        )
        t_build_runners = time.perf_counter() - t_build_runners0

        assert len(runners) == len(expanded_scenarios), (
            f"runners size {len(runners)} != expanded_scenarios size {len(expanded_scenarios)}"
        )
        for i, runner in enumerate(runners):
            runner.scenario_index = int(expanded_scenario_indices[i])
            runner.rollout_id = int(expanded_rollout_ids[i])
            runner.rollout_seed = int(expanded_rollout_seeds[i])
            # Align rollout score with evaluation aggregator per-scenario scoring
            runner.metric_aggregator_score_cfg = aligned_score_cfg
        
        if len(runners) == 0:
            logger.warning("No runners created for current batch")
            step_count += 1
            continue
        
        # 运行 simulation（不带梯度）
        t_execute0 = time.perf_counter()
        with torch.no_grad():
            reports = execute_runners(
                runners=runners,
                worker=common_builder.worker,
                num_gpus=cfg.number_of_gpus_allocated_per_simulation,
                num_cpus=cfg.number_of_cpus_allocated_per_simulation,
                exit_on_failure=cfg.exit_on_failure,
                verbose=cfg.verbose,
            )
        t_execute = time.perf_counter() - t_execute0

        # summarize per-runner durations (from RunnerReport timestamps)
        durations = []
        slowest = None
        slowest_dt = -1.0
        for r in reports:
            if r.end_time is None:
                continue
            dt = float(r.end_time - r.start_time)
            durations.append(dt)
            if dt > slowest_dt:
                slowest_dt = dt
                slowest = r
        if len(durations) > 0:
            durations_sorted = sorted(durations)
            p50 = _pctl(durations_sorted, 0.50)
            p90 = _pctl(durations_sorted, 0.90)
            avg_dt = float(sum(durations_sorted) / len(durations_sorted))
            slow_suffix = ""
            if slowest is not None:
                slow_suffix = (
                    f", slowest={slowest_dt:.2f}s"
                    f" (scenario='{slowest.scenario_name}', log='{slowest.log_name}'"
                    f", rollout_id={getattr(slowest, 'rollout_id', None)}, seed={getattr(slowest, 'rollout_seed', None)})"
                )
            # logger.info(
            #     f"Step {step_count} simulate timing: wall={t_execute:.2f}s, "
            #     f"n_runners={len(reports)}, avg={avg_dt:.2f}s, p50={p50:.2f}s, p90={p90:.2f}s{slow_suffix}"
            # )
        
        # 收集每个 scenario 的多次 rollout scores，并计算 per-scenario mean/std（用于 GRPO）
        t_gather0 = time.perf_counter()
        scenario_scores: List[List[float]] = [[] for _ in range(batch_size)]
        batch_data = []
        
        # logger.info(f"Step {step_count}: Processing {len(reports)} reports")
        
        all_ok = True
        for report in reports:
            if not (report.succeeded and report.score is not None):
                all_ok = False
                break
            if not hasattr(report, "scenario_index"):
                raise AttributeError("RunnerReport missing scenario_index. 请确认 executor.run_simulation 已写入该字段。")
            sidx = int(getattr(report, "scenario_index"))
            if sidx < 0 or sidx >= batch_size:
                raise ValueError(f"Invalid scenario_index {sidx} (batch_size={batch_size})")
            scenario_scores[sidx].append(float(report.score))
        
        # logger.info(f"Step {step_count}: Collected {len(scores)} scores from {len(reports)} reports")
        
        # 从 reports 中获取 RL 数据（因为 Ray 分布式执行，数据在 worker 进程中，需要通过 report 传回）
        runners_with_data = 0
        runners_without_data = 0
        total_data_points = 0
        
        for report in reports:
            if report.rl_data is not None and len(report.rl_data) > 0:
                runners_with_data += 1
                data_points = sum(len(sim_data) for sim_data in report.rl_data)
                total_data_points += data_points
                if not hasattr(report, "scenario_index"):
                    raise AttributeError("RunnerReport missing scenario_index while processing rl_data.")
                scenario_idx = int(getattr(report, "scenario_index"))
                # 写入 scenario_index：同一 batch 内 0..batch_size-1，供按 scenario mean/std 标准化
                for sim_data in report.rl_data:
                    for dp in sim_data:
                        if isinstance(dp, dict):
                            dp['scenario_index'] = scenario_idx
                batch_data.extend(report.rl_data)
                logger.debug(
                    f"Step {step_count}: Report ({report.scenario_name}) scenario_index={scenario_idx} "
                    f"has {len(report.rl_data)} simulation batches, {data_points} total data points"
                )
            else:
                runners_without_data += 1
                # no-op
        t_gather = time.perf_counter() - t_gather0
        
        # logger.info(f"Step {step_count}: Data collection summary:")
        # logger.info(f"  Reports with RL data: {runners_with_data}")
        # logger.info(f"  Reports without RL data: {runners_without_data}")
        # logger.info(f"  Total simulation batches: {len(batch_data)}")
        # logger.info(f"  Total data points: {total_data_points}")
        
        if (not all_ok) or len(batch_data) == 0:
            logger.warning(f"Step {step_count}: No valid data collected (data batches: {len(batch_data)})")
            step_count += 1
            continue

        t_score_stats0 = time.perf_counter()
        scenario_score_means: List[float] = []
        scenario_score_stds: List[float] = []
        
        # 先检查所有scenario的scores是否完整
        for sidx in range(batch_size):
            if len(scenario_scores[sidx]) != rollouts_per_scenario:
                logger.warning(
                    f"Step {step_count}: scenario_index={sidx} has {len(scenario_scores[sidx])}/{rollouts_per_scenario} scores, skip train"
                )
                all_ok = False
                break
        
        if all_ok:
            # 计算每个scenario的mean（分别计算）
            for sidx in range(batch_size):
                scores_arr = torch.tensor(scenario_scores[sidx], dtype=torch.float32)
                scenario_score_means.append(float(scores_arr.mean().item()))
            
            # 计算整个batch的std（所有scenario的所有scores合并计算）
            all_scores = []
            for sidx in range(batch_size):
                all_scores.extend(scenario_scores[sidx])
            all_scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
            batch_std = float(all_scores_tensor.std(unbiased=False).item() + 1e-8)
            
            # 所有scenario使用相同的batch std
            scenario_score_stds = [batch_std] * batch_size
            
            # 记录std信息用于调试
            if step_count % 10 == 0:  # 每10步记录一次
                logger.debug(
                    f"Step {step_count}: batch_std={batch_std:.6f}, "
                    f"score_range=[{all_scores_tensor.min().item():.4f}, {all_scores_tensor.max().item():.4f}], "
                    f"num_scores={len(all_scores)}"
                )
        
        t_score_stats = time.perf_counter() - t_score_stats0

        if not all_ok:
            step_count += 1
            continue
        
        # 训练
        t_train0 = time.perf_counter()
        rl_trainer.train_step(batch_data, scenario_score_means, scenario_score_stds, step_count)
        t_train = time.perf_counter() - t_train0

        t_total = time.perf_counter() - t_step0
        # logger.info(
        #     f"Step {step_count} step timing: total={t_total:.2f}s | "
        #     f"sync_planners={t_sync_planners:.2f}s({_pct(t_sync_planners, t_total)}) | "
        #     f"expand={t_expand:.2f}s({_pct(t_expand, t_total)}) | "
        #     f"build_runners={t_build_runners:.2f}s({_pct(t_build_runners, t_total)}) | "
        #     f"execute={t_execute:.2f}s({_pct(t_execute, t_total)}) | "
        #     f"gather={t_gather:.2f}s({_pct(t_gather, t_total)}) | "
        #     f"score_stats={t_score_stats:.2f}s({_pct(t_score_stats, t_total)}) | "
        #     f"train={t_train:.2f}s({_pct(t_train, t_total)})"
        # )
        
        step_count += 1
        
        # 检查是否达到最大步数（可选）
        if hasattr(rl_cfg, 'max_steps') and step_count >= rl_cfg.max_steps:
            logger.info(f"Reached max steps {rl_cfg.max_steps}")
            break


def clean_up_s3_artifacts() -> None:
    """
    Cleanup lingering s3 artifacts that are written locally.
    This happens because some minor write-to-s3 functionality isn't yet implemented.
    """
    # Lingering artifacts get written locally to a 's3:' directory. Hydra changes
    # the working directory to a subdirectory of this, so we serach the working
    # path for it.
    working_path = os.getcwd()
    s3_dirname = "s3:"
    s3_ind = working_path.find(s3_dirname)
    if s3_ind != -1:
        local_s3_path = working_path[: working_path.find(s3_dirname) + len(s3_dirname)]
        rmtree(local_s3_path)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Calls run_simulation to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    """
    assert cfg.simulation_log_main_path is None, 'Simulation_log_main_path must not be set when running simulation.'

    # Execute simulation with preconfigured planner(s).
    run_simulation(cfg=cfg)

    if is_s3_path(Path(cfg.output_dir)):
        clean_up_s3_artifacts()


if __name__ == '__main__':
    main()
