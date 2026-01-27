from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.abstract_runner import AbstractRunner
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.simulation import Simulation

logger = logging.getLogger(__name__)


def for_each(fn: Callable[[Any], Any], items: List[Any]) -> None:
    """
    Call function on every item in items
    :param fn: function to be called fn(item)
    :param items: list of items
    """
    for item in items:
        fn(item)


class SimulationRunner(AbstractRunner):
    """
    Manager which executes multiple simulations with the same planner
    """

    def __init__(self, simulation: Simulation, planner: AbstractPlanner):
        """
        Initialize the simulations manager
        :param simulation: Simulation which will be executed
        :param planner: to be used to compute the desired ego's trajectory
        """
        self._simulation = simulation
        self._planner = planner

    def _initialize(self) -> None:
        """
        Initialize the planner
        """
        # Execute specific callback
        self._simulation.callback.on_initialization_start(self._simulation.setup, self.planner)

        # Initialize Planner
        self.planner.initialize(self._simulation.initialize())

        # Execute specific callback
        self._simulation.callback.on_initialization_end(self._simulation.setup, self.planner)

    @property
    def planner(self) -> AbstractPlanner:
        """
        :return: Planner used by the SimulationRunner
        """
        return self._planner

    @property
    def simulation(self) -> Simulation:
        """
        :return: Simulation used by the SimulationRunner
        """
        return self._simulation

    @property
    def scenario(self) -> AbstractScenario:
        """
        :return: Get the scenario relative to the simulation.
        """
        return self.simulation.scenario

    def run(self) -> RunnerReport:
        """
        Run through all simulations. The steps of execution follow:
         - Initialize all planners
         - Step through simulations until there no running simulation
        :return: List of SimulationReports containing the results of each simulation
        """
        start_time = time.perf_counter()

        # Initialize reports for all the simulations that will run
        report = RunnerReport(
            succeeded=True,
            error_message=None,
            start_time=start_time,
            end_time=None,
            planner_report=None,
            scenario_name=self._simulation.scenario.scenario_name,
            planner_name=self.planner.name(),
            log_name=self._simulation.scenario.log_name,
            score=None,  # Will be set after simulation ends
            rl_data=None,  # Will be set after simulation ends if RL mode is enabled
        )

        # Execute specific callback
        self.simulation.callback.on_simulation_start(self.simulation.setup)

        # Initialize all simulations
        self._initialize()

        while self.simulation.is_simulation_running():
            # Execute specific callback
            self.simulation.callback.on_step_start(self.simulation.setup, self.planner)

            # Perform step
            planner_input = self._simulation.get_planner_input()
            logger.debug("Simulation iterations: %s" % planner_input.iteration.index)

            # Execute specific callback
            self._simulation.callback.on_planner_start(self.simulation.setup, self.planner)

            # Plan path based on all planner's inputs
            trajectory = self.planner.compute_trajectory(planner_input)

            # Propagate simulation based on planner trajectory
            self._simulation.callback.on_planner_end(self.simulation.setup, self.planner, trajectory)
            self.simulation.propagate(trajectory)

            # Execute specific callback
            self.simulation.callback.on_step_end(self.simulation.setup, self.planner, self.simulation.history.last())

            # Store reports for simulations which just finished running
            current_time = time.perf_counter()
            if not self.simulation.is_simulation_running():
                report.end_time = current_time

        # Execute specific callback
        self.simulation.callback.on_simulation_end(self.simulation.setup, self.planner, self.simulation.history)

        planner_report = self.planner.generate_planner_report()
        report.planner_report = planner_report

        # 获取 score（用于 RL 训练）
        score = None
        aligned_cfg: Optional[Dict[str, Any]] = getattr(self, "metric_aggregator_score_cfg", None)

        def _compute_aligned_score(metric_results: Dict[str, List[Any]], cfg_dict: Dict[str, Any]) -> float:
            """
            Compute per-scenario score aligned with WeightedAverageMetricAggregator:
              score = (sum(w_i * m_i) / sum(w_i)) * prod(m_j in multiple_metrics)
            where m_i are MetricStatistics.metric_score keyed by MetricStatistics.name (metric_statistics_name).
            """
            if "metric_weights" not in cfg_dict:
                raise KeyError("metric_aggregator_score_cfg missing key 'metric_weights'")
            metric_weights = cfg_dict["metric_weights"]
            if not isinstance(metric_weights, dict):
                raise TypeError(f"metric_weights must be dict, got {type(metric_weights)}")

            multiple_metrics = []
            if "multiple_metrics" in cfg_dict:
                multiple_metrics = cfg_dict["multiple_metrics"]
            if not isinstance(multiple_metrics, list):
                raise TypeError(f"multiple_metrics must be list, got {type(multiple_metrics)}")
            for m in multiple_metrics:
                if not isinstance(m, str):
                    raise TypeError(f"multiple_metrics items must be str, got {type(m)}")

            default_weight = 1.0
            if "default" in metric_weights:
                default_weight = float(metric_weights["default"])

            # Build name -> metric_score map (MetricStatistics.name is what aggregator uses as metric_statistics_name)
            name_to_score: Dict[str, Optional[float]] = {}
            for _builder_name, stats_list in metric_results.items():
                for stat in stats_list:
                    if not hasattr(stat, "name"):
                        raise AttributeError("MetricStatistics missing attribute 'name'")
                    stat_name = stat.name
                    if not isinstance(stat_name, str):
                        raise TypeError(f"MetricStatistics.name must be str, got {type(stat_name)}")
                    if stat_name in name_to_score:
                        raise RuntimeError(f"Duplicate MetricStatistics.name encountered: '{stat_name}'")
                    # metric_score can be None
                    if not hasattr(stat, "metric_score"):
                        raise AttributeError(f"MetricStatistics '{stat_name}' missing attribute 'metric_score'")
                    ms = stat.metric_score
                    if ms is None:
                        name_to_score[stat_name] = None
                    else:
                        if not isinstance(ms, (int, float)):
                            raise TypeError(
                                f"MetricStatistics.metric_score for '{stat_name}' must be number, got {type(ms)}"
                            )
                        name_to_score[stat_name] = float(ms)

            metric_scores = 0.0
            sum_weights = 0.0
            multiple_factor = 1.0
            for metric_name, metric_score in name_to_score.items():
                if metric_score is None:
                    continue
                if metric_name in multiple_metrics:
                    multiple_factor *= float(metric_score)
                else:
                    weight = default_weight
                    if metric_name in metric_weights:
                        weight = float(metric_weights[metric_name])
                    sum_weights += weight
                    metric_scores += weight * float(metric_score)

            weighted_average_score = metric_scores / sum_weights if sum_weights else 0.0
            return multiple_factor * weighted_average_score
        
        # 方法1: 从 metric callback 获取
        # 等待 metric 计算完成（如果是异步的）
        for callback in self.simulation.callback.callbacks:
            if hasattr(callback, 'futures') and callback.futures:
                # 等待异步任务完成
                for future in callback.futures:
                    # 对齐评测口径时，不要吞掉 metric 异步失败
                    if aligned_cfg is not None:
                        future.result()
                    else:
                        try:
                            future.result()  # 等待完成
                        except Exception as e:
                            logger.warning(f"Metric computation future failed: {e}")
            
            if hasattr(callback, 'metric_engine') and callback.metric_engine is not None:
                # 计算 metrics 并获取 score
                metric_results = callback.metric_engine.compute_metric_results(
                    history=self.simulation.history,
                    scenario=self.simulation.scenario
                )

                # RL 对齐：使用评测 aggregator 的 per-scenario 打分公式
                if aligned_cfg is not None:
                    score = _compute_aligned_score(metric_results=metric_results, cfg_dict=aligned_cfg)
                    break

                # 默认行为：将所有 MetricStatistics.metric_score 做简单平均（历史行为）
                scores_list = []
                for _metric_name, metric_statistics_list in metric_results.items():
                    for metric_stat in metric_statistics_list:
                        if hasattr(metric_stat, 'metric_score') and metric_stat.metric_score is not None:
                            scores_list.append(metric_stat.metric_score)

                if len(scores_list) > 0:
                    score = sum(scores_list) / len(scores_list)
                    break
        
        # 方法2: 从 planner 获取（如果 planner 有方法）
        if score is None and hasattr(self.planner, 'get_simulation_score'):
            try:
                score = self.planner.get_simulation_score()
            except Exception as e:
                logger.debug(f"Failed to get score from planner: {e}")
        
        report.score = score
        
        # 只有在 RL 模式下才进行 RL 数据收集 / finalize。
        # FlowPlanner 即使 rl_enabled=False 也可能暴露 get_rl_data_collector/finalize_rl_simulation 方法，
        # 但此时 collector 为 None，非 RL 仿真不应产生告警日志。
        rl_enabled = bool(getattr(self.planner, "_rl_enabled", False))

        # 如果 planner 支持 RL，调用 finalize_rl_simulation
        if rl_enabled and hasattr(self.planner, 'finalize_rl_simulation'):
            if score is not None:
                try:
                    # 在调用前检查数据收集器的状态
                    if hasattr(self.planner, 'get_rl_data_collector'):
                        data_collector = self.planner.get_rl_data_collector()
                        if data_collector is not None:
                            current_data_size = len(data_collector.current_simulation_data)
                            batch_data_size = len(data_collector.batch_data)
                            # logger.info(f"Before finalize_rl_simulation: current_data_size={current_data_size}, batch_data_size={batch_data_size}, score={score}")
                        else:
                            logger.warning("RL enabled but get_rl_data_collector returned None")
                    else:
                        logger.warning("RL enabled but planner lacks get_rl_data_collector")
                    
                    self.planner.finalize_rl_simulation(score)
                    
                    # 在调用后检查数据收集器的状态，并将数据提取到 report 中
                    if hasattr(self.planner, 'get_rl_data_collector'):
                        data_collector = self.planner.get_rl_data_collector()
                        if data_collector is not None:
                            current_data_size = len(data_collector.current_simulation_data)
                            batch_data_size = len(data_collector.batch_data)
                            # logger.info(f"After finalize_rl_simulation: current_data_size={current_data_size}, batch_data_size={batch_data_size}")
                            
                            # 将数据收集器的数据提取到 report 中（用于跨进程传递）
                            rl_data = data_collector.get_batch_data()
                            if len(rl_data) > 0:
                                report.rl_data = rl_data
                                # logger.info(f"Extracted {len(rl_data)} simulation batches ({sum(len(sim_data) for sim_data in rl_data)} data points) to report")
                            else:
                                logger.warning("Data collector has no batch_data after finalize")
                except Exception as e:
                    logger.warning(f"Failed to finalize RL simulation: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
            else:
                logger.warning(f"finalize_rl_simulation not called because score is None")
        else:
            # 即使没有 score，也尝试提取数据（可能 score 计算失败但数据已收集）
            if rl_enabled and hasattr(self.planner, 'get_rl_data_collector'):
                data_collector = self.planner.get_rl_data_collector()
                if data_collector is not None:
                    rl_data = data_collector.get_batch_data()
                    if len(rl_data) > 0:
                        report.rl_data = rl_data
                        # logger.info(f"Extracted {len(rl_data)} simulation batches without score")

        return report
