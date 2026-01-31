import concurrent.futures
from concurrent.futures import Future
import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union

from nuplan.planning.simulation.callback.metric_callback import MetricCallback
from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
from nuplan.planning.simulation.runner.abstract_runner import AbstractRunner
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool

logger = logging.getLogger(__name__)


def run_simulation(
    sim_runner: AbstractRunner,
    exit_on_failure: bool = False,
    rl_data_dir: Optional[str] = None,
    rl_step: Optional[int] = None,
) -> RunnerReport:
    """
    Proxy for calling simulation.
    :param sim_runner: A simulation runner which will execute all batched simulations.
    :param exit_on_failure: If true, raises an exception when the simulation fails.
    :return report for the simulation.
    """
    # Store start time so that if the simulations fail, we know how long they ran for
    start_time = time.perf_counter()
    try:
        # RL: 支持对同一 scenario 做多次 rollout（不同 seed）。
        # 若 runner 上设置了 rollout_seed，则在执行前设置全局随机种子。
        if hasattr(sim_runner, "rollout_seed"):
            rollout_seed = getattr(sim_runner, "rollout_seed")
            assert isinstance(rollout_seed, int), f"rollout_seed must be int, got {type(rollout_seed)}"
            import random
            import numpy as np
            import torch

            random.seed(rollout_seed)
            np.random.seed(rollout_seed)
            torch.manual_seed(rollout_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(rollout_seed)

        report = sim_runner.run()

        # 将 rollout 元信息附加到 report，便于上层按 scenario 聚合（不改 RunnerReport dataclass 定义）
        if hasattr(sim_runner, "scenario_index"):
            report.scenario_index = int(getattr(sim_runner, "scenario_index"))
        if hasattr(sim_runner, "rollout_id"):
            report.rollout_id = int(getattr(sim_runner, "rollout_id"))
        if hasattr(sim_runner, "rollout_seed"):
            report.rollout_seed = int(getattr(sim_runner, "rollout_seed"))

        # RL: 将 report.rl_data 落盘，避免通过 Ray object store 回传超大对象导致 spill/卡死
        if rl_data_dir is not None and getattr(report, "rl_data", None) is not None:
            if rl_step is None:
                raise ValueError("rl_step must be provided when rl_data_dir is set")
            if not isinstance(rl_step, int) or rl_step < 0:
                raise ValueError(f"rl_step must be non-negative int, got {rl_step} ({type(rl_step)})")

            import os
            from pathlib import Path
            from uuid import uuid4

            import torch

            rl_data = report.rl_data
            if not isinstance(rl_data, list):
                raise TypeError(f"report.rl_data must be a list, got {type(rl_data)}")

            # 写入 scenario_index/rollout 信息到每个 datapoint，避免 driver 再次加载修改
            scenario_index = getattr(report, "scenario_index", None)
            rollout_id = getattr(report, "rollout_id", None)
            rollout_seed = getattr(report, "rollout_seed", None)
            if scenario_index is not None:
                sidx = int(scenario_index)
                for traj in rl_data:
                    if not isinstance(traj, list):
                        raise TypeError(f"rl_data trajectory must be list, got {type(traj)}")
                    for dp in traj:
                        if isinstance(dp, dict):
                            dp["scenario_index"] = sidx
                            if rollout_id is not None:
                                dp["rollout_id"] = int(rollout_id)
                            if rollout_seed is not None:
                                dp["rollout_seed"] = int(rollout_seed)

            # 生成文件名（可读 + 唯一）
            def _safe(s: str) -> str:
                return "".join((c if c.isalnum() else "_") for c in s)[:80]

            log_name = _safe(getattr(report, "log_name", "log"))
            scenario_name = _safe(getattr(report, "scenario_name", "scenario"))
            rid = int(rollout_id) if rollout_id is not None else -1
            uid = uuid4().hex

            out_dir = Path(rl_data_dir) / f"step_{rl_step:06d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{log_name}__{scenario_name}__rid{rid}__{uid}.pt"

            # 原子写：先写临时文件再 rename
            tmp_path = out_path.with_suffix(".pt.tmp")
            torch.save(rl_data, tmp_path)
            os.replace(tmp_path, out_path)

            # 在 report 上只保留路径/统计，避免回传大对象
            report.rl_data = None
            report.rl_data_path = str(out_path)
            report.rl_data_num_trajectories = int(len(rl_data))
            report.rl_data_num_datapoints = int(sum(len(t) for t in rl_data))

        return report
    except Exception as e:
        error = traceback.format_exc()

        # Print to the terminal
        logger.warning("----------- Simulation failed: with the following trace:")
        traceback.print_exc()
        logger.warning(f"Simulation failed with error:\n {e}")

        # Log the failed scenario log/tokens
        failed_scenarios = f"[{sim_runner.scenario.log_name}, {sim_runner.scenario.scenario_name}]\n"
        logger.warning(f"\nFailed simulation [log,token]:\n {failed_scenarios}")

        logger.warning("----------- Simulation failed!")

        # Fail if desired
        if exit_on_failure:
            raise RuntimeError('Simulation failed')

        end_time = time.perf_counter()
        report = RunnerReport(
            succeeded=False,
            error_message=error,
            start_time=start_time,
            end_time=end_time,
            planner_report=None,
            scenario_name=sim_runner.scenario.scenario_name,
            planner_name=sim_runner.planner.name(),
            log_name=sim_runner.scenario.log_name,
            score=None,
            rl_data=None,
        )

        return report


def execute_runners(
    runners: List[AbstractRunner],
    worker: WorkerPool,
    num_gpus: Optional[Union[int, float]],
    num_cpus: Optional[int],
    exit_on_failure: bool = False,
    verbose: bool = False,
    rl_data_dir: Optional[str] = None,
    rl_step: Optional[int] = None,
) -> List[RunnerReport]:
    """
    Execute multiple simulation runners or metric runners.
    :param runners: A list of simulations to be run.
    :param worker: for submitting tasks.
    :param num_gpus: if None, no GPU will be used, otherwise number (also fractional) of GPU used per simulation.
    :param num_cpus: if None, all available CPU threads are used, otherwise number of threads used.
    :param exit_on_failure: If true, raises an exception when the simulation fails.
    """
    # Validating
    assert len(runners) > 0, 'No scenarios found to simulate!'

    # Start simulations
    number_of_sims = len(runners)
    # logger.info(f"Starting {number_of_sims} simulations using {worker.__class__.__name__}!")
    task = Task(fn=run_simulation, num_gpus=num_gpus, num_cpus=num_cpus)

    # 逐个 Ray object 完成后立刻打印当前场景 score（非 RL 也打印）。
    # Ray worker 支持 submit()，可以 as_completed 流式拿结果；其他 worker 可能不支持 submit，则退化为 map 后打印。
    reports_by_idx: List[Optional[RunnerReport]] = [None] * len(runners)
    try:
        future_to_idx = {
            worker.submit(task, runner, exit_on_failure, rl_data_dir, rl_step): idx for idx, runner in enumerate(runners)
        }
        for fut in concurrent.futures.as_completed(list(future_to_idx.keys())):
            report: RunnerReport = fut.result()
            reports_by_idx[future_to_idx[fut]] = report
            # logger.info(
            #     "Simulation finished: scenario='%s', log='%s', planner='%s', succeeded=%s, score=%s",
            #     report.scenario_name,
            #     report.log_name,
            #     report.planner_name,
            #     report.succeeded,
            #     report.score,
            # )
            # print("report.score {}".format(report.score))
    except NotImplementedError:
        mapped_reports = worker.map(task, runners, exit_on_failure, rl_data_dir, rl_step, verbose=verbose)
        assert len(mapped_reports) == len(runners), f"worker.map returned {len(mapped_reports)} reports for {len(runners)} runners"
        reports_by_idx = list(mapped_reports)
        # for report in reports:
        #     # logger.info(
        #     #     "Simulation finished: scenario='%s', log='%s', planner='%s', succeeded=%s, score=%s",
        #     #     report.scenario_name,
        #     #     report.log_name,
        #     #     report.planner_name,
        #     #     report.succeeded,
        #     #     report.score,
        #     # )
        #     print("report.score {}".format(report.score))
    # 确保所有 runner 都有 report（RL 多 rollout 时同一 scenario 可能出现重复，因此不能用 (scenario_name, planner, log) 去重）
    assert all(r is not None for r in reports_by_idx), "Some runner reports are missing"

    # Iterate over runners, finding the callbacks which may have run asynchronously, and gathering their results
    callback_future_to_idx: Dict[Future[None], int] = {}
    for idx, runner in enumerate(runners):
        if not isinstance(runner, SimulationRunner):
            continue
        for callback in runner.simulation.callback.callbacks:
            if isinstance(callback, MetricCallback) or isinstance(callback, SimulationLogCallback):
                for future in callback.futures:
                    callback_future_to_idx[future] = idx

    for future in concurrent.futures.as_completed(list(callback_future_to_idx.keys())):
        try:
            future.result()
        except Exception:
            error_message = traceback.format_exc()
            runner_report = reports_by_idx[callback_future_to_idx[future]]
            assert runner_report is not None
            runner_report.error_message = error_message
            runner_report.succeeded = False
            runner_report.end_time = time.perf_counter()

    # Notify user about the result of simulations
    failed_simulations = str()
    number_of_successful = 0
    runner_reports: List[RunnerReport] = [r for r in reports_by_idx if r is not None]
    for result in runner_reports:
        if result.succeeded:
            number_of_successful += 1
        else:
            logger.warning("Failed Simulation.\n '%s'", result.error_message)
            # RL 多 rollout：若存在 rollout_id/seed，附加到失败标识便于定位
            rollout_suffix = ""
            if hasattr(result, "rollout_id") or hasattr(result, "rollout_seed"):
                rollout_suffix = f", rollout_id={getattr(result, 'rollout_id', None)}, seed={getattr(result, 'rollout_seed', None)}"
            failed_simulations += f"[{result.log_name}, {result.scenario_name}{rollout_suffix}] \n"

    number_of_failures = number_of_sims - number_of_successful
    # logger.info(f"Number of successful simulations: {number_of_successful}")
    # logger.info(f"Number of failed simulations: {number_of_failures}")

    # Print out all failed simulation unique identifier
    if number_of_failures > 0:
        logger.info(f"Failed simulations [log, token]:\n{failed_simulations}")

    return runner_reports
