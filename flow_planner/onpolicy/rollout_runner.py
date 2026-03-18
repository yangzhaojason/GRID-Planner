from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class NuPlanSimPaths:
    """Filesystem paths required to run nuPlan simulation."""

    project_root: str
    nuplan_devkit_root: str


@dataclass
class NuPlanSimConfig:
    """Configuration for a single simulation run."""

    # Experiment selection
    split: str = "val14"
    challenge: str = "closed_loop_reactive_agents"

    # Ray / distributed worker settings (keep consistent with run scripts)
    ray_tmpdir: str = "/home/datasets_2/ray_tmp"
    threads_per_node: int = 64
    distributed_mode: str = "SINGLE_NODE"
    gpus_per_simulation: float = 0.15


@dataclass
class NuPlanSimRunResult:
    """Result of a single nuPlan simulation invocation."""

    proc: subprocess.CompletedProcess
    experiment_uid: str
    output_dir: Optional[str] = None


def run_nuplan_simulation(
    *,
    paths: NuPlanSimPaths,
    run_dir: str,
    ckpt_path: str,
    config_path: Optional[str] = None,
    sim_cfg: Optional[NuPlanSimConfig] = None,
    env: Optional[Dict[str, str]] = None,
) -> NuPlanSimRunResult:
    """
    Run nuPlan closed-loop simulation using our FlowPlanner wrapper.

    This is a Python-callable equivalent of `launch_sim_nuplan_graph_weighting_rrs_ft.sh`:
      - `run_dir` should contain `.hydra/config.yaml`
      - `ckpt_path` should be an absolute path to a `.pth`

    Returns the CompletedProcess (stdout/stderr captured).
    """
    sim_cfg = sim_cfg or NuPlanSimConfig()

    run_dir_p = Path(run_dir)
    if config_path is None:
        config_path = str(run_dir_p / ".hydra" / "config.yaml")

    # Basic validation (keep errors readable)
    if not Path(paths.project_root).exists():
        raise FileNotFoundError(f"PROJECT_ROOT not found: {paths.project_root}")
    if not Path(paths.nuplan_devkit_root).exists():
        raise FileNotFoundError(f"NUPLAN_DEVKIT_ROOT not found: {paths.nuplan_devkit_root}")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"config_path not found: {config_path}")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"ckpt_path not found: {ckpt_path}")

    # Choose scenario builder based on split (matches existing shell scripts)
    scenario_builder = "nuplan" if sim_cfg.split == "val14" else "nuplan_challenge"

    # Ensure output runs even if ray tmpdir not pre-created
    os.makedirs(sim_cfg.ray_tmpdir, exist_ok=True)

    # Use a unique uid to make output discovery deterministic.
    # nuPlan writes outputs under: ${NUPLAN_EXP_ROOT}/exp/simulation/<challenge>/<experiment_uid>/...
    # Make uid unique to avoid collisions when evaluating multiple checkpoints.
    ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    exp_uid = f"flow_planner/{sim_cfg.split}/onpolicy_opsd/{Path(ckpt_path).stem}_{ts}"

    cmd = [
        "python",
        str(Path(paths.nuplan_devkit_root) / "nuplan" / "planning" / "script" / "run_simulation.py"),
        f"+simulation={sim_cfg.challenge}",
        "planner=flow_planner",
        f"planner.flow_planner.config_path={config_path}",
        f"planner.flow_planner.ckpt_path={ckpt_path}",
        f"scenario_builder={scenario_builder}",
        f"scenario_filter={sim_cfg.split}",
        # Put run artifacts under a unique uid but keep it stable-ish for parsing
        f"experiment_uid={exp_uid}",
        "verbose=true",
        "worker=ray_distributed",
        f"worker.threads_per_node={sim_cfg.threads_per_node}",
        f"distributed_mode={sim_cfg.distributed_mode}",
        f"number_of_gpus_allocated_per_simulation={sim_cfg.gpus_per_simulation}",
        "enable_simulation_progress_bar=false",
        'hydra.searchpath=[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]',
    ]

    merged_env = os.environ.copy()
    merged_env["PYTHONPATH"] = f"{paths.project_root}:{merged_env.get('PYTHONPATH', '')}"
    merged_env.setdefault("HYDRA_FULL_ERROR", "1")
    merged_env.setdefault("RAY_TMPDIR", sim_cfg.ray_tmpdir)
    # Reduce Ray dashboard/metrics noise (matches shell scripts)
    merged_env.setdefault("RAY_DISABLE_DASHBOARD", "1")
    merged_env.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    merged_env.setdefault("RAY_INCLUDE_DASHBOARD", "false")
    merged_env.setdefault("RAY_ENABLE_METRICS_AGENT", "false")
    merged_env.setdefault("RAY_METRICS_EXPORT_PORT", "0")
    if env:
        merged_env.update(env)

    proc = subprocess.run(
        cmd,
        cwd=paths.project_root,
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )

    exp_root = merged_env.get("NUPLAN_EXP_ROOT", None)
    output_dir = None
    if exp_root:
        output_dir = str(Path(exp_root) / "exp" / "simulation" / sim_cfg.challenge / exp_uid)

    return NuPlanSimRunResult(proc=proc, experiment_uid=exp_uid, output_dir=output_dir)

