from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from flow_planner.onpolicy.rollout_runner import NuPlanSimConfig, NuPlanSimPaths, run_nuplan_simulation
from flow_planner.onpolicy.sim_report import parse_final_score_from_runner_report


@dataclass
class TeacherSelectionResult:
    teacher_ckpt_path: str
    ckpt_scores: Dict[str, float]
    best_score: float


def list_candidate_checkpoints(run_dir: str) -> List[str]:
    """
    List candidate checkpoints under a training run dir.
    Includes latest.pth + any model_epoch_*.pth.
    Returns absolute paths.
    """
    p = Path(run_dir)
    cands: List[Path] = []
    latest = p / "latest.pth"
    if latest.exists():
        cands.append(latest)
    cands.extend(sorted(p.glob("model_epoch_*.pth")))
    # de-dup and sort by name (stable)
    uniq = {str(x.resolve()): x for x in cands}
    return [k for k in sorted(uniq.keys())]


def select_teacher_checkpoint_by_sim_score(
    *,
    paths: NuPlanSimPaths,
    run_dir: str,
    candidates: Optional[List[str]] = None,
    sim_cfg: Optional[NuPlanSimConfig] = None,
    sim_output_root: Optional[str] = None,
    persist_json: bool = True,
) -> TeacherSelectionResult:
    """
    Evaluate each candidate checkpoint in closed-loop simulation and pick the best by final score.

    Notes:
    - This function runs simulation; it can be slow.
    - `sim_output_root` can be set to a directory where nuPlan will write its experiment outputs;
      if omitted, nuPlan's default output path will be used (devkit config dependent).
    """
    sim_cfg = sim_cfg or NuPlanSimConfig()
    candidates = candidates or list_candidate_checkpoints(run_dir)
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under run_dir={run_dir}")

    ckpt_scores: Dict[str, float] = {}
    ckpt_reports: Dict[str, str] = {}
    ckpt_exp_uids: Dict[str, str] = {}
    ckpt_returncodes: Dict[str, int] = {}
    invalid_scores: Dict[str, float] = {}
    ckpt_failures: Dict[str, str] = {}
    best_ckpt: Optional[str] = None
    best_score: float = float("-inf")
    last_proc = None

    # Evaluate each ckpt
    for ckpt_path in candidates:
        sim_res = run_nuplan_simulation(
            paths=paths,
            run_dir=run_dir,
            ckpt_path=ckpt_path,
            sim_cfg=sim_cfg,
            env={"NUPLAN_EXP_ROOT": sim_output_root} if sim_output_root else None,
        )
        proc = sim_res.proc
        last_proc = proc
        ckpt_exp_uids[ckpt_path] = sim_res.experiment_uid
        ckpt_returncodes[ckpt_path] = int(proc.returncode)
        if proc.returncode != 0:
            ckpt_scores[ckpt_path] = float("nan")
            ckpt_failures[ckpt_path] = f"simulation_failed | returncode={proc.returncode} | stderr_tail={proc.stderr[-1000:]}"
            continue
        # Try to find report path:
        # Prefer the exact output dir for this experiment_uid (avoids picking up other runs under NUPLAN_EXP_ROOT).
        report_path = None
        if sim_res.output_dir:
            exact_report = Path(sim_res.output_dir) / "runner_report.parquet"
            if exact_report.is_file():
                report_path = str(exact_report)

        if report_path is None:
            ckpt_scores[ckpt_path] = float("nan")
            invalid_scores[ckpt_path] = float("nan")
            ckpt_failures[ckpt_path] = (
                "runner_report_not_found"
                f" | returncode={proc.returncode}"
                f" | output_dir={sim_res.output_dir}"
                f" | stdout_tail={proc.stdout[-1000:]}"
                f" | stderr_tail={proc.stderr[-1000:]}"
            )
            continue

        try:
            score = float(parse_final_score_from_runner_report(report_path).score)
        except Exception as e:
            ckpt_scores[ckpt_path] = float("nan")
            ckpt_reports[ckpt_path] = report_path
            invalid_scores[ckpt_path] = float("nan")
            ckpt_failures[ckpt_path] = f"score_parse_failed | report={report_path} | error={repr(e)}"
            continue

        ckpt_scores[ckpt_path] = score
        ckpt_reports[ckpt_path] = report_path

        # Skip invalid scores (NaN/Inf) but keep diagnostic records.
        if not math.isfinite(score):
            invalid_scores[ckpt_path] = score
            ckpt_failures[ckpt_path] = f"non_finite_score={score} | report={report_path}"
            continue

        if score > best_score:
            best_score = score
            best_ckpt = ckpt_path

    if best_ckpt is None:
        raise RuntimeError(
            "Teacher selection failed: no checkpoint produced a finite simulation score.\n"
            f"run_dir={run_dir}\n"
            f"split={sim_cfg.split}, challenge={sim_cfg.challenge}\n"
            f"num_candidates={len(candidates)}\n"
            f"ckpt_scores={json.dumps(ckpt_scores, indent=2)}\n"
            f"ckpt_reports={json.dumps(ckpt_reports, indent=2)}\n"
            f"invalid_scores={json.dumps(invalid_scores, indent=2)}\n"
            f"ckpt_failures={json.dumps(ckpt_failures, indent=2)}\n"
            f"last_sim_returncode={None if last_proc is None else last_proc.returncode}\n"
            f"last_stdout_tail={'' if last_proc is None else last_proc.stdout[-2000:]}\n"
            f"last_stderr_tail={'' if last_proc is None else last_proc.stderr[-2000:]}"
        )

    res = TeacherSelectionResult(teacher_ckpt_path=best_ckpt, ckpt_scores=ckpt_scores, best_score=best_score)

    if persist_json:
        out = {
            "run_dir": run_dir,
            "split": sim_cfg.split,
            "challenge": sim_cfg.challenge,
            "best_ckpt": best_ckpt,
            "best_score": best_score,
            "ckpt_scores": ckpt_scores,
            "ckpt_reports": ckpt_reports,
            "ckpt_experiment_uids": ckpt_exp_uids,
            "ckpt_returncodes": ckpt_returncodes,
            "ckpt_failures": ckpt_failures,
        }
        Path(run_dir, "teacher_selection.json").write_text(json.dumps(out, indent=2))

    return res
