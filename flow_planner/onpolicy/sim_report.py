from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


@dataclass
class SimScore:
    score: float
    num_scenarios: Optional[float] = None
    components: Dict[str, Any] = None


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _resolve_metrics_parquet(file_path: str) -> str:
    """
    Align with `print_result.py`:
      - If input is runner_report.parquet, require one unambiguous
        aggregator_metric/*weighted_average_metrics*.parquet under the same run directory.
      - Otherwise, treat file_path as the metrics parquet path.
    """
    p = Path(file_path)
    if p.name == "runner_report.parquet":
        aggregator_dir = p.parent / "aggregator_metric"
        if not aggregator_dir.exists():
            raise FileNotFoundError(f"aggregator_metric dir not found under '{p.parent}'")
        candidates = sorted(
            list(aggregator_dir.glob("*weighted_average_metrics*.parquet")),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No aggregator_metric weighted_average_metrics parquet found under '{aggregator_dir}'."
            )
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous metrics under '{aggregator_dir}'; provide an exact metrics parquet path")
        return str(candidates[0])
    return str(p)


def find_runner_report(run_output_dir: str) -> Optional[str]:
    """
    Try to find the nuPlan `runner_report.parquet` under a simulation output directory.
    Returns a unique report path or None; ambiguous directories are rejected.
    """
    root = Path(run_output_dir)
    if not root.exists():
        return None
    # typical path pattern includes .../runner_report.parquet somewhere under the experiment_uid folder
    matches = list(root.rglob("runner_report.parquet"))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Multiple runner reports under '{root}'; provide one experiment directory")
    return str(matches[0])


def _scenario_identities(frame: pd.DataFrame, scenario_column: str, source: str) -> set:
    """Match the per-scenario identities used by nuPlan's runner and aggregator."""
    columns = ["log_name", scenario_column, "planner_name"]
    if any(column not in frame for column in columns):
        raise ValueError(f"Scenario identity fields missing in {source}: expected {columns}")
    for column in columns:
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError(f"Scenario identity field '{column}' is missing or invalid in {source}")
    if frame.duplicated(subset=columns).any():
        raise ValueError(f"Duplicate scenario identities in {source}")
    return set(frame[columns].itertuples(index=False, name=None))


def parse_final_score_from_runner_report(report_parquet_path: str) -> SimScore:
    """
    Parse the FINAL SCORE metrics, aligning with `print_result.py`.

    Input can be:
      - path to runner_report.parquet (will be resolved to aggregator_metric/*weighted_average_metrics*.parquet)
      - path to the metrics parquet directly
    """
    resolved = _resolve_metrics_parquet(report_parquet_path)
    df = pd.read_parquet(resolved)

    if "scenario" not in df.columns:
        raise ValueError(f"Missing scenario column in metrics parquet: {resolved}")
    cand = df[df["scenario"] == "final_score"]
    if len(cand) != 1:
        raise ValueError(f"Expected exactly one final_score row, found {len(cand)}: {resolved}")
    row = cand.iloc[0].to_dict()
    score = float(row.get("score", float("nan")))
    if not math.isfinite(score):
        raise ValueError(f"Non-finite final score in {resolved}: {score}")
    num_scenarios = float(row.get("num_scenarios", float("nan")))
    if not math.isfinite(num_scenarios) or num_scenarios <= 0 or not num_scenarios.is_integer():
        raise ValueError(f"Invalid num_scenarios in {resolved}: {num_scenarios}")

    metrics_path = Path(resolved)
    report_path = (metrics_path.parent.parent if metrics_path.parent.name == "aggregator_metric" else metrics_path.parent) / "runner_report.parquet"
    if not report_path.is_file():
        raise FileNotFoundError(f"Cannot verify simulation completeness: runner report missing at {report_path}")
    runners = pd.read_parquet(report_path)
    if runners.empty or "succeeded" not in runners:
        raise ValueError(f"Missing simulation success records: {report_path}")
    failed = ~runners["succeeded"].eq(True).fillna(False)
    if failed.any():
        raise ValueError(f"{int(failed.sum())} simulation(s) failed or have unknown status: {report_path}")
    planner = row.get("planner_name")
    if not isinstance(planner, str) or "planner_name" not in runners or not runners["planner_name"].eq(planner).all():
        raise ValueError(f"Planner identity mismatch or ambiguity between {resolved} and {report_path}")
    if num_scenarios != len(runners):
        raise ValueError(f"Scenario count mismatch: metrics={num_scenarios:g}, runner reports={len(runners)}")

    # nuPlan gives individual scenarios num_scenarios=None; type aggregates and
    # the final row carry counts. Equal counts alone cannot identify stale data.
    scenario_rows = df[df["num_scenarios"].isna()]
    runner_ids = _scenario_identities(runners, "scenario_name", str(report_path))
    metric_ids = _scenario_identities(scenario_rows, "scenario", resolved)
    if runner_ids != metric_ids:
        raise ValueError(
            "Scenario identities differ between runner reports and aggregated metrics: "
            f"missing={len(runner_ids - metric_ids)}, unexpected={len(metric_ids - runner_ids)}"
        )

    return SimScore(score=score, num_scenarios=num_scenarios, components=row)
