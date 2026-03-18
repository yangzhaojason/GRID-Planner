from __future__ import annotations

import json
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
      - If input is runner_report.parquet, prefer the newest
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
        return str(candidates[0])
    return str(p)


def find_runner_report(run_output_dir: str) -> Optional[str]:
    """
    Try to find the nuPlan `runner_report.parquet` under a simulation output directory.
    Returns absolute path or None.
    """
    root = Path(run_output_dir)
    if not root.exists():
        return None
    # typical path pattern includes .../runner_report.parquet somewhere under the experiment_uid folder
    matches = list(root.rglob("runner_report.parquet"))
    if not matches:
        return None
    # choose the most recently modified report
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0])


def parse_final_score_from_runner_report(report_parquet_path: str) -> SimScore:
    """
    Parse the FINAL SCORE metrics, aligning with `print_result.py`.

    Input can be:
      - path to runner_report.parquet (will be resolved to aggregator_metric/*weighted_average_metrics*.parquet)
      - path to the metrics parquet directly
    """
    resolved = _resolve_metrics_parquet(report_parquet_path)
    df = pd.read_parquet(resolved)

    # Primary: aggregator metrics has a 'scenario' column with 'final_score' row.
    if "scenario" in df.columns:
        cand = df[df["scenario"] == "final_score"]
        if cand.empty:
            raise ValueError(f"No final_score row found in metrics parquet: {resolved}")
    else:
        # Fallback: legacy runner report format (best-effort).
        cand = df
        if "scenario_type" in cand.columns:
            cand = cand[cand["scenario_type"] == "final_score"]
        if cand.empty:
            cand = df.tail(1)

    row = cand.iloc[0].to_dict()
    score = float(row.get("score", float("nan")))
    num_scenarios = row.get("num_scenarios", None)
    try:
        num_scenarios = float(num_scenarios) if num_scenarios is not None else None
    except Exception:
        num_scenarios = None

    return SimScore(score=score, num_scenarios=num_scenarios, components=row)

