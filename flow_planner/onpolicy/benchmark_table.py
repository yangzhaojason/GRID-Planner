from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from flow_planner.onpolicy.sim_report import find_runner_report, parse_final_score_from_runner_report


def _to_float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"", "none", "nan", "null"}:
            return None
    try:
        f = float(v)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _pick_first_number(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
    for k in keys:
        if k in d:
            v = _to_float_or_none(d.get(k))
            if v is not None:
                return v
    return None


def _mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _parse_maybe_struct(v: Any) -> Any:
    """Best-effort parsing for json/python-literal strings stored in parquet object columns."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if len(s) < 2:
        return v
    if not ((s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))):
        return v
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return v


def _is_runtime_hint(key_hint: str) -> bool:
    h = key_hint.lower()
    return any(
        token in h
        for token in (
            "runtime",
            "latency",
            "compute_trajectory",
            "inference",
            "feature_building",
        )
    )


def _is_compute_trajectory_hint(key_hint: str) -> bool:
    h = key_hint.lower()
    return (
        "compute_trajectory" in h
        or "trajectory_runtime" in h
        or "trajectory_latency" in h
        or ("trajectory" in h and ("runtime" in h or "latency" in h))
    )


def _to_ms(value: float, key_hint: str) -> float:
    """
    Convert runtime/latency-like values to ms with a conservative heuristic:
    - explicit *_ms: keep
    - explicit *_s / sec / second: *1000
    - ambiguous runtime values: treat <1.0 as seconds, otherwise as ms
    """
    h = key_hint.lower()
    if "millisecond" in h or "_ms" in h or h.endswith("ms"):
        return value
    if "microsecond" in h or "_us" in h or h.endswith("us"):
        return value / 1_000.0
    if "nanosecond" in h or "_ns" in h or h.endswith("ns"):
        return value / 1_000_000.0
    if "second" in h or "_s" in h or h.endswith("sec"):
        return value * 1_000.0
    return value * 1_000.0 if abs(value) < 1.0 else value


def _collect_runtime_ms_from_obj(
    obj: Any, key_hint: str = "", hint_predicate: Callable[[str], bool] = _is_runtime_hint
) -> List[float]:
    obj = _parse_maybe_struct(obj)
    out: List[float] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            child_hint = f"{key_hint}.{k}" if key_hint else str(k)
            out.extend(_collect_runtime_ms_from_obj(v, child_hint, hint_predicate))
        return out

    if isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(_collect_runtime_ms_from_obj(item, key_hint, hint_predicate))
        return out

    num = _to_float_or_none(obj)
    if num is None:
        return out
    if not hint_predicate(key_hint):
        return out
    out.append(_to_ms(num, key_hint))
    return out


def _resolve_report_input(path_like: str) -> str:
    p = Path(path_like)
    if p.is_dir():
        rr = find_runner_report(str(p))
        if rr is None:
            raise FileNotFoundError(f"runner_report.parquet not found under directory: {p}")
        return rr
    if p.is_file():
        return str(p)
    raise FileNotFoundError(f"Input path does not exist: {p}")


def _resolve_runner_report_path(report_path: str) -> Optional[str]:
    """
    Resolve runner_report.parquet from:
      - report_path itself
      - weighted_average_metrics parquet path
      - nearby parent directories
    """
    p = Path(report_path)
    if p.is_file() and p.name == "runner_report.parquet":
        return str(p)

    if p.is_file() and p.suffix == ".parquet":
        # Common case: .../<run_dir>/aggregator_metric/*weighted_average_metrics*.parquet
        if p.parent.name == "aggregator_metric":
            cand = p.parent.parent / "runner_report.parquet"
            if cand.exists():
                return str(cand)
        # Fallback: probe a few parents
        for parent in (p.parent, p.parent.parent, p.parent.parent.parent):
            cand = parent / "runner_report.parquet"
            if cand.exists():
                return str(cand)
    elif p.is_dir():
        return find_runner_report(str(p))
    return None


def _extract_latency_ms_from_runner_report(
    runner_report_path: str, latency_mode: str
) -> Tuple[Optional[float], str]:
    """
    Best-effort extraction of planner runtime latency (ms) from runner_report.parquet.
    """
    mode = latency_mode.strip().lower()
    if mode not in {"compute_trajectory_only", "auto"}:
        return None, f"invalid latency_mode: {latency_mode}"

    try:
        pd = importlib.import_module("pandas")
    except Exception:
        return None, "auto parse skipped: pandas unavailable"

    try:
        df = pd.read_parquet(runner_report_path)
    except Exception as e:
        return None, f"auto parse failed: {e}"

    # 1) Prefer explicit scalar columns.
    if mode == "compute_trajectory_only":
        direct_ms_keys = (
            "compute_trajectory_latency_ms",
            "compute_trajectory_runtime_ms",
            "compute_trajectory_mean_ms",
            "compute_trajectory_runtimes_mean_ms",
        )
        direct_s_keys = (
            "compute_trajectory_runtime_mean_s",
            "compute_trajectory_runtimes_mean_s",
            "compute_trajectory_latency_s",
            "compute_trajectory_runtimes_mean",
            "compute_trajectory_runtime_mean",
        )
        hint_predicate: Callable[[str], bool] = _is_compute_trajectory_hint
        source_prefix = "auto compute_trajectory from"
    else:
        direct_ms_keys = (
            "latency_ms",
            "mean_latency_ms",
            "planner_runtime_mean_ms",
            "runtime_mean_ms",
            "mean_ms",
        )
        direct_s_keys = (
            "planner_runtime_mean_s",
            "runtime_mean_s",
            "mean_runtime_s",
            "mean_latency_s",
            "compute_trajectory_runtimes_mean",
            "compute_trajectory_runtime_mean",
        )
        hint_predicate = _is_runtime_hint
        source_prefix = "auto from"

    for k in direct_ms_keys:
        if k in df.columns:
            vals = [_to_float_or_none(v) for v in df[k].tolist()]
            vals = [v for v in vals if v is not None]
            m = _mean_or_none(vals)
            if m is not None:
                return m, f"{source_prefix} runner_report.{k}"

    # 2) Explicit second-based scalar columns.
    for k in direct_s_keys:
        if k in df.columns:
            vals = [_to_float_or_none(v) for v in df[k].tolist()]
            vals = [v for v in vals if v is not None]
            m = _mean_or_none(vals)
            if m is not None:
                return m * 1_000.0, f"{source_prefix} runner_report.{k} (sec->ms)"

    # 3) Parse list/object columns (planner_report or any runtime-like columns).
    candidate_cols: List[str] = []
    for c in df.columns:
        c_str = str(c)
        lc = c_str.lower()
        if mode == "compute_trajectory_only":
            if (
                "planner_report" in lc
                or "report" in lc
                or "compute_trajectory" in lc
                or "trajectory_runtime" in lc
                or "trajectory_latency" in lc
            ):
                candidate_cols.append(c_str)
        elif (
            "planner_report" in lc
            or "runtime" in lc
            or "latency" in lc
            or "compute_trajectory" in lc
        ):
            candidate_cols.append(c_str)

    for col in candidate_cols:
        vals: List[float] = []
        for item in df[col].tolist():
            vals.extend(_collect_runtime_ms_from_obj(item, key_hint=col, hint_predicate=hint_predicate))
        m = _mean_or_none(vals)
        if m is not None:
            return m, f"{source_prefix} runner_report.{col}"

    if mode == "compute_trajectory_only":
        return None, "N/A (runner_report has no recognizable compute_trajectory runtime fields)"
    return None, "N/A (runner_report has no recognizable planner runtime fields)"


def _load_latency_ms(
    latency_ms: Optional[float], latency_json: Optional[str], report_path: str, latency_mode: str
) -> Tuple[Optional[float], str]:
    if latency_ms is not None:
        return float(latency_ms), "user provided (--latency-ms)"

    if latency_json:
        p = Path(latency_json)
        if not p.exists():
            raise FileNotFoundError(f"latency_json not found: {p}")
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            for k in (
                "latency_ms",
                "mean_latency_ms",
                "planner_runtime_mean_ms",
                "runtime_mean_ms",
                "mean_ms",
            ):
                v = _to_float_or_none(data.get(k))
                if v is not None:
                    return v, f"user provided (--latency-json:{k})"

    rr = _resolve_runner_report_path(report_path)
    if rr is None:
        return None, "N/A (runner_report.parquet not found for auto latency parse)"
    auto_latency, src = _extract_latency_ms_from_runner_report(rr, latency_mode=latency_mode)
    if auto_latency is None:
        return None, src
    return auto_latency, src


def _fmt(v: Optional[float], ndigits: int = 6) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{ndigits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print benchmark table metrics (Score/Coll/Red-light/Lane dev/Jerk/Latency) from nuPlan outputs."
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to runner_report.parquet, weighted_average_metrics parquet, or a run directory that contains runner_report.parquet.",
    )
    parser.add_argument("--variant", default="variant", help="Name printed in table row.")
    parser.add_argument(
        "--latency-mode",
        type=str,
        default="compute_trajectory_only",
        choices=("compute_trajectory_only", "auto"),
        help="Latency auto-parse mode. Default 'compute_trajectory_only' uses only compute_trajectory runtime fields; 'auto' allows broader runtime/latency fields.",
    )
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=None,
        help="Optional latency (ms). Overrides auto parsing from runner_report.parquet.",
    )
    parser.add_argument(
        "--latency-json",
        type=str,
        default=None,
        help="Optional JSON containing latency keys (latency_ms / mean_latency_ms / planner_runtime_mean_ms / mean_ms). Overrides auto parsing if provided.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also print machine-readable JSON payload.",
    )
    args = parser.parse_args()

    report_path = _resolve_report_input(args.report)
    sim_score = parse_final_score_from_runner_report(report_path)
    row = sim_score.components or {}

    # Official metrics from nuPlan final_score row
    score = _to_float_or_none(row.get("score"))
    no_at_fault_coll = _to_float_or_none(row.get("no_ego_at_fault_collisions"))
    drivable_comp = _to_float_or_none(row.get("drivable_area_compliance"))
    direction_comp = _to_float_or_none(row.get("driving_direction_compliance"))
    comfort_comp = _to_float_or_none(row.get("ego_is_comfortable"))
    jerk_direct = _to_float_or_none(row.get("ego_jerk"))

    # Table columns (lower is better for violation-style columns)
    coll_violation = None if no_at_fault_coll is None else (1.0 - no_at_fault_coll)

    red_light_violation = _pick_first_number(
        row,
        (
            "red_light_violation",
            "red_light_violations",
            "red_light_stopline_violation",
        ),
    )
    red_light_src = "custom / provided by report"
    if red_light_violation is None:
        red_light_comp = _pick_first_number(row, ("red_light_compliance", "traffic_light_compliance"))
        if red_light_comp is not None:
            red_light_violation = 1.0 - red_light_comp
            red_light_src = "derived from red_light_compliance"
        else:
            red_light_src = "N/A (not in default nuPlan final_score)"

    lane_dev = _pick_first_number(row, ("lane_deviation", "ego_lane_deviation", "lane_dev_violation"))
    lane_dev_src = "custom / provided by report"
    if lane_dev is None:
        if drivable_comp is not None and direction_comp is not None:
            lane_dev = 1.0 - 0.5 * (drivable_comp + direction_comp)
            lane_dev_src = "proxy: 1 - mean(drivable_area_compliance, driving_direction_compliance)"
        elif drivable_comp is not None:
            lane_dev = 1.0 - drivable_comp
            lane_dev_src = "proxy: 1 - drivable_area_compliance"
        else:
            lane_dev_src = "N/A (no lane proxy fields found)"

    jerk_metric = jerk_direct
    jerk_src = "ego_jerk from report"
    if jerk_metric is None:
        if comfort_comp is not None:
            jerk_metric = 1.0 - comfort_comp
            jerk_src = "proxy: 1 - ego_is_comfortable"
        else:
            jerk_src = "N/A (no jerk/comfort field found)"

    latency_ms, latency_src = _load_latency_ms(
        args.latency_ms, args.latency_json, report_path, args.latency_mode
    )

    print("===== Benchmark Metrics =====")
    print(f"Variant:   {args.variant}")
    print(f"Report:    {report_path}")
    print(f"Score_up:      {_fmt(score)}")
    print(f"Coll_down:     {_fmt(coll_violation)}")
    print(f"RedLight_down: {_fmt(red_light_violation)}")
    print(f"LaneDev_down:  {_fmt(lane_dev)}")
    print(f"Jerk_down:     {_fmt(jerk_metric)}")
    print(f"Latency_down:  {_fmt(latency_ms, ndigits=3)}")
    print("")
    print("Sources:")
    print("- Score: nuPlan final_score.score")
    print("- Coll:  1 - no_ego_at_fault_collisions")
    print(f"- Red-light: {red_light_src}")
    print(f"- Lane dev:  {lane_dev_src}")
    print(f"- Jerk:      {jerk_src}")
    print(f"- Latency:   {latency_src}")
    print("")
    print("Markdown row:")
    print(
        f"| {args.variant} | {_fmt(score)} | {_fmt(coll_violation)} | {_fmt(red_light_violation)} | "
        f"{_fmt(lane_dev)} | {_fmt(jerk_metric)} | {_fmt(latency_ms, ndigits=3)} |"
    )

    if args.json:
        payload = {
            "variant": args.variant,
            "report_path": report_path,
            "score_up": score,
            "coll_down": coll_violation,
            "red_light_down": red_light_violation,
            "lane_dev_down": lane_dev,
            "jerk_down": jerk_metric,
            "latency_down_ms": latency_ms,
            "latency_mode": args.latency_mode,
            "source": {
                "red_light": red_light_src,
                "lane_dev": lane_dev_src,
                "jerk": jerk_src,
                "latency": latency_src,
            },
            "raw_components": row,
        }
        print("")
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

