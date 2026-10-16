"""Generate synthetic illustration data, never measured benchmark results.

This optional plotting aid is excluded from the training/evaluation workflow.
Outputs marked ``mock://`` must not be used to substantiate paper metrics.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List


STABILITY_VARIANTS = ["full_no_clipping", "full_no_batch_norm", "full"]
DERIVED_METRICS = {"p90_over_p50", "p99_over_p50"}
WEIGHT_METRICS = {"gaiw_p50", "gaiw_p90", "gaiw_p99"}
EVAL_METRICS = {"score_up", "coll_down", "collision_score_loss"}


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _epoch_t(epoch: int, max_epoch: int) -> float:
    if max_epoch <= 1:
        return 0.0
    return float(epoch - 1) / float(max_epoch - 1)


def _wiggle(t: float, amp: float, freq: float, phase: float) -> float:
    return amp * math.sin(freq * math.pi * t + phase) * (1.0 - 0.22 * t)


def _mock_score(variant: str, t: float) -> float:
    if variant == "full":
        base = 0.8190 + 0.0063 * (1.0 - math.exp(-3.1 * t))
        return base + _wiggle(t, 0.00045, 2.4, 0.2)
    if variant == "full_no_batch_norm":
        base = 0.8157 + 0.0032 * (1.0 - math.exp(-2.0 * t))
        return base + _wiggle(t, 0.0015, 3.2, 0.9)
    base = 0.8162 + 0.0024 * (1.0 - math.exp(-2.1 * t))
    return base + _wiggle(t, 0.0011, 3.7, 0.1)


def _mock_collision(variant: str, t: float) -> float:
    if variant == "full":
        base = 0.0570 - 0.0110 * (1.0 - math.exp(-2.8 * t))
        val = base + _wiggle(t, 0.0008, 4.0, 0.15)
    elif variant == "full_no_batch_norm":
        base = 0.0605 - 0.0039 * (1.0 - math.exp(-1.6 * t))
        val = base + _wiggle(t, 0.0021, 3.2, 0.8)
    else:
        base = 0.0591 - 0.0047 * (1.0 - math.exp(-1.9 * t))
        val = base + _wiggle(t, 0.0017, 3.5, 0.35)
    return max(0.040, val)


def _mock_p50(variant: str, t: float) -> float:
    if variant == "full":
        return 1.035 + _wiggle(t, 0.0035, 2.2, 0.1)
    if variant == "full_no_batch_norm":
        return 3.38 + _wiggle(t, 0.12, 2.6, 0.45)
    return 1.045 + _wiggle(t, 0.010, 3.3, 0.2)


def _mock_ratio90(variant: str, t: float) -> float:
    if variant == "full":
        return 1.270 + _wiggle(t, 0.006, 2.8, 0.1)
    if variant == "full_no_batch_norm":
        return 1.295 + _wiggle(t, 0.018, 2.8, 0.5)
    return 1.340 + 0.035 * math.exp(-0.8 * t) + _wiggle(t, 0.018, 3.5, 0.3)


def _mock_ratio99(variant: str, t: float) -> float:
    if variant == "full":
        return 1.330 - 0.020 * (1.0 - math.exp(-2.2 * t)) + _wiggle(t, 0.010, 2.9, 0.25)
    if variant == "full_no_batch_norm":
        return 1.405 - 0.012 * (1.0 - math.exp(-1.7 * t)) + _wiggle(t, 0.018, 3.0, 0.65)
    return 1.530 - 0.040 * (1.0 - math.exp(-1.0 * t)) + _wiggle(t, 0.045, 4.2, 0.45)


def _build_mock_values(epochs: List[int]) -> Dict[str, Dict[str, Dict[int, float]]]:
    max_epoch = max(epochs) if epochs else 1
    out: Dict[str, Dict[str, Dict[int, float]]] = {}
    for variant in STABILITY_VARIANTS:
        out[variant] = {"score_up": {}, "coll_down": {}, "gaiw_p50": {}, "gaiw_p90": {}, "gaiw_p99": {}}
        for epoch in epochs:
            t = _epoch_t(epoch, max_epoch)
            p50 = _mock_p50(variant, t)
            r90 = _mock_ratio90(variant, t)
            r99 = _mock_ratio99(variant, t)
            out[variant]["score_up"][epoch] = _mock_score(variant, t)
            out[variant]["coll_down"][epoch] = _mock_collision(variant, t)
            out[variant]["gaiw_p50"][epoch] = p50
            out[variant]["gaiw_p90"][epoch] = p50 * r90
            out[variant]["gaiw_p99"][epoch] = p50 * r99
    return out


def _collect_epochs(rows: List[Dict[str, str]]) -> List[int]:
    epochs = set()
    for row in rows:
        try:
            epochs.add(int(row.get("epoch", "")))
        except Exception:
            continue
    return sorted(epochs)


def _mock_value(
    variant: str,
    metric: str,
    epoch: int,
    mock_values: Dict[str, Dict[str, Dict[int, float]]],
) -> float:
    if metric == "p90_over_p50":
        return mock_values[variant]["gaiw_p90"][epoch] / mock_values[variant]["gaiw_p50"][epoch]
    if metric == "p99_over_p50":
        return mock_values[variant]["gaiw_p99"][epoch] / mock_values[variant]["gaiw_p50"][epoch]
    return mock_values[variant]["coll_down" if metric == "collision_score_loss" else metric][epoch]


def _mock_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    epochs = _collect_epochs(rows)
    mock_values = _build_mock_values(epochs)
    out: List[Dict[str, str]] = []
    for row in rows:
        variant = row.get("variant", "").strip()
        metric = row.get("metric", "").strip()
        section = row.get("section", "").strip()
        if variant not in STABILITY_VARIANTS:
            out.append(dict(row))
            continue
        try:
            epoch = int(row.get("epoch", ""))
        except Exception:
            out.append(dict(row))
            continue

        if metric in EVAL_METRICS | WEIGHT_METRICS | DERIVED_METRICS:
            new_row = dict(row)
            new_row["suite_root"] = "mock://gaiw_stability_illustration"
            new_row["value"] = f"{_mock_value(variant, metric, epoch, mock_values):.10f}"
            out.append(new_row)
        else:
            out.append(dict(row))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an illustration-only mock GAIW stability CSV from a real cached CSV.")
    parser.add_argument("--input-csv", required=True, help="Real cached CSV produced by gaiw_stability_plot.py")
    parser.add_argument("--output-csv", required=True, help="Output path for the mock CSV")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    rows = _read_rows(input_csv)
    mock_rows = _mock_rows(rows)
    _write_rows(output_csv, mock_rows)
    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
