from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


DEFAULT_VARIANTS = [
    "unified_objective",
    "no_advantage_reweighting",
    "no_opsd_term",
]

VARIANT_LABELS = {
    "unified_objective": "Unified objective",
    "no_advantage_reweighting": "w/o advantage reweighting",
    "no_opsd_term": "w/o OPSD term",
}

VARIANT_COLORS = {
    "unified_objective": "#4C72B0",
    "no_advantage_reweighting": "#DD8452",
    "no_opsd_term": "#C44E52",
}

DEFAULT_DATA_JSON_NAME = "grid_obj_sample_eff_data.json"
DEFAULT_DATA_CSV_NAME = "grid_obj_sample_eff_data.csv"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _metric_points(
    suite_root: Path,
    primary_split: str,
    variant: str,
    metric_key: str,
) -> List[Dict[str, Any]]:
    eval_path = suite_root / "eval" / primary_split / variant / "benchmark_per_epoch.jsonl"
    meta_path = suite_root / "meta" / f"{variant}.json"
    rows = [
        row
        for row in _read_jsonl(eval_path)
        if row.get("status") == "OK" and row.get("epoch") is not None and row.get(metric_key) is not None
    ]
    rows.sort(key=lambda r: int(r["epoch"]))
    meta = _read_json(meta_path) if meta_path.exists() else {}
    rollouts_per_epoch = meta.get("rollouts_per_epoch")
    points: List[Dict[str, Any]] = []
    for row in rows:
        epoch = int(row["epoch"])
        point = {
            "epoch": epoch,
            "value": float(row[metric_key]),
        }
        if rollouts_per_epoch is not None:
            try:
                point["rollouts"] = int(round(float(rollouts_per_epoch) * epoch))
            except Exception:
                pass
        points.append(point)
    return points


def _extract_payload_from_suite(
    suite_root: Path,
    primary_split: str,
    variants: Sequence[str],
) -> Dict[str, Any]:
    variant_payloads: Dict[str, Any] = {}
    x_axis = "epoch"
    for variant in variants:
        meta_path = suite_root / "meta" / f"{variant}.json"
        meta = _read_json(meta_path) if meta_path.exists() else {}
        score_points = _metric_points(suite_root, primary_split, variant, "score_up")
        coll_points = _metric_points(suite_root, primary_split, variant, "coll_down")
        if any("rollouts" in p for p in score_points + coll_points):
            x_axis = "on_policy_rollouts"
        variant_payloads[variant] = {
            "label": VARIANT_LABELS.get(variant, variant),
            "color": VARIANT_COLORS.get(variant, "#333333"),
            "meta": meta,
            "metrics": {
                "score_up": score_points,
                "coll_down": coll_points,
            },
        }
    return {
        "schema_version": 1,
        "source_suite_root": str(suite_root),
        "primary_split": primary_split,
        "x_axis": x_axis,
        "variants": variant_payloads,
    }


def _flatten_payload_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant, body in payload.get("variants", {}).items():
        for metric, points in body.get("metrics", {}).items():
            for point in points:
                rows.append(
                    {
                        "suite_root": payload.get("source_suite_root"),
                        "primary_split": payload.get("primary_split"),
                        "x_axis": payload.get("x_axis"),
                        "variant": variant,
                        "label": body.get("label", variant),
                        "color": body.get("color", "#333333"),
                        "metric": metric,
                        "epoch": point.get("epoch"),
                        "rollouts": point.get("rollouts"),
                        "value": point.get("value"),
                    }
                )
    rows.sort(key=lambda r: (r["variant"], r["metric"], int(r["epoch"])))
    return rows


def _write_data_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_data_csv(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _flatten_payload_rows(payload)
    fieldnames = [
        "suite_root",
        "primary_split",
        "x_axis",
        "variant",
        "label",
        "color",
        "metric",
        "epoch",
        "rollouts",
        "value",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _x_key(payload: Dict[str, Any]) -> str:
    if payload.get("x_axis") == "on_policy_rollouts":
        return "rollouts"
    return "epoch"


def _points_to_xy(payload: Dict[str, Any], points: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    x_key = _x_key(payload)
    xs: List[float] = []
    ys: List[float] = []
    for point in points:
        if point.get("value") is None:
            continue
        x = point.get(x_key)
        if x is None:
            x = point.get("epoch")
        if x is None:
            continue
        xs.append(float(x))
        ys.append(float(point["value"]))
    return xs, ys


def _annotate_if_empty(ax, message: str) -> None:
    if not ax.lines:
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#666666")


def _plot_metric_panel(ax, payload: Dict[str, Any], metric: str, title: str, ylabel: str) -> None:
    for variant in payload.get("variants", {}):
        body = payload["variants"][variant]
        points = body.get("metrics", {}).get(metric, [])
        xs, ys = _points_to_xy(payload, points)
        if xs:
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=4,
                color=body.get("color", "#333333"),
                label=body.get("label", variant),
            )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _annotate_if_empty(ax, f"No {metric} data found")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=9)


def _render_figure(payload: Dict[str, Any], output_png: Path, output_svg: Path, dpi: int) -> None:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    _plot_metric_panel(axes[0], payload, "score_up", "Reactive Score vs. Rollouts", "Closed-loop score")
    _plot_metric_panel(axes[1], payload, "coll_down", "Collision Rate vs. Rollouts", "Collision rate")

    xlabel = "On-policy rollouts" if payload.get("x_axis") == "on_policy_rollouts" else "Epoch"
    for ax in axes:
        ax.set_xlabel(xlabel)
        if payload.get("x_axis") == "on_policy_rollouts":
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    fig.suptitle("Sample efficiency of the unified on-policy objective", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render sample-efficiency plots for the unified objective ablation.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--suite-root", help="Root directory created by launch_grid_obj_ablation_suite.sh")
    source_group.add_argument("--input-json", help="Previously exported plot data JSON")
    parser.add_argument(
        "--primary-split",
        default="val14",
        help="Split used for the sample-efficiency curves when reading from suite outputs.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variant ids to plot.",
    )
    parser.add_argument(
        "--output-png",
        default=None,
        help="PNG output path. Default: <figure-dir>/grid_obj_sample_eff.png",
    )
    parser.add_argument(
        "--output-svg",
        default=None,
        help="SVG output path. Default: <figure-dir>/grid_obj_sample_eff.svg",
    )
    parser.add_argument(
        "--output-data-json",
        default=None,
        help="Optional path to save extracted plot data JSON.",
    )
    parser.add_argument(
        "--output-data-csv",
        default=None,
        help="Optional path to save extracted plot data CSV.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Raster dpi for PNG output.")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.suite_root:
        suite_root = Path(args.suite_root)
        figure_dir = suite_root / "figures"
        payload = _extract_payload_from_suite(suite_root, primary_split=args.primary_split, variants=variants)
        output_data_json = Path(args.output_data_json) if args.output_data_json else figure_dir / DEFAULT_DATA_JSON_NAME
        output_data_csv = Path(args.output_data_csv) if args.output_data_csv else figure_dir / DEFAULT_DATA_CSV_NAME
        _write_data_json(output_data_json, payload)
        _write_data_csv(output_data_csv, payload)
        print(f"Saved {output_data_json}")
        print(f"Saved {output_data_csv}")
    else:
        input_json = Path(args.input_json)
        figure_dir = input_json.parent
        payload = _read_json(input_json)
        if args.output_data_json:
            output_data_json = Path(args.output_data_json)
            _write_data_json(output_data_json, payload)
            print(f"Saved {output_data_json}")
        if args.output_data_csv:
            output_data_csv = Path(args.output_data_csv)
            _write_data_csv(output_data_csv, payload)
            print(f"Saved {output_data_csv}")

    figure_dir.mkdir(parents=True, exist_ok=True)
    output_png = Path(args.output_png) if args.output_png else figure_dir / "grid_obj_sample_eff.png"
    output_svg = Path(args.output_svg) if args.output_svg else figure_dir / "grid_obj_sample_eff.svg"
    _render_figure(payload, output_png, output_svg, args.dpi)
    print(f"Saved {output_png}")
    print(f"Saved {output_svg}")


if __name__ == "__main__":
    main()
