from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STABILITY_VARIANTS = ["full_no_clipping", "full_no_batch_norm", "full"]
VARIANT_LABELS = {
    "full_no_clipping": "Full + no clipping",
    "full_no_batch_norm": "Full + no batch normalization",
    "full": "Our method: GAIW",
}
VARIANT_COLORS = {
    "full_no_clipping": "#B85C5C",
    "full_no_batch_norm": "#D4C566",
    "full": "#244D80",
}
WEIGHT_TAGS = {
    "p50": "gaiw_p50",
    "p90": "gaiw_p90",
    "p99": "gaiw_p99",
}
WEIGHT_STYLES = {
    "p50": "-",
    "p90": "--",
    "p99": ":",
}
DEFAULT_DATA_JSON_NAME = "gaiw_stability_data.json"
DEFAULT_DATA_CSV_NAME = "gaiw_stability_data.csv"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _series_points(xs: List[int], ys: List[float]) -> List[Dict[str, float]]:
    return [{"epoch": int(x), "value": float(y)} for x, y in zip(xs, ys)]


def _points_to_xy(points: List[Dict[str, Any]]) -> Tuple[List[int], List[float]]:
    xs = [int(p["epoch"]) for p in points if p.get("epoch") is not None and p.get("value") is not None]
    ys = [float(p["value"]) for p in points if p.get("epoch") is not None and p.get("value") is not None]
    return xs, ys


def _sort_points(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [p for p in points if p.get("epoch") is not None and p.get("value") is not None],
        key=lambda p: int(p["epoch"]),
    )


def _ratio_points(
    numerator_points: List[Dict[str, Any]],
    denominator_points: List[Dict[str, Any]],
    eps: float = 1e-12,
) -> List[Dict[str, float]]:
    numerator = {int(p["epoch"]): float(p["value"]) for p in numerator_points}
    denominator = {int(p["epoch"]): float(p["value"]) for p in denominator_points}
    out: List[Dict[str, float]] = []
    for epoch in sorted(set(numerator) & set(denominator)):
        denom = denominator[epoch]
        if abs(denom) <= eps:
            continue
        out.append({"epoch": epoch, "value": float(numerator[epoch] / denom)})
    return out


def _eval_series(suite_root: Path, variant: str, key: str) -> Tuple[List[int], List[float]]:
    path = suite_root / "eval" / variant / "benchmark_per_epoch.jsonl"
    rows = [
        r
        for r in _read_jsonl(path)
        if r.get("status") == "OK" and r.get("epoch") is not None and r.get(key) is not None
    ]
    rows.sort(key=lambda r: int(r["epoch"]))
    xs = [int(r["epoch"]) for r in rows]
    ys = [float(r[key]) for r in rows]
    return xs, ys


def _load_scalar_series(tb_dir: Path, tag: str) -> Tuple[List[int], List[float]]:
    if not tb_dir.exists():
        return [], []
    try:
        mod = importlib.import_module("tensorboard.backend.event_processing.event_accumulator")
        event_acc = mod.EventAccumulator(str(tb_dir))
        event_acc.Reload()
    except Exception:
        return [], []
    tags = event_acc.Tags().get("scalars", [])
    if tag not in tags:
        return [], []
    events = event_acc.Scalars(tag)
    return [int(e.step) for e in events], [float(e.value) for e in events]


def _complete_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    variants = payload.setdefault("variants", {})
    for variant in STABILITY_VARIANTS:
        body = variants.setdefault(
            variant,
            {
                "label": VARIANT_LABELS[variant],
                "color": VARIANT_COLORS[variant],
                "eval": {},
                "weights": {},
                "derived": {},
            },
        )
        body.setdefault("label", VARIANT_LABELS.get(variant, variant))
        body.setdefault("color", VARIANT_COLORS.get(variant, "#444444"))
        body.setdefault("eval", {})
        body.setdefault("weights", {})
        body.setdefault("derived", {})
        for section in ("eval", "weights", "derived"):
            for metric, points in list(body[section].items()):
                body[section][metric] = _sort_points(points)
        if "p99_over_p50" not in body["derived"] and "gaiw_p99" in body["weights"] and "gaiw_p50" in body["weights"]:
            body["derived"]["p99_over_p50"] = _ratio_points(body["weights"]["gaiw_p99"], body["weights"]["gaiw_p50"])
        if "p90_over_p50" not in body["derived"] and "gaiw_p90" in body["weights"] and "gaiw_p50" in body["weights"]:
            body["derived"]["p90_over_p50"] = _ratio_points(body["weights"]["gaiw_p90"], body["weights"]["gaiw_p50"])
    return payload


def _extract_payload_from_suite(suite_root: Path) -> Dict[str, Any]:
    variants: Dict[str, Any] = {}
    for variant in STABILITY_VARIANTS:
        tb_dir = suite_root / "runs" / variant / "tensorboard"
        score_points = _series_points(*_eval_series(suite_root, variant, "score_up"))
        coll_points = _series_points(*_eval_series(suite_root, variant, "coll_down"))
        p50_points = _series_points(*_load_scalar_series(tb_dir, WEIGHT_TAGS["p50"]))
        p90_points = _series_points(*_load_scalar_series(tb_dir, WEIGHT_TAGS["p90"]))
        p99_points = _series_points(*_load_scalar_series(tb_dir, WEIGHT_TAGS["p99"]))
        variants[variant] = {
            "label": VARIANT_LABELS[variant],
            "color": VARIANT_COLORS[variant],
            "eval": {
                "score_up": score_points,
                "coll_down": coll_points,
            },
            "weights": {
                "gaiw_p50": p50_points,
                "gaiw_p90": p90_points,
                "gaiw_p99": p99_points,
            },
            "derived": {
                "p99_over_p50": _ratio_points(p99_points, p50_points),
                "p90_over_p50": _ratio_points(p90_points, p50_points),
            },
        }
    return _complete_payload(
        {
            "schema_version": 1,
            "source_suite_root": str(suite_root),
            "variants": variants,
        }
    )


def _payload_from_csv(path: Path) -> Dict[str, Any]:
    rows = _read_csv_rows(path)
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "source_suite_root": rows[0]["suite_root"] if rows else "",
        "variants": {},
    }
    for row in rows:
        variant = row.get("variant", "").strip()
        if not variant:
            continue
        section = row.get("section", "").strip()
        metric = row.get("metric", "").strip()
        if section not in {"eval", "weights", "derived"} or not metric:
            continue
        try:
            epoch = int(row.get("epoch", ""))
            value = float(row.get("value", ""))
        except Exception:
            continue
        body = payload["variants"].setdefault(
            variant,
            {
                "label": row.get("label", "").strip() or VARIANT_LABELS.get(variant, variant),
                "color": row.get("color", "").strip() or VARIANT_COLORS.get(variant, "#444444"),
                "eval": {},
                "weights": {},
                "derived": {},
            },
        )
        body.setdefault(section, {})
        body[section].setdefault(metric, [])
        body[section][metric].append({"epoch": epoch, "value": value})
    return _complete_payload(payload)


def _flatten_payload_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    suite_root = payload.get("source_suite_root")
    variants = payload.get("variants", {})
    for variant in STABILITY_VARIANTS:
        body = variants.get(variant)
        if body is None:
            continue
        for section in ("eval", "weights", "derived"):
            for metric, points in body.get(section, {}).items():
                for point in points:
                    epoch = point.get("epoch")
                    value = point.get("value")
                    if epoch is None or value is None:
                        continue
                    rows.append(
                        {
                            "suite_root": suite_root,
                            "variant": variant,
                            "label": body.get("label", VARIANT_LABELS.get(variant, variant)),
                            "color": body.get("color", VARIANT_COLORS.get(variant, "#333333")),
                            "section": section,
                            "metric": metric,
                            "epoch": int(epoch),
                            "value": float(value),
                        }
                    )
    rows.sort(key=lambda r: (STABILITY_VARIANTS.index(r["variant"]), r["section"], r["metric"], r["epoch"]))
    return rows


def _write_data_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_data_csv(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _flatten_payload_rows(payload)
    fieldnames = ["suite_root", "variant", "label", "color", "section", "metric", "epoch", "value"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _variant_label(payload: Dict[str, Any], variant: str) -> str:
    return payload.get("variants", {}).get(variant, {}).get("label", VARIANT_LABELS.get(variant, variant))


def _variant_color(payload: Dict[str, Any], variant: str) -> str:
    return payload.get("variants", {}).get(variant, {}).get("color", VARIANT_COLORS.get(variant, "#333333"))


def _variant_points(payload: Dict[str, Any], variant: str, section: str, metric: str) -> List[Dict[str, Any]]:
    return payload.get("variants", {}).get(variant, {}).get(section, {}).get(metric, [])


def _annotate_if_empty(ax, message: str) -> None:
    if not ax.lines:
        ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#666666")


def _style_axes(ax) -> None:
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FBFBFC")


def _ema(values: List[float], alpha: float = 0.35) -> List[float]:
    if not values:
        return []
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out


def _plot_raw_and_smoothed(
    ax,
    xs: List[int],
    ys: List[float],
    color: str,
    label: str,
    highlight: bool = False,
    alpha: float = 0.32,
) -> None:
    if not xs:
        return
    smooth = _ema(ys, alpha=0.35)
    raw_linewidth = 1.0 if highlight else 0.9
    smooth_linewidth = 2.6 if highlight else 1.8
    raw_alpha = 0.28 if highlight else 0.20
    smooth_alpha = 0.98 if highlight else alpha
    ax.plot(xs, ys, color=color, linewidth=raw_linewidth, alpha=raw_alpha)
    ax.plot(xs, smooth, color=color, linewidth=smooth_linewidth, alpha=smooth_alpha, label=label)


def _epoch_variant_values(payload: Dict[str, Any], section: str, metric: str) -> Dict[int, Dict[str, float]]:
    by_epoch: Dict[int, Dict[str, float]] = {}
    for variant in STABILITY_VARIANTS:
        for point in _variant_points(payload, variant, section, metric):
            epoch = point.get("epoch")
            value = point.get("value")
            if epoch is None or value is None:
                continue
            by_epoch.setdefault(int(epoch), {})[variant] = float(value)
    return by_epoch


def _normalized_gap_points(payload: Dict[str, Any], section: str, metric: str, higher_is_better: bool) -> Dict[str, List[Dict[str, float]]]:
    by_epoch = _epoch_variant_values(payload, section, metric)
    out = {variant: [] for variant in STABILITY_VARIANTS}
    for epoch in sorted(by_epoch):
        vals = by_epoch[epoch]
        if not vals:
            continue
        epoch_values = list(vals.values())
        best = max(epoch_values) if higher_is_better else min(epoch_values)
        worst = min(epoch_values) if higher_is_better else max(epoch_values)
        denom = worst - best
        for variant, value in vals.items():
            if abs(denom) <= 1e-12:
                gap = 0.0
            else:
                gap = (best - value) / denom if higher_is_better else (value - best) / denom
            out[variant].append({"epoch": int(epoch), "value": float(gap)})
    return out


def _plot_collision_panel(ax, payload: Dict[str, Any]) -> None:
    for variant in STABILITY_VARIANTS:
        xs, ys = _points_to_xy(_variant_points(payload, variant, "eval", "coll_down"))
        if xs:
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=4,
                color=_variant_color(payload, variant),
                label=_variant_label(payload, variant),
            )
    ax.set_title("Collision Rate vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Collision rate")
    _style_axes(ax)
    _annotate_if_empty(ax, "No collision data found")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=9)


def _plot_tail_ratio_panel(ax, payload: Dict[str, Any]) -> None:
    for variant in STABILITY_VARIANTS:
        xs, ys = _points_to_xy(_variant_points(payload, variant, "derived", "p99_over_p50"))
        if xs:
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=4,
                color=_variant_color(payload, variant),
                label=_variant_label(payload, variant),
            )
    ax.set_title("Weight Tail Ratio vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$P99/P50$ of $w(\\tau)$")
    _style_axes(ax)
    _annotate_if_empty(ax, "No P99/P50 data found")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=9)


def _plot_quantiles_panel(ax, payload: Dict[str, Any], line2d_cls: Any) -> None:
    for variant in STABILITY_VARIANTS:
        for q_name, tag in WEIGHT_TAGS.items():
            xs, ys = _points_to_xy(_variant_points(payload, variant, "weights", tag))
            if xs:
                ax.plot(
                    xs,
                    ys,
                    linewidth=2.0,
                    linestyle=WEIGHT_STYLES[q_name],
                    color=_variant_color(payload, variant),
                )
    ax.set_title("Weight Quantiles vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$w(\\tau)$")
    _style_axes(ax)
    _annotate_if_empty(ax, "No weight-quantile data found")
    variant_handles = [
        line2d_cls([0], [0], color=_variant_color(payload, v), linewidth=2.5, label=_variant_label(payload, v))
        for v in STABILITY_VARIANTS
    ]
    quantile_handles = [
        line2d_cls([0], [0], color="#444444", linewidth=2.0, linestyle=WEIGHT_STYLES[q], label=q.upper())
        for q in ("p50", "p90", "p99")
    ]
    leg1 = ax.legend(handles=variant_handles, loc="upper left", frameon=False, fontsize=8, title="Variant")
    ax.add_artist(leg1)
    ax.legend(handles=quantile_handles, loc="upper right", frameon=False, fontsize=8, title="Quantile")


def _plot_showcase_gap_panel(ax, payload: Dict[str, Any]) -> None:
    gap_map = _normalized_gap_points(payload, "eval", "coll_down", higher_is_better=False)
    for variant in STABILITY_VARIANTS:
        xs, ys = _points_to_xy(gap_map.get(variant, []))
        _plot_raw_and_smoothed(
            ax,
            xs,
            ys,
            color=_variant_color(payload, variant),
            label=_variant_label(payload, variant),
            highlight=(variant == "full"),
        )
    ax.set_title("Safety Gap During Training")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Normalized collision gap\n(0 = best, 1 = worst)")
    ax.set_ylim(bottom=0.0)
    _style_axes(ax)
    _annotate_if_empty(ax, "No gap data found")


def _plot_showcase_score_panel(ax, payload: Dict[str, Any]) -> None:
    for variant in STABILITY_VARIANTS:
        xs, ys = _points_to_xy(_variant_points(payload, variant, "eval", "score_up"))
        ys_pct = [100.0 * y for y in ys]
        _plot_raw_and_smoothed(
            ax,
            xs,
            ys_pct,
            color=_variant_color(payload, variant),
            label=_variant_label(payload, variant),
            highlight=(variant == "full"),
        )
    ax.set_title("Closed-loop Score During Training")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Score (%)")
    _style_axes(ax)
    _annotate_if_empty(ax, "No score data found")


def _render_figure(
    payload: Dict[str, Any],
    plot_mode: str,
    output_png: Path,
    output_svg: Path,
    dpi: int,
    figure_title: Optional[str] = None,
    footer_note: Optional[str] = None,
) -> None:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    lines_mod = importlib.import_module("matplotlib.lines")
    line2d_cls = lines_mod.Line2D

    if plot_mode == "paper":
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
        _plot_collision_panel(axes[0], payload)
        _plot_tail_ratio_panel(axes[1], payload)
        fig.suptitle(figure_title or "Training stability for GAIW", fontsize=14)
        fig.tight_layout()
        fig.subplots_adjust(top=0.84)
    elif plot_mode == "quantiles":
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
        _plot_collision_panel(axes[0], payload)
        _plot_quantiles_panel(axes[1], payload, line2d_cls)
        fig.suptitle(figure_title or "Training stability for GAIW", fontsize=14)
        fig.tight_layout()
        fig.subplots_adjust(top=0.84)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.4), sharex=False)
        _plot_showcase_gap_panel(axes[0], payload)
        _plot_showcase_score_panel(axes[1], payload)
        handles = [
            line2d_cls(
                [0],
                [0],
                color=_variant_color(payload, variant),
                linewidth=2.8 if variant == "full" else 2.0,
                label=_variant_label(payload, variant),
            )
            for variant in STABILITY_VARIANTS
        ]
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
        fig.suptitle(
            figure_title or "GAIW: Stable Training Dynamics and Closed-loop Performance",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        fig.text(
            0.5,
            0.02,
            footer_note or "Solid lines show EMA-smoothed trends; faint lines show raw epoch-level measurements.",
            ha="center",
            fontsize=10,
            color="#555555",
        )
        fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))

    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the GAIW stability figure from suite outputs or cached figure data.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--suite-root", help="Root directory created by launch_gaiw_ablation_suite.sh")
    source_group.add_argument("--input-json", help="Previously exported stability data JSON")
    source_group.add_argument("--input-csv", help="Previously exported stability data CSV")
    parser.add_argument(
        "--output-png",
        default=None,
        help="PNG output path. Default: <figure-dir>/gaiw_stability.png",
    )
    parser.add_argument(
        "--output-svg",
        default=None,
        help="SVG output path. Default: <figure-dir>/gaiw_stability.svg",
    )
    parser.add_argument(
        "--output-data-json",
        default=None,
        help="Optional path to save extracted figure data JSON. Default when using --suite-root: <figure-dir>/gaiw_stability_data.json",
    )
    parser.add_argument(
        "--output-data-csv",
        default=None,
        help="Optional path to save extracted figure data CSV. Default when using --suite-root: <figure-dir>/gaiw_stability_data.csv",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("paper", "quantiles", "showcase"),
        default="paper",
        help="`paper` renders collision + P99/P50; `quantiles` renders collision + raw P50/P90/P99; `showcase` renders a stacked conference-style figure.",
    )
    parser.add_argument("--figure-title", default=None, help="Optional figure title override.")
    parser.add_argument("--footer-note", default=None, help="Optional footer note for showcase mode.")
    parser.add_argument("--dpi", type=int, default=220, help="Raster dpi for PNG output.")
    args = parser.parse_args()

    if args.suite_root:
        figure_dir = Path(args.suite_root) / "figures"
        payload = _extract_payload_from_suite(Path(args.suite_root))
        output_data_json = (
            Path(args.output_data_json) if args.output_data_json else figure_dir / DEFAULT_DATA_JSON_NAME
        )
        output_data_csv = (
            Path(args.output_data_csv) if args.output_data_csv else figure_dir / DEFAULT_DATA_CSV_NAME
        )
        _write_data_json(output_data_json, payload)
        _write_data_csv(output_data_csv, payload)
        print(f"Saved {output_data_json}")
        print(f"Saved {output_data_csv}")
    elif args.input_json:
        input_json = Path(args.input_json)
        figure_dir = input_json.parent
        payload = _complete_payload(_read_json(input_json))
        output_data_json = Path(args.output_data_json) if args.output_data_json else None
        output_data_csv = Path(args.output_data_csv) if args.output_data_csv else None
        if output_data_json is not None:
            _write_data_json(output_data_json, payload)
            print(f"Saved {output_data_json}")
        if output_data_csv is not None:
            _write_data_csv(output_data_csv, payload)
            print(f"Saved {output_data_csv}")
    else:
        input_csv = Path(args.input_csv)
        figure_dir = input_csv.parent
        payload = _payload_from_csv(input_csv)
        output_data_json = Path(args.output_data_json) if args.output_data_json else None
        output_data_csv = Path(args.output_data_csv) if args.output_data_csv else None
        if output_data_json is not None:
            _write_data_json(output_data_json, payload)
            print(f"Saved {output_data_json}")
        if output_data_csv is not None:
            _write_data_csv(output_data_csv, payload)
            print(f"Saved {output_data_csv}")

    figure_dir.mkdir(parents=True, exist_ok=True)
    output_png = Path(args.output_png) if args.output_png else figure_dir / "gaiw_stability.png"
    output_svg = Path(args.output_svg) if args.output_svg else figure_dir / "gaiw_stability.svg"
    _render_figure(payload, args.plot_mode, output_png, output_svg, args.dpi, args.figure_title, args.footer_note)
    print(f"Saved {output_png}")
    print(f"Saved {output_svg}")


if __name__ == "__main__":
    main()
