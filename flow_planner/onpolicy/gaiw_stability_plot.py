from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


STABILITY_VARIANTS = ["full_no_clipping", "full_no_batch_norm", "full"]
VARIANT_LABELS = {
    "full_no_clipping": "Full + no clipping",
    "full_no_batch_norm": "Full + no batch normalization",
    "full": "Full + stabilization",
}
VARIANT_COLORS = {
    "full_no_clipping": "#C44E52",
    "full_no_batch_norm": "#DD8452",
    "full": "#4C72B0",
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


def _eval_series(suite_root: Path, variant: str, key: str) -> Tuple[List[int], List[float]]:
    path = suite_root / "eval" / variant / "benchmark_per_epoch.jsonl"
    rows = [r for r in _read_jsonl(path) if r.get("status") == "OK" and r.get("epoch") is not None and r.get(key) is not None]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the GAIW stability figure from ablation suite outputs.")
    parser.add_argument("--suite-root", required=True, help="Root directory created by launch_gaiw_ablation_suite.sh")
    parser.add_argument("--output-png", default=None, help="PNG output path. Default: <suite-root>/figures/gaiw_stability.png")
    parser.add_argument("--output-svg", default=None, help="SVG output path. Default: <suite-root>/figures/gaiw_stability.svg")
    parser.add_argument("--dpi", type=int, default=220, help="Raster dpi for PNG output.")
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    fig_dir = suite_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_png = Path(args.output_png) if args.output_png else fig_dir / "gaiw_stability.png"
    output_svg = Path(args.output_svg) if args.output_svg else fig_dir / "gaiw_stability.svg"

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    lines_mod = importlib.import_module("matplotlib.lines")
    Line2D = lines_mod.Line2D

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel 1: score vs epoch.
    ax = axes[0]
    for variant in STABILITY_VARIANTS:
        xs, ys = _eval_series(suite_root, variant, "score_up")
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4, color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant])
    ax.set_title("Score vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)

    # Panel 2: collision rate vs epoch.
    ax = axes[1]
    for variant in STABILITY_VARIANTS:
        xs, ys = _eval_series(suite_root, variant, "coll_down")
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4, color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant])
    ax.set_title("Collision Rate vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Collision rate")
    ax.grid(True, alpha=0.3)

    # Panel 3: GAIW quantiles vs epoch.
    ax = axes[2]
    for variant in STABILITY_VARIANTS:
        tb_dir = suite_root / "runs" / variant / "tensorboard"
        for q_name, tag in WEIGHT_TAGS.items():
            xs, ys = _load_scalar_series(tb_dir, tag)
            if xs:
                ax.plot(
                    xs,
                    ys,
                    linewidth=2.0,
                    linestyle=WEIGHT_STYLES[q_name],
                    color=VARIANT_COLORS[variant],
                )
    ax.set_title("Weight Statistics vs. Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$w(\\tau)$")
    ax.grid(True, alpha=0.3)

    variant_handles = [
        Line2D([0], [0], color=VARIANT_COLORS[v], linewidth=2.5, label=VARIANT_LABELS[v])
        for v in STABILITY_VARIANTS
    ]
    quantile_handles = [
        Line2D([0], [0], color="#444444", linewidth=2.0, linestyle=WEIGHT_STYLES[q], label=q.upper())
        for q in ("p50", "p90", "p99")
    ]

    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    leg1 = axes[2].legend(handles=variant_handles, loc="upper left", frameon=False, fontsize=8, title="Variant")
    axes[2].add_artist(leg1)
    axes[2].legend(handles=quantile_handles, loc="upper right", frameon=False, fontsize=8, title="Quantile")

    fig.suptitle("Training stability for GAIW", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)

    fig.savefig(output_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")

    print(f"Saved {output_png}")
    print(f"Saved {output_svg}")


if __name__ == "__main__":
    main()
