from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


VARIANT_ORDER = [
    "offline_diffusion_only",
    "unified_objective",
    "no_critic_baseline",
    "no_advantage_reweighting",
    "no_opsd_term",
    "no_advantage_gate",
    "random_teacher",
]

VARIANT_LABELS = {
    "offline_diffusion_only": r"Offline diffusion only ($\mathcal L_{\mathrm{diff}}$)",
    "unified_objective": r"Unified objective (Eq.~\ref{eq:total_loss})",
    "no_critic_baseline": r"w/o critic baseline ($V_\phi(G)=0$)",
    "no_advantage_reweighting": r"w/o advantage reweighting",
    "no_opsd_term": r"w/o OPSD term",
    "no_advantage_gate": r"w/o advantage gate ($\omega(G)=1$)",
    "random_teacher": r"w/ random teacher",
}

SPLIT_ORDER = ["val14", "test14-hard", "test14-random"]
SPLIT_LABELS = {
    "val14": "Val14 (R)",
    "test14-hard": "Test14-hard (R)",
    "test14-random": "Test14 (R)",
}

LATEX_GROUP_BREAK_AFTER = {"unified_objective", "no_opsd_term"}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
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


def _best_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = []
    for row in rows:
        try:
            if row.get("status") == "OK" and math.isfinite(float(row.get("score_up"))):
                valid.append(row)
        except (TypeError, ValueError):
            continue
    if not valid:
        return None
    return max(valid, key=lambda r: float(r["score_up"]))


def _fmt(v: Any, ndigits: int = 6, missing: str = "N/A") -> str:
    if v is None or v == "":
        return missing
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return missing


def _load_variant_record(
    suite_root: Path,
    variant: str,
    primary_split: str,
    metric_split: str,
) -> Dict[str, Any]:
    eval_jsonl = suite_root / "eval" / primary_split / variant / "benchmark_per_epoch.jsonl"
    primary_rows = _read_jsonl(eval_jsonl)
    best_primary = _best_row(primary_rows)
    meta = _read_json(suite_root / "meta" / f"{variant}.json")

    record: Dict[str, Any] = {
        "variant_id": variant,
        "variant_label": VARIANT_LABELS[variant],
        "primary_split": primary_split,
        "metric_split": metric_split,
        "meta": meta,
    }

    if best_primary is None:
        record["status"] = "MISSING"
        return record

    selected_checkpoint = best_primary.get("checkpoint")
    split_payloads: Dict[str, Dict[str, Any]] = {primary_split: dict(best_primary)}
    for split in SPLIT_ORDER:
        if split == primary_split:
            continue
        report_path = suite_root / "reports" / split / variant / "selected_checkpoint.json"
        split_payloads[split] = _read_json(report_path)
        payload = split_payloads[split]
        if payload and (_best_row([payload]) is None or payload.get("checkpoint") != selected_checkpoint):
            split_payloads[split] = {"status": "INVALID", "error": "failed evaluation, non-finite score, or selected checkpoint mismatch", "report_path": str(report_path)}

    metric_payload = split_payloads.get(metric_split, {})
    if metric_payload.get("status") != "OK":
        metric_payload = {}

    record.update(
        {
            "status": "OK" if all(p.get("status") == "OK" for p in split_payloads.values()) else "INCOMPLETE",
            "selected_checkpoint": selected_checkpoint,
            "selected_epoch": best_primary.get("epoch"),
            "val14_score_up": split_payloads.get("val14", {}).get("score_up"),
            "test14_hard_score_up": split_payloads.get("test14-hard", {}).get("score_up"),
            "test14_score_up": split_payloads.get("test14-random", {}).get("score_up"),
            "collision_score_loss": metric_payload.get("collision_score_loss"),
            "driving_compliance_loss": metric_payload.get("driving_compliance_loss"),
            "comfort_failure_rate": metric_payload.get("comfort_failure_rate"),
            "latency_down_ms": metric_payload.get("latency_down_ms"),
            "source": metric_payload.get("source", {}),
            "report_path": metric_payload.get("report_path"),
            "result_dir": metric_payload.get("result_dir"),
            "json_path": metric_payload.get("json_path"),
            "splits": split_payloads,
        }
    )
    return record


def _load_records(suite_root: Path, primary_split: str, metric_split: str) -> List[Dict[str, Any]]:
    return [
        _load_variant_record(suite_root, variant, primary_split=primary_split, metric_split=metric_split)
        for variant in VARIANT_ORDER
    ]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "variant_id",
        "variant_label",
        "status",
        "selected_checkpoint",
        "selected_epoch",
        "val14_score_up",
        "test14_hard_score_up",
        "test14_score_up",
        "collision_score_loss",
        "driving_compliance_loss",
        "comfort_failure_rate",
        "latency_down_ms",
        "source",
        "report_path",
        "result_dir",
        "json_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_markdown(path: Path, rows: List[Dict[str, Any]], metric_split: str) -> None:
    lines = [
        "Collision and driving values are dimensionless metric-score losses; comfort is the fraction failing any comfort bound. These are not physical lane deviation, jerk, or collision event rate.",
        "",
        f"| Variant | {SPLIT_LABELS['val14']} | {SPLIT_LABELS['test14-hard']} | {SPLIT_LABELS['test14-random']} | Collision score loss ({SPLIT_LABELS[metric_split]}) | Driving compliance loss ({SPLIT_LABELS[metric_split]}) | Comfort failure rate ({SPLIT_LABELS[metric_split]}) | Latency ({SPLIT_LABELS[metric_split]}, ms) | Best ckpt |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + row["variant_label"]
            + " | "
            + _fmt(row.get("val14_score_up"))
            + " | "
            + _fmt(row.get("test14_hard_score_up"))
            + " | "
            + _fmt(row.get("test14_score_up"))
            + " | "
            + _fmt(row.get("collision_score_loss"))
            + " | "
            + _fmt(row.get("driving_compliance_loss"))
            + " | "
            + _fmt(row.get("comfort_failure_rate"))
            + " | "
            + _fmt(row.get("latency_down_ms"), ndigits=3)
            + " | "
            + str(row.get("selected_checkpoint", "N/A"))
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = ["% Collision/driving: dimensionless nuPlan metric-score losses. Comfort: any-bound failure fraction, not physical jerk."]
    for row in rows:
        lines.append(
            f"{row['variant_label']} & "
            f"{_fmt(row.get('val14_score_up'), missing='xxx')} & "
            f"{_fmt(row.get('test14_hard_score_up'), missing='xxx')} & "
            f"{_fmt(row.get('test14_score_up'), missing='xxx')} & "
            f"{_fmt(row.get('collision_score_loss'), missing='xxx')} & "
            f"{_fmt(row.get('driving_compliance_loss'), missing='xxx')} & "
            f"{_fmt(row.get('comfort_failure_rate'), missing='xxx')} & "
            f"{_fmt(row.get('latency_down_ms'), ndigits=3, missing='xxx')} \\\\"
        )
        if row["variant_id"] in LATEX_GROUP_BREAK_AFTER:
            lines.append("\\midrule")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_table(path: Path, rows: List[Dict[str, Any]], metric_split: str) -> None:
    body_lines: List[str] = []
    for row in rows:
        body_lines.append(
            f"{row['variant_label']} & "
            f"{_fmt(row.get('val14_score_up'), missing='xxx')} & "
            f"{_fmt(row.get('test14_hard_score_up'), missing='xxx')} & "
            f"{_fmt(row.get('test14_score_up'), missing='xxx')} & "
            f"{_fmt(row.get('collision_score_loss'), missing='xxx')} & "
            f"{_fmt(row.get('driving_compliance_loss'), missing='xxx')} & "
            f"{_fmt(row.get('comfort_failure_rate'), missing='xxx')} & "
            f"{_fmt(row.get('latency_down_ms'), ndigits=3, missing='xxx')} \\\\"
        )
        if row["variant_id"] in LATEX_GROUP_BREAK_AFTER:
            body_lines.append("\\midrule")

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{\\textbf{Unified on-policy objective ablation.} Each row removes one ingredient from Eq.~\\ref{eq:total_loss} while keeping the graph-aware return $R(\\tau,G)$, rollout budget, and evaluation protocol fixed. We report reactive closed-loop scores on nuPlan Val14, Test14-hard, and Test14 ($\\uparrow$), together with collision metric-score loss ($\\downarrow$), driving-compliance score loss ($\\downarrow$), comfort failure rate ($\\downarrow$), and planning latency ($\\downarrow$). The last two rows ablate OPSD-specific choices (advantage gating and teacher selection). Ancillary metrics are reported on "
        + SPLIT_LABELS[metric_split]
        + ". Collision/driving values are dimensionless score losses; comfort is the fraction failing any comfort bound, not physical jerk.}",
        "\\label{tab:ablation_grid_obj}",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\resizebox{\\linewidth}{!}{",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\textbf{Variant} & \\textbf{Val14 (R)} $\\uparrow$ & \\textbf{Test14-hard (R)} $\\uparrow$ & \\textbf{Test14 (R)} $\\uparrow$ & \\textbf{Collision score loss} $\\downarrow$ & \\textbf{Driving compliance loss} $\\downarrow$ & \\textbf{Comfort failure rate} $\\downarrow$ & \\textbf{Latency} $\\downarrow$ \\\\",
        "\\midrule",
        *body_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate unified objective ablation outputs.")
    parser.add_argument("--suite-root", required=True, help="Root directory created by launch_grid_obj_ablation_suite.sh")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write aggregated outputs. Default: <suite-root>/summary",
    )
    parser.add_argument(
        "--primary-split",
        default="val14",
        choices=SPLIT_ORDER,
        help="Primary split used for checkpoint selection.",
    )
    parser.add_argument(
        "--metric-split",
        default="test14-random",
        choices=SPLIT_ORDER,
        help="Split used for Collision score loss/Driving compliance loss/Comfort failure rate/Latency columns.",
    )
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    output_dir = Path(args.output_dir) if args.output_dir else suite_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_records(suite_root, primary_split=args.primary_split, metric_split=args.metric_split)

    (output_dir / "grid_obj_ablation_best_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "grid_obj_ablation_table.csv", rows)
    _write_markdown(output_dir / "grid_obj_ablation_table.md", rows, metric_split=args.metric_split)
    _write_latex_rows(output_dir / "grid_obj_ablation_rows.tex", rows)
    _write_latex_table(output_dir / "grid_obj_ablation_table.tex", rows, metric_split=args.metric_split)

    manifest = {
        "suite_root": str(suite_root),
        "primary_split": args.primary_split,
        "metric_split": args.metric_split,
        "variants": rows,
    }
    (output_dir / "grid_obj_ablation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'grid_obj_ablation_best_rows.json'}")
    print(f"Wrote {output_dir / 'grid_obj_ablation_table.csv'}")
    print(f"Wrote {output_dir / 'grid_obj_ablation_table.md'}")
    print(f"Wrote {output_dir / 'grid_obj_ablation_rows.tex'}")
    print(f"Wrote {output_dir / 'grid_obj_ablation_table.tex'}")
    print(f"Wrote {output_dir / 'grid_obj_ablation_manifest.json'}")


if __name__ == "__main__":
    main()
