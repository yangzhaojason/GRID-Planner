from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


VARIANT_ORDER = [
    "full",
    "no_rule",
    "no_comfort",
    "no_safety",
    "safety_only",
    "rule_only",
    "comfort_only",
]

VARIANT_LABELS = {
    "full": "Full RRS ($R_{\\mathrm{safety}}+R_{\\mathrm{rule}}+R_{\\mathrm{comfort}}$)",
    "no_rule": "w/o rule term ($R_{\\mathrm{safety}}+R_{\\mathrm{comfort}}$)",
    "no_comfort": "w/o comfort term ($R_{\\mathrm{safety}}+R_{\\mathrm{rule}}$)",
    "no_safety": "w/o safety term ($R_{\\mathrm{rule}}+R_{\\mathrm{comfort}}$)",
    "safety_only": "Safety only ($R_{\\mathrm{safety}}$)",
    "rule_only": "Rule only ($R_{\\mathrm{rule}}$)",
    "comfort_only": "Comfort only ($R_{\\mathrm{comfort}}$)",
}

LATEX_GROUP_BREAK_AFTER = {"no_safety"}


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


def _load_best_rows(suite_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        eval_path = suite_root / "eval" / variant / "benchmark_per_epoch.jsonl"
        rows = _read_jsonl(eval_path)
        best = _best_row(rows)
        if best is None:
            records.append(
                {
                    "variant_id": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "status": "MISSING",
                }
            )
            continue

        record = dict(best)
        record["variant_id"] = variant
        record["variant_label"] = VARIANT_LABELS[variant]
        records.append(record)
    return records


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "variant_id",
        "variant_label",
        "checkpoint",
        "epoch",
        "status",
        "score_up",
        "collision_score_loss",
        "red_light_down",
        "driving_compliance_loss",
        "comfort_failure_rate",
        "latency_down_ms",
        "result_dir",
        "source",
        "report_path",
        "json_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "Metric definitions: collision score loss = 1 - at-fault collision score; driving compliance loss = 1 - mean(drivable-area, driving-direction scores); comfort failure rate = 1 - comfort pass rate. These are not physical lane deviation, jerk, or collision event rate.",
        "",
        "| Variant | Score | Collision score loss | Red-light | Driving compliance loss | Comfort failure rate | Latency (ms) | Best ckpt |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + row["variant_label"]
            + " | "
            + _fmt(row.get("score_up"))
            + " | "
            + _fmt(row.get("collision_score_loss"))
            + " | "
            + _fmt(row.get("red_light_down"))
            + " | "
            + _fmt(row.get("driving_compliance_loss"))
            + " | "
            + _fmt(row.get("comfort_failure_rate"))
            + " | "
            + _fmt(row.get("latency_down_ms"), ndigits=3)
            + " | "
            + str(row.get("checkpoint", "N/A"))
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = ["% Collision/driving: dimensionless nuPlan metric-score losses. Comfort: any-bound failure fraction, not physical jerk."]
    for row in rows:
        lines.append(
            f"{row['variant_label']} & "
            f"{_fmt(row.get('score_up'), missing='xxx')} & "
            f"{_fmt(row.get('collision_score_loss'), missing='xxx')} & "
            f"{_fmt(row.get('red_light_down'), missing='xxx')} & "
            f"{_fmt(row.get('driving_compliance_loss'), missing='xxx')} & "
            f"{_fmt(row.get('comfort_failure_rate'), missing='xxx')} & "
            f"{_fmt(row.get('latency_down_ms'), ndigits=3, missing='xxx')} \\\\"
        )
        if row["variant_id"] in LATEX_GROUP_BREAK_AFTER:
            lines.append("\\midrule")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    body_lines: List[str] = []
    for row in rows:
        body_lines.append(
            f"{row['variant_label']} & "
            f"{_fmt(row.get('score_up'), missing='xxx')} & "
            f"{_fmt(row.get('collision_score_loss'), missing='xxx')} & "
            f"{_fmt(row.get('red_light_down'), missing='xxx')} & "
            f"{_fmt(row.get('driving_compliance_loss'), missing='xxx')} & "
            f"{_fmt(row.get('comfort_failure_rate'), missing='xxx')} & "
            f"{_fmt(row.get('latency_down_ms'), ndigits=3, missing='xxx')} \\\\"
        )
        if row["variant_id"] in LATEX_GROUP_BREAK_AFTER:
            body_lines.append("\\midrule")

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{\\textbf{RRS ablation.} Removing individual components of $\\mathcal R_{\\mathrm{RRS}}(\\tau)$ reveals their targeted impact on safety, rule compliance, and comfort. Score ($\\uparrow$; nuPlan Val14 Benchmark), collision and driving metric-score losses ($\\downarrow$), comfort failure rate ( $\\downarrow$), and planning latency ($\\downarrow$) are reported. Collision and driving values are dimensionless score losses; comfort is the fraction failing any acceleration, jerk, or yaw bound. These are not collision event rates or physical lane deviation/jerk.}",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Variant} & \\textbf{Score} $\\uparrow$ & \\textbf{Collision score loss} $\\downarrow$ & \\textbf{Red-light} $\\downarrow$ & \\textbf{Driving compliance loss} $\\downarrow$ & \\textbf{Comfort failure rate} $\\downarrow$ & \\textbf{Latency} $\\downarrow$ \\\\",
        "\\midrule",
        "\\midrule",
        *body_lines,
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:ablation_rrs}",
        "\\end{table*}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate structured RRS ablation benchmark outputs.")
    parser.add_argument("--suite-root", required=True, help="Root directory created by launch_rrs_ablation_suite.sh")
    parser.add_argument("--output-dir", default=None, help="Directory to write aggregated outputs. Default: <suite-root>/summary")
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    output_dir = Path(args.output_dir) if args.output_dir else suite_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_best_rows(suite_root)

    (output_dir / "rrs_ablation_best_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "rrs_ablation_table.csv", rows)
    _write_markdown(output_dir / "rrs_ablation_table.md", rows)
    _write_latex_rows(output_dir / "rrs_ablation_rows.tex", rows)
    _write_latex_table(output_dir / "rrs_ablation_table.tex", rows)

    manifest = {
        "suite_root": str(suite_root),
        "variants": rows,
    }
    (output_dir / "rrs_ablation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {output_dir / 'rrs_ablation_best_rows.json'}")
    print(f"Wrote {output_dir / 'rrs_ablation_table.csv'}")
    print(f"Wrote {output_dir / 'rrs_ablation_table.md'}")
    print(f"Wrote {output_dir / 'rrs_ablation_rows.tex'}")
    print(f"Wrote {output_dir / 'rrs_ablation_table.tex'}")
    print(f"Wrote {output_dir / 'rrs_ablation_manifest.json'}")


if __name__ == "__main__":
    main()
