from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


VARIANT_ORDER = [
    "no_weighting",
    "centrality_only",
    "risk_only",
    "full",
    "full_no_clipping",
    "full_no_batch_norm",
    "full_stabilization",
]

VARIANT_LABELS = {
    "no_weighting": "No weighting ($w(\\tau)=1$)",
    "centrality_only": "Centrality only ($w=\\exp(\\alpha_1\\,\\mathrm{CentFeat})$)",
    "risk_only": "Risk only ($w=\\exp(\\alpha_2\\,\\mathrm{CollProb})$)",
    "full": "Full ($w=\\exp(\\alpha_1\\,\\mathrm{CentFeat}+\\alpha_2\\,\\mathrm{CollProb})$)",
    "full_no_clipping": "Full + no $w$-clipping (remove $[w_{\\min}, w_{\\max}]$)",
    "full_no_batch_norm": "Full + no batch normalization (remove mean-normalization)",
    "full_stabilization": "Full + stabilization (clipping + mean-normalization)",
}

VARIANT_ALIAS = {
    "full_stabilization": "full",
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


def _fmt(v: Any, ndigits: int = 6) -> str:
    if v is None or v == "":
        return "N/A"
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return "N/A"


def _load_best_rows(suite_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        source_variant = VARIANT_ALIAS.get(variant, variant)
        eval_path = suite_root / "eval" / source_variant / "benchmark_per_epoch.jsonl"
        rows = _read_jsonl(eval_path)
        best = _best_row(rows)
        if best is None:
            records.append(
                {
                    "variant_id": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "source_variant_id": source_variant,
                    "status": "MISSING",
                }
            )
            continue

        record = dict(best)
        record["variant_id"] = variant
        record["variant_label"] = VARIANT_LABELS[variant]
        record["source_variant_id"] = source_variant
        records.append(record)
    return records


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "variant_id",
        "variant_label",
        "source_variant_id",
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
    with path.open("w", newline="") as f:
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
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate structured GAIW ablation benchmark outputs.")
    parser.add_argument("--suite-root", required=True, help="Root directory created by launch_gaiw_ablation_suite.sh")
    parser.add_argument("--output-dir", default=None, help="Directory to write aggregated outputs. Default: <suite-root>/summary")
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    output_dir = Path(args.output_dir) if args.output_dir else suite_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_best_rows(suite_root)

    (output_dir / "gaiw_ablation_best_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    _write_csv(output_dir / "gaiw_ablation_table.csv", rows)
    _write_markdown(output_dir / "gaiw_ablation_table.md", rows)

    manifest = {
        "suite_root": str(suite_root),
        "variants": rows,
    }
    (output_dir / "gaiw_ablation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"Wrote {output_dir / 'gaiw_ablation_best_rows.json'}")
    print(f"Wrote {output_dir / 'gaiw_ablation_table.csv'}")
    print(f"Wrote {output_dir / 'gaiw_ablation_table.md'}")
    print(f"Wrote {output_dir / 'gaiw_ablation_manifest.json'}")


if __name__ == "__main__":
    main()
