"""Reporting integrity checks; parquet I/O is mocked so no engine is required."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HAS_PANDAS = importlib.util.find_spec("pandas") is not None


@unittest.skipUnless(HAS_PANDAS, "pandas required for reporting checks")
class ReportIntegrityTest(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        from flow_planner.onpolicy import sim_report
        self.pd, self.report = pd, sim_report
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runner = self.root / "runner_report.parquet"
        self.runner.touch()
        self.metrics = self.root / "aggregator_metric" / "weighted_average_metrics.parquet"
        self.metrics.parent.mkdir()
        self.metrics.touch()
        self.runner_rows = [{"succeeded": True, "planner_name": "FlowPlanner", "scenario_name": "a", "log_name": "log"}]
        self.metric_rows = [
            {"scenario": "final_score", "log_name": None, "planner_name": "FlowPlanner", "score": .8, "num_scenarios": 1},
            {"scenario": "a", "log_name": "log", "planner_name": "FlowPlanner", "score": .8, "num_scenarios": None},
            {"scenario": "changing_lane", "log_name": None, "planner_name": "FlowPlanner", "score": .8, "num_scenarios": 1},
        ]
        def read(path):
            return pd.DataFrame(self.runner_rows if Path(path) == self.runner else self.metric_rows)
        self.mock_read = patch.object(pd, "read_parquet", side_effect=read)
        self.mock_read.start()
        self.addCleanup(self.mock_read.stop)

    def test_valid_report_and_direct_metrics_are_supported(self):
        for path in (self.runner, self.metrics):
            self.assertEqual(self.report.parse_final_score_from_runner_report(str(path)).score, .8)

    def test_partial_failure_is_rejected(self):
        self.runner_rows.append({"succeeded": False, "planner_name": "FlowPlanner", "scenario_name": "b", "log_name": "log"})
        with self.assertRaisesRegex(ValueError, "failed"):
            self.report.parse_final_score_from_runner_report(str(self.runner))

    def test_same_count_stale_scenario_identities_are_rejected(self):
        for updates in ({"scenario": "stale_b"}, {"log_name": "log_old"}, {"planner_name": "OtherPlanner"}):
            with self.subTest(updates=updates):
                self.metric_rows[1].update(scenario="a", log_name="log", planner_name="FlowPlanner")
                self.metric_rows[1].update(updates)
                with self.assertRaisesRegex(ValueError, "Scenario identit"):
                    self.report.parse_final_score_from_runner_report(str(self.runner))

    def test_missing_scenario_rows_are_rejected(self):
        self.metric_rows.pop(1)
        with self.assertRaises(ValueError):
            self.report.parse_final_score_from_runner_report(str(self.runner))

    def test_duplicate_scenario_rows_are_rejected(self):
        self.metric_rows.append(dict(self.metric_rows[1]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.report.parse_final_score_from_runner_report(str(self.runner))

    def test_duplicate_runner_identities_are_rejected(self):
        self.runner_rows.append(dict(self.runner_rows[0]))
        self.metric_rows[0]["num_scenarios"] = 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.report.parse_final_score_from_runner_report(str(self.runner))

    def test_nonfinite_score_and_count_mismatch_are_rejected(self):
        for updates in ({"score": float("nan")}, {"score": float("inf")}, {"num_scenarios": 2}):
            with self.subTest(updates=updates):
                self.metric_rows[0].update(score=.8, num_scenarios=1)
                self.metric_rows[0].update(updates)
                with self.assertRaises(ValueError):
                    self.report.parse_final_score_from_runner_report(str(self.runner))

    def test_missing_runner_and_ambiguous_final_rows_are_rejected(self):
        self.metric_rows.append(dict(self.metric_rows[0], planner_name="AnotherPlanner"))
        with self.assertRaises(ValueError):
            self.report.parse_final_score_from_runner_report(str(self.runner))
        self.metric_rows.pop()
        self.runner.unlink()
        with self.assertRaises(FileNotFoundError):
            self.report.parse_final_score_from_runner_report(str(self.metrics))

    def test_failed_teacher_does_not_reuse_unrelated_report(self):
        from flow_planner.onpolicy import teacher_select as ts
        from flow_planner.onpolicy.rollout_runner import NuPlanSimPaths, NuPlanSimRunResult
        for returncode in (0, 1):
            with self.subTest(returncode=returncode):
                output_dir = self.root if returncode else self.root / "missing_run"
                result = NuPlanSimRunResult(subprocess.CompletedProcess([], returncode, "", "failure"), "new_run", str(output_dir))
                with patch.object(ts, "run_nuplan_simulation", return_value=result):
                    with self.assertRaises(RuntimeError):
                        ts.select_teacher_checkpoint_by_sim_score(paths=NuPlanSimPaths(str(self.root), str(self.root)), run_dir=str(self.root), candidates=["bad.pth"], persist_json=False)

    def test_metric_names_do_not_claim_physical_measurements(self):
        from flow_planner.onpolicy import benchmark_table
        self.metric_rows[0].update(no_ego_at_fault_collisions=.5, drivable_area_compliance=1., driving_direction_compliance=.5, ego_is_comfortable=0., ego_jerk=None)
        output = io.StringIO()
        with patch.object(sys, "argv", ["benchmark_table", "--report", str(self.runner), "--latency-ms", "10", "--json-only"]), contextlib.redirect_stdout(output):
            benchmark_table.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result.get("collision_score_loss"), .5)  # One object collision is a .5 score, not a 50% collision rate.
        self.assertEqual(result.get("driving_compliance_loss"), .25)
        self.assertEqual(result.get("comfort_failure_rate"), 1.)
        self.assertIn("legacy_aliases", result)


class CollectorIntegrityTest(unittest.TestCase):
    def test_heldout_scores_must_be_finite(self):
        from flow_planner.onpolicy.grid_obj_ablation_collect import _load_variant_record
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "eval/val14/unified_objective/benchmark_per_epoch.jsonl"
            primary.parent.mkdir(parents=True)
            primary.write_text(json.dumps({"status": "OK", "score_up": .9, "checkpoint": "epoch1.pth"}) + "\n")
            test = root / "reports/test14-random/unified_objective/selected_checkpoint.json"
            test.parent.mkdir(parents=True)
            for score in (float("inf"), float("nan"), None, "invalid"):
                with self.subTest(score=score):
                    test.write_text(json.dumps({"status": "OK", "score_up": score, "checkpoint": "epoch1.pth", "collision_score_loss": .01}))
                    record = _load_variant_record(root, "unified_objective", "val14", "test14-random")
                    self.assertIsNone(record["test14_score_up"])
                    self.assertIsNone(record["collision_score_loss"])
                    self.assertEqual(record["status"], "INCOMPLETE")

    def test_cached_primary_selection_skips_nonfinite_scores(self):
        script = Path(__file__).resolve().parents[1] / "flow_planner/run_script/launch_grid_obj_ablation_suite.sh"
        function = "persist_best_primary_selection() {" + script.read_text().split("persist_best_primary_selection() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "val14/full/benchmark_per_epoch.jsonl"
            path.parent.mkdir(parents=True)
            rows = [{"status": "OK", "score_up": float("nan"), "checkpoint": "nan.pth"}, {"status": "OK", "score_up": float("inf"), "checkpoint": "inf.pth"}, {"status": "OK", "score_up": .8, "checkpoint": "good.pth"}]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            result = subprocess.run(["bash", "-c", function + "persist_best_primary_selection full\n"], env={**os.environ, "PYTHON_BIN": sys.executable, "SUITE_EVAL_DIR": str(root), "PRIMARY_SPLIT": "val14"}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "good.pth")

    def test_public_tables_use_explicit_metrics_instead_of_legacy_values(self):
        from flow_planner.onpolicy import gaiw_ablation_collect, rrs_ablation_collect, grid_obj_ablation_collect
        row = {"variant_label": "Example", "variant_id": "example", "score_up": .9, "coll_down": 999., "lane_dev_down": 999., "jerk_down": 999., "collision_score_loss": .25, "driving_compliance_loss": .5, "comfort_failure_rate": .75}
        with tempfile.TemporaryDirectory() as tmp:
            for module in (gaiw_ablation_collect, rrs_ablation_collect, grid_obj_ablation_collect):
                path = Path(tmp) / "table.md"
                extra = ("test14-random",) if module is grid_obj_ablation_collect else ()
                module._write_markdown(path, [row], *extra)
                contents = path.read_text()
                self.assertIn("Collision score loss", contents)
                self.assertIn("Comfort failure rate", contents)
                self.assertIn("0.250000", contents)
                self.assertNotIn("999.000000", contents)

    def test_derived_ratios_retain_mock_input_provenance(self):
        from flow_planner.onpolicy.gaiw_stability_plot import _complete_payload, _flatten_payload_rows
        payload = _complete_payload({"source_suite_root": "real", "variants": {"full": {"weights": {"gaiw_p50": [{"epoch": 1, "value": 1., "suite_root": "real"}], "gaiw_p99": [{"epoch": 1, "value": 2., "suite_root": "mock://illustration"}]}}}})
        derived = [r for r in _flatten_payload_rows(payload) if r["metric"] == "p99_over_p50"]
        self.assertIn("mock://illustration", derived[0]["suite_root"])

    def test_mismatched_selected_checkpoint_is_not_reported(self):
        from flow_planner.onpolicy.grid_obj_ablation_collect import _load_variant_record
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = root / "eval/val14/unified_objective/benchmark_per_epoch.jsonl"
            primary.parent.mkdir(parents=True)
            primary.write_text(json.dumps({"status": "OK", "score_up": .9, "checkpoint": "epoch2.pth"}) + "\n")
            test = root / "reports/test14-random/unified_objective/selected_checkpoint.json"
            test.parent.mkdir(parents=True)
            test.write_text(json.dumps({"status": "OK", "score_up": .95, "checkpoint": "epoch1.pth", "collision_score_loss": .01}))
            record = _load_variant_record(root, "unified_objective", "val14", "test14-random")
            self.assertIsNone(record["collision_score_loss"])
            self.assertIsNone(record["test14_score_up"])

    def test_collectors_exclude_nonfinite_scores(self):
        from flow_planner.onpolicy import gaiw_ablation_collect, rrs_ablation_collect, grid_obj_ablation_collect
        rows = [{"status": "OK", "score_up": float("nan")}, {"status": "OK", "score_up": float("inf")}, {"status": "OK", "score_up": .8}]
        for module in (gaiw_ablation_collect, rrs_ablation_collect, grid_obj_ablation_collect):
            self.assertEqual(module._best_row(rows)["score_up"], .8)

    def test_missing_test_metrics_never_fall_back_to_validation(self):
        from flow_planner.onpolicy.grid_obj_ablation_collect import _load_variant_record
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "eval/val14/unified_objective/benchmark_per_epoch.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": "OK", "score_up": .9, "checkpoint": "epoch1.pth", "coll_down": .01, "collision_score_loss": .01}) + "\n")
            record = _load_variant_record(root, "unified_objective", "val14", "test14-random")
            self.assertIsNone(record.get("collision_score_loss"))
            self.assertIsNone(record.get("coll_down"))
            self.assertNotEqual(record["status"], "OK")

    def test_mixed_csv_preserves_each_point_provenance(self):
        from flow_planner.onpolicy.gaiw_stability_plot import _payload_from_csv, _flatten_payload_rows
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.csv"
            rows = [{"suite_root": "real", "variant": "full", "section": "eval", "metric": "score_up", "epoch": "1", "value": ".8"}, {"suite_root": "mock://illustration", "variant": "full", "section": "eval", "metric": "score_up", "epoch": "2", "value": ".9"}]
            with path.open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            payload = _payload_from_csv(path)
            exported = _flatten_payload_rows(payload)
            self.assertEqual([r["suite_root"] for r in exported], ["real", "mock://illustration"])
            self.assertTrue(payload.get("synthetic"))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib required for figure checks")
    def test_synthetic_watermark_in_every_render_mode(self):
        from flow_planner.onpolicy.gaiw_stability_plot import _complete_payload, _render_figure
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"MPLCONFIGDIR": tempfile.gettempdir()}):
            root = Path(tmp)
            payload = _complete_payload({"source_suite_root": "mock://illustration", "variants": {"full": {"eval": {"score_up": [{"epoch": 1, "value": .8}], "collision_score_loss": [{"epoch": 1, "value": .1}]}}}})
            for mode in ("paper", "quantiles", "showcase"):
                _render_figure(payload, mode, root / f"{mode}.png", root / f"{mode}.svg", 40)
                svg = (root / f"{mode}.svg").read_text()
                self.assertIn("SYNTHETIC ILLUSTRATION", svg)
                self.assertNotIn("raw epoch-level measurements", svg)


@unittest.skipUnless(HAS_PANDAS and importlib.util.find_spec("pyarrow"), "pandas and pyarrow required for CLI checks")
class ParquetCommandTest(unittest.TestCase):
    def test_failed_simulation_has_nonzero_cli_exit(self):
        import pandas as pd
        root_repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "runner_report.parquet"
            metrics = root / "aggregator_metric/weighted_average_metrics.parquet"
            metrics.parent.mkdir()
            pd.DataFrame([
                {"scenario": "a", "log_name": "log", "planner_name": "FlowPlanner", "score": .9, "num_scenarios": None},
                {"scenario": "changing_lane", "log_name": None, "planner_name": "FlowPlanner", "score": .9, "num_scenarios": 1},
                {"scenario": "final_score", "log_name": None, "planner_name": "FlowPlanner", "score": .9, "num_scenarios": 1},
            ]).to_parquet(metrics)
            for succeeded, expected in ((True, 0), (False, 1)):
                pd.DataFrame([{"succeeded": succeeded, "planner_name": "FlowPlanner", "scenario_name": "a", "log_name": "log"}]).to_parquet(runner)
                result = subprocess.run([sys.executable, str(root_repo / "print_result.py"), str(runner)], text=True, capture_output=True)
                self.assertEqual(result.returncode, expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
