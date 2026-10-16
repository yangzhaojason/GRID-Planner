"""Filesystem and child-process checks for simulation result discovery."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flow_planner.onpolicy.rollout_runner import NuPlanSimConfig, NuPlanSimPaths, run_nuplan_simulation


class RolloutRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.devkit = self.root / "devkit"
        script = self.devkit / "nuplan/planning/script/run_simulation.py"
        script.parent.mkdir(parents=True)
        script.write_text(
            "import os, sys\n"
            "from pathlib import Path\n"
            "args = dict(arg.split('=', 1) for arg in sys.argv[1:])\n"
            "output = Path(os.environ['NUPLAN_EXP_ROOT']) / 'exp/simulation' / args['+simulation'] / args['experiment_uid']\n"
            "output.mkdir(parents=True)\n"
            "(output / 'runner_report.parquet').touch()\n"
        )
        self.run = self.root / "run"
        (self.run / ".hydra").mkdir(parents=True)
        (self.run / ".hydra/config.yaml").write_text("{}")
        self.ckpt = self.run / "model.pth"
        self.ckpt.touch()
        self.paths = NuPlanSimPaths(str(self.root), str(self.devkit))
        self.sim_cfg = NuPlanSimConfig(ray_tmpdir=str(self.root / "ray"))

    def invoke(self, env=None):
        return run_nuplan_simulation(paths=self.paths, run_dir=str(self.run), ckpt_path=str(self.ckpt), sim_cfg=self.sim_cfg, env=env)

    def test_missing_or_empty_experiment_root_fails_before_launch(self):
        for environment in ({}, {"NUPLAN_EXP_ROOT": ""}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), \
                    patch("flow_planner.onpolicy.rollout_runner.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "NUPLAN_EXP_ROOT"):
                    self.invoke()
                run.assert_not_called()

    def test_explicit_root_override_matches_actual_child_output(self):
        root = self.root / "explicit results"
        with patch.dict(os.environ, {"NUPLAN_EXP_ROOT": str(self.root / "inherited-results")}):
            result = self.invoke(env={"NUPLAN_EXP_ROOT": str(root)})
        self.assertEqual(result.proc.returncode, 0, result.proc.stderr)
        self.assertEqual(Path(result.output_dir), root / "exp/simulation" / self.sim_cfg.challenge / result.experiment_uid)
        self.assertTrue((Path(result.output_dir) / "runner_report.parquet").is_file())

    def test_relative_root_is_resolved_from_child_working_directory(self):
        result = self.invoke(env={"NUPLAN_EXP_ROOT": "relative-results"})
        self.assertEqual(result.proc.returncode, 0, result.proc.stderr)
        expected = self.root.resolve() / "relative-results/exp/simulation" / self.sim_cfg.challenge / result.experiment_uid
        self.assertEqual(Path(result.output_dir), expected)
        self.assertTrue((expected / "runner_report.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
