"""Check shell launch contracts without importing torch or running a GPU job."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from flow_planner.onpolicy.rollout_runner import NuPlanSimConfig, NuPlanSimPaths, run_nuplan_simulation


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "flow_planner" / "run_script"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="grid launcher ")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.capture = self.work / "calls.jsonl"
        fake_python = self.work / "python"
        fake_python.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "if sys.argv[1:2] == ['-']:\n"
            "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
            "with open(os.environ['LAUNCH_CAPTURE'], 'a') as out:\n"
            "    out.write(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), "
            "'env': {k: os.environ.get(k) for k in "
            "['PROJECT_ROOT', 'SAVE_DIR', 'TENSORBOARD_LOG_PATH', 'TRAINING_DATA', "
            "'TRAINING_JSON', 'CUDA_VISIBLE_DEVICES', 'WANDB_API_KEY', "
            "'NUPLAN_EXP_ROOT', 'NUPLAN_DATA_ROOT', 'NUPLAN_MAPS_ROOT']}}) + '\\n')\n"
        )
        fake_python.chmod(0o755)
        self.run_dir = self.work / "selected run"
        (self.run_dir / ".hydra").mkdir(parents=True)
        (self.run_dir / ".hydra" / "config.yaml").write_text("{}\n")
        (self.run_dir / "latest.pth").touch()
        for directory in ("processed", "devkit", "nuplan/maps"):
            (self.work / directory).mkdir(parents=True, exist_ok=True)
        (self.work / "processed" / "flow_planner_training.json").write_text("[]")
        self.env = os.environ.copy()
        for key in ("PROJECT_ROOT", "RUN_DIR", "RUN_TIMESTAMP", "CKPT_PATH", "CKPT_FILE", "CONFIG_FILE", "EXP_SAVE_ROOT", "NUPLAN_EXP_ROOT"):
            self.env.pop(key, None)
        self.env.update(
            PATH=f"{self.work}{os.pathsep}{os.environ.get('PATH', '')}",
            PYTHON_BIN=str(fake_python),
            LAUNCH_CAPTURE=str(self.capture),
            SAVE_DIR=str(self.work / "results"),
            TRAINING_DATA=str(self.work / "processed"),
            TRAINING_JSON=str(self.work / "processed" / "flow_planner_training.json"),
            NUPLAN_DEVKIT_ROOT=str(self.work / "devkit"),
            NUPLAN_DATA_ROOT=str(self.work / "nuplan"),
            NUPLAN_MAPS_ROOT=str(self.work / "nuplan" / "maps"),
            RAY_TMPDIR=str(self.work / "ray"),
            PRETRAINED_CKPT=str(self.run_dir / "latest.pth"),
            NPROC_PER_NODE="2",
            MASTER_PORT="29601",
            CUDA_VISIBLE_DEVICES="4,5",
            WANDB_API_KEY="test-key-must-be-preserved",
        )

    def run_launcher(self, name, *args, **env):
        return subprocess.run(
            ["bash", str(SCRIPTS / name), *args], cwd=self.work,
            env={**self.env, **env}, text=True, capture_output=True, timeout=30,
        )

    def calls(self):
        return [json.loads(line) for line in self.capture.read_text().splitlines()]

    def test_base_training_uses_package_module_from_any_directory(self):
        proc = self.run_launcher("launch_train.sh", "train.epoch=1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        call = self.calls()[-1]
        self.assertIn("flow_planner.trainer", call["argv"])
        self.assertNotIn("../trainer.py", call["argv"])
        self.assertEqual(call["argv"][call["argv"].index("--nproc-per-node") + 1], "2")
        self.assertIn("train.epoch=1", call["argv"])
        self.assertEqual(call["cwd"], str(ROOT))
        self.assertEqual(call["env"]["PROJECT_ROOT"], str(ROOT))
        self.assertEqual(call["env"]["SAVE_DIR"], self.env["SAVE_DIR"])
        self.assertEqual(call["env"]["CUDA_VISIBLE_DEVICES"], "4,5")
        self.assertEqual(call["env"]["WANDB_API_KEY"], self.env["WANDB_API_KEY"])

    def test_finetuning_preserves_checkpoint_argument_with_spaces(self):
        for script in ("launch_train_graph_weighting_ft.sh", "launch_train_graph_weighting_rrs_ft.sh"):
            with self.subTest(script=script):
                proc = self.run_launcher(script, "train.epoch=1")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                call = self.calls()[-1]
                self.assertIn("pretrained_checkpoint=" + self.env["PRETRAINED_CKPT"], call["argv"])
                self.assertIn("train.epoch=1", call["argv"])
                self.assertIn("optimizer.lr=5e-5", call["argv"])
                self.assertEqual(call["env"]["WANDB_API_KEY"], self.env["WANDB_API_KEY"])

    def test_finetuning_requires_an_existing_checkpoint(self):
        proc = self.run_launcher("launch_train_graph_weighting_ft.sh", PRETRAINED_CKPT="")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("PRETRAINED_CKPT", proc.stderr)
        self.assertFalse(self.capture.exists())

    def test_single_checkpoint_launchers_use_explicit_run_and_preserve_paths(self):
        for script in ("launch_sim_nuplan_base.sh", "launch_sim_nuplan_graph_weighting_ft.sh", "launch_sim_nuplan_graph_weighting_rrs_ft.sh"):
            with self.subTest(script=script):
                proc = self.run_launcher(script, RUN_DIR=str(self.run_dir))
                self.assertEqual(proc.returncode, 0, proc.stderr)
                call = self.calls()[-1]
                self.assertIn(f"planner.flow_planner.config_path={self.run_dir}/.hydra/config.yaml", call["argv"])
                self.assertIn(f"planner.flow_planner.ckpt_path={self.run_dir}/latest.pth", call["argv"])
                self.assertEqual(call["env"]["NUPLAN_EXP_ROOT"], self.env["SAVE_DIR"])
                self.assertEqual(call["env"]["CUDA_VISIBLE_DEVICES"], "4,5")

    def test_simulation_does_not_silently_select_the_newest_run(self):
        proc = self.run_launcher("launch_sim_nuplan_graph_weighting_ft.sh")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("RUN_DIR", proc.stderr)
        self.assertFalse(self.capture.exists())

    def test_simulation_accepts_explicit_config_and_checkpoint_without_run(self):
        proc = self.run_launcher(
            "launch_sim_nuplan_graph_weighting_rrs_ft.sh",
            CKPT_PATH=str(self.run_dir / "latest.pth"),
            CONFIG_FILE=str(self.run_dir / ".hydra" / "config.yaml"),
            SPLIT="test14-hard", CUDA_VISIBLE_DEVICES="",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        call = self.calls()[-1]
        self.assertIn("scenario_builder=nuplan_challenge", call["argv"])
        self.assertEqual(call["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_ablation_wrappers_forward_overrides(self):
        proc = self.run_launcher("launch_train_graph_weighting_rrs_ablation_runA.sh", "train.epoch=1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        args = self.calls()[-1]["argv"]
        self.assertIn("core.rrs.lam_col=0", args)
        self.assertEqual(args[-1], "train.epoch=1")

    def test_gaiw_and_rrs_suites_train_with_portable_defaults(self):
        for script, variants in (("launch_gaiw_ablation_suite.sh", "no_weighting"), ("launch_rrs_ablation_suite.sh", "full")):
            with self.subTest(script=script):
                proc = self.run_launcher(script, VARIANTS=variants, RUN_EVAL="false", RUN_SUMMARY="false", RUN_PLOT="false")
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                call = self.calls()[-1]
                self.assertIn("flow_planner.trainer", call["argv"])
                self.assertIn("pretrained_checkpoint=" + self.env["PRETRAINED_CKPT"], call["argv"])

    def test_all_shell_scripts_parse(self):
        for script in SCRIPTS.glob("*.sh"):
            with self.subTest(script=script.name):
                proc = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_onpolicy_distillation_uses_selected_run_and_output_root(self):
        proc = self.run_launcher(
            "launch_onpolicy_opsd_cycle.sh", RUN_DIR=str(self.run_dir),
            START_FROM_STEP="distill", TEACHER_CKPT=str(self.run_dir / "latest.pth"),
            NUM_CYCLES="1",
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        call = self.calls()[-1]
        self.assertIn("flow_planner.onpolicy.distill_trainer", call["argv"])
        self.assertIn("core.opsd.teacher_ckpt_path=" + str(self.run_dir / "latest.pth"), call["argv"])
        self.assertEqual(call["env"]["NUPLAN_EXP_ROOT"], self.env["SAVE_DIR"])

    def test_objective_suite_can_train_with_evaluation_disabled(self):
        proc = self.run_launcher(
            "launch_grid_obj_ablation_suite.sh", BASE_RUN_DIR=str(self.run_dir),
            VARIANTS="offline_diffusion_only", RUN_PRIMARY_EVAL="false",
            RUN_REPORT_EVAL="false", RUN_SUMMARY="false", RUN_PLOT="false",
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("flow_planner.trainer", self.calls()[-1]["argv"])

    def test_python_simulation_uses_current_interpreter_and_effective_ray_temp(self):
        script = self.work / "devkit" / "nuplan" / "planning" / "script" / "run_simulation.py"
        script.parent.mkdir(parents=True)
        script.write_text("import os, sys\nprint(sys.executable)\nprint(os.environ['RAY_TMPDIR'])\n")
        selected_temp = self.work / "effective ray temp"
        cfg_temp = self.work / "unused ray temp"
        result = run_nuplan_simulation(
            paths=NuPlanSimPaths(project_root=str(ROOT), nuplan_devkit_root=str(self.work / "devkit")),
            run_dir=str(self.run_dir), ckpt_path=str(self.run_dir / "latest.pth"),
            sim_cfg=NuPlanSimConfig(ray_tmpdir=str(cfg_temp)),
            env={"RAY_TMPDIR": str(selected_temp), "NUPLAN_EXP_ROOT": str(self.work / "results"), "PATH": ""},
        )
        self.assertEqual(result.proc.returncode, 0, result.proc.stderr)
        self.assertEqual(result.proc.stdout.splitlines(), [sys.executable, str(selected_temp)])
        self.assertTrue(selected_temp.is_dir())
        self.assertFalse(cfg_temp.exists())

    def test_python_simulation_config_respects_ray_environment(self):
        with mock.patch.dict(os.environ, {"RAY_TMPDIR": str(self.work / "ray")}):
            self.assertEqual(NuPlanSimConfig().ray_tmpdir, str(self.work / "ray"))


if __name__ == "__main__":
    unittest.main()
