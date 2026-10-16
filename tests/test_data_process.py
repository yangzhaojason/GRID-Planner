"""Dependency-free checks for the preprocessing command and dataset manifest."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DataProcessTest(unittest.TestCase):
    def test_help_from_another_directory_without_training_dependencies(self):
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, str(ROOT / "data_process.py"), "--help"],
                cwd=cwd, text=True, capture_output=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output_json", result.stdout)
        self.assertIn("--log_names", result.stdout)

    def test_shell_wrapper_preserves_paths_and_interpreter(self):
        with tempfile.TemporaryDirectory(prefix="grid data ") as directory:
            root = Path(directory)
            capture = root / "arguments.json"
            (root / "data_process.py").write_text(
                "import json, os, sys\n"
                "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n"
            )
            env = {
                **os.environ,
                "PROJECT_ROOT": str(root), "PYTHON_BIN": sys.executable,
                "CAPTURE": str(capture),
                "NUPLAN_DATA_PATH": str(root / "raw data"),
                "NUPLAN_MAP_PATH": str(root / "map data"),
                "TRAINING_DATA": str(root / "processed data"),
                "TRAINING_JSON": str(root / "index files" / "train.json"),
                "TOTAL_SCENARIOS": "12",
            }
            env.pop("TRAIN_SET_PATH", None)
            result = subprocess.run(
                ["bash", str(ROOT / "data_process.sh"), "--shuffle_scenarios", "false"],
                cwd=root, env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = json.loads(capture.read_text())
            self.assertEqual(arguments[arguments.index("--data_path") + 1], env["NUPLAN_DATA_PATH"])
            self.assertEqual(arguments[arguments.index("--output_json") + 1], env["TRAINING_JSON"])
            self.assertEqual(arguments[arguments.index("--total_scenarios") + 1], "12")
            self.assertEqual(arguments[-2:], ["--shuffle_scenarios", "false"])

    def load_module(self):
        spec = importlib.util.spec_from_file_location("data_process", ROOT / "data_process.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_index_contains_only_sorted_npz_files(self):
        module = self.load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("b.npz", "a.npz", "notes.txt"):
                (root / name).touch()
            (root / "not-a-file.npz").mkdir()
            output = root / "indices" / "train.json"
            self.assertEqual(module.write_data_list(root, output), 2)
            self.assertEqual(json.loads(output.read_text()), ["a.npz", "b.npz"])

    def test_false_shuffle_and_repository_split_default(self):
        module = self.load_module()
        args = module.build_parser().parse_args(
            ["--data_path", "/dataset", "--map_path", "/maps", "--shuffle_scenarios", "false"]
        )
        self.assertFalse(args.shuffle_scenarios)
        self.assertEqual(Path(args.log_names), ROOT / "nuplan_train.json")


if __name__ == "__main__":
    unittest.main()
