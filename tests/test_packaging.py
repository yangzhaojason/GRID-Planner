"""Build and install the distribution without the optional training environment.

Requires only the build tools: pip, setuptools, and wheel.
"""
import email
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


class PackagingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        source = cls.root / "source"
        source.mkdir()
        for name in ("setup.py", "pyproject.toml", "MANIFEST.in", "requirements.txt", "README.md", "LICENSE"):
            if (ROOT / name).is_file():
                shutil.copy2(ROOT / name, source / name)
        shutil.copytree(
            ROOT / "flow_planner", source / "flow_planner",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        result = subprocess.run(
            [sys.executable, "-c", "from setuptools.build_meta import build_sdist; build_sdist('dist')"],
            cwd=source, text=True, capture_output=True,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        # Build the wheel from the source distribution, so missing build inputs
        # cannot be masked by files present only in the checkout.
        unpacked = cls.root / "unpacked"
        with tarfile.open(next((source / "dist").glob("*.tar.gz"))) as archive:
            options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            archive.extractall(unpacked, **options)
        sdist_source = next(unpacked.iterdir())
        result = subprocess.run(
            [sys.executable, "-c", "from setuptools.build_meta import build_wheel; build_wheel('dist')"],
            cwd=sdist_source, text=True, capture_output=True,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.wheel = next((sdist_source / "dist").glob("*.whl"))
        with zipfile.ZipFile(cls.wheel) as archive:
            cls.members = set(archive.namelist())
            metadata_name = next(name for name in cls.members if name.endswith(".dist-info/METADATA"))
            cls.metadata = email.message_from_bytes(archive.read(metadata_name))

    def test_wheel_contains_all_python_modules_and_runtime_resources(self):
        expected = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "flow_planner").rglob("*")
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".sh"}
        }
        self.assertFalse(expected - self.members, f"Missing wheel files: {sorted(expected - self.members)}")
        self.assertFalse(any("__pycache__" in name or name.endswith(".pyc") for name in self.members))

    def test_distribution_metadata(self):
        self.assertEqual(self.metadata["Name"], "grid-planner")
        self.assertIn("GRID-Planner", self.metadata["Summary"] or "")
        self.assertIn("Zhao Yang", self.metadata["Author"] or "")
        self.assertEqual(self.metadata.get("License-Expression", self.metadata.get("License")), "MIT")

    def test_installed_package_imports_and_configuration_resources(self):
        target = self.root / "installed"
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--target", str(target), str(self.wheel)],
            cwd=self.root, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = subprocess.run(
            [sys.executable, "-I", "-c", """
import importlib
from importlib.resources import files
from pathlib import Path
import sys

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
for name in (
    'flow_planner',
    'flow_planner.onpolicy.rollout_runner',
    'flow_planner.core.common.json_utils',
    'flow_planner.script',
    'flow_planner.nuplan_simulation.scenario_filter',
):
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(target), module.__file__
assert 'defaults:' in files('flow_planner.script').joinpath('flow_planner_standard.yaml').read_text()
assert files('flow_planner.script').joinpath('data/dataset/nuplan_data.yaml').is_file()
assert files('flow_planner.nuplan_simulation.scenario_filter').joinpath('val14.yaml').is_file()
assert files('flow_planner').joinpath('run_script/launch_train.sh').is_file()
""", str(target)],
            cwd=self.root, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
