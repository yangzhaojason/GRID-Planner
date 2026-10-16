"""CPU regression checks for checkpoint portability and launcher setup.

These checks require the training dependencies; the lightweight suite skips
this module when PyTorch is not installed.
"""
import copy
import importlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class CheckpointTest(unittest.TestCase):
    def make_training_state(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        model(torch.ones(2, 3)).sum().backward()
        optimizer.step()
        scheduler.step()
        ema = SimpleNamespace(ema=copy.deepcopy(model))
        return model, optimizer, scheduler, ema

    def test_resume_ddp_checkpoint_into_plain_model(self):
        from flow_planner.train_utils.save_model import resume_model, save_model

        model, optimizer, scheduler, ema = self.make_training_state()
        # DDP prefixes model parameters with its actual registered module name.
        wrapped = torch.nn.Module()
        wrapped.add_module("module", model)
        wrapped_ema = torch.nn.Module()
        wrapped_ema.add_module("module", ema.ema)
        with tempfile.TemporaryDirectory() as directory:
            save_model(wrapped, optimizer, scheduler, directory, 4, 1.0, "run-id", wrapped_ema)
            restored, restored_optimizer, restored_scheduler, restored_ema = self.make_training_state()
            result = resume_model(directory, restored, restored_optimizer, restored_scheduler, restored_ema, "cpu")

        self.assertEqual(result[3:5], (5, "run-id"))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)
            torch.testing.assert_close(restored_ema.ema.state_dict()[name], value)
        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
        self.assertEqual(restored_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
        self.assertFalse(restored_ema.ema.training)
        self.assertTrue(all(not p.requires_grad for p in restored_ema.ema.parameters()))

    def test_weight_loader_accepts_plain_and_ddp_ema_checkpoints(self):
        from flow_planner.train_utils.save_model import load_model_weights

        source = torch.nn.Linear(3, 2)
        for prefix in ("", "module."):
            with self.subTest(prefix=prefix):
                state = {prefix + key: value.clone() for key, value in source.state_dict().items()}
                target = torch.nn.Linear(3, 2)
                load_model_weights(target, {"ema_state_dict": state})
                for key, value in source.state_dict().items():
                    torch.testing.assert_close(target.state_dict()[key], value)

    def test_weight_loader_retains_strict_shape_validation(self):
        from flow_planner.train_utils.save_model import load_model_weights

        with self.assertRaises(RuntimeError):
            load_model_weights(torch.nn.Linear(3, 2), {"model": torch.nn.Linear(4, 2).state_dict()}, use_ema=False)

    def test_resume_without_ema_initializes_shadow_from_restored_model(self):
        from flow_planner.train_utils.save_model import resume_model

        source = torch.nn.Linear(3, 2)
        with torch.no_grad():
            source.weight.fill_(7)
            source.bias.fill_(4)
        for prefix in ("", "module."):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pth"
                torch.save({"model": {prefix + k: v for k, v in source.state_dict().items()}, "epoch": 5}, path)
                model, optimizer, scheduler, ema = self.make_training_state()
                initialized_scheduler = copy.deepcopy(scheduler.state_dict())
                initialized_optimizer = copy.deepcopy(optimizer.state_dict())
                resume_model(str(path), model, optimizer, scheduler, ema, "cpu")
                for name, value in source.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name], value)
                    torch.testing.assert_close(ema.ema.state_dict()[name], value)
                self.assertFalse(ema.ema.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in ema.ema.parameters()))
                self.assertEqual(scheduler.state_dict(), initialized_scheduler)
                self.assertEqual(optimizer.param_groups[0]["lr"], initialized_optimizer["param_groups"][0]["lr"])

    def test_resume_rejects_incompatible_existing_ema(self):
        from flow_planner.train_utils.save_model import resume_model

        model, optimizer, scheduler, ema = self.make_training_state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save({"model": model.state_dict(), "ema_state_dict": torch.nn.Linear(4, 2).state_dict()}, path)
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                resume_model(str(path), model, optimizer, scheduler, ema, "cpu")

    def test_resume_rejects_invalid_present_optimizer_scheduler_and_aux(self):
        from flow_planner.train_utils.save_model import resume_model
        from flow_planner.core.flow_matching_core import FlowMatchingCore

        core = FlowMatchingCore.__new__(FlowMatchingCore)
        core.value_net = torch.nn.Linear(3, 2)
        cases = (
            ("optimizer", {"state": {}, "param_groups": []}, ValueError),
            ("schedule", None, TypeError),
            ("aux_state_dict", {"value_net": torch.nn.Linear(4, 2).state_dict()}, RuntimeError),
            ("aux_state_dict", {"value_net": {"weight": torch.ones(2, 3)}}, RuntimeError),
        )
        for key, value, error in cases:
            with self.subTest(state=key), tempfile.TemporaryDirectory() as directory:
                model, optimizer, scheduler, ema = self.make_training_state()
                path = Path(directory) / "checkpoint.pth"
                checkpoint = {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "schedule": scheduler.state_dict(), "ema_state_dict": ema.ema.state_dict(), "epoch": 25,
                }
                checkpoint[key] = value
                torch.save(checkpoint, path)
                with self.assertRaises(error):
                    resume_model(str(path), model, optimizer, scheduler, ema, "cpu", core=core)


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class DistributedSetupTest(unittest.TestCase):
    def config(self):
        return SimpleNamespace(ddp=SimpleNamespace(distributed=True, port=23456, init_process_group="nccl"))

    def test_no_launcher_environment_falls_back_to_non_distributed(self):
        from flow_planner.train_utils.ddp import ddp_setup_universal

        cfg = self.config()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ddp_setup_universal(cfg=cfg), (0, 0, 1))
        self.assertFalse(cfg.ddp.distributed)

    def test_slurm_uses_ddp_port_and_local_gpu(self):
        from flow_planner.train_utils.ddp import ddp_setup_universal

        cfg = self.config()
        env = {"SLURM_PROCID": "5", "SLURM_NTASKS": "8", "SLURM_NODELIST": "worker-[1-2]"}
        with patch.dict(os.environ, env, clear=True), \
                patch("torch.cuda.device_count", return_value=4), \
                patch("torch.cuda.set_device") as set_device, \
                patch("flow_planner.train_utils.ddp.subprocess.getoutput", return_value="worker-1"), \
                patch("flow_planner.train_utils.ddp.init_process_group") as init_group, \
                patch("torch.distributed.barrier"):
            self.assertEqual(ddp_setup_universal(cfg=cfg), (5, 1, 8))
            self.assertEqual(os.environ["MASTER_PORT"], "23456")
            self.assertEqual(os.environ["MASTER_ADDR"], "worker-1")
        set_device.assert_called_once_with(1)
        init_group.assert_called_once_with(backend="nccl", world_size=8, rank=5)


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class DataDeviceTest(unittest.TestCase):
    def test_device_object_moves_every_tensor_including_speed_limit_flags(self):
        from dataclasses import fields
        from flow_planner.data.dataset.nuplan import NuPlanDataSample

        values = {
            field.name: torch.ones(1, dtype=torch.bool if "has_speedlimit" in field.name else torch.float32)
            for field in fields(NuPlanDataSample) if field.name != "batched"
        }
        sample = NuPlanDataSample(batched=True, **values)
        self.assertIs(sample.to(torch.device("meta")), sample)
        self.assertTrue(sample.batched)
        for name in values:
            with self.subTest(field=name):
                self.assertEqual(getattr(sample, name).device.type, "meta")


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class TrainingLifecycleTest(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("nuplan") is None, "nuPlan devkit is not installed")
    def test_cpu_planner_loads_plain_checkpoint_from_a_cuda_config(self):
        from hydra import compose, initialize_config_module
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from flow_planner.planner import FlowPlanner

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "PROJECT_ROOT": str(Path(__file__).resolve().parents[1]), "SAVE_DIR": directory,
        }):
            with initialize_config_module(config_module="flow_planner.script", version_base=None):
                cfg = compose(config_name="flow_planner_standard", overrides=["device=cpu", "ddp.distributed=false"])
            model = instantiate(cfg.model)
            self.assertEqual(model.device, "cpu")
            checkpoint = str(Path(directory) / "model.pth")
            torch.save({"ema_state_dict": model.state_dict()}, checkpoint)
            cfg.device = "cuda"
            config_path = str(Path(directory) / "config.yaml")
            OmegaConf.save(cfg, config_path)
            planner = FlowPlanner(
                config_path, checkpoint,
                TrajectorySampling(num_poses=20, time_horizon=2.0),
                TrajectorySampling(num_poses=80, time_horizon=8.0),
                device="cpu",
            )
            planner.initialize(SimpleNamespace(map_api=None, route_roadblock_ids=[]))
            self.assertEqual(planner._planner.device, "cpu")
            self.assertEqual(planner.core.device, "cpu")
            self.assertEqual(next(planner._planner.parameters()).device.type, "cpu")
            for key, value in model.state_dict().items():
                torch.testing.assert_close(planner._planner.state_dict()[key], value)

    def test_single_cpu_epoch_saves_and_exits_without_a_process_group(self):
        from omegaconf import OmegaConf

        class TinyModel(torch.nn.Linear):
            def get_optimizer_params(self):
                return [{"params": self.parameters()}]

        class TinyCore:
            def train_step(self, model, data):
                loss = model(data).square().mean()
                return {"total_loss": loss}

            def attach_teacher(self, teacher, checkpoint, device):
                from flow_planner.train_utils.save_model import load_model, load_model_weights
                load_model_weights(teacher, load_model(checkpoint))
                teacher.to(device)

        class TinyEMA:
            def __init__(self, model):
                self.ema = copy.deepcopy(model)

            def update(self, model):
                self.ema.load_state_dict(model.state_dict())

        def instantiate(config, **kwargs):
            component = config.component
            if component == "model":
                return TinyModel(3, 2)
            if component == "data":
                return [torch.ones(3), torch.ones(3) * 2]
            if component == "core":
                return TinyCore()
            if component == "optimizer":
                return torch.optim.SGD(kwargs["params"], lr=kwargs["lr"])
            if component == "scheduler":
                return torch.optim.lr_scheduler.StepLR(kwargs["optimizer"], 1)
            if component == "ema":
                return TinyEMA(kwargs["model"])
            raise AssertionError(component)

        for module_name, function_name in (("flow_planner.trainer", "trainer"), ("flow_planner.onpolicy.distill_trainer", "distill_trainer")):
            with self.subTest(trainer=module_name), tempfile.TemporaryDirectory() as directory:
                checkpoint = str(Path(directory) / "teacher.pth")
                torch.save({"ema_state_dict": TinyModel(3, 2).state_dict()}, checkpoint)
                cfg = OmegaConf.create({
                    "seed": 12, "device": "cpu", "ddp": {"distributed": False},
                    "save_dir": directory, "job_name": "cpu-smoke", "num_workers": 0,
                    "pin_mem": False, "pretrained_checkpoint": checkpoint, "resume_path": None,
                    "save_every_since": 200,
                    "train": {"batch_size": 2, "epoch": 1, "save_utd": 1},
                    "model": {"component": "model"},
                    "core": {"component": "core", "opsd": {"teacher_ckpt_path": checkpoint}},
                    "data": {"dataset": {"train": {"component": "data"}}},
                    "optimizer": {"component": "optimizer", "lr": 0.01},
                    "scheduler": {"component": "scheduler"}, "ema": {"component": "ema"},
                })
                module = importlib.import_module(module_name)
                with patch.object(module, "instantiate", side_effect=instantiate), patch.object(module, "collect_batch", torch.stack):
                    getattr(module, function_name).__wrapped__(cfg)
                saved = torch.load(Path(directory) / "latest.pth", weights_only=True, map_location="cpu")
                self.assertEqual(saved["epoch"], 1)
                self.assertTrue(torch.isfinite(saved["loss"]))
                self.assertFalse(torch.distributed.is_initialized())


if __name__ == "__main__":
    unittest.main()
