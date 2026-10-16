"""Real Hydra model, forward/backward, and CFG sampling on a synthetic CPU scene.

No benchmark data or driving-performance claims are derived from this fixture.
"""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


def make_scene():
    from flow_planner.data.dataset.nuplan import NuPlanDataSample

    batch = 2
    shapes = {
        "ego_past": (batch, 21, 14), "ego_current": (batch, 16),
        "ego_future": (batch, 80, 3), "neighbor_past": (batch, 32, 21, 11),
        "neighbor_future": (batch, 0, 80, 3),
        "neighbor_future_observed": (batch, 32, 80, 3),
        "lanes": (batch, 70, 20, 12), "lanes_speedlimit": (batch, 70, 1),
        "lanes_has_speedlimit": (batch, 70, 1), "routes": (batch, 25, 20, 12),
        "routes_speedlimit": (batch, 25, 1), "routes_has_speedlimit": (batch, 25, 1),
        "map_objects": (batch, 5, 10),
    }
    fields = {
        key: torch.zeros(shape, dtype=torch.bool if "has_speedlimit" in key else torch.float32)
        for key, shape in shapes.items()
    }
    fields["ego_current"][:, 2] = 1
    fields["ego_current"][:, 4] = 5
    fields["ego_current"][:, 10:12] = torch.tensor([2., 4.8])
    fields["ego_past"][:, :, 0] = torch.linspace(-10, 0, 21)
    fields["ego_past"][:, :, 2] = 1
    fields["ego_future"][:, :, 0] = torch.linspace(.5, 40, 80)
    fields["neighbor_past"][:, 0, :, 0] = 20
    fields["neighbor_past"][:, 0, :, 1] = 3
    fields["neighbor_past"][:, 0, :, 2] = 1
    fields["neighbor_past"][:, 0, :, 6:8] = torch.tensor([2., 4.8])
    fields["neighbor_future_observed"][:, 0, :, 0] = 20
    fields["neighbor_future_observed"][:, 0, :, 1] = 3
    for key in ("lanes", "routes"):
        fields[key][:, 0, :, 0] = torch.linspace(-10, 80, 20)
        fields[key][:, 0, :, 2] = 1
        fields[key + "_speedlimit"][:, 0] = 15
        fields[key + "_has_speedlimit"][:, 0] = True
    return NuPlanDataSample(batched=True, **fields)


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class FullModelSmokeTest(unittest.TestCase):
    def test_training_reward_backward_and_cfg_sampling(self):
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        root = Path(__file__).resolve().parents[1]
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "PROJECT_ROOT": str(root), "SAVE_DIR": directory,
            "TRAINING_DATA": directory, "TRAINING_JSON": str(Path(directory) / "list.json"),
            "WORLD_SIZE": "1", "LOCAL_RANK": "0",
        }):
            torch.manual_seed(3402)
            with initialize_config_dir(config_dir=str(root / "flow_planner/script"), version_base=None):
                cfg = compose(config_name="flow_planner_standard", overrides=[
                    "device=cpu", "ddp.distributed=false", "core.input_aug.state_aug=null",
                    "core.graph_weighting.enabled=true", "core.rrs.enabled=true", "core.rrs.nu=0.0008",
                ])
            model = instantiate(cfg.model)
            core = instantiate(cfg.core)
            data = make_scene()
            model.train()
            losses = core.train_step(model, data)
            self.assertIn("rrs_loss", losses)
            self.assertTrue(torch.isfinite(losses["total_loss"]).item())
            losses["total_loss"].backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(grad).all().item() for grad in gradients))
            model.eval()
            with torch.no_grad():
                output = model(data, mode="inference", use_cfg=True, cfg_weight=1.1)
            self.assertEqual(tuple(output.shape), (2, 1, 80, 4))
            self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
