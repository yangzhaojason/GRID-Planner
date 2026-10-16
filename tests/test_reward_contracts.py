"""Training must not silently drop configured reward or distillation terms."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("Reward contracts require the training dependencies") from exc

from flow_planner.core.flow_matching_core import FlowMatchingCore
from flow_planner.core.opds_distill_core import OPSDDistillCore
from flow_planner.data.dataset.nuplan import NuPlanDataSample
from flow_planner.model.flow_planner_model.flow_planner import FlowPlanner


def scene():
    return NuPlanDataSample(batched=True, ego_current=torch.zeros(1, 10))


def planner_fixture():
    """Exercise the real forward_train with deterministic lightweight model inputs."""
    return SimpleNamespace(
        cfg_prob=0., device="cpu", _model_type="x_start", assemble_method="average",
        planner_params={"future_len": 4, "action_len": 2, "action_overlap": 1, "state_dim": 4},
        prepare_model_input=lambda *args, **kwargs: ({}, torch.zeros(1, 1, 5, 4)),
        extract_encoder_inputs=lambda inputs: {}, encoder=lambda: {},
        extract_decoder_inputs=lambda *args: {}, decoder=lambda x, t, **kwargs: x,
        flow_ode=SimpleNamespace(sample=lambda x, kind: (x, x, torch.zeros(1))),
        basic_loss=torch.nn.MSELoss(reduction="none"),
        data_processor=SimpleNamespace(state_postprocess=lambda x: x),
    )


def incomplete_model(data, mode):
    return None, {"ego_planning_loss": torch.tensor(1., requires_grad=True)}


class RewardContractTest(unittest.TestCase):
    def test_assembly_errors_propagate_from_forward_train(self):
        planner = planner_fixture()
        with patch("flow_planner.model.flow_planner_model.flow_planner.assemble_actions", side_effect=ValueError("invalid trajectory token assembly")):
            with self.assertRaisesRegex(ValueError, "invalid trajectory token assembly"):
                FlowPlanner.forward_train(planner, scene())

    def test_denormalization_errors_propagate_from_forward_train(self):
        planner = planner_fixture()
        with patch.object(planner.data_processor, "state_postprocess", side_effect=ValueError("invalid normalization state")):
            with self.assertRaisesRegex(ValueError, "invalid normalization state"):
                FlowPlanner.forward_train(planner, scene())

    def test_active_rrs_rejects_missing_predictions(self):
        for nu, rl_enabled in ((.5, False), (0., True)):
            with self.subTest(nu=nu, rl_enabled=rl_enabled):
                core = FlowMatchingCore(input_aug=lambda data: data, device="cpu", rrs={"enabled": True, "nu": nu}, rl={"enabled": rl_enabled}, ego_planning_loss=1.)
                # The contract must fail before any optional on-policy sampling.
                with patch.object(core, "_apply_standard_rl_objective", side_effect=lambda **kw: kw["total_loss"]):
                    with self.assertRaisesRegex(RuntimeError, "pred_future_denorm"):
                        core.train_step(incomplete_model, scene())

    def test_inactive_reward_does_not_require_predictions(self):
        core = FlowMatchingCore(input_aug=lambda data: data, device="cpu", rrs={"enabled": True, "nu": 0.}, ego_planning_loss=1.)
        result = core.train_step(incomplete_model, scene())
        self.assertEqual(result["total_loss"].item(), 1.)

    def test_enabled_rl_cannot_silently_run_without_rrs(self):
        core = FlowMatchingCore(input_aug=lambda data: data, device="cpu", rrs={"enabled": False}, rl={"enabled": True}, ego_planning_loss=1.)
        with self.assertRaisesRegex(RuntimeError, "core.rrs.enabled=true"):
            core.train_step(incomplete_model, scene())

    def test_attached_distillation_rejects_missing_student_predictions(self):
        core = OPSDDistillCore(input_aug=lambda data: data, device="cpu", rrs={"enabled": True, "nu": .5}, opsd={"enabled": True}, ego_planning_loss=1.)
        core._teacher_model = torch.nn.Identity()
        with self.assertRaisesRegex(RuntimeError, "pred_future_denorm"):
            core.train_step(incomplete_model, scene())


if __name__ == "__main__":
    unittest.main()
