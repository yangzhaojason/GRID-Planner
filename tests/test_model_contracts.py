"""CPU tensor regressions for scene conditioning and trajectory token losses."""

import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("Model tensor tests require the training dependencies") from exc
from torch import nn
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler

from flow_planner.data.dataset.nuplan import NuPlanDataSample
from flow_planner.model.flow_planner_model.flow_planner import FlowPlanner
from flow_planner.model.flow_planner_model.flow_utils.flow_ode import FlowODE
from flow_planner.model.model_utils.input_preprocess import ModelInputProcessor
from flow_planner.model.model_utils.time_sampler import TimeSampler
from flow_planner.reward.rrs import RelationalRewardSurrogate


class IdentityNormalizer:
    def __call__(self, value):
        return value

    def inverse(self, value):
        return value


class SceneEncoder(nn.Module):
    """Expose scene identities so incorrect CFG pairing cannot be hidden by a network."""

    def forward(self, neighbors, routes, **kwargs):
        return {"scene": routes[:, 0, 0, 0] + neighbors[:, 0, -1, 0]}


class SceneDecoder(nn.Module):
    def forward(self, x, t, scene, **kwargs):
        return scene[:, None, None, None].expand_as(x)


class TokenDecoder(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = nn.Parameter(tokens)

    def forward(self, x, t, **kwargs):
        return self.tokens.expand(x.shape[0], -1, -1, -1)


def make_sample(scene_ids, future_len=4):
    batch_size = len(scene_ids)
    ids = torch.tensor(scene_ids, dtype=torch.float32)
    current = torch.zeros(batch_size, 10)
    current[:, 2] = 1
    neighbors = torch.zeros(batch_size, 2, 2, 8)
    neighbors[:, 0, :, 0] = ids[:, None]
    neighbors[:, 0, :, 2] = 1
    neighbors[:, 0, :, 6:8] = 2
    routes = torch.zeros(batch_size, 1, 2, 12)
    routes[..., 0] = 10 * ids[:, None, None]
    return NuPlanDataSample(
        batched=True,
        ego_past=torch.zeros(batch_size, 2, 7),
        ego_current=current,
        ego_future=torch.zeros(batch_size, future_len, 3),
        neighbor_past=neighbors,
        routes=routes,
        lanes=routes.clone(),
        lanes_speedlimit=torch.zeros(batch_size, 1, 1),
        lanes_has_speedlimit=torch.zeros(batch_size, 1, 1),
        map_objects=torch.zeros(batch_size, 1, 10),
    )


def make_model(future_len=4, action_len=2, action_overlap=1, decoder=None, cfg_type="neighbors"):
    flow_ode = FlowODE(
        path=AffineProbPath(scheduler=CondOTScheduler()),
        time_sampler=TimeSampler(device="cpu", eps=1e-3),
        cfg_weight=1.8,
        sample_temperature=1.0,
        sample_steps=1,
        sample_method="euler",
    )
    return FlowPlanner(
        model_encoder=SceneEncoder(),
        model_decoder=decoder if decoder is not None else SceneDecoder(),
        flow_ode=flow_ode,
        data_processor=ModelInputProcessor(future_len, None, IdentityNormalizer(), 0),
        device="cpu",
        assemble_method="average",
        cfg_prob=0.0,
        cfg_weight=1.8,
        cfg_type=cfg_type,
        neighbor_num=2,
        cfg_neighbor_num=1,
        future_len=future_len,
        action_len=action_len,
        action_overlap=action_overlap,
        state_dim=4,
    )


class ModelContractTest(unittest.TestCase):
    def test_sampler_honors_explicit_cfg_weight(self):
        model = make_model().eval()
        for weight in (0.0, 1.0, 2.0):
            with self.subTest(cfg_weight=weight):
                actual = model(make_sample([2.0]), mode="inference", use_cfg=True, cfg_weight=weight)
                torch.testing.assert_close(actual, torch.full_like(actual, 20.0 + 2.0 * weight))

    def test_sampler_defaults_to_configured_cfg_weight(self):
        model = make_model().eval()
        actual = model(make_sample([2.0]), mode="inference", use_cfg=True, cfg_weight=None)
        torch.testing.assert_close(actual, torch.full_like(actual, 23.6))

    def test_cfg_batch_matches_each_scene_sampled_individually(self):
        model = make_model().eval()
        for scene_ids in ([1.0, 3.0], [1.0, 3.0, 7.0]):
            with self.subTest(batch_size=len(scene_ids)):
                batch = model(make_sample(scene_ids), mode="inference", use_cfg=True, cfg_weight=1.8)
                individual = torch.cat([
                    model(make_sample([scene]), mode="inference", use_cfg=True, cfg_weight=1.8)
                    for scene in scene_ids
                ])
                torch.testing.assert_close(batch, individual)

    def test_training_conditioning_dropout_preserves_caller_scene(self):
        for cfg_type, field in (("neighbors", "neighbor_past"), ("lanes", "lanes")):
            with self.subTest(cfg_type=cfg_type):
                model = make_model(cfg_type=cfg_type)
                data = make_sample([1.0])
                original = getattr(data, field).clone()
                model_inputs, _ = model.prepare_model_input(
                    torch.zeros(1, 1, dtype=torch.int32), data, use_cfg=False, is_training=True,
                )
                self.assertEqual(torch.count_nonzero(model_inputs[field]).item(), 0)
                torch.testing.assert_close(getattr(data, field), original)

    def test_conditioning_dropout_does_not_hide_collision_from_rrs(self):
        model = make_model()
        data = make_sample([1.0])
        trajectory = torch.zeros(1, 4, 4)
        trajectory[..., 2] = 1
        reward = RelationalRewardSurrogate({"safety_neighbor_source": "past_cv"})
        _, before = reward(trajectory, data)
        model.prepare_model_input(
            torch.zeros(1, 1, dtype=torch.int32), data, use_cfg=False, is_training=True,
        )
        _, after = reward(trajectory, data)
        self.assertGreater(before["rrs_safety_pen"].item(), 0)
        torch.testing.assert_close(after["rrs_safety_pen"], before["rrs_safety_pen"])

    def test_consistency_loss_includes_final_overlap_and_its_gradient(self):
        tokens = torch.zeros(1, 3, 2, 4)
        tokens[:, -1, 0, 0] = 2
        decoder = TokenDecoder(tokens)
        model = make_model(decoder=decoder)
        _, loss = model(make_sample([1.0]), mode="train")
        self.assertAlmostEqual(loss["consistency_loss"].item(), 2.0)
        loss["consistency_loss"].backward()
        self.assertAlmostEqual(decoder.tokens.grad[0, -1, 0, 0].item(), 2.0)

    def test_two_tokens_have_one_consistency_pair(self):
        tokens = torch.zeros(1, 2, 2, 4)
        tokens[:, -1, 0, 0] = 2
        model = make_model(future_len=3, decoder=TokenDecoder(tokens))
        _, loss = model(make_sample([1.0], future_len=3), mode="train")
        self.assertAlmostEqual(loss["consistency_loss"].item(), 4.0)

    def test_single_token_has_zero_consistency_loss(self):
        model = make_model(future_len=2, decoder=TokenDecoder(torch.zeros(1, 1, 2, 4)))
        _, loss = model(make_sample([1.0], future_len=2), mode="train")
        self.assertEqual(loss["consistency_loss"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
