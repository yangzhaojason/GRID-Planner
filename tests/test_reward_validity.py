"""Padded future observations must not create agents at the ego origin."""

import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise unittest.SkipTest("Reward tensor tests require the training dependencies") from exc

from flow_planner.core.flow_matching_core import FlowMatchingCore
from flow_planner.data.dataset.nuplan import NuPlanDataSample
from flow_planner.reward.rrs import RelationalRewardSurrogate


def make_scene(future_x):
    horizon = len(future_x)
    observed = torch.zeros(1, 2, horizon, 3)
    observed[0, 0, :, 0] = torch.tensor(future_x)
    past = torch.zeros(1, 2, 2, 8)
    if any(future_x):
        past[0, 0, :, 0] = 100
        past[0, 0, :, 2] = 1
        past[0, 0, :, 6:8] = 2
    sample = NuPlanDataSample(
        batched=True,
        ego_current=torch.zeros(1, 10),
        ego_future=torch.zeros(1, horizon, 3),
        neighbor_past=past,
        neighbor_future_observed=observed,
    )
    trajectory = torch.zeros(1, horizon, 4)
    trajectory[..., 2] = 1
    return sample, trajectory


class RewardValidityTest(unittest.TestCase):
    def test_missing_future_frames_do_not_create_gaiw_collision_risk(self):
        core = FlowMatchingCore(
            device="cpu",
            graph_weighting={"enabled": True, "alpha1": 0.0, "alpha2": 1.0, "normalize_by_mean": False},
        )
        for future_x in ([100, 100, 0, 0], [100, 0, 100, 100], [0, 0, 0, 0]):
            with self.subTest(future_x=future_x):
                sample, _ = make_scene(future_x)
                torch.testing.assert_close(core._compute_graph_aware_weight(sample), torch.ones(1))

    def test_missing_future_frames_do_not_create_rrs_distance_or_collision_penalties(self):
        reward = RelationalRewardSurrogate({"safety_neighbor_source": "future_observed"})
        for future_x in ([100, 100, 0, 0], [100, 0, 100, 100], [0, 0, 0, 0]):
            with self.subTest(future_x=future_x):
                sample, trajectory = make_scene(future_x)
                _, components = reward(trajectory, sample)
                torch.testing.assert_close(components["rrs_safety_d_pen"], torch.zeros(1))
                torch.testing.assert_close(components["rrs_safety_col_pen"], torch.zeros(1))

    def test_ttc_requires_observations_at_both_ends_of_the_velocity_interval(self):
        reward = RelationalRewardSurrogate({"safety_neighbor_source": "future_observed"})
        for future_x in ([100, 100, 0, 0], [100, 0, 100, 100]):
            with self.subTest(future_x=future_x):
                sample, trajectory = make_scene(future_x)
                _, components = reward(trajectory, sample)
                torch.testing.assert_close(components["rrs_safety_ttc_pen"], torch.zeros(1))

    def test_valid_close_observations_still_contribute_risk(self):
        core = FlowMatchingCore(
            device="cpu",
            graph_weighting={"enabled": True, "alpha1": 0.0, "alpha2": 1.0, "normalize_by_mean": False},
        )
        sample, trajectory = make_scene([1, 1, 0, 0])
        reward = RelationalRewardSurrogate({"safety_neighbor_source": "future_observed"})
        _, components = reward(trajectory, sample)
        self.assertGreater(core._compute_graph_aware_weight(sample).item(), 1.0)
        self.assertGreater(components["rrs_safety_d_pen"].item(), 0.0)
        self.assertGreater(components["rrs_safety_col_pen"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
