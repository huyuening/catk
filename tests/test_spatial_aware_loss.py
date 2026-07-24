import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_metric_utils():
    if importlib.util.find_spec("omegaconf") is None:
        omegaconf_stub = types.ModuleType("omegaconf")
        omegaconf_stub.DictConfig = dict
        sys.modules["omegaconf"] = omegaconf_stub

    module_name = "catk_test_metric_utils"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "src/smart/metrics/utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


METRIC_UTILS = _load_metric_utils()
get_prob_targets = METRIC_UTILS.get_prob_targets
get_prob_targets_spatial_aware_smoothing = (
    METRIC_UTILS.get_prob_targets_spatial_aware_smoothing
)


class SpatialAwareTargetTest(unittest.TestCase):
    @staticmethod
    def _contours(center_x):
        center_x = torch.as_tensor(center_x, dtype=torch.float32)
        half_length = 2.0
        half_width = 1.0
        corners = torch.stack(
            (
                torch.stack(
                    (
                        center_x + half_length,
                        torch.full_like(center_x, half_width),
                    ),
                    dim=-1,
                ),
                torch.stack(
                    (
                        center_x + half_length,
                        torch.full_like(center_x, -half_width),
                    ),
                    dim=-1,
                ),
                torch.stack(
                    (
                        center_x - half_length,
                        torch.full_like(center_x, -half_width),
                    ),
                    dim=-1,
                ),
                torch.stack(
                    (
                        center_x - half_length,
                        torch.full_like(center_x, half_width),
                    ),
                    dim=-1,
                ),
            ),
            dim=-2,
        )
        return corners.unsqueeze(0)

    @staticmethod
    def _shape():
        return torch.tensor([[2.0, 4.0]], dtype=torch.float32)

    def test_distribution_is_normalized_and_reserves_point_one_for_neighbors(self):
        target = torch.tensor([[[0.2, 0.0, 0.0]]], dtype=torch.float32)
        token_traj = self._contours([0.0, 1.0, 4.0])

        probability = get_prob_targets_spatial_aware_smoothing(
            target=target,
            token_agent_shape=self._shape(),
            token_traj=token_traj,
            label_smoothing=0.1,
        )

        self.assertTrue(
            torch.allclose(probability.sum(dim=-1), torch.ones(1, 1))
        )
        self.assertAlmostEqual(float(probability[0, 0, 0]), 0.9, places=6)
        self.assertAlmostEqual(
            float(probability[0, 0, 1:].sum()), 0.1, places=6
        )
        self.assertGreater(
            float(probability[0, 0, 1]),
            float(probability[0, 0, 2]),
        )

    def test_neighbors_are_ranked_from_raw_gt_not_quantized_target(self):
        target = torch.tensor([[[0.2, 0.0, 0.0]]], dtype=torch.float32)
        token_traj = self._contours([0.0, 0.55, -0.45])

        probability = get_prob_targets_spatial_aware_smoothing(
            target=target,
            token_agent_shape=self._shape(),
            token_traj=token_traj,
            label_smoothing=0.1,
        )

        self.assertAlmostEqual(float(probability[0, 0, 0]), 0.9, places=6)
        self.assertGreater(
            float(probability[0, 0, 1]),
            float(probability[0, 0, 2]),
        )

    def test_zero_smoothing_matches_legacy_one_hot_target(self):
        target = torch.tensor([[[0.2, 0.0, 0.0]]], dtype=torch.float32)
        token_traj = self._contours([0.0, 1.0, 4.0])

        legacy = get_prob_targets(target, self._shape(), token_traj)
        spatial = get_prob_targets_spatial_aware_smoothing(
            target=target,
            token_agent_shape=self._shape(),
            token_traj=token_traj,
            label_smoothing=0.0,
        )

        self.assertTrue(torch.equal(spatial, legacy))

    def test_single_token_vocabulary_returns_one_hot(self):
        probability = get_prob_targets_spatial_aware_smoothing(
            target=torch.tensor([[[0.2, 0.0, 0.0]]], dtype=torch.float32),
            token_agent_shape=self._shape(),
            token_traj=self._contours([0.0]),
            label_smoothing=0.1,
        )

        self.assertTrue(torch.equal(probability, torch.ones(1, 1, 1)))

    def test_invalid_smoothing_is_rejected(self):
        target = torch.tensor([[[0.2, 0.0, 0.0]]], dtype=torch.float32)
        token_traj = self._contours([0.0, 1.0])

        for value in (-0.1, 1.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "label_smoothing"):
                    get_prob_targets_spatial_aware_smoothing(
                        target=target,
                        token_agent_shape=self._shape(),
                        token_traj=token_traj,
                        label_smoothing=value,
                    )


if __name__ == "__main__":
    unittest.main()
