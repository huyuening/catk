import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import yaml


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


def _get_trajtok_original_helper():
    return getattr(
        METRIC_UTILS,
        "get_prob_targets_trajtok_original",
    )


def _install_torchmetrics_stub():
    torchmetrics_module = types.ModuleType("torchmetrics")
    metric_module = types.ModuleType("torchmetrics.metric")

    class Metric(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def add_state(self, name, default, dist_reduce_fx=None):
            self.register_buffer(name, default.clone())

        def forward(self, *args, **kwargs):
            self.update(*args, **kwargs)
            return self.compute()

    metric_module.Metric = Metric
    torchmetrics_module.metric = metric_module
    sys.modules["torchmetrics"] = torchmetrics_module
    sys.modules["torchmetrics.metric"] = metric_module


def _load_cross_entropy():
    using_stub = importlib.util.find_spec("torchmetrics") is None
    if using_stub:
        _install_torchmetrics_stub()

    try:
        package_name = "catk_test_metrics"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / "src/smart/metrics")]
        sys.modules[package_name] = package

        for module_name in ("utils", "cross_entropy"):
            qualified_name = f"{package_name}.{module_name}"
            spec = importlib.util.spec_from_file_location(
                qualified_name,
                ROOT / f"src/smart/metrics/{module_name}.py",
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified_name] = module
            spec.loader.exec_module(module)

        return sys.modules[f"{package_name}.cross_entropy"].CrossEntropy
    finally:
        if using_stub:
            sys.modules.pop("torchmetrics.metric", None)
            sys.modules.pop("torchmetrics", None)


CrossEntropy = _load_cross_entropy()


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


class TrajTokOriginalTargetTest(unittest.TestCase):
    def test_matches_trajtok_5920c89_reference_fixture(self):
        token_traj = SpatialAwareTargetTest._contours(
            [0.0, 1.0, 4.0]
        )

        probability = _get_trajtok_original_helper()(
            gt_idx=torch.tensor([[0]], dtype=torch.long),
            token_traj=token_traj,
            label_smoothing=0.1,
        )

        expected = torch.tensor(
            [[[0.9, 0.0885974765, 0.00553817255]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(probability, expected))

    def test_retains_second_division_and_non_unit_mass(self):
        token_traj = SpatialAwareTargetTest._contours(
            [0.0, 1.0, 4.0]
        )

        probability = _get_trajtok_original_helper()(
            gt_idx=torch.tensor([[0]], dtype=torch.long),
            token_traj=token_traj,
            label_smoothing=0.1,
        )

        self.assertAlmostEqual(
            float(probability.sum()),
            0.9941356182,
            places=6,
        )
        self.assertFalse(
            torch.allclose(probability.sum(), torch.tensor(1.0))
        )

    def test_selected_gt_idx_changes_spatial_center(self):
        token_traj = SpatialAwareTargetTest._contours(
            [0.0, 1.0, 4.0]
        )
        helper = _get_trajtok_original_helper()

        centered_at_zero = helper(
            torch.tensor([[0]]),
            token_traj,
            0.1,
        )
        centered_at_one = helper(
            torch.tensor([[1]]),
            token_traj,
            0.1,
        )

        self.assertAlmostEqual(
            float(centered_at_zero[0, 0, 0]),
            0.9,
            places=6,
        )
        self.assertAlmostEqual(
            float(centered_at_one[0, 0, 1]),
            0.9,
            places=6,
        )
        self.assertFalse(torch.equal(centered_at_zero, centered_at_one))


class CrossEntropySelectionTest(unittest.TestCase):
    @staticmethod
    def _metric(
        spatial_aware_smoothing,
        spatial_aware_smoothing_mode="raw_gt_normalized",
    ):
        metric = CrossEntropy(
            use_gt_raw=True,
            gt_thresh_scale_length=-1.0,
            label_smoothing=0.1,
            rollout_as_gt=False,
            spatial_aware_smoothing=spatial_aware_smoothing,
            spatial_aware_smoothing_mode=spatial_aware_smoothing_mode,
        )
        metric.eval()
        return metric

    @staticmethod
    def _inputs():
        n_agent, n_step = 1, 16
        pred_positions = torch.zeros(n_agent, 18, 2)
        gt_positions = pred_positions.clone()
        gt_positions[..., 0] = 0.2
        headings = torch.zeros(n_agent, 18)
        valid_18 = torch.ones(n_agent, 18, dtype=torch.bool)
        logits = torch.tensor([2.0, 0.5, -1.0]).view(1, 1, 3)
        logits = logits.expand(n_agent, n_step, 3).clone()
        token_agent_shape = SpatialAwareTargetTest._shape()
        token_traj = SpatialAwareTargetTest._contours([0.0, 1.0, 4.0])

        return {
            "next_token_logits": logits,
            "next_token_valid": torch.ones(
                n_agent,
                n_step,
                dtype=torch.bool,
            ),
            "pred_pos": pred_positions,
            "pred_head": headings,
            "pred_valid": valid_18,
            "gt_pos_raw": gt_positions,
            "gt_head_raw": headings,
            "gt_valid_raw": valid_18,
            "gt_pos": gt_positions,
            "gt_head": headings,
            "gt_valid": valid_18,
            "token_agent_shape": token_agent_shape,
            "token_traj": token_traj,
            "gt_idx": torch.zeros(n_agent, n_step, dtype=torch.long),
        }

    @classmethod
    def _expected_target(cls):
        return torch.tensor(
            [[[0.2, 0.0, 0.0]] * 16],
            dtype=torch.float32,
        )

    def test_spatial_path_uses_spatial_distribution_without_double_smoothing(
        self,
    ):
        metric = self._metric(spatial_aware_smoothing=True)
        inputs = self._inputs()

        metric.update(**inputs)

        probability = get_prob_targets_spatial_aware_smoothing(
            target=self._expected_target(),
            token_agent_shape=inputs["token_agent_shape"],
            token_traj=inputs["token_traj"],
            label_smoothing=0.1,
        )
        expected = torch.nn.functional.cross_entropy(
            inputs["next_token_logits"].transpose(1, 2),
            probability.transpose(1, 2),
            reduction="none",
            label_smoothing=0.0,
        ).mean()
        self.assertTrue(torch.allclose(metric.compute(), expected))

    def test_legacy_path_keeps_uniform_builtin_smoothing(self):
        metric = self._metric(spatial_aware_smoothing=False)
        inputs = self._inputs()

        metric.update(**inputs)

        probability = get_prob_targets(
            target=self._expected_target(),
            token_agent_shape=inputs["token_agent_shape"],
            token_traj=inputs["token_traj"],
        )
        expected = torch.nn.functional.cross_entropy(
            inputs["next_token_logits"].transpose(1, 2),
            probability.transpose(1, 2),
            reduction="none",
            label_smoothing=0.1,
        ).mean()
        self.assertTrue(torch.allclose(metric.compute(), expected))

    def test_trajtok_original_mode_uses_original_soft_target(self):
        metric = self._metric(
            spatial_aware_smoothing=True,
            spatial_aware_smoothing_mode="trajtok_original",
        )
        inputs = self._inputs()

        metric.update(**inputs)

        probability = torch.tensor(
            [0.9, 0.0885974765, 0.00553817255],
            dtype=torch.float32,
        ).view(1, 1, 3)
        probability = probability.expand(1, 16, 3)
        expected = torch.nn.functional.cross_entropy(
            inputs["next_token_logits"].transpose(1, 2),
            probability.transpose(1, 2),
            reduction="none",
            label_smoothing=0.0,
        ).mean()
        self.assertTrue(torch.allclose(metric.compute(), expected))

    def test_trajtok_original_mode_requires_gt_idx(self):
        metric = self._metric(
            spatial_aware_smoothing=True,
            spatial_aware_smoothing_mode="trajtok_original",
        )
        inputs = self._inputs()
        inputs.pop("gt_idx")

        with self.assertRaisesRegex(ValueError, "gt_idx"):
            metric.update(**inputs)

    def test_unknown_spatial_mode_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "spatial_aware_smoothing_mode",
        ):
            self._metric(
                spatial_aware_smoothing=True,
                spatial_aware_smoothing_mode="unknown",
            )


class SpatialAwareConfigTest(unittest.TestCase):
    @classmethod
    def _load(cls, relative_path):
        with (ROOT / relative_path).open(encoding="utf-8") as file:
            return yaml.safe_load(file)

    def test_pre_bc_enables_spatial_smoothing(self):
        config = self._load("configs/experiment/pre_bc.yaml")
        self.assertTrue(
            config["model"]["model_config"]["training_loss"][
                "spatial_aware_smoothing"
            ]
        )

    def test_base_defaults_to_current_raw_gt_mode(self):
        config = self._load("configs/model/smart.yaml")
        self.assertEqual(
            config["model_config"]["training_loss"][
                "spatial_aware_smoothing_mode"
            ],
            "raw_gt_normalized",
        )

    def test_trajtok_original_experiment_only_overrides_mode(self):
        config = self._load(
            "configs/experiment/pre_bc_trajtok_original.yaml"
        )
        self.assertEqual(config["defaults"], ["pre_bc", "_self_"])
        self.assertEqual(
            config["model"]["model_config"]["training_loss"],
            {"spatial_aware_smoothing_mode": "trajtok_original"},
        )

    def test_trajtok_original_and_clsft_experiments_compose(self):
        try:
            import hydra
        except ModuleNotFoundError:
            self.skipTest("Hydra is not installed in this test environment")

        with hydra.initialize_config_dir(
            config_dir=str(ROOT / "configs"),
            version_base=None,
        ):
            pre_bc_config = hydra.compose(
                config_name="run.yaml",
                overrides=["experiment=pre_bc_trajtok_original"],
            )
            clsft_config = hydra.compose(
                config_name="run.yaml",
                overrides=["experiment=clsft"],
            )

        pre_bc_loss = pre_bc_config.model.model_config.training_loss
        self.assertTrue(pre_bc_loss.spatial_aware_smoothing)
        self.assertEqual(
            pre_bc_loss.spatial_aware_smoothing_mode,
            "trajtok_original",
        )
        self.assertEqual(pre_bc_loss.label_smoothing, 0.1)

        clsft_loss = clsft_config.model.model_config.training_loss
        self.assertFalse(clsft_loss.spatial_aware_smoothing)
        self.assertEqual(
            clsft_loss.spatial_aware_smoothing_mode,
            "raw_gt_normalized",
        )

    def test_history_dynamics_inherits_pre_bc(self):
        config = self._load(
            "configs/experiment/pre_bc_history_dynamics.yaml"
        )
        self.assertIn("pre_bc", config["defaults"])

    def test_base_and_clsft_keep_spatial_smoothing_disabled(self):
        base = self._load("configs/model/smart.yaml")
        clsft = self._load("configs/experiment/clsft.yaml")
        base_loss = base["model_config"]["training_loss"]
        clsft_loss = clsft["model"]["model_config"]["training_loss"]

        self.assertFalse(base_loss["spatial_aware_smoothing"])
        self.assertEqual(
            base_loss["spatial_aware_smoothing_mode"],
            "raw_gt_normalized",
        )
        self.assertFalse(clsft_loss["spatial_aware_smoothing"])
        self.assertNotIn("spatial_aware_smoothing_mode", clsft_loss)


if __name__ == "__main__":
    unittest.main()
