import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = (
    "/mnt/pfs/waymo_motion_1_3_0/"
    "preprocessed_scenario/validation_gt"
)


class TrainingFastWOSACConfigTest(unittest.TestCase):
    def _load(self, relative_path):
        return yaml.safe_load(
            (ROOT / relative_path).read_text(encoding="utf-8")
        )

    def _assert_raw_protocol(self, name):
        config = self._load(f"configs/experiment/{name}.yaml")
        trainer = config["trainer"]
        model = config["model"]["model_config"]
        sampling = model["validation_rollout_sampling"]
        self.assertEqual(trainer["limit_val_batches"], 0.1)
        self.assertEqual(trainer["check_val_every_n_epoch"], 1)
        self.assertTrue(model["val_open_loop"])
        self.assertTrue(model["val_closed_loop"])
        self.assertEqual(model["wosac_backend"], "fast")
        self.assertEqual(model["wosac_metrics_version"], "2025")
        self.assertTrue(model["fast_wosac_require_preprocessed_gt"])
        self.assertEqual(model["n_batch_wosac_metric"], -1)
        self.assertEqual(model["n_rollout_closed_val"], 32)
        self.assertEqual(sampling["criterium"], "topk_prob")
        self.assertEqual(sampling["num_k"], 48)
        self.assertEqual(sampling["temp"], 1.0)
        self.assertEqual(
            model["fast_wosac_gt_dir"],
            "${paths.validation_gt_dir}",
        )

    def test_pre_bc_protocol(self):
        self._assert_raw_protocol("pre_bc")

    def test_clsft_protocol(self):
        self._assert_raw_protocol("clsft")

    def test_shared_path_uses_environment_with_deployment_default(self):
        config = self._load("configs/paths/default.yaml")
        self.assertEqual(
            config["validation_gt_dir"],
            "${oc.env:FAST_WOSAC_GT_DIR," + DEFAULT_GT + "}",
        )

    def test_derived_experiments_inherit_protocol(self):
        try:
            import hydra
        except ModuleNotFoundError:
            self.skipTest("Hydra is not installed")
        with hydra.initialize_config_dir(
            config_dir=str(ROOT / "configs"),
            version_base=None,
        ):
            for experiment in (
                "pre_bc_history_dynamics",
                "pre_bc_trajtok_original",
                "clsft_history_dynamics",
            ):
                config = hydra.compose(
                    config_name="run.yaml",
                    overrides=[f"experiment={experiment}"],
                )
                self.assertEqual(config.trainer.limit_val_batches, 0.1)
                self.assertEqual(
                    config.model.model_config.wosac_backend,
                    "fast",
                )
                self.assertTrue(
                    config.model.model_config
                    .fast_wosac_require_preprocessed_gt
                )
                self.assertEqual(
                    config.model.model_config
                    .validation_rollout_sampling.num_k,
                    48,
                )

    def test_environment_overrides_gt_directory(self):
        try:
            import hydra
        except ModuleNotFoundError:
            self.skipTest("Hydra is not installed")
        with patch.dict(
            os.environ,
            {"FAST_WOSAC_GT_DIR": "/tmp/custom-validation-gt"},
        ):
            with hydra.initialize_config_dir(
                config_dir=str(ROOT / "configs"),
                version_base=None,
            ):
                config = hydra.compose(
                    config_name="run.yaml",
                    overrides=["experiment=pre_bc"],
                )
            self.assertEqual(
                config.paths.validation_gt_dir,
                "/tmp/custom-validation-gt",
            )


if __name__ == "__main__":
    unittest.main()
