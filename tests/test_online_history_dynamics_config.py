import unittest
from pathlib import Path

import yaml


class OnlineHistoryDynamicsConfigTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    @classmethod
    def _load(cls, relative_path):
        return yaml.safe_load(
            (cls.ROOT / relative_path).read_text(encoding="utf-8")
        )

    def test_base_configuration_defaults_to_cached_reconstruction(self):
        config = self._load("configs/model/smart.yaml")
        history = config["model_config"]["history_dynamics"]
        self.assertFalse(history["is_active"])
        self.assertEqual(history["mode"], "cached_reconstructed")

    def test_online_experiment_enables_only_online_history_mode(self):
        config = self._load(
            "configs/experiment/pre_bc_history_dynamics_online_raw.yaml"
        )
        self.assertEqual(config["defaults"], ["pre_bc", "_self_"])
        self.assertEqual(
            config["model"]["model_config"]["history_dynamics"],
            {"is_active": True, "mode": "online_raw"},
        )

    def test_existing_reconstructed_experiment_keeps_cached_mode(self):
        reconstructed = self._load(
            "configs/experiment/pre_bc_history_dynamics.yaml"
        )
        self.assertNotIn(
            "mode",
            reconstructed["model"]["model_config"]["history_dynamics"],
        )

    def test_online_experiment_composes_with_pre_bc_protocol(self):
        try:
            import hydra
        except ModuleNotFoundError:
            self.skipTest("Hydra is not installed in this test environment")
        with hydra.initialize_config_dir(
            config_dir=str(self.ROOT / "configs"), version_base=None
        ):
            config = hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "experiment=pre_bc_history_dynamics_online_raw"
                ],
            )
        self.assertTrue(
            config.model.model_config.history_dynamics.is_active
        )
        self.assertEqual(
            config.model.model_config.history_dynamics.mode, "online_raw"
        )
        self.assertEqual(config.trainer.max_epochs, 32)
        self.assertEqual(config.trainer.limit_val_batches, 0.1)
        self.assertEqual(config.model.model_config.wosac_backend, "fast")


if __name__ == "__main__":
    unittest.main()
