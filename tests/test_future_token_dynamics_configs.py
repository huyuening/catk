import unittest
from pathlib import Path

import yaml


class FutureTokenDynamicsConfigTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    RAW_EXPERIMENTS = {
        "pre_bc_history_future_token_dynamics": "pre_bc_history_dynamics",
        "clsft_history_future_token_dynamics": "clsft_history_dynamics",
        "inference_history_future_token_dynamics": "inference_history_dynamics",
    }
    RECONSTRUCTED_EXPERIMENTS = {
        f"{name}_reconstructed": name for name in RAW_EXPERIMENTS
    }

    @classmethod
    def _load(cls, relative_path):
        with (cls.ROOT / relative_path).open(encoding="utf-8") as file:
            return yaml.safe_load(file)

    @classmethod
    def _load_experiment(cls, name):
        return cls._load(f"configs/experiment/{name}.yaml")

    def test_base_model_has_complete_disabled_configuration(self):
        config = self._load("configs/model/smart.yaml")

        self.assertEqual(
            config["model_config"]["future_token_dynamics"],
            {
                "is_active": False,
                "lookup_file": None,
                "source": "raw",
                "normalization_scale": [5.0, 1.0, 5.0],
                "initial_gate": 1.0,
            },
        )

    def test_raw_experiments_enable_raw_training_lookup(self):
        for experiment, parent in self.RAW_EXPERIMENTS.items():
            with self.subTest(experiment=experiment):
                config = self._load_experiment(experiment)

                self.assertEqual(config["defaults"], [parent, "_self_"])
                model_config = config["model"]["model_config"]
                self.assertEqual(
                    model_config["future_token_dynamics"],
                    {
                        "is_active": True,
                        "lookup_file": None,
                        "source": "raw",
                    },
                )
                self.assertEqual(
                    model_config["token_processor"]["agent_token_file"],
                    "agent_vocab_555_s2.pkl",
                )

    def test_reconstructed_experiments_select_matching_source_and_vocabulary(self):
        for experiment, parent in self.RECONSTRUCTED_EXPERIMENTS.items():
            with self.subTest(experiment=experiment):
                config = self._load_experiment(experiment)

                self.assertEqual(config["defaults"], [parent, "_self_"])
                model_config = config["model"]["model_config"]
                self.assertEqual(
                    model_config["future_token_dynamics"],
                    {"source": "reconstructed"},
                )
                self.assertEqual(
                    model_config["token_processor"]["agent_token_file"],
                    "agent_vocab_reconstructed.pkl",
                )

    def test_all_experiments_compose_with_history_and_matching_provenance(self):
        try:
            import hydra
        except ModuleNotFoundError:
            self.skipTest("Hydra is not installed in this test environment")

        config_dir = self.ROOT / "configs"
        experiment_names = (
            list(self.RAW_EXPERIMENTS)
            + list(self.RECONSTRUCTED_EXPERIMENTS)
        )
        with hydra.initialize_config_dir(
            config_dir=str(config_dir),
            version_base=None,
        ):
            for experiment in experiment_names:
                with self.subTest(experiment=experiment):
                    config = hydra.compose(
                        config_name="run.yaml",
                        overrides=[
                            f"experiment={experiment}",
                            "model.model_config.future_token_dynamics.lookup_file=/tmp/example.pt",
                            "ckpt_path=/tmp/catk-placeholder.ckpt",
                        ],
                    )
                    model_config = config.model.model_config
                    reconstructed = experiment.endswith("_reconstructed")

                    self.assertTrue(model_config.history_dynamics.is_active)
                    self.assertTrue(
                        model_config.future_token_dynamics.is_active
                    )
                    self.assertEqual(
                        model_config.future_token_dynamics.lookup_file,
                        "/tmp/example.pt",
                    )
                    self.assertEqual(
                        model_config.future_token_dynamics.source,
                        "reconstructed" if reconstructed else "raw",
                    )
                    self.assertEqual(
                        model_config.token_processor.agent_token_file,
                        (
                            "agent_vocab_reconstructed.pkl"
                            if reconstructed
                            else "agent_vocab_555_s2.pkl"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
