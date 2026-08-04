import inspect
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]


class TextControlConfigTest(unittest.TestCase):
    @staticmethod
    def load(relative_path):
        return yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))

    def test_base_model_has_complete_disabled_text_control_defaults(self):
        model = self.load("configs/model/smart.yaml")["model_config"]

        self.assertEqual(
            model["text_control"],
            {
                "is_active": False,
                "freeze_base": True,
                "model_name_or_path": "distilbert-base-uncased",
                "local_files_only": False,
                "max_length": 384,
                "mean_pool": False,
                "output_dim": 256,
                "lora_rank": 16,
                "lora_alpha": 0.4,
                "lora_dropout": 0.05,
                "lora_last_n_layers": 6,
                "film_blocks": [0, 1, 2, 3, 4, 5],
                "film_hidden_dim": 256,
                "film_dropout": 0.0,
                "film_init_mode": "identity",
                "film_identity_noise_std": 0.001,
                "control_dropout": 0.3,
            },
        )

    def test_baseline_model_loss_and_data_defaults_are_preserved(self):
        model = self.load("configs/model/smart.yaml")["model_config"]
        data = self.load("configs/data/waymo.yaml")

        self.assertEqual(model["token_processor"]["agent_token_file"], "agent_vocab_555_s2.pkl")
        self.assertFalse(model["history_dynamics"]["is_active"])
        self.assertFalse(model["future_token_dynamics"]["is_active"])
        self.assertEqual(model["training_loss"]["label_smoothing"], 0.1)
        self.assertFalse(model["training_loss"]["spatial_aware_smoothing"])
        self.assertEqual(
            model["training_loss"]["spatial_aware_smoothing_mode"],
            "raw_gt_normalized",
        )
        self.assertEqual(data["train_batch_size"], 4)
        self.assertIsNone(data["text_prompt_root"])

    def test_experiment_declares_exact_pre_bc_text_training_overrides(self):
        config = self.load("configs/experiment/text_control_pre_bc.yaml")
        model = config["model"]["model_config"]

        self.assertEqual(config["defaults"], ["pre_bc_history_dynamics", "_self_"])
        self.assertEqual(config["action"], "finetune")
        self.assertEqual(model["lr"], 5e-5)
        self.assertFalse(model["finetune"])
        self.assertTrue(model["history_dynamics"]["is_active"])
        self.assertFalse(model["training_loss"]["spatial_aware_smoothing"])
        self.assertEqual(model["training_loss"]["label_smoothing"], 0.0)
        self.assertTrue(model["text_control"]["is_active"])
        self.assertTrue(model["text_control"]["freeze_base"])
        self.assertEqual(
            model["text_control"]["model_name_or_path"],
            "${oc.env:TEXT_MODEL_PATH,distilbert-base-uncased}",
        )
        self.assertNotIn("cfg", model["text_control"])
        for forbidden in ("use_cfg", "cfg_scale", "per_step", "sketch", "latent"):
            self.assertNotIn(forbidden, model["text_control"])

        sampling = model["training_rollout_sampling"]
        self.assertEqual(sampling["criterium"], "topk_prob_sampled_with_dist")
        self.assertEqual(sampling["num_k"], 32)
        self.assertEqual(sampling["temp"], 1e-5)
        self.assertEqual(model["n_rollout_closed_val"], 32)
        self.assertEqual(model["n_batch_wosac_metric"], -1)
        self.assertEqual(model["wosac_backend"], "fast")
        self.assertEqual(model["wosac_metrics_version"], "2025")

        self.assertEqual(config["trainer"]["max_epochs"], 10)
        self.assertEqual(config["trainer"]["limit_val_batches"], 0.1)
        self.assertEqual(config["trainer"]["check_val_every_n_epoch"], 1)
        self.assertEqual(config["data"]["train_batch_size"], 4)
        self.assertEqual(config["data"]["val_batch_size"], 4)
        self.assertEqual(config["data"]["test_batch_size"], 4)
        self.assertEqual(
            config["data"]["text_prompt_root"],
            "${oc.env:TEXT_PROMPT_ROOT}",
        )
        self.assertEqual(config["logger"]["wandb"]["resume"], "never")
        self.assertIsNone(config["logger"]["wandb"]["id"])

    def test_resolved_hydra_config_matches_training_contract(self):
        try:
            import hydra
        except ModuleNotFoundError:
            self.skipTest("Hydra is not installed in this test environment")

        environment = {
            "TEXT_PROMPT_ROOT": "/tmp/text-control-tags",
            "TEXT_MODEL_PATH": "/tmp/distilbert",
            "PRE_BC_CKPT": "/tmp/pre-bc.ckpt",
        }
        initialize_options = {"config_dir": str(ROOT / "configs")}
        if "version_base" in inspect.signature(
            hydra.initialize_config_dir
        ).parameters:
            initialize_options["version_base"] = None
        with patch.dict(os.environ, environment):
            with hydra.initialize_config_dir(**initialize_options):
                config = hydra.compose(
                    config_name="run.yaml",
                    overrides=[
                        "experiment=text_control_pre_bc",
                        "hydra/hydra_logging=default",
                        "hydra/job_logging=default",
                    ],
                )

        model = config.model.model_config
        self.assertTrue(model.history_dynamics.is_active)
        self.assertFalse(model.training_loss.spatial_aware_smoothing)
        self.assertEqual(model.training_loss.label_smoothing, 0.0)
        self.assertTrue(model.text_control.is_active)
        self.assertTrue(model.text_control.freeze_base)
        self.assertFalse(model.finetune)
        self.assertEqual(model.training_rollout_sampling.num_k, 32)
        self.assertEqual(model.training_rollout_sampling.temp, 1e-5)
        self.assertEqual(model.n_rollout_closed_val, 32)
        self.assertEqual(model.wosac_metrics_version, "2025")
        self.assertEqual(model.wosac_backend, "fast")
        self.assertEqual(config.data.train_batch_size, 4)
        self.assertEqual(config.trainer.max_epochs, 10)
        self.assertEqual(config.trainer.limit_val_batches, 0.1)
        self.assertEqual(config.action, "finetune")
        self.assertNotIn("cfg", model.text_control)
        self.assertEqual(config.data.text_prompt_root, "/tmp/text-control-tags")
        self.assertEqual(
            config.data.train_text_mapping_path,
            "/tmp/text-control-tags/train_scenario_mapping.json",
        )

    def test_launch_script_is_syntax_valid_and_never_resumes_wandb(self):
        script = ROOT / "scripts" / "train_text_control_pre_bc.sh"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        text = script.read_text(encoding="utf-8")
        self.assertIn("unset WANDB_RUN_ID WANDB_RESUME", text)
        self.assertIn("logger.wandb.resume=never", text)
        self.assertIn("logger.wandb.id=null", text)
        self.assertIn("PRE_BC_CKPT", text)
        self.assertIn("TEXT_PROMPT_ROOT", text)
        self.assertIn(
            "/root/workspace/catk/logs/pre_bc_history_dynamics_hard_ce_b200/"
            "runs/2026-07-30_21-15-08/checkpoints/last.ckpt",
            text,
        )
        self.assertIn("text_control_pre_bc_history_dynamics_hard_ce", text)
        self.assertNotIn("trajtok_original", text)


if __name__ == "__main__":
    unittest.main()
