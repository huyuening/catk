from pathlib import Path
import pickle
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import yaml

from src.smart.inference.text_control import (
    TextControlInferenceRequest,
    run_text_control_inference,
)
from src.smart.modules.text_control import EncodedTextControl
from tests.test_checkpoint_warm_start import (
    TEXT_PREFIX,
    TinyModel,
    load_warm_start_state_dict,
)
import tests.test_text_control_decoder as decoder_fixtures
from tests.test_text_control_inference import FakeModel
from tests.test_text_control_training import SMART, make_data, model_config


ROOT = Path(__file__).resolve().parents[1]


class TextControlIntegrationTest(unittest.TestCase):
    def test_pre_bc_warm_start_and_resolved_experiment_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pre-bc.ckpt"
            torch.save(
                {"state_dict": TinyModel(with_text=False).state_dict()},
                checkpoint,
            )
            report = load_warm_start_state_dict(
                TinyModel(with_text=True),
                checkpoint,
                allowed_missing_prefixes=(TEXT_PREFIX,),
            )

        self.assertTrue(report.missing_keys)
        self.assertTrue(all(key.startswith(TEXT_PREFIX) for key in report.missing_keys))
        self.assertFalse(report.unexpected_keys)
        self.assertFalse(report.restored_trainer_state)

        experiment = yaml.safe_load(
            (ROOT / "configs/experiment/text_control_pre_bc.yaml").read_text(
                encoding="utf-8"
            )
        )
        model = experiment["model"]["model_config"]
        self.assertTrue(model["history_dynamics"]["is_active"])
        self.assertTrue(model["training_loss"]["spatial_aware_smoothing"])
        self.assertEqual(
            model["training_loss"]["spatial_aware_smoothing_mode"],
            "trajtok_original",
        )
        self.assertTrue(model["text_control"]["is_active"])
        self.assertFalse(model["finetune"])
        self.assertFalse(
            {"use_cfg", "cfg_scale", "per_step"}
            & set(model["text_control"])
        )

    @staticmethod
    def _install_fake_rollout_embedding(decoder):
        def fake_embedding(**kwargs):
            n_agent, n_step = kwargs["agent_token_index"].shape
            feat = torch.zeros(n_agent, n_step, 4)
            token_emb = torch.zeros_like(feat)
            vocabulary = torch.zeros(2, 4)
            vehicle = kwargs["agent_type"] == 0
            categorical = [torch.zeros(n_agent, 4), torch.zeros(n_agent, 4)]
            return (
                feat,
                token_emb,
                vocabulary,
                vocabulary,
                vocabulary,
                vehicle,
                ~vehicle,
                torch.zeros_like(vehicle),
                categorical,
            )

        decoder.agent_token_embedding = fake_embedding

    def test_no_prompt_logits_and_eight_second_rollout_are_bitwise_equal(self):
        case = decoder_fixtures.TextControlDecoderTest()
        decoder = case._decoder(num_layers=2, num_future_steps=80)
        adapter = decoder_fixtures._RecordingAdapter(4, 2, delta=5.0)
        decoder.text_control_adapter = adapter
        decoder.t_attn_layers = nn.ModuleList(
            [decoder_fixtures._CacheSpyIdentity(), decoder_fixtures._CacheSpyIdentity()]
        )
        decoder.pt2a_attn_layers = nn.ModuleList(
            [decoder_fixtures._EventIdentity("map", []) for _ in range(2)]
        )
        decoder.a2a_attn_layers = nn.ModuleList(
            [decoder_fixtures._EventIdentity("agent", []) for _ in range(2)]
        )
        decoder.build_temporal_edge = lambda **kwargs: case._empty_edges()
        decoder.build_map2agent_edge = lambda **kwargs: case._empty_edges()
        decoder.build_interaction_edge = lambda **kwargs: case._empty_edges()
        decoder.token_predict_head = nn.Linear(4, 2, bias=False)
        self._install_fake_rollout_embedding(decoder)
        decoder.eval()
        tokenized, map_feature = case._rollout_inputs()
        tokenized["gt_z_raw"] = torch.zeros(1)

        with patch.object(
            decoder_fixtures.agent_decoder_module,
            "sample_next_token_traj",
            side_effect=case._fixed_sample,
        ):
            baseline = decoder.inference(
                tokenized,
                map_feature,
                SimpleNamespace(),
                encoded_text_control=None,
            )
            all_auto = decoder.inference(
                tokenized,
                map_feature,
                SimpleNamespace(),
                encoded_text_control=EncodedTextControl(
                    features=torch.zeros(1, 4),
                    mask=torch.tensor([False]),
                ),
            )

        torch.testing.assert_close(
            baseline["next_token_logits"],
            all_auto["next_token_logits"],
            rtol=0,
            atol=0,
        )
        self.assertEqual(tuple(baseline["pred_traj_10hz"].shape[-2:]), (80, 2))
        torch.testing.assert_close(
            baseline["pred_traj_10hz"],
            all_auto["pred_traj_10hz"],
            rtol=0,
            atol=0,
        )

    def test_selected_prompt_changes_target_and_auto_reacts_only_via_attention(self):
        case = decoder_fixtures.TextControlDecoderTest()
        decoder = case._decoder(num_layers=6)
        adapter = decoder_fixtures._RecordingAdapter(4, 6, delta=3.0)
        case._prepare_forward_harness(decoder, adapter, reaction=True)
        tokenized, map_feature = case._forward_inputs()

        baseline = decoder(tokenized, map_feature, encoded_text_control=None)
        controlled = decoder(
            tokenized,
            map_feature,
            encoded_text_control=EncodedTextControl(
                features=torch.zeros(2, 4),
                mask=torch.tensor([True, False]),
            ),
        )

        self.assertFalse(
            torch.equal(
                baseline["next_token_logits"][0],
                controlled["next_token_logits"][0],
            )
        )
        self.assertTrue(all(adapter.unmasked_were_unchanged))
        self.assertFalse(
            torch.equal(
                baseline["next_token_logits"][1],
                controlled["next_token_logits"][1],
            )
        )

    def test_gradients_and_optimizer_step_leave_every_pre_bc_tensor_unchanged(self):
        model = SMART(model_config())
        model.train()
        frozen_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
        loss = model.training_step(make_data(), batch_idx=0)
        loss.backward()

        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(trainable_names)
        for name in trainable_names:
            self.assertTrue(
                "lora_A" in name
                or "lora_B" in name
                or "projection" in name
                or "film_layers" in name,
                name,
            )
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertIsNone(parameter.grad, name)

        optimizers, _ = model.configure_optimizers()
        optimizers[0].step()
        for name, expected in frozen_before.items():
            actual = dict(model.named_parameters())[name]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_validation_and_counterfactual_rollouts_do_not_double_decode(self):
        validation_model = SMART(model_config(closed=True))
        validation_model.n_rollout_closed_val = 32
        validation_model.eval()
        validation_model.validation_step(make_data(), batch_idx=0)
        self.assertEqual(validation_model.encoder.text_encode_calls, 1)
        self.assertEqual(len(validation_model.encoder.encoded_seen), 33)
        encoded = validation_model.encoder.encoded_seen[0]
        self.assertTrue(
            all(value is encoded for value in validation_model.encoder.encoded_seen)
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario_path = root / "scenario.pkl"
            n_agent = 3
            scenario = {
                "scenario_id": "integration-scene",
                "agent": {
                    "id": torch.tensor([10, 20, 30]),
                    "position": torch.zeros(n_agent, 91, 3),
                    "heading": torch.zeros(n_agent, 91),
                    "velocity": torch.zeros(n_agent, 91, 2),
                    "valid_mask": torch.ones(n_agent, 91, dtype=torch.bool),
                },
                "map_save": {
                    "traj_pos": torch.zeros(1, 3, 2),
                    "traj_theta": torch.zeros(1),
                },
                "pt_token": {
                    "type": torch.zeros(1, dtype=torch.long),
                    "pl_type": torch.zeros(1, dtype=torch.long),
                    "light_type": torch.zeros(1, dtype=torch.long),
                },
            }
            with scenario_path.open("wb") as handle:
                pickle.dump(scenario, handle)
            inference_model = FakeModel()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_text_control_inference(
                    TextControlInferenceRequest(
                        checkpoint=root / "unused.ckpt",
                        scenario_pickle=scenario_path,
                        target_agent_id=20,
                        prompt="The target vehicle is accelerating.",
                        output_dir=root / "output",
                        n_rollouts=32,
                        seed=5,
                    ),
                    model=inference_model,
                )

        self.assertEqual(inference_model.encoder.encode_calls, 1)
        self.assertEqual(inference_model.encoder.inference_calls, 32)
        self.assertEqual(inference_model.encoder.unconditional_inference_calls, 0)

        simulated_states = torch.cat(
            [
                result.trajectories,
                result.z.unsqueeze(-1),
                result.headings.unsqueeze(-1),
            ],
            dim=-1,
        ).permute(1, 0, 2, 3).contiguous()
        self.assertEqual(tuple(simulated_states.shape), (32, 3, 80, 4))


if __name__ == "__main__":
    unittest.main()
