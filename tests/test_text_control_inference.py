import builtins
from contextlib import contextmanager
import io
from pathlib import Path
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.smart.inference.text_control import (
    TextControlInferenceRequest,
    build_single_agent_override,
    make_history_only_inference_view,
    run_text_control_inference,
)


@contextmanager
def deny_open_patterns(patterns):
    original = builtins.open

    def guarded(file, *args, **kwargs):
        lowered = str(file).lower()
        if any(pattern in lowered for pattern in patterns):
            raise AssertionError(f"forbidden file access: {file}")
        return original(file, *args, **kwargs)

    with patch("builtins.open", side_effect=guarded), patch(
        "io.open", side_effect=guarded
    ):
        yield


class FakeTokenProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, data):
        self.calls.append(data)
        agent = data["agent"]
        return {}, {
            "agent_ids": agent["id"],
            "history_last": agent["position"][:, 10, :2],
        }


class FakeEncoder:
    def __init__(self):
        self.encode_calls = 0
        self.inference_calls = 0
        self.unconditional_inference_calls = 0
        self.training_flags = []
        self.encoded_objects = []

    def encode_text_control(self, prompts, mask, device, *, training):
        self.encode_calls += 1
        self.training_flags.append(training)
        return SimpleNamespace(
            prompts=list(prompts),
            mask=mask.to(device),
        )

    def inference(
        self,
        tokenized_map,
        tokenized_agent,
        sampling_scheme,
        encoded_text_control=None,
    ):
        self.inference_calls += 1
        self.encoded_objects.append(encoded_text_control)
        if sampling_scheme.criterium != "topk_prob":
            raise AssertionError("fake inference received a future-aware sampler")
        n_agent = tokenized_agent["agent_ids"].shape[0]
        noise = torch.rand(n_agent, 80, 2)
        trajectory = tokenized_agent["history_last"].unsqueeze(1) + noise
        trajectory = trajectory.clone()
        trajectory[encoded_text_control.mask, :, 0] += 10.0
        return {
            "pred_traj_10hz": trajectory,
            "pred_z_10hz": torch.zeros(n_agent, 80),
            "pred_head_10hz": torch.zeros(n_agent, 80),
            "pred_idx": torch.full(
                (n_agent, 18),
                self.inference_calls,
                dtype=torch.long,
            ),
        }


class FakeModel(nn.Module):
    def __init__(self, criterium="topk_prob"):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.token_processor = FakeTokenProcessor()
        self.encoder = FakeEncoder()
        self.validation_rollout_sampling = SimpleNamespace(
            criterium=criterium,
            num_k=2,
            temp=1.0,
        )


class TextControlInferenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def scenario(future_value=0.0):
        n_agent = 3
        position = torch.zeros(n_agent, 91, 3)
        position[:, :11, 0] = torch.arange(11, dtype=torch.float32)
        position[:, 11:] = future_value
        heading = torch.zeros(n_agent, 91)
        heading[:, 11:] = future_value
        velocity = torch.ones(n_agent, 91, 2)
        velocity[:, 11:] = future_value
        valid = torch.ones(n_agent, 91, dtype=torch.bool)
        valid[1, 11:] = False
        return {
            "scenario_id": "scenario-inference-test",
            "agent": {
                "id": torch.tensor([10, 20, 30]),
                "type": torch.zeros(n_agent, dtype=torch.long),
                "role": torch.zeros(n_agent, 3, dtype=torch.bool),
                "shape": torch.ones(n_agent, 3),
                "position": position,
                "heading": heading,
                "velocity": velocity,
                "valid_mask": valid,
                "future_acceleration": torch.full(
                    (n_agent, 91), future_value
                ),
                "text_prompt": ["future-derived", "", ""],
                "text_prompt_mask": torch.tensor([True, False, False]),
            },
            "map_save": {"traj_pos": torch.zeros(1, 3, 2), "traj_theta": torch.zeros(1)},
            "pt_token": {
                "type": torch.zeros(1, dtype=torch.long),
                "pl_type": torch.zeros(1, dtype=torch.long),
                "light_type": torch.zeros(1, dtype=torch.long),
            },
        }

    def save_scenario(self, scenario, name="scenario.pkl"):
        path = self.root / name
        with path.open("wb") as handle:
            pickle.dump(scenario, handle)
        return path

    def request(self, scenario_path, *, output="result", n_rollouts=4, seed=7, prompt=None):
        return TextControlInferenceRequest(
            checkpoint=self.root / "unused-model.ckpt",
            scenario_pickle=scenario_path,
            target_agent_id=20,
            prompt=prompt or "The target vehicle is changing lanes left.",
            output_dir=self.root / output,
            n_rollouts=n_rollouts,
            seed=seed,
        )

    def test_override_assigns_prompt_to_only_requested_agent(self):
        prompt = "The target vehicle is changing lanes left."
        prompts, mask = build_single_agent_override(
            agent_ids=torch.tensor([10, 20, 30]),
            target_agent_id=20,
            prompt=prompt,
        )

        self.assertEqual(prompts, ["", prompt, ""])
        torch.testing.assert_close(mask, torch.tensor([False, True, False]))

    def test_missing_or_duplicate_target_id_is_fatal(self):
        for ids, target in ((torch.tensor([10, 10]), 10), (torch.tensor([10, 20]), 30)):
            with self.subTest(ids=ids.tolist(), target=target):
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    build_single_agent_override(ids, target, "accelerate")

    def test_empty_prompt_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_single_agent_override(torch.tensor([10]), 10, "   ")

    def test_history_view_erases_future_without_mutating_source(self):
        source = self.scenario(future_value=99.0)
        original_future = source["agent"]["position"][:, 11:].clone()

        view = make_history_only_inference_view(source)

        self.assertTrue(torch.equal(source["agent"]["position"][:, 11:], original_future))
        self.assertFalse(view["agent"]["position"][:, 11:].any())
        self.assertFalse(view["agent"]["heading"][:, 11:].any())
        self.assertFalse(view["agent"]["velocity"][:, 11:].any())
        self.assertFalse(view["agent"]["future_acceleration"][:, 11:].any())
        expected_valid = view["agent"]["valid_mask"][:, 10:11].expand(-1, 80)
        torch.testing.assert_close(
            view["agent"]["valid_mask"][:, 11:],
            expected_valid,
        )
        self.assertNotIn("text_prompt", view["agent"])
        self.assertNotIn("text_prompt_mask", view["agent"])

    def test_custom_inference_never_opens_tag_or_future_gt_files(self):
        scenario_path = self.save_scenario(self.scenario())
        with deny_open_patterns(("tag", "mapping", "validation_gt", "future_gt")):
            result = run_text_control_inference(
                self.request(scenario_path),
                model=FakeModel(),
            )
        self.assertEqual(tuple(result.trajectories.shape[-2:]), (80, 2))

    def test_runtime_output_is_invariant_to_hidden_future_contents(self):
        first_path = self.save_scenario(self.scenario(0.0), "first.pkl")
        second_path = self.save_scenario(self.scenario(1e6), "second.pkl")

        first = run_text_control_inference(
            self.request(first_path, output="first"),
            model=FakeModel(),
        )
        second = run_text_control_inference(
            self.request(second_path, output="second"),
            model=FakeModel(),
        )

        torch.testing.assert_close(first.trajectories, second.trajectories, rtol=0, atol=0)

    def test_one_conditional_decoder_call_per_rollout_and_one_encoding(self):
        scenario_path = self.save_scenario(self.scenario())
        model = FakeModel()

        result = run_text_control_inference(
            self.request(scenario_path, n_rollouts=32),
            model=model,
        )

        self.assertEqual(model.encoder.inference_calls, 32)
        self.assertEqual(model.encoder.unconditional_inference_calls, 0)
        self.assertEqual(model.encoder.encode_calls, 1)
        self.assertEqual(model.encoder.training_flags, [False])
        self.assertTrue(
            all(value is model.encoder.encoded_objects[0] for value in model.encoder.encoded_objects)
        )
        self.assertEqual(tuple(result.trajectories.shape), (3, 32, 80, 2))

    def test_fixed_seed_is_repeatable_and_preserves_every_agent(self):
        scenario_path = self.save_scenario(self.scenario())
        first = run_text_control_inference(
            self.request(scenario_path, output="repeat-a", seed=19),
            model=FakeModel(),
        )
        second = run_text_control_inference(
            self.request(scenario_path, output="repeat-b", seed=19),
            model=FakeModel(),
        )

        torch.testing.assert_close(first.trajectories, second.trajectories, rtol=0, atol=0)
        torch.testing.assert_close(first.agent_ids, torch.tensor([10, 20, 30]))
        self.assertEqual(first.trajectories.shape[0], 3)
        self.assertTrue((self.root / "repeat-a" / "rollouts.pt").is_file())
        self.assertTrue((self.root / "repeat-a" / "request.json").is_file())

    def test_future_aware_sampling_is_rejected_before_decoder_call(self):
        scenario_path = self.save_scenario(self.scenario())
        model = FakeModel(criterium="topk_prob_sampled_with_dist")

        with self.assertRaisesRegex(ValueError, "topk_prob"):
            run_text_control_inference(
                self.request(scenario_path),
                model=model,
            )
        self.assertEqual(model.encoder.inference_calls, 0)


if __name__ == "__main__":
    unittest.main()
