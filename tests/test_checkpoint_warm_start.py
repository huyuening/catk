import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "utils" / "checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("catk_checkpoint_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load module spec for {MODULE_PATH}")
CHECKPOINT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKPOINT
SPEC.loader.exec_module(CHECKPOINT)
load_warm_start_state_dict = CHECKPOINT.load_warm_start_state_dict


TEXT_PREFIX = "encoder.agent_encoder.text_control_adapter."


class TinyAgentBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.history_dynamics_emb = nn.Linear(3, 4)
        self.token_predict_head = nn.Linear(4, 2)


class TinyAgentWithText(TinyAgentBase):
    def __init__(self):
        super().__init__()
        self.text_control_adapter = nn.Module()
        self.text_control_adapter.projection = nn.Linear(4, 4)
        self.text_control_adapter.film = nn.Linear(4, 8)


class TinyEncoder(nn.Module):
    def __init__(self, *, with_text):
        super().__init__()
        self.map_encoder = nn.Linear(2, 4)
        self.agent_encoder = TinyAgentWithText() if with_text else TinyAgentBase()


class TinyModel(nn.Module):
    def __init__(self, *, with_text=True):
        super().__init__()
        self.encoder = TinyEncoder(with_text=with_text)


class WarmStartCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, payload, name="checkpoint.ckpt"):
        path = self.root / name
        torch.save(payload, path)
        return path

    def test_only_new_text_keys_may_be_missing(self):
        base = TinyModel(with_text=False)
        target = TinyModel(with_text=True)
        path = self.save({"state_dict": base.state_dict()})

        report = load_warm_start_state_dict(
            target,
            path,
            allowed_missing_prefixes=(TEXT_PREFIX,),
        )

        self.assertTrue(report.missing_keys)
        self.assertTrue(
            all(key.startswith(TEXT_PREFIX) for key in report.missing_keys)
        )
        self.assertFalse(report.unexpected_keys)
        torch.testing.assert_close(
            target.encoder.map_encoder.weight,
            base.encoder.map_encoder.weight,
        )

    def test_missing_history_dynamics_key_is_fatal(self):
        base = TinyModel(with_text=False)
        state = dict(base.state_dict())
        missing = "encoder.agent_encoder.history_dynamics_emb.weight"
        state.pop(missing)

        with self.assertRaisesRegex(RuntimeError, "history_dynamics_emb"):
            load_warm_start_state_dict(
                TinyModel(with_text=True),
                self.save({"state_dict": state}),
                allowed_missing_prefixes=(TEXT_PREFIX,),
            )

    def test_unexpected_checkpoint_key_is_always_fatal(self):
        state = dict(TinyModel(with_text=False).state_dict())
        state["encoder.agent_encoder.obsolete.weight"] = torch.zeros(1)

        with self.assertRaisesRegex(RuntimeError, "obsolete.weight"):
            load_warm_start_state_dict(
                TinyModel(with_text=True),
                self.save({"state_dict": state}),
                allowed_missing_prefixes=(TEXT_PREFIX,),
            )

    def test_warm_start_does_not_restore_training_state(self):
        checkpoint = {
            "state_dict": TinyModel(with_text=False).state_dict(),
            "epoch": 31,
            "global_step": 200000,
            "optimizer_states": [{"state": {"do-not-load": True}}],
            "lr_schedulers": [{"last_epoch": 31}],
        }

        report = load_warm_start_state_dict(
            TinyModel(with_text=True),
            self.save(checkpoint),
            allowed_missing_prefixes=(TEXT_PREFIX,),
        )

        self.assertEqual(report.loaded_epoch, 31)
        self.assertEqual(report.loaded_global_step, 200000)
        self.assertFalse(report.restored_trainer_state)

    def test_raw_state_dict_is_supported(self):
        base = TinyModel(with_text=False)
        target = TinyModel(with_text=True)

        report = load_warm_start_state_dict(
            target,
            self.save(base.state_dict(), "raw.pt"),
            allowed_missing_prefixes=(TEXT_PREFIX,),
        )

        self.assertIsNone(report.loaded_epoch)
        self.assertIsNone(report.loaded_global_step)
        torch.testing.assert_close(
            target.encoder.agent_encoder.token_predict_head.bias,
            base.encoder.agent_encoder.token_predict_head.bias,
        )

    def test_uniform_module_prefix_is_removed(self):
        base = TinyModel(with_text=False)
        prefixed = {
            f"module.{key}": value for key, value in base.state_dict().items()
        }
        target = TinyModel(with_text=True)

        load_warm_start_state_dict(
            target,
            self.save({"state_dict": prefixed}),
            allowed_missing_prefixes=(TEXT_PREFIX,),
        )

        torch.testing.assert_close(
            target.encoder.map_encoder.bias,
            base.encoder.map_encoder.bias,
        )

    def test_mixed_module_prefix_is_not_guessed(self):
        state = dict(TinyModel(with_text=False).state_dict())
        key = next(iter(state))
        state[f"module.{key}"] = state.pop(key)

        with self.assertRaisesRegex(RuntimeError, "module\."):
            load_warm_start_state_dict(
                TinyModel(with_text=True),
                self.save({"state_dict": state}),
                allowed_missing_prefixes=(TEXT_PREFIX,),
            )

    def test_non_finite_checkpoint_tensor_is_fatal(self):
        state = dict(TinyModel(with_text=False).state_dict())
        key = "encoder.map_encoder.weight"
        state[key] = state[key].clone()
        state[key][0, 0] = float("nan")

        with self.assertRaisesRegex(RuntimeError, "non-finite.*map_encoder"):
            load_warm_start_state_dict(
                TinyModel(with_text=True),
                self.save({"state_dict": state}),
                allowed_missing_prefixes=(TEXT_PREFIX,),
            )

    def test_same_name_shape_mismatch_is_fatal_before_loading(self):
        target = TinyModel(with_text=True)
        original = target.encoder.map_encoder.weight.detach().clone()
        state = dict(TinyModel(with_text=False).state_dict())
        state["encoder.map_encoder.weight"] = torch.zeros(5, 2)

        with self.assertRaisesRegex(RuntimeError, "shape.*map_encoder"):
            load_warm_start_state_dict(
                target,
                self.save({"state_dict": state}),
                allowed_missing_prefixes=(TEXT_PREFIX,),
            )
        torch.testing.assert_close(target.encoder.map_encoder.weight, original)

    def test_same_name_dtype_class_mismatch_is_fatal(self):
        state = dict(TinyModel(with_text=False).state_dict())
        key = "encoder.map_encoder.weight"
        state[key] = state[key].to(torch.int64)

        with self.assertRaisesRegex(RuntimeError, "dtype class.*map_encoder"):
            load_warm_start_state_dict(
                TinyModel(with_text=True),
                self.save({"state_dict": state}),
                allowed_missing_prefixes=(TEXT_PREFIX,),
            )

    def test_checkpoint_must_load_at_least_one_base_tensor(self):
        text_only = {
            key: value
            for key, value in TinyModel(with_text=True).state_dict().items()
            if key.startswith(TEXT_PREFIX)
        }

        with self.assertRaisesRegex(RuntimeError, "base.*tensor"):
            load_warm_start_state_dict(
                TinyModel(with_text=True),
                self.save({"state_dict": text_only}),
                allowed_missing_prefixes=(
                    "encoder.map_encoder.",
                    "encoder.agent_encoder.history_dynamics_emb.",
                    "encoder.agent_encoder.token_predict_head.",
                ),
            )

    def test_empty_allowed_prefix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            load_warm_start_state_dict(
                TinyModel(with_text=True),
                self.save(TinyModel(with_text=False).state_dict()),
                allowed_missing_prefixes=("",),
            )


if __name__ == "__main__":
    unittest.main()
