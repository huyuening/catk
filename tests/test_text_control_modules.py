import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch
from torch import nn


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "smart"
    / "modules"
    / "text_control.py"
)
SPEC = importlib.util.spec_from_file_location("catk_text_control", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load module spec for {MODULE_PATH}")
TEXT_CONTROL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TEXT_CONTROL
SPEC.loader.exec_module(TEXT_CONTROL)


class FakeAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_lin = nn.Linear(hidden_size, hidden_size)
        self.k_lin = nn.Linear(hidden_size, hidden_size)
        self.v_lin = nn.Linear(hidden_size, hidden_size)
        self.out_lin = nn.Linear(hidden_size, hidden_size)

    def forward(self, value):
        mixed = self.q_lin(value) + self.k_lin(value) + self.v_lin(value)
        return self.out_lin(torch.tanh(mixed))


class FakeTransformerLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = FakeAttention(hidden_size)

    def forward(self, value):
        return value + self.attention(value)


class FakeTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layer = nn.ModuleList(
            [FakeTransformerLayer(hidden_size) for _ in range(num_layers)]
        )


class FakeBackbone(nn.Module):
    def __init__(self, hidden_size=12, num_layers=8):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.embeddings = nn.Embedding(64, hidden_size)
        self.transformer = FakeTransformer(hidden_size, num_layers)

    def forward(self, input_ids, attention_mask=None):
        value = self.embeddings(input_ids)
        for layer in self.transformer.layer:
            value = layer(value)
        return types.SimpleNamespace(last_hidden_state=value)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        texts,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        self.calls.append(list(texts))
        width = 4
        input_ids = torch.zeros((len(texts), width), dtype=torch.long)
        for row, text in enumerate(texts):
            input_ids[row] = torch.tensor(
                [len(text) % 63, 2, 3, 4], dtype=torch.long
            )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class FakeTransformers:
    def __init__(self, backbone=None, tokenizer=None):
        self.backbone = backbone or FakeBackbone()
        self.tokenizer = tokenizer or FakeTokenizer()
        self.model_calls = []
        self.tokenizer_calls = []

        owner = self

        class AutoModel:
            @staticmethod
            def from_pretrained(path, **kwargs):
                owner.model_calls.append((path, dict(kwargs)))
                return owner.backbone

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                owner.tokenizer_calls.append((path, dict(kwargs)))
                return owner.tokenizer

        self.module = types.ModuleType("transformers")
        self.module.AutoModel = AutoModel
        self.module.AutoTokenizer = AutoTokenizer


class TransformersContext:
    def __init__(self, fake):
        self.fake = fake
        self.previous = None

    def __enter__(self):
        self.previous = sys.modules.get("transformers")
        sys.modules["transformers"] = self.fake.module
        return self.fake

    def __exit__(self, exc_type, exc_value, traceback):
        if self.previous is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = self.previous


def adapter_config(**overrides):
    config = {
        "model_name_or_path": "fake-distilbert",
        "local_files_only": True,
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
    }
    config.update(overrides)
    return config


class LoRALinearTest(unittest.TestCase):
    def test_zero_delta_preserves_pretrained_linear_output(self):
        base = nn.Linear(7, 5)
        original = copy.deepcopy(base)
        wrapped = TEXT_CONTROL.LoRALinear.from_linear(
            base, rank=2, alpha=0.4, dropout=0.0
        )
        value = torch.randn(4, 7)

        torch.testing.assert_close(wrapped(value), original(value), rtol=0, atol=0)
        self.assertFalse(wrapped.base.weight.requires_grad)
        self.assertTrue(wrapped.lora_A.weight.requires_grad)
        self.assertTrue(wrapped.lora_B.weight.requires_grad)

    def test_only_last_six_attention_layers_receive_lora(self):
        backbone = FakeBackbone(num_layers=8)

        names = TEXT_CONTROL.install_distilbert_attention_lora(
            backbone,
            rank=16,
            alpha=0.4,
            dropout=0.05,
            last_n_layers=6,
        )

        self.assertEqual(len(names), 24)
        self.assertIsInstance(
            backbone.transformer.layer[2].attention.q_lin,
            TEXT_CONTROL.LoRALinear,
        )
        self.assertIsInstance(
            backbone.transformer.layer[7].attention.out_lin,
            TEXT_CONTROL.LoRALinear,
        )
        self.assertIsInstance(
            backbone.transformer.layer[1].attention.q_lin,
            nn.Linear,
        )

    def test_missing_requested_transformer_layers_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "last_n_layers"):
            TEXT_CONTROL.install_distilbert_attention_lora(
                FakeBackbone(num_layers=4),
                rank=2,
                alpha=1.0,
                dropout=0.0,
                last_n_layers=6,
            )


class FiLMLayerTest(unittest.TestCase):
    def make_layer(self):
        return TEXT_CONTROL.FiLMLayer(
            feature_dim=4,
            conditioning_dim=8,
            hidden_dim=8,
            dropout=0.0,
            init_mode="identity",
            identity_noise_std=0.0,
        )

    def set_constant_affine(self, layer, gamma, beta):
        with torch.no_grad():
            layer.film_mlp[-1].weight.zero_()
            layer.film_mlp[-1].bias[:4].fill_(gamma)
            layer.film_mlp[-1].bias[4:].fill_(beta)

    def test_exact_identity_affine_preserves_rank_three_features(self):
        layer = self.make_layer()
        self.set_constant_affine(layer, gamma=1.0, beta=0.0)
        value = torch.randn(3, 5, 4)
        condition = torch.randn(3, 8)
        mask = torch.tensor([True, False, True])

        output = layer(value, condition, mask)

        torch.testing.assert_close(output, value, rtol=0, atol=0)

    def test_mask_never_directly_changes_auto_agents(self):
        layer = self.make_layer()
        self.set_constant_affine(layer, gamma=2.0, beta=1.0)
        value = torch.randn(3, 4)
        mask = torch.tensor([False, True, False])

        output = layer(value, torch.randn(3, 8), mask)

        torch.testing.assert_close(output[[0, 2]], value[[0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(output[1], value[1] * 2.0 + 1.0)

    def test_shape_and_mask_contract_is_validated(self):
        layer = self.make_layer()
        with self.assertRaisesRegex(ValueError, "rank 2 or 3"):
            layer(torch.randn(2, 3, 4, 5), torch.randn(2, 8), torch.ones(2).bool())
        with self.assertRaisesRegex(ValueError, "Boolean"):
            layer(torch.randn(2, 4), torch.randn(2, 8), torch.ones(2))
        with self.assertRaisesRegex(ValueError, "agent"):
            layer(
                torch.randn(2, 4),
                torch.randn(3, 8),
                torch.ones(2, dtype=torch.bool),
            )


class TextControlAdapterTest(unittest.TestCase):
    def make_adapter(self, fake=None, **config_overrides):
        fake = fake or FakeTransformers()
        context = TransformersContext(fake)
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        adapter = TEXT_CONTROL.TextControlAdapter(
            hidden_dim=128,
            num_blocks=6,
            config=adapter_config(**config_overrides),
        )
        return adapter, fake

    def test_encode_returns_one_static_feature_per_agent(self):
        adapter, fake = self.make_adapter()

        encoded = adapter.encode(
            ["turn left", "", "accelerate"],
            torch.tensor([True, False, True]),
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(encoded.features.shape), (3, 256))
        torch.testing.assert_close(
            encoded.mask, torch.tensor([True, False, True])
        )
        torch.testing.assert_close(
            encoded.features[1], torch.zeros(256), rtol=0, atol=0
        )
        self.assertEqual(fake.tokenizer.calls, [["turn left", "accelerate"]])

    def test_all_false_mask_skips_tokenizer_and_returns_none(self):
        adapter, fake = self.make_adapter()

        encoded = adapter.encode(
            ["", ""],
            torch.zeros(2, dtype=torch.bool),
            device=torch.device("cpu"),
        )

        self.assertIsNone(encoded)
        self.assertEqual(fake.tokenizer.calls, [])

    def test_control_dropout_is_sampled_once_before_all_blocks(self):
        adapter, _ = self.make_adapter(control_dropout=1.0)

        encoded = adapter.encode(
            ["turn left", "accelerate"],
            torch.tensor([True, True]),
            device=torch.device("cpu"),
            apply_control_dropout=True,
        )

        self.assertIsNone(encoded)

    def test_condition_validates_block_index_and_keeps_none_exact(self):
        adapter, _ = self.make_adapter()
        value = torch.randn(2, 3, 128)
        self.assertIs(adapter.condition(value, None, block_index=0), value)
        with self.assertRaisesRegex(IndexError, "block"):
            adapter.condition(value, None, block_index=6)

    def test_trainable_whitelist_excludes_frozen_text_backbone(self):
        adapter, _ = self.make_adapter()

        names = set(adapter.unfreeze_control_parameters())

        self.assertTrue(any("lora_A" in name for name in names))
        self.assertTrue(any("lora_B" in name for name in names))
        self.assertTrue(any("projection" in name for name in names))
        self.assertTrue(any("film_layers" in name for name in names))
        self.assertFalse(any("embeddings" in name for name in names))
        actual = {name for name, parameter in adapter.named_parameters() if parameter.requires_grad}
        self.assertEqual(actual, names)

    def test_adapter_does_not_override_torch_module_apply(self):
        self.assertIs(TEXT_CONTROL.TextControlAdapter.apply, nn.Module.apply)

    def test_huggingface_local_files_policy_is_forwarded(self):
        _, fake = self.make_adapter()
        self.assertEqual(
            fake.model_calls,
            [("fake-distilbert", {"local_files_only": True})],
        )
        self.assertEqual(
            fake.tokenizer_calls,
            [("fake-distilbert", {"local_files_only": True})],
        )


if __name__ == "__main__":
    unittest.main()
