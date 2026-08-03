import importlib.util
from pathlib import Path
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.smart.modules.text_control import EncodedTextControl


ROOT = Path(__file__).resolve().parents[1]


class _SilentLogger:
    def __init__(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


def _load_finetune_module():
    fake_utils = types.ModuleType("src.utils")
    fake_utils.RankedLogger = _SilentLogger
    path = ROOT / "src" / "smart" / "utils" / "finetune.py"
    spec = importlib.util.spec_from_file_location("catk_finetune_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"src.utils": fake_utils}):
        spec.loader.exec_module(module)
    return module


FINETUNE = _load_finetune_module()


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class FakeLightningModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_epoch = 0
        self.global_rank = 0
        self.logger = None
        self.logged = {}

    def save_hyperparameters(self, *args, **kwargs):
        pass

    def log(self, name, value, *args, **kwargs):
        self.logged[name] = value

    def print(self, *args, **kwargs):
        pass


class FakeMetric:
    def __init__(self, *args, **kwargs):
        self.updates = []

    def update(self, *args, **kwargs):
        self.updates.append((args, kwargs))

    def compute(self):
        return torch.tensor(0.0)

    def reset(self):
        pass


class FakeWOSACMetric(FakeMetric):
    def compute(self):
        return {"val_closed/metametric": 0.0}


class FakeSubmission(FakeMetric):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.is_active = False

    def aggregate_rollouts(self, *args, **kwargs):
        pass

    def save_sub_file(self):
        pass


class FakeCrossEntropy(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor(2.0, requires_grad=True)


class FakeTokenProcessor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n_token_agent = 2

    def forward(self, data):
        return data["_tokenized_map"], data["_tokenized_agent"]


class FakeTextBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(8, 4)
        self.attention = nn.Module()
        self.attention.lora_A = nn.Linear(4, 2, bias=False)
        self.attention.lora_B = nn.Linear(2, 4, bias=False)
        self.attention.base = nn.Linear(4, 4)


class FakeTextAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = FakeTextBackbone()
        self.encoder.projection = nn.Linear(4, 4)
        self.film_layers = nn.ModuleList([nn.Linear(4, 8) for _ in range(6)])
        self.encode_calls = 0
        self.encode_training_flags = []

    def unfreeze_control_parameters(self):
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.encoder.backbone.attention.lora_A.weight.requires_grad = True
        self.encoder.backbone.attention.lora_B.weight.requires_grad = True
        for parameter in self.encoder.projection.parameters():
            parameter.requires_grad = True
        for parameter in self.film_layers.parameters():
            parameter.requires_grad = True

    def encode(
        self,
        prompts,
        mask,
        device,
        *,
        apply_control_dropout=False,
    ):
        self.encode_calls += 1
        self.encode_training_flags.append(bool(apply_control_dropout))
        effective = mask.to(device).clone()
        effective &= torch.tensor(
            [bool(prompt.strip()) for prompt in prompts],
            dtype=torch.bool,
            device=device,
        )
        if not bool(effective.any()):
            return None
        return EncodedTextControl(
            features=torch.zeros(len(prompts), 4, device=device),
            mask=effective,
        )


class FakeAgentEncoder(nn.Module):
    def __init__(self, text_control=None):
        super().__init__()
        self.token_predict_head = nn.Linear(4, 2)
        self.history_dynamics_emb = nn.Linear(3, 4)
        self.t_attn_layers = nn.ModuleList([nn.Linear(4, 4)])
        self.pt2a_attn_layers = nn.ModuleList([nn.Linear(4, 4)])
        self.a2a_attn_layers = nn.ModuleList([nn.Linear(4, 4)])
        self.text_control_adapter = None
        if text_control is not None and bool(text_control.get("is_active", False)):
            self.text_control_adapter = FakeTextAdapter()


class FakeSMARTDecoder(nn.Module):
    def __init__(self, *args, text_control=None, **kwargs):
        super().__init__()
        self.map_encoder = nn.Linear(4, 4)
        self.agent_encoder = FakeAgentEncoder(text_control=text_control)
        self.text_encode_calls = 0
        self.encoded_seen = []

    def encode_text_control(self, prompts, mask, device, *, training):
        self.text_encode_calls += 1
        adapter = self.agent_encoder.text_control_adapter
        if adapter is None:
            return None
        return adapter.encode(
            prompts,
            mask,
            device,
            apply_control_dropout=training,
        )

    @staticmethod
    def _open_output(tokenized_agent):
        n_agent = tokenized_agent["gt_idx"].shape[0]
        return {
            "next_token_logits": torch.zeros(n_agent, 16, 2),
            "next_token_valid": torch.ones(n_agent, 16, dtype=torch.bool),
        }

    def forward(
        self,
        tokenized_map,
        tokenized_agent,
        encoded_text_control=None,
    ):
        self.encoded_seen.append(encoded_text_control)
        return self._open_output(tokenized_agent)

    def inference(
        self,
        tokenized_map,
        tokenized_agent,
        sampling_scheme,
        encoded_text_control=None,
    ):
        self.encoded_seen.append(encoded_text_control)
        n_agent = tokenized_agent["gt_idx"].shape[0]
        output = self._open_output(tokenized_agent)
        output.update(
            {
                "pred_traj_10hz": torch.zeros(n_agent, 80, 2),
                "pred_z_10hz": torch.zeros(n_agent, 80),
                "pred_head_10hz": torch.zeros(n_agent, 80),
            }
        )
        return output


def _missing_text_freezer(model):
    raise NotImplementedError("set_model_for_text_control is not implemented")


def _flatten_prompts(value):
    output = []

    def visit(item):
        if isinstance(item, str):
            output.append(item)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            raise TypeError("text prompts must contain only string leaves")

    visit(value)
    return output


def _load_smart_class():
    hydra = types.ModuleType("hydra")
    hydra.core = SimpleNamespace(
        hydra_config=SimpleNamespace(
            HydraConfig=SimpleNamespace(
                get=lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir="/tmp"))
            )
        )
    )
    lightning = types.ModuleType("lightning")
    lightning.LightningModule = FakeLightningModule

    metrics = types.ModuleType("src.smart.metrics")
    metrics.CrossEntropy = FakeCrossEntropy
    metrics.FastWOSACMetrics = FakeWOSACMetric
    metrics.TokenCls = FakeMetric
    metrics.WOSACMetrics = FakeWOSACMetric
    metrics.WOSACSubmission = FakeSubmission
    metrics.minADE = FakeMetric

    decoder = types.ModuleType("src.smart.modules.smart_decoder")
    decoder.SMARTDecoder = FakeSMARTDecoder
    processor = types.ModuleType("src.smart.tokens.token_processor")
    processor.TokenProcessor = FakeTokenProcessor
    datasets = types.ModuleType("src.smart.datasets")
    datasets.__path__ = []
    text_prompts = types.ModuleType("src.smart.datasets.text_prompts")
    text_prompts.flatten_batched_prompts = _flatten_prompts
    finetune = types.ModuleType("src.smart.utils.finetune")
    finetune.set_model_for_finetuning = FINETUNE.set_model_for_finetuning
    finetune.set_model_for_text_control = getattr(
        FINETUNE,
        "set_model_for_text_control",
        _missing_text_freezer,
    )
    vis = types.ModuleType("src.utils.vis_waymo")
    vis.VisWaymo = object
    wosac = types.ModuleType("src.utils.wosac_utils")
    wosac.get_scenario_id_int_tensor = lambda *args, **kwargs: torch.tensor([0])
    wosac.get_scenario_rollouts = lambda *args, **kwargs: []

    replacements = {
        "hydra": hydra,
        "lightning": lightning,
        "src.smart.metrics": metrics,
        "src.smart.modules.smart_decoder": decoder,
        "src.smart.tokens.token_processor": processor,
        "src.smart.datasets": datasets,
        "src.smart.datasets.text_prompts": text_prompts,
        "src.smart.utils.finetune": finetune,
        "src.utils.vis_waymo": vis,
        "src.utils.wosac_utils": wosac,
    }
    path = ROOT / "src" / "smart" / "model" / "smart.py"
    spec = importlib.util.spec_from_file_location("catk_smart_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module.SMART


SMART = _load_smart_class()


def model_config(*, text_control_active=True, finetune=False, closed=False):
    return AttrDict(
        lr=1e-4,
        lr_warmup_steps=2,
        lr_total_steps=10,
        lr_min_ratio=0.1,
        decoder=AttrDict(num_historical_steps=11),
        val_open_loop=True,
        val_closed_loop=closed,
        token_processor=AttrDict(),
        history_dynamics=AttrDict(is_active=True),
        future_token_dynamics=AttrDict(is_active=False),
        text_control=AttrDict(is_active=text_control_active),
        finetune=finetune,
        wosac_backend="fast",
        wosac_metrics_version="2025",
        fast_wosac_gt_dir=None,
        fast_wosac_require_preprocessed_gt=False,
        wosac_submission=AttrDict(is_active=False),
        training_loss=AttrDict(),
        n_rollout_closed_val=3,
        n_vis_batch=0,
        n_vis_scenario=0,
        n_vis_rollout=0,
        n_batch_wosac_metric=0,
        training_rollout_sampling=SimpleNamespace(num_k=0),
        validation_rollout_sampling=SimpleNamespace(num_k=1),
    )


def make_data(
    *,
    prompts=None,
    prompt_mask=None,
    train_mask=None,
):
    prompts = prompts if prompts is not None else [["accelerating", "turning"], [""]]
    prompt_mask = prompt_mask if prompt_mask is not None else [True, True, False]
    train_mask = train_mask if train_mask is not None else [True, False, True]
    n_agent = len(prompt_mask)
    tokenized_agent = {
        "gt_idx": torch.zeros(n_agent, 18, dtype=torch.long),
        "token_agent_shape": torch.ones(n_agent, 2),
        "token_traj": torch.zeros(n_agent, 2, 4, 2),
        "valid_mask": torch.ones(n_agent, 18, dtype=torch.bool),
    }
    return {
        "scenario_id": ["scenario-training-test"],
        "tfrecord_path": ["unused.tfrecord"],
        "agent": {
            "id": torch.arange(n_agent),
            "batch": torch.zeros(n_agent, dtype=torch.long),
            "position": torch.zeros(n_agent, 91, 3),
            "valid_mask": torch.ones(n_agent, 91, dtype=torch.bool),
            "text_prompt": prompts,
            "text_prompt_mask": torch.tensor(prompt_mask, dtype=torch.bool),
            "train_mask": torch.tensor(train_mask, dtype=torch.bool),
        },
        "_tokenized_map": {},
        "_tokenized_agent": tokenized_agent,
    }


class TextControlTrainingTest(unittest.TestCase):
    def test_text_only_freezer_has_exact_trainable_boundary(self):
        self.assertTrue(
            hasattr(FINETUNE, "set_model_for_text_control"),
            "set_model_for_text_control is missing",
        )
        decoder = FakeSMARTDecoder(text_control={"is_active": True})

        names = FINETUNE.set_model_for_text_control(decoder)
        trainable = {name for name, p in decoder.named_parameters() if p.requires_grad}

        self.assertEqual(set(names), trainable)
        self.assertTrue(any("lora_A" in name for name in trainable))
        self.assertTrue(any("projection" in name for name in trainable))
        self.assertTrue(any("film_layers" in name for name in trainable))
        self.assertFalse(any("map_encoder" in name for name in trainable))
        self.assertFalse(any("token_predict_head" in name for name in trainable))
        self.assertFalse(any("history_dynamics_emb" in name for name in trainable))
        self.assertFalse(any("backbone.embeddings" in name for name in trainable))

    def test_smart_constructor_applies_text_only_freeze(self):
        model = SMART(model_config())
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(any("lora_A" in name for name in names))
        self.assertTrue(any("projection" in name for name in names))
        self.assertTrue(any("film_layers" in name for name in names))
        self.assertFalse(any("map_encoder" in name for name in names))
        self.assertFalse(any("token_predict_head" in name for name in names))
        self.assertFalse(any("history_dynamics_emb" in name for name in names))
        self.assertFalse(any("backbone.embeddings" in name for name in names))

    def test_ordinary_finetune_and_text_freeze_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "finetune.*text control|text control.*finetune"):
            SMART(model_config(finetune=True))

    def test_training_mask_intersects_prompt_and_train_masks(self):
        model = SMART(model_config())
        model.train()
        data = make_data(
            prompts=[["accelerating"], ["turning", ""]],
            prompt_mask=[True, True, False],
            train_mask=[True, False, True],
        )

        encoded = model._prepare_text_control(data)

        torch.testing.assert_close(
            encoded.mask,
            torch.tensor([True, False, False]),
        )
        self.assertTrue(
            model.encoder.agent_encoder.text_control_adapter.encode_training_flags[-1]
        )
        self.assertAlmostEqual(
            float(model.logged["train/text_control_fraction"]),
            1.0 / 3.0,
        )

    def test_prompt_alignment_error_identifies_scenario(self):
        model = SMART(model_config())
        data = make_data()
        data["agent"]["text_prompt"] = ["one prompt only"]
        with self.assertRaisesRegex(ValueError, "scenario-training-test"):
            model._prepare_text_control(data)

    def test_open_and_closed_validation_reuse_one_text_encoding(self):
        model = SMART(model_config(closed=True))
        model.eval()

        model.validation_step(make_data(), batch_idx=0)

        self.assertEqual(model.encoder.text_encode_calls, 1)
        self.assertEqual(len(model.encoder.encoded_seen), 4)
        first = model.encoder.encoded_seen[0]
        self.assertTrue(all(value is first for value in model.encoder.encoded_seen))
        self.assertFalse(
            model.encoder.agent_encoder.text_control_adapter.encode_training_flags[-1]
        )

    def test_training_loss_receives_original_catk_arguments(self):
        model = SMART(model_config())
        data = make_data()

        model.training_step(data, batch_idx=0)

        kwargs = model.training_loss.kwargs
        torch.testing.assert_close(
            kwargs["gt_idx"],
            data["_tokenized_agent"]["gt_idx"][:, 2:],
        )
        self.assertIs(kwargs["train_mask"], data["agent"]["train_mask"])
        self.assertIs(kwargs["token_traj"], data["_tokenized_agent"]["token_traj"])
        self.assertIs(
            kwargs["token_agent_shape"],
            data["_tokenized_agent"]["token_agent_shape"],
        )

    def test_optimizer_contains_only_trainable_parameters(self):
        model = SMART(model_config())

        optimizers, _ = model.configure_optimizers()

        optimized = {
            id(parameter)
            for group in optimizers[0].param_groups
            for parameter in group["params"]
        }
        expected = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        self.assertEqual(optimized, expected)

    def test_empty_prompt_batch_marks_every_adapter_parameter_used(self):
        model = SMART(model_config())
        model.train()
        data = make_data(
            prompts=[["", ""], [""]],
            prompt_mask=[False, False, False],
            train_mask=[True, True, True],
        )

        loss = model.training_step(data, batch_idx=0)
        loss.backward()

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)

    def test_disabled_text_control_requires_no_prompt_fields(self):
        model = SMART(model_config(text_control_active=False))
        data = make_data()
        del data["agent"]["text_prompt"]
        del data["agent"]["text_prompt_mask"]

        self.assertIsNone(model._prepare_text_control(data))


if __name__ == "__main__":
    unittest.main()
