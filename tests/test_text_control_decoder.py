import importlib.machinery
import importlib.util
import math
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

if not hasattr(torch, "arctan2"):
    torch.arctan2 = torch.atan2


def _install_dependency_stubs():
    if "omegaconf" not in sys.modules and importlib.util.find_spec("omegaconf") is None:
        module = types.ModuleType("omegaconf")
        module.__spec__ = importlib.machinery.ModuleSpec("omegaconf", loader=None)
        module.DictConfig = dict
        sys.modules["omegaconf"] = module

    if "torch_cluster" not in sys.modules and importlib.util.find_spec("torch_cluster") is None:
        module = types.ModuleType("torch_cluster")
        module.__spec__ = importlib.machinery.ModuleSpec("torch_cluster", loader=None)
        module.radius = lambda *args, **kwargs: torch.empty(2, 0, dtype=torch.long)
        module.radius_graph = lambda *args, **kwargs: torch.empty(2, 0, dtype=torch.long)
        sys.modules["torch_cluster"] = module

    if "torch_geometric" in sys.modules or importlib.util.find_spec("torch_geometric") is not None:
        return

    root = types.ModuleType("torch_geometric")
    root.__spec__ = importlib.machinery.ModuleSpec(
        "torch_geometric", loader=None, is_package=True
    )
    root.__path__ = []
    nn_module = types.ModuleType("torch_geometric.nn")
    nn_module.__spec__ = importlib.machinery.ModuleSpec(
        "torch_geometric.nn", loader=None, is_package=True
    )
    nn_module.__path__ = []
    conv = types.ModuleType("torch_geometric.nn.conv")
    conv.__spec__ = importlib.machinery.ModuleSpec(
        "torch_geometric.nn.conv", loader=None
    )

    class MessagePassing(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def propagate(self, *args, **kwargs):
            raise AssertionError("test attention layers must replace MessagePassing")

    conv.MessagePassing = MessagePassing
    data = types.ModuleType("torch_geometric.data")
    data.__spec__ = importlib.machinery.ModuleSpec(
        "torch_geometric.data", loader=None
    )
    data.HeteroData = dict
    utils = types.ModuleType("torch_geometric.utils")
    utils.__spec__ = importlib.machinery.ModuleSpec(
        "torch_geometric.utils", loader=None
    )
    utils.softmax = lambda value, *args, **kwargs: value
    utils.dense_to_sparse = lambda value, *args, **kwargs: (
        value.nonzero().T,
        value[value != 0],
    )
    utils.subgraph = lambda subset, edge_index, *args, **kwargs: edge_index
    root.nn = nn_module
    root.data = data
    root.utils = utils
    sys.modules["torch_geometric"] = root
    sys.modules["torch_geometric.data"] = data
    sys.modules["torch_geometric.nn"] = nn_module
    sys.modules["torch_geometric.nn.conv"] = conv
    sys.modules["torch_geometric.utils"] = utils


_install_dependency_stubs()

import src.smart.modules.agent_decoder as agent_decoder_module
import src.smart.modules.smart_decoder as smart_decoder_module
from src.smart.modules.agent_decoder import SMARTAgentDecoder
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.modules.text_control import EncodedTextControl


class _EventIdentity(nn.Module):
    def __init__(self, name, events, *, reaction=False, n_agent=2):
        super().__init__()
        self.name = name
        self.events = events
        self.reaction = reaction
        self.n_agent = n_agent

    def forward(self, feature, *args):
        self.events.append(self.name)
        value = feature[1] if isinstance(feature, tuple) else feature
        if not self.reaction:
            return value
        shaped = value.view(-1, self.n_agent, value.shape[-1])
        output = shaped.clone()
        output[:, 1] = output[:, 1] + shaped[:, 0]
        return output.flatten(0, 1)


class _CacheSpyIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, feature, *args):
        value = feature[1] if isinstance(feature, tuple) else feature
        self.inputs.append(value.detach().clone())
        return value


class _RecordingAdapter(nn.Module):
    def __init__(self, hidden_dim, num_blocks, config=None, events=None, delta=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.events = events
        self.delta = float(delta)
        self.outputs = {index: [] for index in range(num_blocks)}
        self.unmasked_were_unchanged = []
        self.encode_calls = 0

    def encode(self, prompts, mask, device, *, apply_control_dropout=False):
        self.encode_calls += 1
        return EncodedTextControl(
            features=torch.zeros(len(prompts), self.hidden_dim, device=device),
            mask=mask.to(device),
        )

    def condition(self, features, encoded, block_index):
        if self.events is not None:
            self.events.append(f"film{block_index}")
        if encoded is None:
            output = features
        else:
            output = features.clone()
            mask = encoded.mask.to(features.device)
            if features.ndim == 3:
                output[mask] = output[mask] + self.delta
            else:
                output[mask] = output[mask] + self.delta
            self.unmasked_were_unchanged.append(
                torch.equal(output[~mask], features[~mask])
            )
        self.outputs[block_index].append(output.detach().clone())
        return output


class _SentinelAdapter(nn.Module):
    def __init__(self, *, hidden_dim, num_blocks, config):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.encoder.backbone.weight.fill_(7.0)


class TextControlDecoderTest(unittest.TestCase):
    @staticmethod
    def _decoder(*, num_layers=6, num_future_steps=80, text_control=None):
        return SMARTAgentDecoder(
            hidden_dim=4,
            num_historical_steps=11,
            num_future_steps=num_future_steps,
            time_span=30,
            pl2a_radius=30.0,
            a2a_radius=60.0,
            num_freq_bands=2,
            num_layers=num_layers,
            num_heads=1,
            head_dim=4,
            dropout=0.0,
            hist_drop_prob=0.0,
            n_token_agent=2,
            endpoint_interpolation={"is_active": False},
            history_dynamics={"is_active": False},
            future_token_dynamics={"is_active": False},
            text_control=text_control,
        )

    @staticmethod
    def _forward_inputs(n_agent=2, n_step=4):
        tokenized = {
            "valid_mask": torch.ones(n_agent, n_step, dtype=torch.bool),
            "sampled_pos": torch.zeros(n_agent, n_step, 2),
            "sampled_heading": torch.zeros(n_agent, n_step),
            "sampled_idx": torch.zeros(n_agent, n_step, dtype=torch.long),
            "trajectory_token_veh": torch.zeros(2, 8),
            "trajectory_token_ped": torch.zeros(2, 8),
            "trajectory_token_cyc": torch.zeros(2, 8),
            "type": torch.zeros(n_agent, dtype=torch.long),
            "shape": torch.ones(n_agent, 3),
            "batch": torch.zeros(n_agent, dtype=torch.long),
            "num_graphs": 1,
            "gt_pos_raw": torch.zeros(n_agent, n_step, 2),
            "gt_head_raw": torch.zeros(n_agent, n_step),
            "gt_valid_raw": torch.ones(n_agent, n_step, dtype=torch.bool),
            "gt_pos": torch.zeros(n_agent, n_step, 2),
            "gt_heading": torch.zeros(n_agent, n_step),
        }
        map_feature = {
            "position": torch.zeros(1, 2),
            "orientation": torch.zeros(1),
            "batch": torch.zeros(1, dtype=torch.long),
            "pt_token": torch.zeros(1, 4),
        }
        return tokenized, map_feature

    @staticmethod
    def _empty_edges(hidden_dim=4):
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, hidden_dim)

    def _prepare_forward_harness(self, decoder, adapter, *, reaction=False):
        n_agent, n_step = 2, 4
        initial = torch.arange(n_agent * n_step * 4, dtype=torch.float32).view(
            n_agent, n_step, 4
        )
        decoder.agent_token_embedding = lambda **kwargs: initial.clone()
        decoder.build_temporal_edge = lambda **kwargs: self._empty_edges()
        decoder.build_map2agent_edge = lambda **kwargs: self._empty_edges()
        decoder.build_interaction_edge = lambda **kwargs: self._empty_edges()
        events = adapter.events if adapter.events is not None else []
        decoder.t_attn_layers = nn.ModuleList(
            [_EventIdentity(f"t{i}", events) for i in range(decoder.num_layers)]
        )
        decoder.pt2a_attn_layers = nn.ModuleList(
            [_EventIdentity(f"map{i}", events) for i in range(decoder.num_layers)]
        )
        decoder.a2a_attn_layers = nn.ModuleList(
            [
                _EventIdentity(
                    f"agent{i}", events, reaction=reaction, n_agent=n_agent
                )
                for i in range(decoder.num_layers)
            ]
        )
        decoder.text_control_adapter = adapter
        decoder.token_predict_head = nn.Identity()
        return initial

    def test_disabled_control_does_not_add_state_dict_keys(self):
        baseline = self._decoder(text_control=None)
        disabled = self._decoder(text_control={"is_active": False})
        self.assertEqual(set(baseline.state_dict()), set(disabled.state_dict()))
        self.assertIsNone(baseline.text_control_adapter)
        self.assertIsNone(disabled.text_control_adapter)

    def test_agent_weight_init_never_reinitializes_text_backbone(self):
        with patch.object(
            agent_decoder_module,
            "TextControlAdapter",
            _SentinelAdapter,
            create=True,
        ):
            decoder = self._decoder(text_control={"is_active": True})
        expected = torch.full((4, 4), 7.0)
        torch.testing.assert_close(
            decoder.text_control_adapter.encoder.backbone.weight,
            expected,
            rtol=0,
            atol=0,
        )

    def test_no_prompt_forward_matches_unconditional_forward_exactly(self):
        decoder = self._decoder()
        adapter = _RecordingAdapter(4, decoder.num_layers)
        self._prepare_forward_harness(decoder, adapter)
        tokenized, map_feature = self._forward_inputs()

        base = decoder(tokenized, map_feature, encoded_text_control=None)
        auto = decoder(
            tokenized,
            map_feature,
            encoded_text_control=EncodedTextControl(
                features=torch.zeros(2, 4),
                mask=torch.zeros(2, dtype=torch.bool),
            ),
        )

        torch.testing.assert_close(
            auto["next_token_logits"],
            base["next_token_logits"],
            rtol=0,
            atol=0,
        )

    def test_film_runs_after_every_attention_triplet(self):
        events = []
        decoder = self._decoder()
        adapter = _RecordingAdapter(4, decoder.num_layers, events=events)
        self._prepare_forward_harness(decoder, adapter)
        tokenized, map_feature = self._forward_inputs()

        decoder(
            tokenized,
            map_feature,
            encoded_text_control=EncodedTextControl(
                features=torch.zeros(2, 4),
                mask=torch.tensor([True, False]),
            ),
        )

        expected = []
        for index in range(6):
            expected.extend(
                [f"t{index}", f"map{index}", f"agent{index}", f"film{index}"]
            )
        self.assertEqual(events, expected)

    def test_auto_agent_changes_only_through_later_agent_attention(self):
        decoder = self._decoder()
        adapter = _RecordingAdapter(4, decoder.num_layers, delta=3.0)
        self._prepare_forward_harness(decoder, adapter, reaction=True)
        tokenized, map_feature = self._forward_inputs()
        baseline = decoder(tokenized, map_feature, encoded_text_control=None)

        adapter.outputs = {index: [] for index in range(decoder.num_layers)}
        controlled = decoder(
            tokenized,
            map_feature,
            encoded_text_control=EncodedTextControl(
                features=torch.zeros(2, 4),
                mask=torch.tensor([True, False]),
            ),
        )

        self.assertTrue(all(adapter.unmasked_were_unchanged))
        self.assertFalse(
            torch.equal(
                controlled["next_token_logits"][1],
                baseline["next_token_logits"][1],
            )
        )

    @staticmethod
    def _rollout_inputs():
        time = torch.arange(6, dtype=torch.float32) * 0.1
        center = torch.stack((time, torch.zeros_like(time)), dim=-1)
        heading = torch.zeros_like(time)
        forward = torch.stack((heading.cos(), heading.sin()), dim=-1)
        left = torch.stack((-heading.sin(), heading.cos()), dim=-1)
        contour = torch.stack(
            (
                center + 2.0 * forward + left,
                center + 2.0 * forward - left,
                center - 2.0 * forward - left,
                center - 2.0 * forward + left,
            ),
            dim=-2,
        )
        token_all = torch.stack((contour, contour), dim=0)
        tokenized = {
            "valid_mask": torch.ones(1, 18, dtype=torch.bool),
            "gt_pos": torch.zeros(1, 18, 2),
            "gt_heading": torch.zeros(1, 18),
            "gt_idx": torch.zeros(1, 18, dtype=torch.long),
            "trajectory_token_veh": token_all[:, -1].flatten(1, 2),
            "trajectory_token_ped": token_all[:, -1].flatten(1, 2),
            "trajectory_token_cyc": token_all[:, -1].flatten(1, 2),
            "token_traj": token_all[:, -1].unsqueeze(0),
            "token_traj_all": token_all.unsqueeze(0),
            "type": torch.zeros(1, dtype=torch.long),
            "shape": torch.ones(1, 3),
            "batch": torch.zeros(1, dtype=torch.long),
            "num_graphs": 1,
            "gt_pos_raw": torch.zeros(1, 18, 2),
            "gt_head_raw": torch.zeros(1, 18),
            "gt_valid_raw": torch.ones(1, 18, dtype=torch.bool),
            "token_agent_shape": torch.tensor([[2.0, 4.0]]),
        }
        map_feature = {
            "position": torch.zeros(1, 2),
            "orientation": torch.zeros(1),
            "batch": torch.zeros(1, dtype=torch.long),
            "pt_token": torch.zeros(1, 4),
        }
        return tokenized, map_feature

    @staticmethod
    def _fixed_sample(**kwargs):
        index = torch.zeros(
            kwargs["pos_now"].shape[0],
            dtype=torch.long,
            device=kwargs["pos_now"].device,
        )
        agent = torch.arange(index.shape[0], device=index.device)
        return index, kwargs["token_traj_all"][agent, index]

    def test_recurrent_cache_contains_conditioned_feature(self):
        decoder = self._decoder(num_layers=2, num_future_steps=10)
        adapter = _RecordingAdapter(4, 2, delta=5.0)
        cache_spy = _CacheSpyIdentity()
        decoder.text_control_adapter = adapter
        decoder.t_attn_layers = nn.ModuleList([_CacheSpyIdentity(), cache_spy])
        decoder.pt2a_attn_layers = nn.ModuleList(
            [_EventIdentity("unused", []) for _ in range(2)]
        )
        decoder.a2a_attn_layers = nn.ModuleList(
            [_EventIdentity("unused", []) for _ in range(2)]
        )
        decoder.build_temporal_edge = lambda **kwargs: self._empty_edges()
        decoder.build_map2agent_edge = lambda **kwargs: self._empty_edges()
        decoder.build_interaction_edge = lambda **kwargs: self._empty_edges()
        decoder.token_predict_head = nn.Linear(4, 2, bias=False)

        def fake_embedding(**kwargs):
            n_agent, n_step = kwargs["agent_token_index"].shape
            feat = torch.zeros(n_agent, n_step, 4)
            token_emb = torch.zeros_like(feat)
            vocabulary = torch.zeros(2, 4)
            mask = kwargs["agent_type"] == 0
            categorical = [torch.zeros(n_agent, 4), torch.zeros(n_agent, 4)]
            return (
                feat,
                token_emb,
                vocabulary,
                vocabulary,
                vocabulary,
                mask,
                ~mask,
                torch.zeros_like(mask),
                categorical,
            )

        decoder.agent_token_embedding = fake_embedding
        decoder.train()
        tokenized, map_feature = self._rollout_inputs()
        control = EncodedTextControl(
            features=torch.zeros(1, 4),
            mask=torch.tensor([True]),
        )

        with patch.object(
            agent_decoder_module,
            "sample_next_token_traj",
            side_effect=self._fixed_sample,
        ):
            decoder.inference(
                tokenized,
                map_feature,
                SimpleNamespace(),
                encoded_text_control=control,
            )

        self.assertEqual(len(cache_spy.inputs), 2)
        torch.testing.assert_close(
            cache_spy.inputs[0],
            adapter.outputs[0][0].flatten(0, 1),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            cache_spy.inputs[1],
            adapter.outputs[0][1],
            rtol=0,
            atol=0,
        )

    def test_smart_decoder_delegates_one_static_text_encoding(self):
        class FakeMapDecoder(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

        class FakeAgentDecoder(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.received_text_control = kwargs.get("text_control")
                self.text_control_adapter = _RecordingAdapter(4, 1)

        config = {"is_active": True, "marker": "same-object"}
        with patch.object(smart_decoder_module, "SMARTMapDecoder", FakeMapDecoder), patch.object(
            smart_decoder_module, "SMARTAgentDecoder", FakeAgentDecoder
        ):
            decoder = SMARTDecoder(
                hidden_dim=4,
                num_historical_steps=11,
                num_future_steps=80,
                pl2pl_radius=150.0,
                time_span=30,
                pl2a_radius=30.0,
                a2a_radius=60.0,
                num_freq_bands=2,
                num_map_layers=3,
                num_agent_layers=6,
                num_heads=1,
                head_dim=4,
                dropout=0.0,
                hist_drop_prob=0.0,
                n_token_agent=2,
                text_control=config,
            )

        encoded = decoder.encode_text_control(
            ["The target vehicle is accelerating."],
            torch.tensor([True]),
            torch.device("cpu"),
            training=True,
        )
        self.assertIs(decoder.agent_encoder.received_text_control, config)
        self.assertIsInstance(encoded, EncodedTextControl)
        self.assertEqual(decoder.agent_encoder.text_control_adapter.encode_calls, 1)


if __name__ == "__main__":
    unittest.main()
