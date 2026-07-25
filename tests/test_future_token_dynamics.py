import importlib.machinery
import importlib.util
import math
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

if not hasattr(torch, "arctan2"):
    torch.arctan2 = torch.atan2

from src.smart.tokens.future_token_dynamics import (
    build_future_token_dynamics_lookup,
    gather_future_token_dynamics,
)

try:
    from src.smart.tokens.future_token_dynamics import (
        gather_transition_dynamics,
    )
except ImportError:
    gather_transition_dynamics = None

from src.smart.tokens.transition_dynamics_artifact import (
    make_transition_dynamics_artifact,
    save_transition_dynamics_artifact,
)


def _load_token_processor():
    if (
        "omegaconf" not in sys.modules
        and importlib.util.find_spec("omegaconf") is None
    ):
        omegaconf_stub = types.ModuleType("omegaconf")
        omegaconf_stub.__spec__ = importlib.machinery.ModuleSpec(
            "omegaconf",
            loader=None,
        )
        omegaconf_stub.DictConfig = dict
        sys.modules["omegaconf"] = omegaconf_stub

    if (
        "torch_geometric" not in sys.modules
        and importlib.util.find_spec("torch_geometric") is None
    ):
        torch_geometric_stub = types.ModuleType("torch_geometric")
        torch_geometric_stub.__spec__ = importlib.machinery.ModuleSpec(
            "torch_geometric",
            loader=None,
            is_package=True,
        )
        torch_geometric_stub.__path__ = []
        torch_geometric_data_stub = types.ModuleType("torch_geometric.data")
        torch_geometric_data_stub.__spec__ = importlib.machinery.ModuleSpec(
            "torch_geometric.data",
            loader=None,
        )
        torch_geometric_data_stub.HeteroData = dict
        torch_geometric_nn_stub = types.ModuleType("torch_geometric.nn")
        torch_geometric_nn_stub.__spec__ = importlib.machinery.ModuleSpec(
            "torch_geometric.nn",
            loader=None,
            is_package=True,
        )
        torch_geometric_nn_stub.__path__ = []
        torch_geometric_conv_stub = types.ModuleType("torch_geometric.nn.conv")
        torch_geometric_conv_stub.__spec__ = importlib.machinery.ModuleSpec(
            "torch_geometric.nn.conv",
            loader=None,
        )

        class MessagePassing(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        torch_geometric_conv_stub.MessagePassing = MessagePassing
        torch_geometric_utils_stub = types.ModuleType("torch_geometric.utils")
        torch_geometric_utils_stub.__spec__ = importlib.machinery.ModuleSpec(
            "torch_geometric.utils",
            loader=None,
        )
        torch_geometric_utils_stub.softmax = lambda value, *args, **kwargs: value
        torch_geometric_utils_stub.dense_to_sparse = (
            lambda value, *args, **kwargs: (
                value.nonzero().T,
                value[value != 0],
            )
        )
        torch_geometric_utils_stub.subgraph = (
            lambda subset, edge_index, *args, **kwargs: edge_index
        )
        torch_geometric_stub.data = torch_geometric_data_stub
        torch_geometric_stub.nn = torch_geometric_nn_stub
        torch_geometric_stub.utils = torch_geometric_utils_stub
        sys.modules["torch_geometric"] = torch_geometric_stub
        sys.modules["torch_geometric.data"] = torch_geometric_data_stub
        sys.modules["torch_geometric.nn"] = torch_geometric_nn_stub
        sys.modules["torch_geometric.nn.conv"] = torch_geometric_conv_stub
        sys.modules["torch_geometric.utils"] = torch_geometric_utils_stub

    from src.smart.tokens.token_processor import TokenProcessor

    return TokenProcessor


TokenProcessor = _load_token_processor()

from src.smart.modules.future_token_dynamics import (
    FutureTokenDynamicsConditioner,
)


def _load_agent_decoder():
    if (
        "torch_cluster" not in sys.modules
        and importlib.util.find_spec("torch_cluster") is None
    ):
        torch_cluster_stub = types.ModuleType("torch_cluster")
        torch_cluster_stub.__spec__ = importlib.machinery.ModuleSpec(
            "torch_cluster",
            loader=None,
        )

        def unavailable(*args, **kwargs):
            raise AssertionError("graph radius function was not replaced in the test")

        torch_cluster_stub.radius = unavailable
        torch_cluster_stub.radius_graph = unavailable
        sys.modules["torch_cluster"] = torch_cluster_stub

    from src.smart.modules.agent_decoder import SMARTAgentDecoder

    return SMARTAgentDecoder


SMARTAgentDecoder = _load_agent_decoder()


class FutureTokenDynamicsLookupTest(unittest.TestCase):
    @staticmethod
    def _contours(center, heading, *, length=4.0, width=2.0):
        center = torch.as_tensor(center)
        heading = torch.as_tensor(heading, dtype=center.dtype, device=center.device)
        forward = torch.stack((heading.cos(), heading.sin()), dim=-1)
        left = torch.stack((-heading.sin(), heading.cos()), dim=-1)
        half_length = 0.5 * length
        half_width = 0.5 * width
        return torch.stack(
            (
                center + half_length * forward + half_width * left,
                center + half_length * forward - half_width * left,
                center - half_length * forward - half_width * left,
                center - half_length * forward + half_width * left,
            ),
            dim=-2,
        )

    @classmethod
    def _constant_acceleration_token(cls, acceleration, dtype=torch.float64):
        time = torch.arange(6, dtype=dtype) * 0.1
        x = 2.0 * time + 0.5 * acceleration * time.square()
        center = torch.stack((x, torch.zeros_like(x)), dim=-1)
        return cls._contours(center, torch.zeros_like(time))

    def test_constant_acceleration_has_longitudinal_dynamics_only(self):
        token = self._constant_acceleration_token(2.0).unsqueeze(0)

        lookup = build_future_token_dynamics_lookup(token)

        expected = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64)
        self.assertTrue(torch.allclose(lookup, expected, atol=1e-10, rtol=1e-10))

    def test_constant_radius_motion_has_angular_and_lateral_dynamics(self):
        time = torch.arange(6, dtype=torch.float64) * 0.1
        radius = 10.0
        angular_speed = 0.4
        heading = angular_speed * time
        center = torch.stack(
            (
                radius * heading.sin(),
                radius * (1.0 - heading.cos()),
            ),
            dim=-1,
        )
        token = self._contours(center, heading).unsqueeze(0)

        lookup = build_future_token_dynamics_lookup(token)

        self.assertAlmostEqual(float(lookup[0, 0]), 0.0, delta=6e-2)
        self.assertAlmostEqual(
            float(lookup[0, 1]),
            angular_speed,
            delta=1e-10,
        )
        self.assertAlmostEqual(
            float(lookup[0, 2]),
            radius * angular_speed**2,
            delta=3e-2,
        )

    def test_heading_is_unwrapped_before_angular_speed_is_computed(self):
        time = torch.arange(6, dtype=torch.float64) * 0.1
        angular_speed = 0.8
        unwrapped_heading = 3.0 + angular_speed * time
        wrapped_heading = torch.remainder(
            unwrapped_heading + math.pi,
            2.0 * math.pi,
        ) - math.pi
        center = torch.zeros(6, 2, dtype=torch.float64)
        token = self._contours(center, wrapped_heading).unsqueeze(0)

        lookup = build_future_token_dynamics_lookup(token)

        self.assertAlmostEqual(float(lookup[0, 1]), angular_speed, delta=1e-10)

    def test_rigid_rotation_and_translation_leave_body_dynamics_unchanged(self):
        time = torch.arange(6, dtype=torch.float64) * 0.1
        heading = 0.35 * time
        center = torch.stack(
            (
                8.0 * heading.sin(),
                8.0 * (1.0 - heading.cos()),
            ),
            dim=-1,
        )
        token = self._contours(center, heading)
        rotation = 1.1
        rotation_matrix = torch.tensor(
            [
                [math.cos(rotation), -math.sin(rotation)],
                [math.sin(rotation), math.cos(rotation)],
            ],
            dtype=torch.float64,
        )
        transformed = token @ rotation_matrix.T
        transformed = transformed + torch.tensor(
            [13.0, -7.0],
            dtype=torch.float64,
        )

        reference = build_future_token_dynamics_lookup(token.unsqueeze(0))
        changed = build_future_token_dynamics_lookup(transformed.unsqueeze(0))

        self.assertTrue(torch.allclose(changed, reference, atol=1e-9, rtol=1e-9))

    def test_lookup_preserves_token_order_shape_and_dtype(self):
        token_slow = self._constant_acceleration_token(1.0, torch.float32)
        token_fast = self._constant_acceleration_token(3.0, torch.float32)
        vocabulary = torch.stack((token_fast, token_slow), dim=0)

        lookup = build_future_token_dynamics_lookup(vocabulary)

        self.assertEqual(tuple(lookup.shape), (2, 3))
        self.assertEqual(lookup.dtype, torch.float32)
        self.assertTrue(
            torch.allclose(
                lookup[:, 0],
                torch.tensor([3.0, 1.0]),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_class_specific_gather_uses_each_agents_type_and_token_index(self):
        token_index = torch.tensor([[1, 0], [0, 1], [1, 1]])
        agent_type = torch.tensor([0, 1, 2])
        vehicle = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pedestrian = vehicle + 10.0
        cyclist = vehicle + 20.0

        gathered = gather_future_token_dynamics(
            token_index=token_index,
            agent_type=agent_type,
            dynamics_veh=vehicle,
            dynamics_ped=pedestrian,
            dynamics_cyc=cyclist,
        )

        expected = torch.tensor(
            [
                [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]],
                [[24.0, 25.0, 26.0], [24.0, 25.0, 26.0]],
            ]
        )
        self.assertTrue(torch.equal(gathered, expected))

    def test_lookup_clips_to_physical_limits(self):
        token = self._constant_acceleration_token(100.0).unsqueeze(0)

        lookup = build_future_token_dynamics_lookup(token)

        self.assertEqual(float(lookup[0, 0]), 15.0)
        self.assertTrue(
            torch.all(
                lookup.abs()
                <= torch.tensor([15.0, 3.0, 15.0], dtype=lookup.dtype)
            )
        )

    def test_invalid_shape_and_non_finite_values_identify_context(self):
        with self.assertRaisesRegex(
            ValueError,
            "agent_vocab.pkl.*veh.*\\[n_token, 6, 4, 2\\]",
        ):
            build_future_token_dynamics_lookup(
                torch.zeros(2, 5, 4, 2),
                context="agent_vocab.pkl class veh",
            )

        bad = torch.zeros(2, 6, 4, 2)
        bad[1, 3, 0, 0] = float("nan")
        with self.assertRaisesRegex(
            ValueError,
            "agent_vocab.pkl.*ped.*non-finite",
        ):
            build_future_token_dynamics_lookup(
                bad,
                context="agent_vocab.pkl class ped",
            )

    def test_gather_rejects_unknown_agent_type(self):
        lookup = torch.zeros(2, 3)
        with self.assertRaisesRegex(ValueError, "agent_type"):
            gather_future_token_dynamics(
                token_index=torch.tensor([0]),
                agent_type=torch.tensor([3]),
                dynamics_veh=lookup,
                dynamics_ped=lookup,
                dynamics_cyc=lookup,
            )

    def test_pair_gather_uses_previous_current_indices_and_agent_type(self):
        self.assertIsNotNone(
            gather_transition_dynamics,
            "token-transition gather is not implemented",
        )
        vehicle = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
        pedestrian = vehicle + 100.0
        cyclist = vehicle + 200.0

        gathered = gather_transition_dynamics(
            previous_token_index=torch.tensor([[0, 1], [1, 0], [1, 1]]),
            current_token_index=torch.tensor([[1, 0], [0, 1], [1, 0]]),
            agent_type=torch.tensor([0, 1, 2]),
            dynamics_veh=vehicle,
            dynamics_ped=pedestrian,
            dynamics_cyc=cyclist,
        )

        expected = torch.stack(
            (
                torch.stack((vehicle[0, 1], vehicle[1, 0])),
                torch.stack((pedestrian[1, 0], pedestrian[0, 1])),
                torch.stack((cyclist[1, 1], cyclist[1, 0])),
            )
        )
        torch.testing.assert_close(gathered, expected)


class FutureTokenDynamicsTokenProcessorTest(unittest.TestCase):
    class _FakeHeteroData(dict):
        num_graphs = 1

    @staticmethod
    def _contours_for_acceleration(acceleration):
        time = torch.arange(6, dtype=torch.float32) * 0.1
        x = time + 0.5 * acceleration * time.square()
        center = torch.stack((x, torch.zeros_like(x)), dim=-1)
        return FutureTokenDynamicsLookupTest._contours(
            center,
            torch.zeros_like(time),
        )

    @classmethod
    def _vocabulary(cls):
        return {
            "token_all": {
                "veh": torch.stack(
                    (
                        cls._contours_for_acceleration(1.0),
                        cls._contours_for_acceleration(2.0),
                    )
                ).numpy(),
                "ped": torch.stack(
                    (
                        cls._contours_for_acceleration(3.0),
                        cls._contours_for_acceleration(4.0),
                    )
                ).numpy(),
                "cyc": torch.stack(
                    (
                        cls._contours_for_acceleration(5.0),
                        cls._contours_for_acceleration(6.0),
                    )
                ).numpy(),
            }
        }

    @staticmethod
    def _processor(active=True, config=None):
        processor = TokenProcessor.__new__(TokenProcessor)
        torch.nn.Module.__init__(processor)
        processor.future_token_dynamics_active = active
        processor.future_token_dynamics_config = (
            config
            if config is not None
            else {"is_active": active}
        )
        processor.history_dynamics_active = False
        processor.agent_token_sampling = SimpleNamespace(num_k=1, temp=1.0)
        processor.shift = 5
        return processor

    @staticmethod
    def _write_pickle(directory, payload, name="agent_vocab.pkl"):
        path = Path(directory) / name
        with path.open("wb") as file:
            pickle.dump(payload, file)
        return path

    @staticmethod
    def _write_transition_artifact(directory, vocabulary_path, source="raw"):
        values = (
            torch.arange(36, dtype=torch.float32)
            .to(torch.float16)
            .view(3, 2, 2, 3)
            .numpy()
        )
        artifact = make_transition_dynamics_artifact(
            values,
            vocabulary_path=vocabulary_path,
            source=source,
            dt=0.1,
            clipping_limits=(15.0, 3.0, 15.0),
            shrinkage_count=8.0,
            statistics={},
        )
        output = save_transition_dynamics_artifact(
            Path(directory) / "lookup.pt",
            artifact,
            vocabulary_path=vocabulary_path,
        )
        return output, values

    def test_active_processor_requires_transition_lookup_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_pickle(directory, self._vocabulary())
            processor = self._processor(active=True)

            with self.assertRaisesRegex(ValueError, "lookup_file"):
                processor.init_agent_token(str(path))

    def test_active_processor_loads_vocabulary_bound_transition_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_pickle(directory, self._vocabulary())
            lookup_path, values = self._write_transition_artifact(
                directory,
                path,
            )
            processor = self._processor(
                active=True,
                config={
                    "is_active": True,
                    "lookup_file": str(lookup_path),
                    "source": "raw",
                },
            )

            processor.init_agent_token(str(path))

        self.assertEqual(
            tuple(processor.agent_token_dynamics_veh.shape),
            (2, 2, 3),
        )
        self.assertEqual(processor.agent_token_dynamics_veh.dtype, torch.float16)
        torch.testing.assert_close(
            processor.agent_token_dynamics_veh,
            torch.from_numpy(values[0]),
        )
        torch.testing.assert_close(
            processor.agent_token_dynamics_ped,
            torch.from_numpy(values[1]),
        )
        torch.testing.assert_close(
            processor.agent_token_dynamics_cyc,
            torch.from_numpy(values[2]),
        )
        state_keys = set(processor.state_dict())
        self.assertNotIn("agent_token_dynamics_veh", state_keys)
        self.assertNotIn("agent_token_dynamics_ped", state_keys)
        self.assertNotIn("agent_token_dynamics_cyc", state_keys)

    def test_inactive_processor_does_not_build_future_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_pickle(directory, self._vocabulary())
            processor = self._processor(active=False)

            processor.init_agent_token(str(path))

        self.assertFalse(hasattr(processor, "agent_token_dynamics_veh"))
        self.assertFalse(hasattr(processor, "agent_token_dynamics_ped"))
        self.assertFalse(hasattr(processor, "agent_token_dynamics_cyc"))

    def test_tokenized_agent_exposes_lookup_tables_without_copying_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_pickle(directory, self._vocabulary())
            lookup_path, _ = self._write_transition_artifact(directory, path)
            processor = self._processor(
                active=True,
                config={
                    "is_active": True,
                    "lookup_file": str(lookup_path),
                    "source": "raw",
                },
            )
            processor.init_agent_token(str(path))

        n_agent = 3
        valid = torch.ones(n_agent, 91, dtype=torch.bool)
        position = torch.zeros(n_agent, 91, 3)
        position[:, :, 0] = torch.arange(91, dtype=torch.float32) * 0.1
        data = self._FakeHeteroData(
            {
                "agent": {
                    "type": torch.tensor([0, 1, 2]),
                    "shape": torch.ones(n_agent, 3),
                    "role": torch.zeros(n_agent, 3, dtype=torch.bool),
                    "batch": torch.zeros(n_agent, dtype=torch.long),
                    "valid_mask": valid,
                    "heading": torch.zeros(n_agent, 91),
                    "position": position,
                    "velocity": torch.zeros(n_agent, 91, 2),
                }
            }
        )

        tokenized = processor.tokenize_agent(data)

        for agent_class in ("veh", "ped", "cyc"):
            key = f"agent_token_dynamics_{agent_class}"
            self.assertIs(tokenized[key], getattr(processor, key))

    def test_invalid_vocabulary_error_identifies_path_and_agent_class(self):
        vocabulary = self._vocabulary()
        vocabulary["token_all"]["veh"] = np.zeros((2, 5, 4, 2), np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_pickle(directory, vocabulary)
            processor = self._processor(active=True)

            with self.assertRaisesRegex(
                ValueError,
                "agent_vocab.pkl.*veh.*\\[n_token, 6, 4, 2\\]",
            ):
                processor.init_agent_token(str(path))

    def test_non_finite_vocabulary_error_identifies_path_and_agent_class(self):
        vocabulary = self._vocabulary()
        vocabulary["token_all"]["ped"][1, 3, 0, 0] = float("inf")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_pickle(directory, vocabulary)
            processor = self._processor(active=True)

            with self.assertRaisesRegex(
                ValueError,
                "agent_vocab.pkl.*ped.*non-finite",
            ):
                processor.init_agent_token(str(path))

    def test_agent_classes_must_have_one_shared_token_count(self):
        vocabulary = self._vocabulary()
        vocabulary["token_all"]["cyc"] = vocabulary["token_all"]["cyc"][:1]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_pickle(directory, vocabulary)
            processor = self._processor(active=True)

            with self.assertRaisesRegex(ValueError, "same token count"):
                processor.init_agent_token(str(path))


class FutureTokenDynamicsConditionerTest(unittest.TestCase):
    @staticmethod
    def _lookups():
        vehicle = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pedestrian = vehicle + 10.0
        cyclist = vehicle + 20.0
        return vehicle, pedestrian, cyclist

    @staticmethod
    def _active_config(**overrides):
        config = {
            "is_active": True,
            "normalization_scale": [1.0, 1.0, 1.0],
            "initial_gate": 1.0,
        }
        config.update(overrides)
        return config

    def test_open_loop_masks_history_and_uses_own_teacher_forced_token_afterward(
        self,
    ):
        conditioner = FutureTokenDynamicsConditioner(
            hidden_dim=3,
            config=self._active_config(),
        )
        conditioner.embedding = torch.nn.Identity()
        token_index = torch.tensor(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [0, 0, 1, 1],
            ]
        )
        agent_type = torch.tensor([0, 1, 2])
        lookups = self._lookups()

        result = conditioner.add_open_loop(
            feature=torch.zeros(3, 4, 3),
            token_index=token_index,
            agent_type=agent_type,
            dynamics_veh=lookups[0],
            dynamics_ped=lookups[1],
            dynamics_cyc=lookups[2],
            num_historical_tokens=2,
        )

        self.assertTrue(torch.equal(result[:, :2], torch.zeros(3, 2, 3)))
        expected_future = torch.tensor(
            [
                [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]],
                [[24.0, 25.0, 26.0], [24.0, 25.0, 26.0]],
            ]
        )
        self.assertTrue(torch.equal(result[:, 2:], expected_future))

    def test_selected_rollout_token_conditions_only_newly_appended_feature(self):
        conditioner = FutureTokenDynamicsConditioner(
            hidden_dim=3,
            config=self._active_config(),
        )
        conditioner.embedding = torch.nn.Identity()
        lookups = self._lookups()

        result = conditioner.add_selected(
            feature=torch.zeros(3, 3),
            token_index=torch.tensor([1, 0, 1]),
            agent_type=torch.tensor([0, 1, 2]),
            dynamics_veh=lookups[0],
            dynamics_ped=lookups[1],
            dynamics_cyc=lookups[2],
        )

        expected = torch.tensor(
            [
                [4.0, 5.0, 6.0],
                [11.0, 12.0, 13.0],
                [24.0, 25.0, 26.0],
            ]
        )
        self.assertTrue(torch.equal(result, expected))

    def test_normalization_and_gate_scale_the_added_feature(self):
        conditioner = FutureTokenDynamicsConditioner(
            hidden_dim=3,
            config=self._active_config(
                normalization_scale=[2.0, 4.0, 5.0],
                initial_gate=0.5,
            ),
        )
        conditioner.embedding = torch.nn.Identity()
        lookups = self._lookups()

        result = conditioner.add_selected(
            feature=torch.ones(1, 3),
            token_index=torch.tensor([1]),
            agent_type=torch.tensor([0]),
            dynamics_veh=lookups[0],
            dynamics_ped=lookups[1],
            dynamics_cyc=lookups[2],
        )

        expected = torch.tensor([[2.0, 1.625, 1.6]])
        self.assertTrue(torch.allclose(result, expected))

    def test_inactive_conditioner_is_exact_no_op_without_checkpoint_keys(self):
        conditioner = FutureTokenDynamicsConditioner(
            hidden_dim=8,
            config={"is_active": False},
        )
        feature = torch.randn(2, 4, 8)
        selected_feature = feature[:, 0]

        open_loop = conditioner.add_open_loop(
            feature=feature,
            token_index=torch.zeros(2, 4, dtype=torch.long),
            agent_type=torch.tensor([0, 1]),
        )
        selected = conditioner.add_selected(
            feature=selected_feature,
            token_index=torch.zeros(2, dtype=torch.long),
            agent_type=torch.tensor([0, 1]),
        )

        self.assertIs(open_loop, feature)
        self.assertIs(selected, selected_feature)
        self.assertEqual(conditioner.state_dict(), {})

    def test_active_state_has_embedding_and_gate_but_not_normalization_buffer(self):
        conditioner = FutureTokenDynamicsConditioner(
            hidden_dim=8,
            config=self._active_config(),
        )

        state_keys = set(conditioner.state_dict())

        self.assertIn("gate", state_keys)
        self.assertTrue(any(key.startswith("embedding.") for key in state_keys))
        self.assertNotIn("normalization_scale", state_keys)

    def test_invalid_normalization_scale_is_rejected(self):
        for scale in ([1.0, 2.0], [1.0, 0.0, 1.0]):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(
                    ValueError,
                    "future_token_dynamics.normalization_scale",
                ):
                    FutureTokenDynamicsConditioner(
                        hidden_dim=8,
                        config=self._active_config(normalization_scale=scale),
                    )

    def test_active_conditioner_requires_all_lookup_tables(self):
        conditioner = FutureTokenDynamicsConditioner(
            hidden_dim=3,
            config=self._active_config(),
        )

        with self.assertRaisesRegex(KeyError, "agent_token_dynamics"):
            conditioner.add_open_loop(
                feature=torch.zeros(1, 3, 3),
                token_index=torch.zeros(1, 3, dtype=torch.long),
                agent_type=torch.zeros(1, dtype=torch.long),
            )


class FutureTokenDynamicsDecoderTest(unittest.TestCase):
    class _TemporalIdentity(torch.nn.Module):
        def forward(self, feature, *args):
            if isinstance(feature, tuple):
                return feature[1]
            return feature

    class _BipartiteIdentity(torch.nn.Module):
        def forward(self, feature, *args):
            if isinstance(feature, tuple):
                return feature[1]
            return feature

    @staticmethod
    def _decoder(future_config, *, num_future_steps=10):
        return SMARTAgentDecoder(
            hidden_dim=3,
            num_historical_steps=11,
            num_future_steps=num_future_steps,
            time_span=30,
            pl2a_radius=30.0,
            a2a_radius=60.0,
            num_freq_bands=2,
            num_layers=1,
            num_heads=1,
            head_dim=3,
            dropout=0.0,
            hist_drop_prob=0.0,
            n_token_agent=2,
            endpoint_interpolation={"is_active": False},
            history_dynamics={"is_active": False},
            future_token_dynamics=future_config,
        )

    @staticmethod
    def _embedding_inputs():
        n_agent, n_step, n_token = 3, 4, 2
        position = torch.zeros(n_agent, n_step, 2)
        position[:, :, 0] = torch.arange(n_step, dtype=torch.float32)
        heading_vector = torch.zeros(n_agent, n_step, 2)
        heading_vector[..., 0] = 1.0
        trajectory = torch.arange(n_token * 8, dtype=torch.float32).view(
            n_token,
            8,
        )
        return {
            "agent_token_index": torch.tensor(
                [
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                    [0, 0, 1, 1],
                ]
            ),
            "trajectory_token_veh": trajectory,
            "trajectory_token_ped": trajectory + 20.0,
            "trajectory_token_cyc": trajectory + 40.0,
            "pos_a": position,
            "head_vector_a": heading_vector,
            "agent_type": torch.tensor([0, 1, 2]),
            "agent_shape": torch.ones(n_agent, 3),
        }

    @staticmethod
    def _lookups():
        vehicle = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pedestrian = vehicle + 10.0
        cyclist = vehicle + 20.0
        return {
            "agent_token_dynamics_veh": vehicle,
            "agent_token_dynamics_ped": pedestrian,
            "agent_token_dynamics_cyc": cyclist,
        }

    def test_teacher_forcing_masks_history_before_adding_own_token_dynamics(self):
        torch.manual_seed(17)
        disabled = self._decoder({"is_active": False})
        torch.manual_seed(23)
        active = self._decoder(
            {
                "is_active": True,
                "normalization_scale": [1.0, 1.0, 1.0],
                "initial_gate": 1.0,
            }
        )
        active.load_state_dict(disabled.state_dict(), strict=False)
        active.future_token_dynamics.embedding = torch.nn.Identity()
        inputs = self._embedding_inputs()

        reference = disabled.agent_token_embedding(**inputs)
        changed = active.agent_token_embedding(**inputs, **self._lookups())

        self.assertTrue(torch.equal(changed[:, :2], reference[:, :2]))
        expected_delta = torch.tensor(
            [
                [[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]],
                [[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]],
                [[24.0, 25.0, 26.0], [24.0, 25.0, 26.0]],
            ]
        )
        self.assertTrue(
            torch.allclose(
                changed[:, 2:] - reference[:, 2:],
                expected_delta,
            )
        )

    @classmethod
    def _rollout_inputs(cls, vehicle_lookup):
        time = torch.arange(6, dtype=torch.float32) * 0.1
        token_zero = FutureTokenDynamicsLookupTest._contours(
            torch.stack((time, torch.zeros_like(time)), dim=-1),
            torch.zeros_like(time),
        )
        token_one = FutureTokenDynamicsLookupTest._contours(
            torch.stack((2.0 * time, torch.zeros_like(time)), dim=-1),
            torch.zeros_like(time),
        )
        token_all = torch.stack((token_zero, token_one), dim=0)
        tokenized_agent = {
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
            "agent_token_dynamics_veh": vehicle_lookup,
            "agent_token_dynamics_ped": torch.zeros(2, 3),
            "agent_token_dynamics_cyc": torch.zeros(2, 3),
        }
        map_feature = {
            "position": torch.zeros(1, 2),
            "orientation": torch.zeros(1),
            "batch": torch.zeros(1, dtype=torch.long),
            "pt_token": torch.zeros(1, 3),
        }
        return tokenized_agent, map_feature

    @staticmethod
    def _fixed_sample(**kwargs):
        token_index = torch.ones(
            kwargs["pos_now"].shape[0],
            dtype=torch.long,
            device=kwargs["pos_now"].device,
        )
        agent_index = torch.arange(
            token_index.shape[0],
            device=token_index.device,
        )
        trajectory = kwargs["token_traj_all"][agent_index, token_index]
        return token_index, trajectory

    def test_rollout_lookup_cannot_change_first_logits_but_changes_next_logits(self):
        decoder = self._decoder(
            {
                "is_active": True,
                "normalization_scale": [1.0, 1.0, 1.0],
                "initial_gate": 1.0,
            },
            num_future_steps=10,
        )
        decoder.future_token_dynamics.embedding = torch.nn.Identity()
        decoder.t_attn_layers = torch.nn.ModuleList([self._TemporalIdentity()])
        decoder.pt2a_attn_layers = torch.nn.ModuleList([self._BipartiteIdentity()])
        decoder.a2a_attn_layers = torch.nn.ModuleList([self._TemporalIdentity()])
        decoder.build_temporal_edge = lambda **kwargs: (
            torch.empty(2, 0, dtype=torch.long),
            torch.empty(0, 3),
        )
        decoder.build_map2agent_edge = lambda **kwargs: (
            torch.empty(2, 0, dtype=torch.long),
            torch.empty(0, 3),
        )
        decoder.build_interaction_edge = lambda **kwargs: (
            torch.empty(2, 0, dtype=torch.long),
            torch.empty(0, 3),
        )
        decoder.token_predict_head = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            decoder.token_predict_head.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )
        decoder.train()
        baseline_inputs = self._rollout_inputs(
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        )
        changed_inputs = self._rollout_inputs(
            torch.tensor([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        )

        with patch(
            "src.smart.modules.agent_decoder.sample_next_token_traj",
            side_effect=self._fixed_sample,
        ):
            baseline = decoder.inference(
                *baseline_inputs,
                sampling_scheme=SimpleNamespace(),
            )
            changed = decoder.inference(
                *changed_inputs,
                sampling_scheme=SimpleNamespace(),
            )

        self.assertTrue(
            torch.equal(
                baseline["next_token_logits"][:, 0],
                changed["next_token_logits"][:, 0],
            )
        )
        self.assertFalse(
            torch.equal(
                baseline["next_token_logits"][:, 1],
                changed["next_token_logits"][:, 1],
            )
        )


class FutureTokenDynamicsConfigTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    ORIGINAL_EXPERIMENTS = {
        "pre_bc_history_future_token_dynamics": "pre_bc_history_dynamics",
        "clsft_history_future_token_dynamics": "clsft_history_dynamics",
        "inference_history_future_token_dynamics": "inference_history_dynamics",
    }
    RECONSTRUCTED_EXPERIMENTS = {
        f"{name}_reconstructed": name for name in ORIGINAL_EXPERIMENTS
    }

    @classmethod
    def _load_experiment(cls, name):
        path = cls.ROOT / "configs" / "experiment" / f"{name}.yaml"
        with path.open(encoding="utf-8") as file:
            return yaml.safe_load(file)

    def test_base_model_disables_future_token_dynamics(self):
        with (
            self.ROOT / "configs" / "model" / "smart.yaml"
        ).open(encoding="utf-8") as file:
            config = yaml.safe_load(file)

        future = config["model_config"]["future_token_dynamics"]

        self.assertFalse(future["is_active"])
        self.assertEqual(future["normalization_scale"], [5.0, 1.0, 5.0])
        self.assertEqual(future["initial_gate"], 1.0)

    def test_original_variants_inherit_history_and_enable_future_dynamics(self):
        for experiment, parent in self.ORIGINAL_EXPERIMENTS.items():
            with self.subTest(experiment=experiment):
                config = self._load_experiment(experiment)

                self.assertEqual(config["defaults"], [parent, "_self_"])
                model_config = config["model"]["model_config"]
                self.assertTrue(
                    model_config["future_token_dynamics"]["is_active"]
                )
                self.assertEqual(
                    model_config["token_processor"]["agent_token_file"],
                    "agent_vocab_555_s2.pkl",
                )

    def test_reconstructed_variants_only_override_vocabulary_file(self):
        for experiment, parent in self.RECONSTRUCTED_EXPERIMENTS.items():
            with self.subTest(experiment=experiment):
                config = self._load_experiment(experiment)

                self.assertEqual(set(config), {"defaults", "model"})
                self.assertEqual(config["defaults"], [parent, "_self_"])
                self.assertEqual(set(config["model"]), {"model_config"})
                model_config = config["model"]["model_config"]
                self.assertEqual(set(model_config), {"token_processor"})
                self.assertEqual(
                    model_config["token_processor"],
                    {"agent_token_file": "agent_vocab_reconstructed.pkl"},
                )

    def test_all_six_experiments_compose_with_both_dynamics_branches(self):
        try:
            import hydra
        except ModuleNotFoundError:
            self.skipTest("Hydra is not installed in this test environment")

        experiment_names = (
            list(self.ORIGINAL_EXPERIMENTS)
            + list(self.RECONSTRUCTED_EXPERIMENTS)
        )
        config_dir = self.ROOT / "configs"
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
                            "ckpt_path=/tmp/catk-placeholder.ckpt",
                        ],
                    )
                    model_config = config.model.model_config
                    self.assertTrue(model_config.history_dynamics.is_active)
                    self.assertTrue(
                        model_config.future_token_dynamics.is_active
                    )
                    expected_vocabulary = (
                        "agent_vocab_reconstructed.pkl"
                        if experiment.endswith("_reconstructed")
                        else "agent_vocab_555_s2.pkl"
                    )
                    self.assertEqual(
                        model_config.token_processor.agent_token_file,
                        expected_vocabulary,
                    )


if __name__ == "__main__":
    unittest.main()
