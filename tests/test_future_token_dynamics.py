import importlib.util
import math
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

if not hasattr(torch, "arctan2"):
    torch.arctan2 = torch.atan2

from src.smart.tokens.future_token_dynamics import (
    build_future_token_dynamics_lookup,
    gather_future_token_dynamics,
)


def _load_token_processor():
    if importlib.util.find_spec("omegaconf") is None:
        omegaconf_stub = types.ModuleType("omegaconf")
        omegaconf_stub.DictConfig = dict
        sys.modules["omegaconf"] = omegaconf_stub

    if importlib.util.find_spec("torch_geometric") is None:
        torch_geometric_stub = types.ModuleType("torch_geometric")
        torch_geometric_stub.__path__ = []
        torch_geometric_data_stub = types.ModuleType("torch_geometric.data")
        torch_geometric_data_stub.HeteroData = dict
        torch_geometric_stub.data = torch_geometric_data_stub
        sys.modules["torch_geometric"] = torch_geometric_stub
        sys.modules["torch_geometric.data"] = torch_geometric_data_stub

    from src.smart.tokens.token_processor import TokenProcessor

    return TokenProcessor


TokenProcessor = _load_token_processor()


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
    def _processor(active=True):
        processor = TokenProcessor.__new__(TokenProcessor)
        torch.nn.Module.__init__(processor)
        processor.future_token_dynamics_active = active
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

    def test_active_processor_registers_class_separated_nonpersistent_lookups(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_pickle(directory, self._vocabulary())
            processor = self._processor(active=True)

            processor.init_agent_token(str(path))

        self.assertEqual(tuple(processor.agent_token_dynamics_veh.shape), (2, 3))
        self.assertEqual(tuple(processor.agent_token_dynamics_ped.shape), (2, 3))
        self.assertEqual(tuple(processor.agent_token_dynamics_cyc.shape), (2, 3))
        self.assertTrue(
            torch.allclose(
                processor.agent_token_dynamics_veh[:, 0],
                torch.tensor([1.0, 2.0]),
                atol=1e-4,
                rtol=1e-4,
            )
        )
        self.assertFalse(
            torch.equal(
                processor.agent_token_dynamics_veh,
                processor.agent_token_dynamics_ped,
            )
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
            processor = self._processor(active=True)
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


if __name__ == "__main__":
    unittest.main()
