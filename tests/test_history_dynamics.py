import unittest

import numpy as np
import torch

from src.smart.tokens.history_dynamics import (
    estimate_raw_history_dynamics,
    extract_history_dynamics,
)


class HistoryDynamicsTest(unittest.TestCase):
    @staticmethod
    def _trajectory(position_xy):
        position_xy = np.asarray(position_xy, dtype=np.float64)
        position = np.zeros((91, 3), dtype=np.float64)
        position[: len(position_xy), :2] = position_xy
        heading = np.zeros(91, dtype=np.float64)
        valid = np.zeros(91, dtype=bool)
        valid[: len(position_xy)] = True
        return position, heading, valid

    def test_constant_longitudinal_acceleration_is_sampled_for_both_tokens(self):
        time = np.arange(11, dtype=np.float64) * 0.1
        acceleration = 2.0
        x = 3.0 * time + 0.5 * acceleration * time**2
        position, heading, valid = self._trajectory(
            np.column_stack((x, np.zeros_like(x)))
        )

        result = extract_history_dynamics(
            position, heading, valid, agent_type=0, timestamps=time
        )

        self.assertEqual(result.values.shape, (2, 3))
        np.testing.assert_array_equal(result.valid, [True, True])
        np.testing.assert_allclose(result.values[:, 0], acceleration, atol=2e-2)
        np.testing.assert_allclose(result.values[:, 1:], 0.0, atol=2e-2)

    def test_reverse_body_heading_keeps_longitudinal_acceleration_signed(self):
        time = np.arange(11, dtype=np.float64) * 0.1
        x = 3.0 * time + time**2
        position, heading, valid = self._trajectory(
            np.column_stack((x, np.zeros_like(x)))
        )
        heading[:11] = np.pi

        result = extract_history_dynamics(
            position, heading, valid, agent_type=0, timestamps=time
        )

        np.testing.assert_allclose(result.values[:, 0], -2.0, atol=2e-2)

    def test_same_body_frame_definition_supports_every_catk_agent_type(self):
        time = np.arange(11, dtype=np.float64) * 0.1
        x = 3.0 * time + time**2
        position, heading, valid = self._trajectory(
            np.column_stack((x, np.zeros_like(x)))
        )

        for agent_type in (0, 1, 2):
            with self.subTest(agent_type=agent_type):
                result = extract_history_dynamics(
                    position,
                    heading,
                    valid,
                    agent_type=agent_type,
                    timestamps=time,
                )
                np.testing.assert_array_equal(result.valid, [True, True])
                np.testing.assert_allclose(result.values[:, 0], 2.0, atol=2e-2)

    def test_curved_xytheta_produces_angular_and_lateral_motion(self):
        time = np.arange(11, dtype=np.float64) * 0.1
        radius = 10.0
        angular_speed = 0.4
        theta = angular_speed * time
        position, heading, valid = self._trajectory(
            np.column_stack(
                (
                    radius * np.sin(theta),
                    radius * (1.0 - np.cos(theta)),
                )
            )
        )
        heading[:11] = theta

        result = extract_history_dynamics(
            position, heading, valid, agent_type=0, timestamps=time
        )

        np.testing.assert_allclose(
            result.values[:, 1], angular_speed, atol=3e-2
        )
        np.testing.assert_allclose(
            result.values[:, 2], radius * angular_speed**2, atol=6e-2
        )

    def test_internal_gaps_are_filled_in_one_eleven_frame_reconstruction(self):
        time = np.arange(11, dtype=np.float64) * 0.1
        x = 2.0 * time + time**2
        position, heading, valid = self._trajectory(
            np.column_stack((x, np.zeros_like(x)))
        )
        valid[:11] = False
        valid[[0, 2, 5, 7, 10]] = True

        result = extract_history_dynamics(
            position, heading, valid, agent_type=0, timestamps=time
        )

        np.testing.assert_array_equal(result.reconstructed_valid, np.ones(11, bool))
        np.testing.assert_array_equal(result.valid, [True, True])
        np.testing.assert_allclose(result.values[:, 0], 2.0, atol=7e-2)

    def test_sparse_token_support_is_masked_instead_of_embedded_as_zero_action(self):
        time = np.arange(11, dtype=np.float64) * 0.1
        position, heading, valid = self._trajectory(
            np.column_stack((time, np.zeros_like(time)))
        )
        valid[:11] = False
        valid[[0, 10]] = True

        result = extract_history_dynamics(
            position, heading, valid, agent_type=0, timestamps=time
        )

        np.testing.assert_array_equal(result.valid, [False, False])
        np.testing.assert_allclose(result.values, 0.0)

    def test_future_positions_and_headings_cannot_change_history_features(self):
        time = np.arange(91, dtype=np.float64) * 0.1
        x = time + 0.5 * time**2
        position, heading, valid = self._trajectory(
            np.column_stack((x, np.zeros_like(x)))
        )
        reference = extract_history_dynamics(
            position, heading, valid, agent_type=0, timestamps=time
        )

        position[11:] = np.random.default_rng(7).normal(size=(80, 3)) * 1000.0
        heading[11:] = np.random.default_rng(8).normal(size=80)
        changed = extract_history_dynamics(
            position, heading, valid, agent_type=0, timestamps=time
        )

        np.testing.assert_allclose(changed.values, reference.values)
        np.testing.assert_array_equal(changed.valid, reference.valid)

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "position"):
            extract_history_dynamics(
                np.zeros((11,)),
                np.zeros(11),
                np.ones(11, dtype=bool),
                agent_type=0,
            )


class RawHistoryDynamicsTest(unittest.TestCase):
    @staticmethod
    def _history(x, y=None, heading=None):
        x = torch.as_tensor(x, dtype=torch.float32)
        y = (
            torch.zeros_like(x)
            if y is None
            else torch.as_tensor(y, dtype=torch.float32)
        )
        position = torch.zeros(1, len(x), 3, dtype=torch.float32)
        position[0, :, :2] = torch.stack((x, y), dim=-1)
        if heading is None:
            heading = torch.zeros(1, len(x), dtype=torch.float32)
        else:
            heading = torch.as_tensor(
                heading, dtype=torch.float32
            ).view(1, -1)
        valid = torch.ones(1, len(x), dtype=torch.bool)
        return position, heading, valid

    def test_quadratic_motion_returns_endpoint_body_acceleration(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        acceleration = 2.0
        x = 3.0 * time + 0.5 * acceleration * time.square()
        position, heading, valid = self._history(x)

        values, feature_valid = estimate_raw_history_dynamics(
            position, heading, valid
        )

        self.assertEqual(tuple(values.shape), (1, 2, 3))
        torch.testing.assert_close(
            feature_valid, torch.ones(1, 2, dtype=torch.bool)
        )
        torch.testing.assert_close(
            values[0, :, 0],
            torch.full((2,), acceleration),
            atol=2e-4,
            rtol=0,
        )
        torch.testing.assert_close(
            values[0, :, 1:], torch.zeros(2, 2)
        )

    def test_constant_velocity_has_zero_dynamics(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        position, heading, valid = self._history(4.0 * time)

        values, feature_valid = estimate_raw_history_dynamics(
            position, heading, valid
        )

        torch.testing.assert_close(
            values, torch.zeros(1, 2, 3), atol=2e-4, rtol=0
        )
        self.assertTrue(feature_valid.all())

    def test_heading_difference_wraps_across_pi(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        heading = torch.zeros(11)
        heading[4], heading[5] = torch.pi - 0.01, -torch.pi + 0.01
        heading[9], heading[10] = torch.pi - 0.01, -torch.pi + 0.01
        position, heading, valid = self._history(time, heading=heading)

        values, feature_valid = estimate_raw_history_dynamics(
            position, heading, valid
        )

        torch.testing.assert_close(
            values[0, :, 1],
            torch.full((2,), 0.2),
            atol=2e-4,
            rtol=0,
        )
        self.assertTrue(feature_valid.all())

    def test_invalid_or_nonfinite_support_is_zero_and_masked(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        position, heading, valid = self._history(time.square())
        valid[0, 3] = False
        position[0, 9, 0] = float("nan")

        values, feature_valid = estimate_raw_history_dynamics(
            position, heading, valid
        )

        torch.testing.assert_close(
            feature_valid, torch.zeros(1, 2, dtype=torch.bool)
        )
        torch.testing.assert_close(values, torch.zeros(1, 2, 3))

    def test_feature_ranges_are_clipped(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        x = 50.0 * time.square()
        heading = torch.arange(11, dtype=torch.float32)
        position, heading, valid = self._history(x, heading=heading)

        values, feature_valid = estimate_raw_history_dynamics(
            position, heading, valid
        )

        self.assertTrue(feature_valid.all())
        self.assertTrue(
            torch.all(
                values.abs() <= torch.tensor([15.0, 3.0, 15.0])
            )
        )
        torch.testing.assert_close(
            values[0, :, 1], torch.full((2,), 3.0)
        )

    def test_malformed_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "position"):
            estimate_raw_history_dynamics(
                torch.zeros(11),
                torch.zeros(1, 11),
                torch.ones(1, 11, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "dt"):
            estimate_raw_history_dynamics(
                torch.zeros(1, 11, 3),
                torch.zeros(1, 11),
                torch.ones(1, 11, dtype=torch.bool),
                dt=0.0,
            )


if __name__ == "__main__":
    unittest.main()
