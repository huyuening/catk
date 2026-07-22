import unittest

import numpy as np

from src.smart.tokens.history_dynamics import extract_history_dynamics


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


if __name__ == "__main__":
    unittest.main()
