import unittest

import torch

from src.smart.tokens.history_dynamics import estimate_history_dynamics


class HistoryDynamicsTest(unittest.TestCase):
    def _trajectory(self, position_xy, agent_type=0):
        position = torch.zeros(1, 91, 2, dtype=torch.float32)
        position[0, : position_xy.size(0)] = position_xy
        valid = torch.zeros(1, 91, dtype=torch.bool)
        valid[0, : position_xy.size(0)] = True
        types = torch.tensor([agent_type], dtype=torch.uint8)
        return position, valid, types

    def test_constant_longitudinal_acceleration(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        acceleration = 2.0
        x = 3.0 * time + 0.5 * acceleration * time.square()
        position, valid, agent_type = self._trajectory(
            torch.stack([x, torch.zeros_like(x)], dim=-1)
        )

        dynamics = estimate_history_dynamics(position, valid, agent_type)

        self.assertEqual(dynamics.shape, (1, 2, 3))
        torch.testing.assert_close(
            dynamics[0, :, 0],
            torch.full((2,), acceleration),
            atol=2e-2,
            rtol=0,
        )
        torch.testing.assert_close(
            dynamics[0, :, 1:], torch.zeros(2, 2), atol=2e-2, rtol=0
        )

    def test_motion_frame_definition_is_shared_by_all_agent_types(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        x = 2.0 * time + 0.5 * time.square()
        position_xy = torch.stack([x, torch.zeros_like(x)], dim=-1)
        outputs = []
        for agent_type_value in range(3):
            position, valid, agent_type = self._trajectory(
                position_xy, agent_type=agent_type_value
            )
            outputs.append(estimate_history_dynamics(position, valid, agent_type))

        torch.testing.assert_close(outputs[0], outputs[1])
        torch.testing.assert_close(outputs[0], outputs[2])

    def test_first_token_uses_the_shared_full_history_reconstruction(self):
        time_first = torch.arange(6, dtype=torch.float32) * 0.1
        time_second = torch.arange(6, dtype=torch.float32) * 0.1
        position_xy = torch.zeros(11, 2)
        position_xy[:6, 0] = 2.0 * time_first
        position_xy[5:, 0] = (
            position_xy[5, 0]
            + 2.0 * time_second
            + 0.5 * 2.0 * time_second.square()
        )
        position, valid, agent_type = self._trajectory(position_xy)

        dynamics = estimate_history_dynamics(position, valid, agent_type)

        # A separately fitted first interval would remain at zero acceleration.
        # The non-zero value proves both token features use the one 11-frame fit.
        self.assertGreater(abs(float(dynamics[0, 0, 0])), 0.1)

    def test_curved_motion_has_consistent_omega_and_lateral_acceleration(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        radius = 10.0
        angular_speed = 0.4
        angle = angular_speed * time
        position_xy = torch.stack(
            [radius * torch.sin(angle), radius * (1.0 - torch.cos(angle))], dim=-1
        )
        position, valid, agent_type = self._trajectory(position_xy)

        dynamics = estimate_history_dynamics(
            position,
            valid,
            agent_type,
            min_speed_mps=(0.1, 0.1, 0.1),
        )

        self.assertTrue(torch.all(dynamics[0, :, 1] > 0.0))
        self.assertTrue(torch.all(dynamics[0, :, 2] > 0.0))
        torch.testing.assert_close(
            dynamics[0, :, 2],
            dynamics[0, :, 1] * (radius * angular_speed),
            atol=0.15,
            rtol=0.05,
        )

    def test_stationary_agents_are_finite_for_all_agent_types(self):
        position_xy = torch.zeros(11, 2)
        for agent_type_value in range(3):
            position, valid, agent_type = self._trajectory(
                position_xy, agent_type=agent_type_value
            )
            dynamics = estimate_history_dynamics(position, valid, agent_type)
            self.assertTrue(torch.isfinite(dynamics).all())
            torch.testing.assert_close(dynamics, torch.zeros_like(dynamics))

    def test_future_positions_cannot_change_history_features(self):
        time = torch.arange(91, dtype=torch.float32) * 0.1
        base_position = torch.stack([time, torch.zeros_like(time)], dim=-1)
        position, valid, agent_type = self._trajectory(base_position)
        reference = estimate_history_dynamics(position, valid, agent_type)

        position[:, 11:] = torch.randn_like(position[:, 11:]) * 1000.0
        changed = estimate_history_dynamics(position, valid, agent_type)

        torch.testing.assert_close(changed, reference)

    def test_global_translation_does_not_change_features(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        position_xy = torch.stack(
            [2.0 * time + 0.5 * time.square(), 0.25 * time.square()], dim=-1
        )
        position, valid, agent_type = self._trajectory(position_xy)
        reference = estimate_history_dynamics(position, valid, agent_type)

        translated = position.clone()
        translated[:, :11] += torch.tensor([10_000.0, -20_000.0])
        changed = estimate_history_dynamics(translated, valid, agent_type)

        torch.testing.assert_close(changed, reference, atol=2e-2, rtol=1e-2)

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "position"):
            estimate_history_dynamics(
                torch.zeros(11, 2),
                torch.ones(11, dtype=torch.bool),
                torch.tensor([0]),
            )


if __name__ == "__main__":
    unittest.main()
