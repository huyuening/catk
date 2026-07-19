import unittest

import torch

from src.smart.modules.endpoint_interpolation import EndpointInterpolator


class EndpointInterpolatorTest(unittest.TestCase):
    @staticmethod
    def _curved_endpoints(n_agent: int = 1):
        step = torch.arange(1, 17, dtype=torch.float32)
        endpoints = torch.stack([step, 0.02 * step.square()], dim=-1)
        return endpoints.unsqueeze(0).expand(n_agent, -1, -1).clone()

    def test_global_cubic_preserves_generated_endpoints(self):
        interpolator = EndpointInterpolator(
            {
                "is_active": True,
                "method": "global_cubic",
                "heading_method": "endpoint_cubic",
                "low_speed_reconstruction": False,
                "moving_only": False,
                "moving_segment_only": False,
                "smooth_output": False,
            }
        )
        start_pos = torch.zeros(1, 2)
        start_head = torch.zeros(1)
        endpoint_pos = self._curved_endpoints()
        displacement = torch.cat(
            [
                endpoint_pos[:, :1] - start_pos.unsqueeze(1),
                endpoint_pos[:, 1:] - endpoint_pos[:, :-1],
            ],
            dim=1,
        )
        endpoint_head = torch.atan2(displacement[..., 1], displacement[..., 0])

        pred_traj, pred_head = interpolator.reconstruct(
            raw_traj=torch.randn(1, 80, 2),
            raw_head=torch.randn(1, 80),
            start_pos=start_pos,
            start_head=start_head,
            endpoint_pos=endpoint_pos,
            endpoint_head=endpoint_head,
            agent_type=torch.zeros(1, dtype=torch.long),
        )

        self.assertEqual(tuple(pred_traj.shape), (1, 80, 2))
        self.assertEqual(tuple(pred_head.shape), (1, 80))
        self.assertTrue(torch.isfinite(pred_traj).all())
        self.assertTrue(torch.isfinite(pred_head).all())
        self.assertTrue(torch.allclose(pred_traj[:, 4::5], endpoint_pos))
        self.assertTrue(torch.allclose(pred_head[:, 4::5], endpoint_head))

    def test_inactive_module_is_an_exact_no_op(self):
        interpolator = EndpointInterpolator({"is_active": False})
        raw_traj = torch.randn(1, 80, 2)
        raw_head = torch.randn(1, 80)
        pred_traj, pred_head = interpolator.reconstruct(
            raw_traj=raw_traj,
            raw_head=raw_head,
            start_pos=torch.zeros(1, 2),
            start_head=torch.zeros(1),
            endpoint_pos=torch.zeros(1, 16, 2),
            endpoint_head=torch.zeros(1, 16),
        )
        self.assertIs(pred_traj, raw_traj)
        self.assertIs(pred_head, raw_head)

    def test_low_speed_and_static_reconstruction_match_trajtok_policy(self):
        interpolator = EndpointInterpolator(
            {
                "is_active": True,
                "method": "global_cubic",
                "heading_method": "endpoint_cubic",
                "low_speed_reconstruction": True,
                "low_speed_min_segment_threshold_mps": 0.1,
                "low_speed_heading_method": "endpoint_smoothstep",
                "static_reconstruction": True,
                "static_max_segment_speed_threshold_mps": 0.5,
                "static_endpoint_span_threshold_m": 1.0,
                "static_pedestrian_max_segment_speed_threshold_mps": 0.2,
                "static_pedestrian_endpoint_span_threshold_m": 0.5,
                "static_cyclist_max_segment_speed_threshold_mps": 0.3,
                "static_cyclist_endpoint_span_threshold_m": 0.7,
                "smooth_output": True,
                "xy_smoothing_iterations": 0,
                "heading_smoothing_iterations": 0,
            }
        )
        start_pos = torch.zeros(2, 2)
        start_head = torch.zeros(2)
        endpoint_pos = self._curved_endpoints(2)
        # One stopped segment selects conservative low-speed interpolation.
        endpoint_pos[0, 7] = endpoint_pos[0, 6]
        # A vehicle whose endpoints stay within 1 m is reconstructed as static.
        endpoint_pos[1, :, 0] = torch.linspace(0.01, 0.8, 16)
        endpoint_pos[1, :, 1] = 0.0
        endpoint_head = torch.zeros(2, 16)

        pred_traj, _ = interpolator.reconstruct(
            raw_traj=torch.randn(2, 80, 2),
            raw_head=torch.randn(2, 80),
            start_pos=start_pos,
            start_head=start_head,
            endpoint_pos=endpoint_pos,
            endpoint_head=endpoint_head,
            agent_type=torch.zeros(2, dtype=torch.long),
        )

        self.assertTrue(torch.allclose(pred_traj[0, 4::5], endpoint_pos[0]))
        self.assertTrue(torch.equal(pred_traj[1], torch.zeros_like(pred_traj[1])))

    def test_moving_only_policy_leaves_other_states_raw(self):
        interpolator = EndpointInterpolator(
            {
                "is_active": True,
                "method": "global_cubic",
                "heading_method": "endpoint_cubic",
                "moving_only": True,
                "moving_speed_threshold_mps": 0.5,
                "moving_segment_only": True,
                "moving_segment_speed_threshold_mps": 0.5,
                "low_speed_reconstruction": False,
                "static_reconstruction": False,
                "smooth_output": False,
            }
        )
        start_pos = torch.zeros(2, 2)
        start_head = torch.zeros(2)
        endpoint_pos = self._curved_endpoints(2)
        # Agent 0 is moving overall but has one stopped segment.
        endpoint_pos[0, 7] = endpoint_pos[0, 6]
        # Agent 1 stays below the moving threshold for the entire rollout.
        endpoint_pos[1, :, 0] = torch.linspace(0.01, 0.16, 16)
        endpoint_pos[1, :, 1] = 0.0
        endpoint_head = torch.zeros(2, 16)
        raw_traj = torch.randn(2, 80, 2)
        raw_head = torch.randn(2, 80)

        pred_traj, pred_head = interpolator.reconstruct(
            raw_traj=raw_traj,
            raw_head=raw_head,
            start_pos=start_pos,
            start_head=start_head,
            endpoint_pos=endpoint_pos,
            endpoint_head=endpoint_head,
            agent_type=torch.zeros(2, dtype=torch.long),
        )

        stopped_segment = slice(7 * 5, 8 * 5)
        self.assertTrue(
            torch.equal(pred_traj[0, stopped_segment], raw_traj[0, stopped_segment])
        )
        self.assertTrue(
            torch.equal(pred_head[0, stopped_segment], raw_head[0, stopped_segment])
        )
        self.assertTrue(torch.equal(pred_traj[1], raw_traj[1]))
        self.assertTrue(torch.equal(pred_head[1], raw_head[1]))
        self.assertFalse(torch.equal(pred_traj[0, :5], raw_traj[0, :5]))

    def test_rejects_incompatible_future_length(self):
        interpolator = EndpointInterpolator({"is_active": True})
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            interpolator.reconstruct(
                raw_traj=torch.zeros(1, 79, 2),
                raw_head=torch.zeros(1, 79),
                start_pos=torch.zeros(1, 2),
                start_head=torch.zeros(1),
                endpoint_pos=torch.zeros(1, 16, 2),
                endpoint_head=torch.zeros(1, 16),
            )


if __name__ == "__main__":
    unittest.main()
