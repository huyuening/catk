import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.smart.modules.endpoint_interpolation import EndpointInterpolator
from tools.compare_endpoint_interpolation import (
    endpoint_delta_summary,
    motion_split_summary,
)


class EndpointComparisonTest(unittest.TestCase):
    def test_endpoint_delta_uses_every_fifth_future_step(self):
        raw = np.zeros((1, 1, 80, 4), dtype=np.float32)
        post = raw.copy()
        post[..., 4::5, 0] = 2.0
        post[..., 4::5, 3] = 0.25

        summary = endpoint_delta_summary(raw, post)

        self.assertEqual(summary["indices"], list(range(4, 80, 5)))
        self.assertAlmostEqual(summary["mean_m"], 2.0)
        self.assertAlmostEqual(summary["max_m"], 2.0)
        self.assertAlmostEqual(summary["heading_mean_rad"], 0.25, places=6)

    def test_motion_modes_follow_endpoint_interpolator_policy(self):
        config = {
            "is_active": True,
            "low_speed_reconstruction": True,
            "low_speed_min_segment_threshold_mps": 0.1,
            "static_reconstruction": True,
            "static_max_segment_speed_threshold_mps": 0.5,
            "static_endpoint_span_threshold_m": 1.0,
            "static_pedestrian_max_segment_speed_threshold_mps": 0.2,
            "static_pedestrian_endpoint_span_threshold_m": 0.5,
            "static_cyclist_max_segment_speed_threshold_mps": 0.3,
            "static_cyclist_endpoint_span_threshold_m": 0.7,
        }
        interpolator = EndpointInterpolator(config=config, shift=5)
        tokenized_agent = {
            "gt_pos": torch.zeros(2, 2, 2),
            "type": torch.tensor([0, 0]),
        }
        decoder = SimpleNamespace(
            num_historical_steps=11,
            shift=5,
            endpoint_interpolator=interpolator,
        )
        model = SimpleNamespace(
            encoder=SimpleNamespace(agent_encoder=decoder),
            token_processor=lambda data: (None, tokenized_agent),
        )

        raw = np.zeros((1, 2, 80, 4), dtype=np.float32)
        moving_endpoints = np.arange(1, 17, dtype=np.float32)
        raw[0, 0, 4::5, 0] = moving_endpoints
        for segment in range(16):
            start = 0.0 if segment == 0 else moving_endpoints[segment - 1]
            end = moving_endpoints[segment]
            raw[0, 0, segment * 5 : (segment + 1) * 5, 0] = np.linspace(
                start, end, 6, dtype=np.float32
            )[1:]
        raw[0, 1, :, 0] = np.linspace(0.0, 0.8, 80, dtype=np.float32)

        summary = motion_split_summary(model, None, raw)

        self.assertEqual(summary["mode"][0, 0], "endpoint_interpolation")
        self.assertEqual(summary["mode"][0, 1], "static_reconstruction")
        self.assertEqual(summary["counts"]["endpoint_interpolation"], 1)
        self.assertEqual(summary["counts"]["static_reconstruction"], 1)


if __name__ == "__main__":
    unittest.main()
