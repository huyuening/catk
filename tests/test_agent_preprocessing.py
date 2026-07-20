import unittest

import numpy as np
import torch

from src.agent_preprocessing import get_agent_features


def _track_infos():
    states = np.zeros((1, 91, 9), dtype=np.float32)
    states[0, :, 0] = np.arange(91, dtype=np.float32)
    states[0, :, 3:6] = np.array([4.0, 1.8, 1.4], dtype=np.float32)
    states[0, 10, 3:6] = np.array([4.8, 2.0, 1.5], dtype=np.float32)
    states[0, 90, 3:6] = np.array([20.0, 20.0, 20.0], dtype=np.float32)

    valid = np.zeros((1, 91), dtype=bool)
    valid[0, [0, 2, 10, 90]] = True
    return {
        "object_id": np.array([123], dtype=np.int64),
        "object_type": np.array([0], dtype=np.uint8),
        "states": states,
        "valid": valid,
        "role": np.array([[True, False, True]], dtype=bool),
    }


class AgentPreprocessingTest(unittest.TestCase):
    def test_shape_comes_from_last_history_frame(self):
        features = get_agent_features(
            _track_infos(),
            split="training",
            num_historical_steps=11,
            num_steps=91,
        )

        torch.testing.assert_close(
            features["shape"][0], torch.tensor([4.8, 2.0, 1.5])
        )

    def test_support_keeps_legacy_interpolation(self):
        features = get_agent_features(
            _track_infos(),
            split="training",
            num_historical_steps=11,
            num_steps=91,
        )

        self.assertTrue(features["valid_mask"][0, 1])
        self.assertNotIn("trajectory_reconstructed", features)


if __name__ == "__main__":
    unittest.main()
