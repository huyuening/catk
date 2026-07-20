import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.smart.tokens.compare_trajectory_token_reconstruction import (
    _canonical_cache_paths,
    _local_segment,
    _resolve_input_tfrecords,
    _vocab_export_path,
    kdisk_cluster,
    polygon_contours,
)


class TrajectoryTokenComparisonTest(unittest.TestCase):
    def test_input_directory_resolves_only_canonical_tfrecord_shards(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_dir = Path(temporary_directory) / "training"
            input_dir.mkdir()
            second = input_dir / "training.tfrecord-00001-of-01000"
            first = input_dir / "training.tfrecord-00000-of-01000"
            first.touch()
            second.touch()
            (input_dir / "training.tfrecord-00000-of-01000-new").touch()
            (input_dir / "training_20s.tfrecord-00000-of-01000").touch()
            (input_dir / "README.txt").touch()

            self.assertEqual(
                _resolve_input_tfrecords(input_dir),
                [first.resolve(), second.resolve()],
            )

    def test_vocab_export_defaults_to_catk_token_directory(self):
        args = argparse.Namespace(
            vocab_output_dir="src/smart/tokens",
            vocab_output_name="agent_vocab_reconstructed.pkl",
        )

        output_path = _vocab_export_path(args)

        self.assertEqual(output_path.name, "agent_vocab_reconstructed.pkl")
        self.assertEqual(output_path.parent.name, "tokens")

    def test_macos_conflict_copies_are_not_consumed_as_scenarios(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            canonical = cache_dir / "0123456789abcdef.pkl"
            canonical.touch()
            (cache_dir / "0123456789abcdef 2.pkl").touch()

            self.assertEqual(_canonical_cache_paths(cache_dir), [canonical])

    def test_local_segment_uses_first_pose_as_origin(self):
        position = np.asarray(
            [[1.0, 2.0], [1.0, 3.0], [1.0, 4.0]], dtype=np.float32
        )
        heading = np.full(3, np.pi / 2.0, dtype=np.float32)

        local = _local_segment(position, heading)

        np.testing.assert_allclose(local[:, 0], [0.0, 1.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(local[:, 1:], 0.0, atol=1e-6)

    def test_polygon_contour_respects_width_and_length(self):
        contour = polygon_contours(
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            (2.0, 4.8),
        )[0]

        self.assertAlmostEqual(float(contour[:, 0].max()), 2.4, places=6)
        self.assertAlmostEqual(float(contour[:, 0].min()), -2.4, places=6)
        self.assertAlmostEqual(float(contour[:, 1].max()), 1.0, places=6)
        self.assertAlmostEqual(float(contour[:, 1].min()), -1.0, places=6)

    def test_kdisk_merges_endpoints_within_tolerance(self):
        trajectories = np.zeros((3, 6, 3), dtype=np.float32)
        trajectories[1, :, 0] = 0.01
        trajectories[2, :, 0] = 1.0

        tokens = kdisk_cluster(
            trajectories,
            num_clusters=3,
            tolerance_m=0.05,
            width_length=(1.0, 1.0),
            seed=2,
        )

        self.assertEqual(tokens.shape, (2, 6, 3))
        np.testing.assert_allclose(tokens[0, :, 0], 0.005, atol=1e-6)
        np.testing.assert_allclose(tokens[1, :, 0], 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
