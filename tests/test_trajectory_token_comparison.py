import argparse
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.smart.tokens.compare_trajectory_token_reconstruction import (
    _canonical_cache_paths,
    _local_segment,
    _resolve_input_tfrecords,
    _vocab_export_path,
    collect_segments,
    kdisk_cluster,
    polygon_contours,
    validate_cache_pairs,
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

    def test_existing_cache_zero_shape_is_reported_but_still_collected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original_dir = root / "original"
            reconstructed_dir = root / "reconstructed"
            original_dir.mkdir()
            reconstructed_dir.mkdir()

            valid_mask = np.ones((2, 91), dtype=bool)
            position = np.zeros((2, 91, 3), dtype=np.float32)
            position[0, :, 0] = np.arange(91, dtype=np.float32) * 0.1
            position[1, :, 0] = np.arange(91, dtype=np.float32) * 0.2
            base_agent = {
                "num_nodes": 2,
                "valid_mask": valid_mask,
                "role": np.zeros((2, 3), dtype=bool),
                "id": np.asarray([1, 2], dtype=np.int64),
                "type": np.asarray([0, 0], dtype=np.uint8),
                "position": position,
                "heading": np.zeros((2, 91), dtype=np.float32),
                "velocity": np.zeros((2, 91, 2), dtype=np.float32),
                "shape": np.asarray(
                    [[4.8, 2.0, 1.5], [4.8, 0.0, 1.5]], dtype=np.float32
                ),
            }
            for cache_dir, reconstructed in (
                (original_dir, False),
                (reconstructed_dir, True),
            ):
                agent = dict(base_agent)
                agent["trajectory_reconstructed"] = np.full(
                    2, reconstructed, dtype=bool
                )
                cache = {
                    "scenario_id": "zero-shape",
                    "current_time_index": 10,
                    "agent": agent,
                }
                with (cache_dir / "zero-shape.pkl").open("wb") as stream:
                    pickle.dump(cache, stream)

            validation = validate_cache_pairs(original_dir, reconstructed_dir)
            segments, candidates = collect_segments(original_dir, max_per_class=100)

            self.assertEqual(
                validation["unresolved_last_history_shape_agent_count"], 1
            )
            self.assertEqual(candidates["veh"], 36)
            self.assertAlmostEqual(float(segments["veh"][:, -1, 0].max()), 1.0)


if __name__ == "__main__":
    unittest.main()
