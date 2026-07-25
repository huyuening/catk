import argparse
import pickle
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.smart.tokens.compare_trajectory_token_reconstruction import (
    WorkerConfig,
    _canonical_cache_paths,
    _load_scenario_class,
    _local_segment,
    _process_scenario_task,
    _reconstruction_implementation,
    _resolve_input_tfrecords,
    _vocab_export_path,
    _write_output_readme,
    collect_segments,
    kdisk_cluster,
    parse_args,
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

    def test_reconstruction_provenance_distinguishes_bundled_batch(self):
        self.assertEqual(
            _reconstruction_implementation("filter", None),
            "catk_bundled_filter",
        )
        self.assertEqual(
            _reconstruction_implementation("batch", None),
            "catk_bundled_batch",
        )
        self.assertEqual(
            _reconstruction_implementation("batch", "/external"),
            "external",
        )

    def test_parser_accepts_bundled_batch_without_external_root(self):
        with patch.object(
            sys,
            "argv",
            ["compare", "--input-path", "/training", "--method", "batch"],
        ):
            args = parse_args()

        self.assertEqual(args.method, "batch")
        self.assertIsNone(args.reconstruction_root)

    def test_bundled_batch_reproduction_command_omits_external_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            args = argparse.Namespace(
                num_clusters=2048,
                write_reconstructed_tfrecord=False,
                vocab_output_dir="src/smart/tokens",
                vocab_output_name="agent_vocab_reconstructed_batch.pkl",
                input_tfrecord="/training",
                reconstruction_root=None,
                method="batch",
                filter_strength="strong",
                num_workers=24,
                worker_backend="process",
            )
            _write_output_readme(output_dir, args)
            reproduction = (output_dir / "README.md").read_text()

        self.assertIn("--method batch", reproduction)
        self.assertNotIn("--reconstruction-root", reproduction)

    def test_worker_runs_bundled_batch_without_external_checkout(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            scenario_class = _load_scenario_class()
        scenario = scenario_class()
        scenario.scenario_id = "bundled-batch-smoke"
        scenario.current_time_index = 10
        scenario.sdc_track_index = 0
        scenario.timestamps_seconds.extend(
            (np.arange(91, dtype=float) * 0.1).tolist()
        )
        track = scenario.tracks.add()
        track.id = 1
        track.object_type = 1
        for index in range(91):
            state = track.states.add()
            state.center_x = 0.2 * index
            state.center_y = 0.0
            state.heading = 0.0
            state.length = 4.8
            state.width = 2.0
            state.height = 1.5
            state.velocity_x = 2.0
            state.valid = True

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original_dir = root / "original"
            reconstructed_dir = root / "reconstructed"
            original_dir.mkdir()
            reconstructed_dir.mkdir()
            config = WorkerConfig(
                reconstruction_root=None,
                method="batch",
                filter_strength="strong",
                max_gap_frames=-1,
                batch_linear_jerk_weight=1.0,
                batch_angular_jerk_weight=1.0,
                serialize_reconstructed=False,
                original_cache_dir=str(original_dir),
                reconstructed_cache_dir=str(reconstructed_dir),
            )

            result = _process_scenario_task(
                (0, scenario.SerializeToString(), config)
            )

            self.assertEqual(result["scenario_id"], "bundled-batch-smoke")
            self.assertEqual(result["original_agent_count"], 1)
            self.assertEqual(result["reconstructed_agent_count"], 1)
            self.assertTrue(
                (original_dir / "bundled-batch-smoke.pkl").is_file()
            )
            self.assertTrue(
                (reconstructed_dir / "bundled-batch-smoke.pkl").is_file()
            )

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
