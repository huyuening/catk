import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.smart.tokens.exact_metric_store import (
    BufferKey,
    EvaluationCheckpoint,
    EvaluationIdentity,
    ExactMetricStore,
    load_checkpoint,
    restore_checkpoint,
    write_checkpoint,
)
from src.smart.tokens.reconstruction_evaluation import ScenarioMetricBatch


def metric_batch(
    scope: str,
    values: np.ndarray,
) -> ScenarioMetricBatch:
    values = np.asarray(values, dtype=np.float64)
    return ScenarioMetricBatch(
        scenario_id="scenario",
        agent_count=len(values),
        agent_values={scope: {"xy_rmse_m": values}},
        frame_values={
            scope: {
                "raw_linear_jerk_mps3": values - 50.0,
            }
        },
    )


def identity(size: int = 100) -> EvaluationIdentity:
    return EvaluationIdentity(
        input_shards=(
            {
                "path": "/data/training.tfrecord-00000-of-01000",
                "size": size,
                "mtime_ns": 123,
            },
        ),
        reconstruction={
            "method": "batch",
            "filter_strength": "strong",
            "max_gap_frames": -1,
            "batch_linear_jerk_weight": 1.0,
            "batch_angular_jerk_weight": 1.0,
        },
        max_scenarios=None,
        metric_schema="exact-reconstruction-v1",
    )


class ExactMetricStoreTest(unittest.TestCase):
    def test_percentiles_match_numpy_linear_method(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExactMetricStore(Path(directory))
            values = np.arange(100, dtype=np.float64)
            store.append_batch(metric_batch("vehicle", values))
            store.flush_and_sync()

            actual = store.percentiles(
                BufferKey("agent", "vehicle", "xy_rmse_m")
            )
            store.close()

        self.assertEqual(actual.p01, 0.99)
        self.assertEqual(actual.p99, 98.01)
        self.assertAlmostEqual(actual.p99_minus_p01, 97.02)

    def test_percentiles_match_numpy_for_signed_odd_and_singleton_streams(self):
        generator = np.random.default_rng(17)
        streams = (
            np.asarray([3.5], dtype=np.float64),
            np.asarray([-9.0, -2.0, 0.0, 4.0, 11.0]),
            generator.normal(size=101).astype(np.float64),
        )
        for values in streams:
            with self.subTest(count=len(values)):
                with tempfile.TemporaryDirectory() as directory:
                    store = ExactMetricStore(Path(directory))
                    store.append_batch(metric_batch("vehicle", values))
                    expected = np.percentile(
                        values,
                        [1.0, 99.0],
                        method="linear",
                    )

                    actual = store.percentiles(
                        BufferKey(
                            "agent",
                            "vehicle",
                            "xy_rmse_m",
                        )
                    )
                    store.close()

                self.assertAlmostEqual(
                    actual.p01,
                    float(expected[0]),
                )
                self.assertAlmostEqual(
                    actual.p99,
                    float(expected[1]),
                )

    def test_store_filters_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExactMetricStore(Path(directory))
            store.append_batch(
                metric_batch(
                    "vehicle",
                    np.asarray([1.0, np.nan, np.inf, 4.0]),
                )
            )

            counts = store.snapshot_counts()
            store.close()

        key = BufferKey(
            "agent",
            "vehicle",
            "xy_rmse_m",
        ).encoded
        self.assertEqual(counts[key], 2)

    def test_combined_percentiles_use_every_type(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ExactMetricStore(root)
            store.append_batch(
                metric_batch("vehicle", np.asarray([0.0, 1.0]))
            )
            store.append_batch(
                metric_batch("pedestrian", np.asarray([8.0, 9.0]))
            )
            store.flush_and_sync()

            actual = store.combined_percentiles(
                "agent",
                "xy_rmse_m",
                ["pedestrian", "vehicle"],
            )
            combined_directory = root / "combined"
            store.close()

        self.assertAlmostEqual(actual.p01, 0.03)
        self.assertAlmostEqual(actual.p99, 8.97)
        self.assertFalse(combined_directory.exists())

    def test_empty_stream_has_null_percentiles(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExactMetricStore(Path(directory))

            actual = store.percentiles(
                BufferKey("agent", "vehicle", "xy_rmse_m")
            )
            store.close()

        self.assertIsNone(actual.p01)
        self.assertIsNone(actual.p99)
        self.assertIsNone(actual.p99_minus_p01)

    def test_truncate_discards_uncommitted_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ExactMetricStore(Path(directory))
            store.append_batch(
                metric_batch("vehicle", np.asarray([1.0, 2.0]))
            )
            store.flush_and_sync()
            committed = store.snapshot_counts()
            store.append_batch(
                metric_batch("vehicle", np.asarray([100.0, 200.0]))
            )
            store.flush_and_sync()

            store.truncate_to(committed)
            actual = store.percentiles(
                BufferKey("agent", "vehicle", "xy_rmse_m")
            )
            counts = store.snapshot_counts()
            store.close()

        key = BufferKey(
            "agent",
            "vehicle",
            "xy_rmse_m",
        ).encoded
        self.assertEqual(counts[key], 2)
        self.assertAlmostEqual(actual.p01, 1.01)
        self.assertAlmostEqual(actual.p99, 1.99)


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_round_trip_and_identity_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            checkpoint = EvaluationCheckpoint(
                identity=identity(),
                completed_shards=[
                    "/data/training.tfrecord-00000-of-01000"
                ],
                buffer_counts={},
                accumulator_state={
                    "scenarios": 496,
                    "agents": 10,
                    "agent_moments": {},
                    "frame_moments": {},
                },
                reconstruction_counts={"total_tracks": 10},
            )
            write_checkpoint(path, checkpoint)
            restored = load_checkpoint(path, identity())

            self.assertEqual(
                restored.completed_shards,
                checkpoint.completed_shards,
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                load_checkpoint(path, identity(size=101))

    def test_restore_truncates_bytes_after_last_committed_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ExactMetricStore(root / "metrics")
            store.append_batch(
                metric_batch("vehicle", np.asarray([1.0, 2.0]))
            )
            store.flush_and_sync()
            committed = store.snapshot_counts()
            checkpoint = EvaluationCheckpoint(
                identity=identity(),
                completed_shards=[
                    "/data/training.tfrecord-00000-of-01000"
                ],
                buffer_counts=committed,
                accumulator_state={
                    "scenarios": 1,
                    "agents": 2,
                    "agent_moments": {},
                    "frame_moments": {},
                },
                reconstruction_counts={},
            )
            path = root / "checkpoint.json"
            write_checkpoint(path, checkpoint)
            store.append_batch(
                metric_batch("vehicle", np.asarray([99.0]))
            )
            store.flush_and_sync()

            restored = restore_checkpoint(
                store,
                path,
                identity(),
            )
            counts = store.snapshot_counts()
            store.close()

        self.assertEqual(
            restored.completed_shards,
            ["/data/training.tfrecord-00000-of-01000"],
        )
        self.assertEqual(counts, committed)


if __name__ == "__main__":
    unittest.main()
