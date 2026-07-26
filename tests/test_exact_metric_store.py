import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.smart.tokens.exact_metric_store import (
    BufferKey,
    ExactMetricStore,
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


if __name__ == "__main__":
    unittest.main()
