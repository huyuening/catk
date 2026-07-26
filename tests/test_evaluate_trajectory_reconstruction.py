import json
import struct
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from src.smart.tokens.evaluate_trajectory_reconstruction import (
    ReconstructionSettings,
    ScenarioTask,
    _load_scenario_class,
    bounded_ordered_map,
    count_tfrecord_records,
    evaluate_scenario_task,
    load_reconstruction_settings,
    resolve_input_paths,
)


def write_tfrecord(path: Path, payloads: list[bytes]) -> None:
    with path.open("wb") as stream:
        for payload in payloads:
            stream.write(struct.pack("<Q", len(payload)))
            stream.write(b"\0" * 4)
            stream.write(payload)
            stream.write(b"\0" * 4)


class EvaluationCliInputTest(unittest.TestCase):
    def test_loads_only_batch_reconstruction_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            path.write_text(
                json.dumps(
                    {
                        "method": "batch",
                        "filter_strength": "strong",
                        "max_gap_frames": -1,
                        "batch_linear_jerk_weight": 1.0,
                        "batch_angular_jerk_weight": 2.0,
                        "num_workers": 99,
                    }
                ),
                encoding="utf-8",
            )
            settings = load_reconstruction_settings(path)

        self.assertEqual(settings.method, "batch")
        self.assertEqual(
            settings.batch_angular_jerk_weight,
            2.0,
        )

    def test_rejects_non_batch_run_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_config.json"
            path.write_text(
                json.dumps(
                    {
                        "method": "filter",
                        "filter_strength": "strong",
                        "max_gap_frames": -1,
                        "batch_linear_jerk_weight": 1.0,
                        "batch_angular_jerk_weight": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "method.*batch",
            ):
                load_reconstruction_settings(path)

    def test_resolves_canonical_training_shards_and_counts_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "training.tfrecord-00000-of-00002"
            second = root / "training.tfrecord-00001-of-00002"
            write_tfrecord(first, [b"a", b"bc"])
            write_tfrecord(second, [b"def"])
            (
                root
                / "training.tfrecord-00000-of-00002-new"
            ).touch()

            resolved = resolve_input_paths([str(root)])

            self.assertEqual(
                resolved,
                [first.resolve(), second.resolve()],
            )
            self.assertEqual(count_tfrecord_records(first), 2)


class _ReverseFuture:
    def __init__(self, executor, index, value):
        self.executor = executor
        self.index = index
        self.value = value
        self.done = False
        self.consumed = False

    def result(self):
        self.executor.complete_pending_in_reverse()
        if not self.consumed:
            self.executor.outstanding -= 1
            self.consumed = True
        return self.value


class _ReverseExecutor:
    def __init__(self):
        self.futures = []
        self.completion_order = []
        self.outstanding = 0
        self.max_outstanding = 0

    def submit(self, function, item):
        future = _ReverseFuture(
            self,
            len(self.futures),
            function(item),
        )
        self.futures.append(future)
        self.outstanding += 1
        self.max_outstanding = max(
            self.max_outstanding,
            self.outstanding,
        )
        return future

    def complete_pending_in_reverse(self):
        for future in reversed(self.futures):
            if not future.done:
                future.done = True
                self.completion_order.append(future.index)


class EvaluationWorkerTest(unittest.TestCase):
    def test_worker_reconstructs_and_evaluates_without_writing_files(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            scenario_class = _load_scenario_class()
        scenario = scenario_class()
        scenario.scenario_id = "worker-test"
        scenario.current_time_index = 10
        scenario.sdc_track_index = 0
        timestamps = np.arange(91, dtype=np.float64) * 0.1
        scenario.timestamps_seconds.extend(timestamps.tolist())
        track = scenario.tracks.add()
        track.id = 7
        track.object_type = 1
        for time in timestamps:
            state = track.states.add()
            state.center_x = float(
                2.0 * time + 0.02 * time**3
            )
            state.center_y = float(0.05 * time**2)
            state.heading = float(
                np.arctan2(
                    0.1 * time,
                    2.0 + 0.06 * time**2,
                )
            )
            state.length = 4.8
            state.width = 2.0
            state.height = 1.5
            state.velocity_x = float(
                2.0 + 0.06 * time**2
            )
            state.velocity_y = float(0.1 * time)
            state.valid = True

        settings = ReconstructionSettings(
            method="batch",
            filter_strength="strong",
            max_gap_frames=-1,
            batch_linear_jerk_weight=1.0,
            batch_angular_jerk_weight=1.0,
        )
        task = ScenarioTask(
            source_file="/data/training.tfrecord-00000-of-00001",
            record_index=0,
            payload=scenario.SerializeToString(),
            settings=settings,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            result = evaluate_scenario_task(task)

            self.assertEqual(list(root.iterdir()), [])

        self.assertEqual(result.metrics.scenario_id, "worker-test")
        self.assertEqual(result.metrics.agent_count, 1)
        self.assertIn(
            "vehicle",
            result.metrics.frame_values,
        )
        self.assertGreater(
            result.reconstruction_counts["processed_tracks"],
            0,
        )

    def test_bounded_map_preserves_order_and_pending_limit(self):
        executor = _ReverseExecutor()

        actual = list(
            bounded_ordered_map(
                executor,
                lambda value: value * 10,
                range(7),
                limit=3,
            )
        )

        self.assertEqual(actual, [0, 10, 20, 30, 40, 50, 60])
        self.assertEqual(executor.completion_order[:3], [2, 1, 0])
        self.assertLessEqual(executor.max_outstanding, 3)


if __name__ == "__main__":
    unittest.main()
