import argparse
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.smart.tokens.evaluate_trajectory_reconstruction import (
    ReconstructionSettings,
    ScenarioEvaluationResult,
    ScenarioTask,
    _load_scenario_class,
    bounded_ordered_map,
    count_tfrecord_records,
    evaluate_scenario_task,
    iter_tfrecord,
    load_reconstruction_settings,
    resolve_input_paths,
    run_evaluation,
    write_final_outputs,
)
from src.smart.tokens.exact_metric_store import ExactMetricStore
from src.smart.tokens.reconstruction_evaluation import (
    EvaluationAccumulator,
    ScenarioMetricBatch,
)


def write_tfrecord(path: Path, payloads: list[bytes]) -> None:
    with path.open("wb") as stream:
        for payload in payloads:
            stream.write(struct.pack("<Q", len(payload)))
            stream.write(b"\0" * 4)
            stream.write(payload)
            stream.write(b"\0" * 4)


def summary_metric_batch(
    scope: str,
    values: np.ndarray,
) -> ScenarioMetricBatch:
    values = np.asarray(values, dtype=np.float64)
    return ScenarioMetricBatch(
        scenario_id=f"{scope}-scenario",
        agent_count=len(values),
        agent_values={scope: {"xy_rmse_m": values}},
        frame_values={
            scope: {
                "raw_linear_jerk_mps3": values - 4.0,
            }
        },
    )


def build_synthetic_scenario():
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
    return scenario


class EvaluationCliInputTest(unittest.TestCase):
    def test_module_help_exposes_resume_and_scratch_contract(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.smart.tokens.evaluate_trajectory_reconstruction",
                "--help",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in (
            "--reconstruction-run-config",
            "--scratch-dir",
            "--resume",
            "--keep-scratch",
            "--max-scenarios",
        ):
            self.assertIn(option, completed.stdout)

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
            self.assertEqual(
                resolve_input_paths(
                    [str(root / "training.tfrecord-*")]
                ),
                [first.resolve(), second.resolve()],
            )
            self.assertEqual(count_tfrecord_records(first), 2)

    def test_tfrecord_reader_rejects_metadata_change_while_open(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.tfrecord-00000-of-00001"
            write_tfrecord(path, [b"first", b"second"])
            original = path.stat()
            records = iter_tfrecord(
                path,
                expected_size=original.st_size,
                expected_mtime_ns=original.st_mtime_ns,
            )
            self.assertEqual(next(records), (0, b"first"))
            os.utime(
                path,
                ns=(
                    original.st_atime_ns,
                    original.st_mtime_ns + 1,
                ),
            )

            with self.assertRaisesRegex(ValueError, "metadata changed"):
                list(records)

    def test_rejects_nested_output_and_scratch_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "training.tfrecord-00000-of-00001"
            write_tfrecord(shard, [])
            run_config = root / "reconstruction_run_config.json"
            run_config.write_text(
                json.dumps(
                    {
                        "method": "batch",
                        "filter_strength": "strong",
                        "max_gap_frames": -1,
                        "batch_linear_jerk_weight": 1.0,
                        "batch_angular_jerk_weight": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            scratch = root / "scratch"
            args = argparse.Namespace(
                input_path=[str(shard)],
                reconstruction_run_config=run_config,
                output_dir=scratch / "output",
                scratch_dir=scratch,
                workers=1,
                max_scenarios=None,
                progress_every=0,
                resume=False,
                keep_scratch=False,
            )

            with self.assertRaisesRegex(
                ValueError,
                "non-nested",
            ):
                run_evaluation(args)


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
    def test_worker_failure_reports_shard_record_and_scenario(self):
        scenario = build_synthetic_scenario()
        task = ScenarioTask(
            source_file="/data/training.tfrecord-00007-of-01000",
            record_index=23,
            payload=scenario.SerializeToString(),
            settings=ReconstructionSettings(
                method="batch",
                filter_strength="strong",
                max_gap_frames=-1,
                batch_linear_jerk_weight=1.0,
                batch_angular_jerk_weight=1.0,
            ),
        )

        with patch(
            "src.smart.tokens.evaluate_trajectory_reconstruction."
            "reconstruct_scenario_for_vocabulary",
            side_effect=ValueError("synthetic reconstruction error"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "00007-of-01000, record 23, scenario worker-test",
            ):
                evaluate_scenario_task(task)

    def test_worker_reconstructs_and_evaluates_without_writing_files(self):
        scenario = build_synthetic_scenario()

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


class EvaluationOutputTest(unittest.TestCase):
    def test_writes_exact_summary_schema_and_only_five_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "output"
            store = ExactMetricStore(root / "scratch")
            accumulator = EvaluationAccumulator()
            for batch in (
                summary_metric_batch(
                    "vehicle",
                    np.asarray([0.0, 1.0]),
                ),
                summary_metric_batch(
                    "pedestrian",
                    np.asarray([8.0, 9.0]),
                ),
            ):
                accumulator.add_batch(batch)
                store.append_batch(batch)

            paths = write_final_outputs(
                output_dir=output_dir,
                accumulator=accumulator,
                store=store,
                reconstruction_counts={
                    "total_tracks": 4,
                    "processed_tracks": 4,
                },
                run_config={
                    "reconstruction": {"method": "batch"},
                    "software": {
                        "python": "test",
                        "numpy": "test",
                        "catk_git_commit": "test",
                    },
                },
            )
            with (output_dir / "agent_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as stream:
                agent_rows = list(csv.DictReader(stream))

            expected_fields = {
                "scope",
                "level",
                "metric",
                "variant",
                "unit",
                "count",
                "mean",
                "std",
                "min",
                "max",
                "range",
                "p01",
                "p99",
                "p99_minus_p01",
            }
            self.assertEqual(set(agent_rows[0]), expected_fields)
            overall_xy = next(
                row
                for row in agent_rows
                if row["scope"] == "all"
                and row["metric"] == "xy_rmse"
            )
            self.assertAlmostEqual(
                float(overall_xy["p99_minus_p01"]),
                float(overall_xy["p99"])
                - float(overall_xy["p01"]),
            )
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                [
                    "agent_summary.csv",
                    "frame_jerk_summary.csv",
                    "reconstruction_summary.json",
                    "run_config.json",
                    "summary.json",
                ],
            )
            self.assertEqual(
                sorted(path.name for path in paths),
                sorted(path.name for path in output_dir.iterdir()),
            )
            summary = json.loads(
                (output_dir / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["failure_count"], 0)
            self.assertEqual(
                summary["reconstruction_provenance"]["method"],
                "batch",
            )


class EvaluationResumeTest(unittest.TestCase):
    def test_partial_smoke_finalization_preserves_committed_buffer_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "training"
            input_dir.mkdir()
            first = input_dir / "training.tfrecord-00000-of-00002"
            second = input_dir / "training.tfrecord-00001-of-00002"
            write_tfrecord(first, [b"first-0", b"first-1"])
            write_tfrecord(second, [b"second-0", b"second-1"])
            run_config = root / "reconstruction_run_config.json"
            run_config.write_text(
                json.dumps(
                    {
                        "method": "batch",
                        "filter_strength": "strong",
                        "max_gap_frames": -1,
                        "batch_linear_jerk_weight": 1.0,
                        "batch_angular_jerk_weight": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_path=[str(input_dir)],
                reconstruction_run_config=run_config,
                output_dir=root / "output",
                scratch_dir=root / "scratch",
                workers=1,
                max_scenarios=3,
                progress_every=0,
                resume=False,
                keep_scratch=True,
            )
            values = iter([100.0, 200.0, -999.0])

            def result_for(task):
                batch = summary_metric_batch(
                    "vehicle",
                    np.asarray([next(values)]),
                )
                return ScenarioEvaluationResult(
                    source_file=task.source_file,
                    record_index=task.record_index,
                    scenario_id=batch.scenario_id,
                    metrics=batch,
                    reconstruction_counts={
                        "total_tracks": 1,
                        "processed_tracks": 1,
                    },
                )

            with patch(
                "src.smart.tokens.evaluate_trajectory_reconstruction."
                "evaluate_scenario_task",
                side_effect=result_for,
            ):
                run_evaluation(args)

            buffer = np.fromfile(
                root
                / "scratch"
                / "metrics"
                / "buffers"
                / "agent"
                / "vehicle"
                / "xy_rmse_m.f64",
                dtype="<f8",
            )
            np.testing.assert_array_equal(
                buffer,
                np.asarray([100.0, 200.0, -999.0]),
            )
            checkpoint = json.loads(
                (
                    root
                    / "scratch"
                    / "checkpoint.json"
                ).read_text(encoding="utf-8")
            )
            committed_key = "agent|vehicle|xy_rmse_m"
            self.assertEqual(
                checkpoint["buffer_counts"][committed_key],
                2,
            )

    def test_shard_metadata_change_rolls_back_uncommitted_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "training"
            input_dir.mkdir()
            shard = input_dir / "training.tfrecord-00000-of-00001"
            write_tfrecord(shard, [b"record"])
            run_config = root / "reconstruction_run_config.json"
            run_config.write_text(
                json.dumps(
                    {
                        "method": "batch",
                        "filter_strength": "strong",
                        "max_gap_frames": -1,
                        "batch_linear_jerk_weight": 1.0,
                        "batch_angular_jerk_weight": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_path=[str(input_dir)],
                reconstruction_run_config=run_config,
                output_dir=root / "output",
                scratch_dir=root / "scratch",
                workers=1,
                max_scenarios=None,
                progress_every=0,
                resume=False,
                keep_scratch=True,
            )

            def mutate_input(task):
                stat = shard.stat()
                os.utime(
                    shard,
                    ns=(
                        stat.st_atime_ns,
                        stat.st_mtime_ns + 1,
                    ),
                )
                batch = summary_metric_batch(
                    "vehicle",
                    np.asarray([1.0]),
                )
                return ScenarioEvaluationResult(
                    source_file=task.source_file,
                    record_index=task.record_index,
                    scenario_id=batch.scenario_id,
                    metrics=batch,
                    reconstruction_counts={
                        "total_tracks": 1,
                        "processed_tracks": 1,
                    },
                )

            with patch(
                "src.smart.tokens.evaluate_trajectory_reconstruction."
                "evaluate_scenario_task",
                side_effect=mutate_input,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "metadata changed",
                ):
                    run_evaluation(args)

            checkpoint = json.loads(
                (
                    root
                    / "scratch"
                    / "checkpoint.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["completed_shards"], [])
            self.assertEqual(
                list((root / "scratch" / "metrics").rglob("*.f64")),
                [],
            )

    def test_smoke_prefix_is_not_committed_as_a_complete_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "training"
            input_dir.mkdir()
            shard = input_dir / "training.tfrecord-00000-of-00001"
            write_tfrecord(shard, [b"first", b"second"])
            run_config = root / "reconstruction_run_config.json"
            run_config.write_text(
                json.dumps(
                    {
                        "method": "batch",
                        "filter_strength": "strong",
                        "max_gap_frames": -1,
                        "batch_linear_jerk_weight": 1.0,
                        "batch_angular_jerk_weight": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_path=[str(input_dir)],
                reconstruction_run_config=run_config,
                output_dir=root / "output",
                scratch_dir=root / "scratch",
                workers=1,
                max_scenarios=1,
                progress_every=0,
                resume=False,
                keep_scratch=True,
            )

            def result_for(task):
                batch = summary_metric_batch(
                    "vehicle",
                    np.asarray([1.0]),
                )
                return ScenarioEvaluationResult(
                    source_file=task.source_file,
                    record_index=task.record_index,
                    scenario_id=batch.scenario_id,
                    metrics=batch,
                    reconstruction_counts={
                        "total_tracks": 1,
                        "processed_tracks": 1,
                    },
                )

            with patch(
                "src.smart.tokens.evaluate_trajectory_reconstruction."
                "evaluate_scenario_task",
                side_effect=result_for,
            ) as evaluate:
                run_evaluation(args)

            self.assertEqual(evaluate.call_count, 1)
            checkpoint = json.loads(
                (
                    root
                    / "scratch"
                    / "checkpoint.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["completed_shards"], [])
            self.assertEqual(
                checkpoint["accumulator_state"]["scenarios"],
                0,
            )

    def test_real_single_scenario_run_deletes_scalar_scratch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "training"
            input_dir.mkdir()
            shard = input_dir / "training.tfrecord-00000-of-00001"
            write_tfrecord(
                shard,
                [build_synthetic_scenario().SerializeToString()],
            )
            run_config = root / "reconstruction_run_config.json"
            run_config.write_text(
                json.dumps(
                    {
                        "method": "batch",
                        "filter_strength": "strong",
                        "max_gap_frames": -1,
                        "batch_linear_jerk_weight": 1.0,
                        "batch_angular_jerk_weight": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "output"
            scratch_dir = root / "scratch"
            args = argparse.Namespace(
                input_path=[str(input_dir)],
                reconstruction_run_config=run_config,
                output_dir=output_dir,
                scratch_dir=scratch_dir,
                workers=1,
                max_scenarios=None,
                progress_every=0,
                resume=True,
                keep_scratch=False,
            )

            paths = run_evaluation(args)

            self.assertFalse(scratch_dir.exists())
            self.assertEqual(len(paths), 5)
            summary = json.loads(
                (output_dir / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["scenario_count"], 1)
            self.assertEqual(summary["agent_count"], 1)

    def test_commits_complete_shards_and_resumes_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "training"
            input_dir.mkdir()
            first = input_dir / "training.tfrecord-00000-of-00002"
            second = input_dir / "training.tfrecord-00001-of-00002"
            write_tfrecord(first, [b"first"])
            write_tfrecord(second, [b"second"])
            second_stat = second.stat()
            run_config = root / "reconstruction_run_config.json"
            config_value = {
                "method": "batch",
                "filter_strength": "strong",
                "max_gap_frames": -1,
                "batch_linear_jerk_weight": 1.0,
                "batch_angular_jerk_weight": 1.0,
            }
            run_config.write_text(
                json.dumps(config_value),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input_path=[str(input_dir)],
                reconstruction_run_config=run_config,
                output_dir=root / "output",
                scratch_dir=root / "scratch",
                workers=1,
                max_scenarios=None,
                progress_every=0,
                resume=False,
                keep_scratch=True,
            )

            def result_for(task):
                batch = summary_metric_batch(
                    "vehicle",
                    np.asarray([float(task.record_index + 1)]),
                )
                return ScenarioEvaluationResult(
                    source_file=task.source_file,
                    record_index=task.record_index,
                    scenario_id=batch.scenario_id,
                    metrics=batch,
                    reconstruction_counts={
                        "total_tracks": 1,
                        "processed_tracks": 1,
                    },
                )

            def fail_second_shard(task):
                if task.source_file == str(second.resolve()):
                    raise RuntimeError("synthetic shard failure")
                return result_for(task)

            with patch(
                "src.smart.tokens.evaluate_trajectory_reconstruction."
                "evaluate_scenario_task",
                side_effect=fail_second_shard,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "synthetic shard failure",
                ):
                    run_evaluation(args)

            checkpoint_path = (
                root
                / "scratch"
                / "checkpoint.json"
            )
            checkpoint = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                checkpoint["completed_shards"],
                [str(first.resolve())],
            )

            args.resume = True
            config_value["batch_linear_jerk_weight"] = 2.0
            run_config.write_text(
                json.dumps(config_value),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                run_evaluation(args)
            config_value["batch_linear_jerk_weight"] = 1.0
            run_config.write_text(
                json.dumps(config_value),
                encoding="utf-8",
            )

            with second.open("ab") as stream:
                stream.write(b"x")
            with self.assertRaisesRegex(ValueError, "identity"):
                run_evaluation(args)
            with second.open("r+b") as stream:
                stream.truncate(second_stat.st_size)
            os.utime(
                second,
                ns=(
                    second_stat.st_atime_ns,
                    second_stat.st_mtime_ns,
                ),
            )

            with patch(
                "src.smart.tokens.evaluate_trajectory_reconstruction."
                "evaluate_scenario_task",
                side_effect=result_for,
            ) as evaluate:
                paths = run_evaluation(args)

            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(
                sorted(path.name for path in paths),
                sorted(
                    [
                        "agent_summary.csv",
                        "frame_jerk_summary.csv",
                        "reconstruction_summary.json",
                        "run_config.json",
                        "summary.json",
                    ]
                ),
            )
            final_checkpoint = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                final_checkpoint["completed_shards"],
                [str(first.resolve()), str(second.resolve())],
            )


if __name__ == "__main__":
    unittest.main()
