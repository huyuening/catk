import csv
import gzip
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "smart"
    / "datasets"
    / "build_text_control_tags.py"
)
SPEC = importlib.util.spec_from_file_location("catk_build_text_control_tags", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load module spec for {MODULE_PATH}")
BUILD_TAGS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD_TAGS
SPEC.loader.exec_module(BUILD_TAGS)


class _FakeFuture:
    def __init__(self, *, value=(20, 2), error=None, cancellable=True):
        self.value = value
        self.error = error
        self.cancellable = cancellable
        self.cancel_calls = 0

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value

    def cancel(self):
        self.cancel_calls += 1
        return self.cancellable


class _FakeExecutor:
    def __init__(self, futures):
        self.futures = iter(futures)
        self.submissions = []
        self.shutdown_calls = []

    def submit(self, *args):
        future = next(self.futures)
        self.submissions.append((args, future))
        return future

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


class BuildTextControlTagsTest(unittest.TestCase):
    @staticmethod
    def row(
        frame,
        action,
        speed=8.0,
        acceleration=0.0,
        *,
        scenario_id="waymo-a",
        global_index=17,
        track_id=123,
        current_index=10,
    ):
        return {
            "scenario_id": str(scenario_id),
            "global_index": str(global_index),
            "dataset_current_time_index": str(current_index),
            "frame_index": str(frame),
            "track_id": str(track_id),
            "action": str(action),
            "absolute_speed_mps": str(speed),
            "longitudinal_acceleration_mps2": str(acceleration),
        }

    def test_turning_and_acceleration_are_emitted_together(self):
        tags = BUILD_TAGS.derive_frame_tags(
            self.row(20, "LEFT_TURN", speed=7.0, acceleration=1.0)
        )
        self.assertEqual(tags, {"LeftTurn", "Accelerate"})

    def test_non_turning_motion_has_straight_and_speed_labels(self):
        tags = BUILD_TAGS.derive_frame_tags(
            self.row(20, "KEEP_SPEED", speed=7.0, acceleration=0.0)
        )
        self.assertEqual(tags, {"Straight", "KeepSpeed"})

    def test_stopped_agent_is_parked_without_straight(self):
        tags = BUILD_TAGS.derive_frame_tags(
            self.row(20, "STOP", speed=0.0, acceleration=0.0)
        )
        self.assertEqual(tags, {"Parked"})

    def test_u_turn_is_excluded_from_text_v1(self):
        tags = BUILD_TAGS.derive_frame_tags(
            self.row(20, "U_TURN", speed=7.0, acceleration=0.0)
        )
        self.assertEqual(tags, {"KeepSpeed"})

    def test_only_future_eighty_frames_are_converted(self):
        rows = [
            self.row(frame, "LEFT_TURN", speed=7.0, acceleration=0.0)
            for frame in range(0, 91)
        ]

        intervals = BUILD_TAGS.build_intervals(rows)

        self.assertIn("LeftTurn(123 at 11-91)", intervals)
        self.assertNotIn("LeftTurn(123 at 0-91)", intervals)

    def test_gaps_split_half_open_intervals(self):
        rows = [
            self.row(frame, "ACCELERATE", acceleration=1.0)
            for frame in list(range(11, 21)) + list(range(24, 36))
        ]

        intervals = BUILD_TAGS.build_intervals(rows)

        self.assertIn("Accelerate(123 at 11-21)", intervals)
        self.assertIn("Accelerate(123 at 24-36)", intervals)

    def test_deceleration_followed_by_parked_becomes_stopping(self):
        rows = [
            self.row(frame, "DECELERATE", speed=3.0, acceleration=-1.0)
            for frame in range(11, 26)
        ]
        rows.extend(
            self.row(frame, "STOP", speed=0.0, acceleration=0.0)
            for frame in range(27, 42)
        )

        intervals = BUILD_TAGS.build_intervals(rows)

        self.assertIn("Stopping(123 at 11-26)", intervals)
        self.assertIn("Parked(123 at 27-42)", intervals)
        self.assertNotIn("Decelerate(123 at 11-26)", intervals)

    def test_invalid_numeric_row_reports_field_and_value(self):
        row = self.row(20, "KEEP_SPEED")
        row["absolute_speed_mps"] = "not-a-number"
        with self.assertRaisesRegex(ValueError, "absolute_speed_mps"):
            BUILD_TAGS.derive_frame_tags(row)


class BuildTextControlTagsCliTest(unittest.TestCase):
    fieldnames = [
        "scenario_id",
        "global_index",
        "dataset_current_time_index",
        "frame_index",
        "track_id",
        "action",
        "absolute_speed_mps",
        "longitudinal_acceleration_mps2",
    ]

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input_path = self.root / "actions.csv.gz"
        self.output_root = self.root / "tags"
        self.mapping_path = self.output_root / "train_scenario_mapping.json"

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self, rows):
        with gzip.open(self.input_path, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_cli_writes_ecosim_layout_and_sorted_mapping(self):
        rows = []
        rows.extend(
            BuildTextControlTagsTest.row(
                frame,
                "RIGHT_LANE_CHANGE",
                scenario_id="waymo-b",
                global_index=18,
                track_id=222,
            )
            for frame in range(11, 31)
        )
        rows.extend(
            BuildTextControlTagsTest.row(
                frame,
                "LEFT_TURN",
                scenario_id="waymo-a",
                global_index=17,
                track_id=111,
            )
            for frame in range(11, 31)
        )
        self.write_rows(rows)

        exit_code = BUILD_TAGS.main(
            [
                "--input",
                str(self.input_path),
                "--output-root",
                str(self.output_root),
                "--split",
                "train",
                "--mapping-output",
                str(self.mapping_path),
                "--workers",
                "1",
            ]
        )

        self.assertEqual(exit_code, 0)
        mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        self.assertEqual(
            mapping,
            {"waymo-a": "scene_17", "waymo-b": "scene_18"},
        )
        first_tags = json.loads(
            (
                self.output_root
                / "tag_prompts"
                / "waymo_train_v_action"
                / "tags"
                / "17"
                / "scene_17.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            first_tags,
            ["LeftTurn(111 at 11-31)", "KeepSpeed(111 at 11-31)"],
        )

    def test_parser_defaults_to_eighty_workers_and_one_thousand_scenarios(self):
        args = BUILD_TAGS.build_parser().parse_args(
            [
                "--input", str(self.input_path),
                "--output-root", str(self.output_root),
                "--split", "train",
                "--mapping-output", str(self.mapping_path),
            ]
        )
        self.assertEqual(args.workers, 80)
        self.assertEqual(args.progress_every, 1000)

    def test_invalid_parallel_settings_fail_before_reading_input(self):
        common = {
            "input_paths": [self.root / "missing.csv.gz"],
            "output_root": self.output_root,
            "split": "train",
            "mapping_output": self.mapping_path,
        }
        with self.assertRaisesRegex(ValueError, "workers"):
            BUILD_TAGS.convert_action_rows(**common, workers=0)
        with self.assertRaisesRegex(ValueError, "progress_every"):
            BUILD_TAGS.convert_action_rows(**common, progress_every=0)

    def test_serial_conversion_reports_exact_progress(self):
        rows = []
        for scenario_index in range(2):
            rows.extend(
                BuildTextControlTagsTest.row(
                    frame,
                    "LEFT_TURN",
                    scenario_id=f"waymo-{scenario_index}",
                    global_index=17 + scenario_index,
                    track_id=100 + scenario_index,
                )
                for frame in range(11, 31)
            )
        self.write_rows(rows)
        progress = io.StringIO()
        BUILD_TAGS.convert_action_rows(
            input_paths=[self.input_path],
            output_root=self.output_root,
            split="train",
            mapping_output=self.mapping_path,
            workers=1,
            progress_every=1,
            progress_stream=progress,
        )
        output = progress.getvalue()
        self.assertIn("status=start", output)
        self.assertIn("workers=1", output)
        self.assertIn("completed=1", output)
        self.assertIn("status=complete", output)
        self.assertIn("completed=2", output)
        self.assertIn("rows=40", output)
        self.assertIn("pending=0", output)

    def cli_args(self, output_root, mapping_path, *, workers):
        return [
            sys.executable,
            str(MODULE_PATH),
            "--input", str(self.input_path),
            "--output-root", str(output_root),
            "--split", "train",
            "--mapping-output", str(mapping_path),
            "--workers", str(workers),
            "--progress-every", "1",
        ]

    @staticmethod
    def artifact_bytes(root):
        return {
            path.relative_to(root): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_parallel_and_serial_cli_outputs_are_identical(self):
        rows = []
        for scenario_index in range(6):
            rows.extend(
                BuildTextControlTagsTest.row(
                    frame,
                    "RIGHT_LANE_CHANGE" if scenario_index % 2 else "LEFT_TURN",
                    scenario_id=f"waymo-{scenario_index}",
                    global_index=100 + scenario_index,
                    track_id=1000 + scenario_index,
                )
                for frame in range(11, 31)
            )
        self.write_rows(rows)
        serial_root = self.root / "serial"
        parallel_root = self.root / "parallel"
        serial = subprocess.run(
            self.cli_args(serial_root, serial_root / "mapping.json", workers=1),
            cwd=MODULE_PATH.parents[3],
            text=True,
            capture_output=True,
            check=False,
        )
        parallel = subprocess.run(
            self.cli_args(parallel_root, parallel_root / "mapping.json", workers=2),
            cwd=MODULE_PATH.parents[3],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(serial.returncode, 0, serial.stderr)
        self.assertEqual(parallel.returncode, 0, parallel.stderr)
        self.assertEqual(
            self.artifact_bytes(serial_root),
            self.artifact_bytes(parallel_root),
        )
        self.assertIn("status=complete", parallel.stderr)
        self.assertIn("completed=6", parallel.stderr)

    def test_parallel_scheduler_waits_at_exactly_twice_the_worker_count(self):
        self.write_rows(
            [
                BuildTextControlTagsTest.row(
                    frame,
                    "LEFT_TURN",
                    scenario_id=f"waymo-{scenario_index}",
                    global_index=17 + scenario_index,
                    track_id=100 + scenario_index,
                )
                for scenario_index in range(5)
                for frame in range(11, 31)
            ]
        )
        executor = _FakeExecutor([_FakeFuture() for _ in range(5)])
        wait_calls = []

        def complete_one(pending, *, return_when):
            wait_calls.append((len(executor.submissions), len(pending), return_when))
            completed = next(
                future for _, future in executor.submissions if future in pending
            )
            return {completed}, set(pending) - {completed}

        with patch.object(BUILD_TAGS, "ProcessPoolExecutor", return_value=executor), patch.object(
            BUILD_TAGS, "wait", side_effect=complete_one
        ):
            BUILD_TAGS.convert_action_rows(
                input_paths=[self.input_path],
                output_root=self.output_root,
                split="train",
                mapping_output=self.mapping_path,
                workers=2,
                progress_stream=io.StringIO(),
            )

        self.assertEqual(wait_calls[0], (4, 4, BUILD_TAGS.FIRST_COMPLETED))
        self.assertEqual(len(executor.submissions), 5)

    def test_parallel_worker_failure_cancels_queued_work_and_skips_mapping(self):
        self.write_rows(
            [
                BuildTextControlTagsTest.row(
                    frame,
                    "LEFT_TURN",
                    scenario_id=f"waymo-{scenario_index}",
                    global_index=17 + scenario_index,
                    track_id=100 + scenario_index,
                )
                for scenario_index in range(4)
                for frame in range(11, 31)
            ]
        )
        failure = RuntimeError("worker failed")
        failed_future = _FakeFuture(error=failure)
        queued_futures = [_FakeFuture() for _ in range(3)]
        executor = _FakeExecutor([failed_future, *queued_futures])

        def fail_first(pending, *, return_when):
            self.assertIs(return_when, BUILD_TAGS.FIRST_COMPLETED)
            self.assertEqual(len(pending), 4)
            return {failed_future}, set(pending) - {failed_future}

        with patch.object(BUILD_TAGS, "ProcessPoolExecutor", return_value=executor), patch.object(
            BUILD_TAGS, "wait", side_effect=fail_first
        ):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                BUILD_TAGS.convert_action_rows(
                    input_paths=[self.input_path],
                    output_root=self.output_root,
                    split="train",
                    mapping_output=self.mapping_path,
                    workers=2,
                    progress_stream=io.StringIO(),
                )

        self.assertTrue(all(future.cancel_calls == 1 for future in queued_futures))
        self.assertEqual(
            executor.shutdown_calls,
            [{"wait": True, "cancel_futures": True}],
        )
        self.assertFalse(self.mapping_path.exists())

    def test_shell_wrapper_forwards_parallel_defaults(self):
        script_path = MODULE_PATH.parents[3] / "scripts" / "build_text_control_tags.sh"
        script = script_path.read_text(encoding="utf-8")
        self.assertIn('TAG_WORKERS="${TAG_WORKERS:-80}"', script)
        self.assertIn(
            'TAG_PROGRESS_EVERY="${TAG_PROGRESS_EVERY:-1000}"',
            script,
        )
        self.assertIn('--workers "${TAG_WORKERS}"', script)
        self.assertIn('--progress-every "${TAG_PROGRESS_EVERY}"', script)

    def test_cli_rejects_test_split(self):
        self.write_rows([])
        with self.assertRaises(SystemExit):
            BUILD_TAGS.main(
                [
                    "--input",
                    str(self.input_path),
                    "--output-root",
                    str(self.output_root),
                    "--split",
                    "test",
                    "--mapping-output",
                    str(self.mapping_path),
                ]
            )


if __name__ == "__main__":
    unittest.main()
