import csv
import gzip
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


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
