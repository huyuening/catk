import math
import unittest

import numpy as np

from src.womd_labeling.agent_action_classification import (
    LaneVertexIndex,
    compute_signed_distance_to_lane_v1_fixed,
    detect_vehicle_lane_changes,
    label_scenario_actions,
    normalize_angle,
)
from src.womd_labeling.proto import scenario_pb2


def add_lane(scenario, lane_id, y, frame_count=21):
    feature = scenario.map_features.add()
    feature.id = lane_id
    feature.lane.type = 2
    for frame_index in range(frame_count):
        point = feature.lane.polyline.add()
        point.x = float(frame_index)
        point.y = float(y)


def add_track(
    scenario,
    *,
    object_type=1,
    frame_count=21,
    y_values=None,
    headings=None,
    velocity_x_values=None,
    valid_values=None,
):
    if not scenario.timestamps_seconds:
        scenario.timestamps_seconds.extend(
            frame_index * 0.1 for frame_index in range(frame_count)
        )
    track = scenario.tracks.add()
    track.id = len(scenario.tracks)
    track.object_type = object_type
    y_values = y_values if y_values is not None else np.zeros(frame_count)
    headings = headings if headings is not None else np.zeros(frame_count)
    velocity_x_values = (
        velocity_x_values
        if velocity_x_values is not None
        else np.full(frame_count, 5.0)
    )
    valid_values = (
        valid_values
        if valid_values is not None
        else np.ones(frame_count, dtype=bool)
    )
    for frame_index in range(frame_count):
        state = track.states.add()
        state.center_x = float(frame_index)
        state.center_y = float(y_values[frame_index])
        state.heading = float(headings[frame_index])
        state.velocity_x = float(velocity_x_values[frame_index])
        state.velocity_y = 0.0
        state.length = 4.8
        state.width = 2.0
        state.height = 1.6
        state.valid = bool(valid_values[frame_index])
    return track


class AgentActionClassificationTest(unittest.TestCase):
    def test_normalize_angle_and_signed_distance_match_reference(self):
        self.assertAlmostEqual(float(normalize_angle(3.0 * math.pi)), -math.pi)
        polyline = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        trajectory = np.asarray([[1.0, 2.0], [1.0, -2.0]])
        distances = compute_signed_distance_to_lane_v1_fixed(
            trajectory,
            polyline,
        )
        np.testing.assert_allclose(distances, [2.0, -2.0])

    def test_lane_index_matches_reference_all_lane_search(self):
        scenario = scenario_pb2.Scenario()
        add_lane(scenario, 10, 0.0)
        add_lane(scenario, 20, 5.0)
        trajectory = np.asarray(
            [[2.2, 1.0], [7.6, 4.2], [14.4, -0.7]],
            dtype=float,
        )
        lane_index = LaneVertexIndex(scenario)
        actual_distances, actual_lane_ids = (
            lane_index.closest_signed_distances(trajectory)
        )

        lane_ids = list(lane_index.polylines)
        all_distances = np.vstack(
            [
                compute_signed_distance_to_lane_v1_fixed(
                    trajectory,
                    lane_index.polylines[lane_id],
                )
                for lane_id in lane_ids
            ]
        )
        closest = np.argmin(np.abs(all_distances), axis=0)
        expected_distances = all_distances[
            closest,
            np.arange(len(trajectory)),
        ]
        expected_lane_ids = np.asarray([lane_ids[index] for index in closest])

        np.testing.assert_allclose(actual_distances, expected_distances)
        np.testing.assert_array_equal(actual_lane_ids, expected_lane_ids)

    def test_stop_has_priority_and_reverse_speed_uses_absolute_value(self):
        scenario = scenario_pb2.Scenario()
        add_track(scenario, velocity_x_values=np.full(21, 0.1))
        reverse = add_track(scenario, velocity_x_values=np.full(21, -2.0))

        records, _ = label_scenario_actions(scenario, 10)

        self.assertEqual(records[0]["action"], "STOP")
        self.assertEqual(records[1]["action"], "KEEP_SPEED")
        self.assertAlmostEqual(records[1]["longitudinal_velocity_mps"], -2.0)
        self.assertTrue(reverse.states[10].valid)

    def test_vehicle_acceleration_and_deceleration(self):
        scenario = scenario_pb2.Scenario()
        times = np.arange(21) * 0.1
        add_track(scenario, velocity_x_values=5.0 + times)
        add_track(scenario, velocity_x_values=8.0 - times)

        records, _ = label_scenario_actions(scenario, 10)

        self.assertEqual(records[0]["action"], "ACCELERATE")
        self.assertEqual(records[1]["action"], "DECELERATE")
        self.assertAlmostEqual(
            records[0]["longitudinal_acceleration_mps2"],
            1.0,
            places=5,
        )

    def test_turn_and_u_turn_use_valid_frame_windows(self):
        scenario = scenario_pb2.Scenario()
        left_headings = np.concatenate([np.zeros(11), np.linspace(0.05, 0.5, 10)])
        add_track(scenario, headings=left_headings)

        records, _ = label_scenario_actions(scenario, 10)
        self.assertEqual(records[0]["action"], "LEFT_TURN")
        self.assertEqual(records[0]["future_valid_frame_index"], 20)

        u_turn_scenario = scenario_pb2.Scenario()
        headings = np.linspace(0.0, math.pi, 41)
        add_track(u_turn_scenario, frame_count=41, headings=headings)
        u_turn_records, _ = label_scenario_actions(u_turn_scenario, 10)
        self.assertEqual(u_turn_records[0]["action"], "U_TURN")
        self.assertEqual(u_turn_records[0]["future_long_valid_frame_index"], 40)

    def test_valid_frame_window_skips_invalid_global_frames(self):
        scenario = scenario_pb2.Scenario()
        valid = np.asarray([index % 2 == 0 for index in range(31)])
        headings = np.zeros(31)
        add_track(
            scenario,
            frame_count=31,
            headings=headings,
            valid_values=valid,
        )

        records, _ = label_scenario_actions(scenario, 10)

        self.assertEqual(records[0]["past_valid_frame_index"], 0)
        self.assertEqual(records[0]["future_valid_frame_index"], 30)

    def test_default_labels_every_valid_agent_frame(self):
        scenario = scenario_pb2.Scenario()
        valid = np.asarray([True, True, False, True, True])
        add_track(
            scenario,
            frame_count=5,
            velocity_x_values=np.full(5, 2.0),
            valid_values=valid,
        )

        records, diagnostics = label_scenario_actions(scenario)

        self.assertEqual(
            [record["frame_index"] for record in records],
            [0, 1, 3, 4],
        )
        self.assertTrue(all(record["action"] == "KEEP_SPEED" for record in records))
        self.assertEqual(diagnostics["valid_state_frames"], 4)
        self.assertEqual(diagnostics["invalid_state_frames"], 1)
        self.assertEqual(diagnostics["action_labeled_frames"], 4)

    def test_left_and_right_lane_changes_match_reference_intervals(self):
        left_scenario = scenario_pb2.Scenario()
        add_lane(left_scenario, 1, 0.0)
        add_lane(left_scenario, 2, 4.0)
        left_track = add_track(
            left_scenario,
            y_values=np.linspace(0.0, 4.0, 21),
            velocity_x_values=np.full(21, 10.0),
        )
        lane_index = LaneVertexIndex(left_scenario)

        left_events = detect_vehicle_lane_changes(
            left_track,
            lane_index,
            left_scenario.timestamps_seconds,
        )
        left_records, _ = label_scenario_actions(left_scenario, 10)

        self.assertEqual(left_events, [(0, 18, "left")])
        self.assertEqual(left_records[0]["action"], "LEFT_LANE_CHANGE")
        self.assertEqual(left_records[0]["lane_change_end_frame_index"], 18)

        right_scenario = scenario_pb2.Scenario()
        add_lane(right_scenario, 1, 0.0)
        add_lane(right_scenario, 2, 4.0)
        right_track = add_track(
            right_scenario,
            y_values=np.linspace(4.0, 0.0, 21),
            velocity_x_values=np.full(21, 10.0),
        )
        right_records, _ = label_scenario_actions(right_scenario, 10)
        self.assertEqual(right_records[0]["action"], "RIGHT_LANE_CHANGE")

    def test_pedestrian_uses_half_stop_and_acceleration_thresholds(self):
        scenario = scenario_pb2.Scenario()
        add_track(
            scenario,
            object_type=2,
            velocity_x_values=np.full(21, 0.15),
        )
        times = np.arange(21) * 0.1
        add_track(
            scenario,
            object_type=2,
            velocity_x_values=1.0 + 0.3 * times,
        )

        records, _ = label_scenario_actions(scenario, 10)

        self.assertEqual(records[0]["action"], "KEEP_SPEED")
        self.assertEqual(records[1]["action"], "ACCELERATE")


if __name__ == "__main__":
    unittest.main()
