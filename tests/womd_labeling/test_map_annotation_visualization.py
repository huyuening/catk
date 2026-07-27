import math
import unittest

import numpy as np

from src.womd_labeling.map_annotation_visualization import (
    DEFAULT_MAP_FRAME_INDEX,
    MapVisualizationConfig,
    _additional_same_direction_lane_ids,
    _ego_lane_chain_ids,
    _panel_text,
    format_region_summary,
    format_road_environment_summary,
    select_render_frame,
    world_to_ego,
)
from src.womd_labeling.proto import scenario_pb2


class MapAnnotationVisualizationTest(unittest.TestCase):
    @staticmethod
    def _add_lane(
        scenario,
        lane_id,
        points,
        *,
        entry_lanes=(),
        exit_lanes=(),
    ):
        feature = scenario.map_features.add()
        feature.id = lane_id
        for x, y in points:
            point = feature.lane.polyline.add()
            point.x = x
            point.y = y
        feature.lane.entry_lanes.extend(entry_lanes)
        feature.lane.exit_lanes.extend(exit_lanes)
        return feature

    def test_world_to_ego_rotates_ego_heading_to_positive_x(self):
        transformed = world_to_ego(
            [(10.0, 20.0), (9.0, 10.0)],
            origin_xy=(10.0, 10.0),
            ego_heading_rad=math.pi / 2.0,
        )

        np.testing.assert_allclose(transformed[0], [10.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(transformed[1], [0.0, 1.0], atol=1e-9)

    def test_summary_uses_requested_road_intersection_hierarchy(self):
        road = {
            "region_type": "ROAD_SEGMENT",
            "same_direction_lane_count": 3,
        }
        intersection = {
            "region_type": "INTERSECTION",
            "same_direction_lane_count": 4,
            "junction_side_lane_count": 2,
            "junction_arm_count": 4,
        }
        inside = {
            "region_type": "IN_INTERSECTION",
            "junction_arm_count": 3,
        }
        roundabout = {
            "region_type": "INTERSECTION",
            "junction_kind": "roundabout",
            "junction_arm_count": 4,
        }

        self.assertEqual(format_region_summary(road), "主车位置：路段（3 车道）")
        self.assertEqual(
            format_region_summary(intersection),
            "主车位置：路口（4 支路口）",
        )
        self.assertEqual(
            format_region_summary(inside),
            "主车位置：路口（3 支路口）",
        )
        self.assertEqual(
            format_region_summary(roundabout),
            "主车位置：路口（环形路口）",
        )
        self.assertIn("控制类型：无控制", _panel_text(roundabout, "zh"))

    def test_road_environment_summary_includes_lane_count(self):
        parking = {
            "road_environment": "PARKING_LOT",
            "road_environment_lane_count": 1,
        }
        freeway = {
            "road_environment": "FREEWAY",
            "road_environment_lane_count": 5,
        }

        self.assertEqual(
            format_road_environment_summary(parking),
            "道路环境：停车场（1 车道）",
        )
        self.assertEqual(
            format_road_environment_summary(freeway),
            "道路环境：高速公路（5 车道）",
        )
        self.assertIn("道路环境：停车场（1 车道）", _panel_text(parking, "zh"))

    def test_select_render_frame_falls_back_to_first_valid_sdc_frame(self):
        scenario = scenario_pb2.Scenario()
        scenario.sdc_track_index = 0
        track = scenario.tracks.add()
        track.states.add().valid = False
        track.states.add().valid = True
        annotation = {
            "ego_frames": [
                {"frame_index": 0, "region_type": "UNKNOWN"},
                {"frame_index": 1, "region_type": "ROAD_SEGMENT"},
            ]
        }

        frame_index, frame = select_render_frame(scenario, annotation, 0)

        self.assertEqual(frame_index, 1)
        self.assertEqual(frame["region_type"], "ROAD_SEGMENT")

    def test_default_render_frame_is_index_10(self):
        scenario = scenario_pb2.Scenario()
        scenario.sdc_track_index = 0
        track = scenario.tracks.add()
        for _ in range(DEFAULT_MAP_FRAME_INDEX + 1):
            track.states.add().valid = True
        annotation = {
            "ego_frames": [
                {"frame_index": 0, "region_type": "ROAD_SEGMENT"},
                {
                    "frame_index": DEFAULT_MAP_FRAME_INDEX,
                    "region_type": "INTERSECTION",
                },
            ]
        }

        frame_index, frame = select_render_frame(scenario, annotation)

        self.assertEqual(frame_index, DEFAULT_MAP_FRAME_INDEX)
        self.assertEqual(frame["region_type"], "INTERSECTION")

    def test_config_rejects_reversed_ranges(self):
        with self.assertRaises(ValueError):
            MapVisualizationConfig(x_min_m=5.0, x_max_m=-5.0)

    def test_ego_lane_chain_extends_until_topology_branches(self):
        scenario = scenario_pb2.Scenario()
        self._add_lane(
            scenario,
            1,
            [(-10.0, 0.0), (0.0, 0.0)],
            exit_lanes=(2,),
        )
        self._add_lane(
            scenario,
            2,
            [(0.0, 0.0), (10.0, 0.0)],
            entry_lanes=(1,),
            exit_lanes=(3, 4),
        )
        self._add_lane(
            scenario,
            3,
            [(10.0, 0.0), (20.0, 1.0)],
            entry_lanes=(2,),
        )
        self._add_lane(
            scenario,
            4,
            [(10.0, 0.0), (20.0, -1.0)],
            entry_lanes=(2,),
        )

        self.assertEqual(_ego_lane_chain_ids(scenario, 1), (1, 2))

    def test_ego_lane_chain_stops_at_ambiguous_merge(self):
        scenario = scenario_pb2.Scenario()
        self._add_lane(
            scenario,
            1,
            [(-10.0, 1.0), (0.0, 0.0)],
            exit_lanes=(3,),
        )
        self._add_lane(
            scenario,
            2,
            [(-10.0, -1.0), (0.0, 0.0)],
            exit_lanes=(3,),
        )
        self._add_lane(
            scenario,
            3,
            [(0.0, 0.0), (10.0, 0.0)],
            entry_lanes=(1, 2),
        )

        self.assertEqual(_ego_lane_chain_ids(scenario, 3), (3,))

    def test_additional_same_direction_lanes_exclude_junction_lanes(self):
        frame = {
            "same_direction_lane_ids": [188, 190, 191, 192],
        }
        junction = {
            "incoming_lane_ids": [190, 191, 192],
            "outgoing_lane_ids": [187],
            "core_lane_ids": [181, 182],
        }

        self.assertEqual(
            _additional_same_direction_lane_ids(frame, junction),
            (188,),
        )

    def test_old_annotation_without_lane_ids_has_no_additional_lanes(self):
        self.assertEqual(
            _additional_same_direction_lane_ids(
                {"same_direction_lane_count": 4},
                {"incoming_lane_ids": [190, 191, 192]},
            ),
            (),
        )


if __name__ == "__main__":
    unittest.main()
