from types import SimpleNamespace
import unittest

from src.womd_labeling.map_annotation import RegionType
from src.womd_labeling.road_type_statistics import (
    DrivewayPolygonIndex,
    RoadCategory,
    ThreeClassRoadConfig,
    classify_ego_frame,
    classify_ego_frame_three_class,
    decode_count_key,
    summarize_scenario_road_types,
)
from src.womd_labeling.proto import scenario_pb2


def frame(
    region_type,
    *,
    frame_index=0,
    position=(0.0, 0.0),
    lane_count=None,
    side_lane_count=None,
    arm_count=None,
    distance_to_junction_m=None,
    valid=True,
):
    return SimpleNamespace(
        frame_index=frame_index,
        valid=valid,
        region_type=region_type,
        position_xy=position,
        same_direction_lane_count=lane_count,
        junction_side_lane_count=side_lane_count,
        junction_arm_count=arm_count,
        distance_to_junction_m=distance_to_junction_m,
        confidence=0.9,
        reason=None,
    )


def driveway_scenario():
    scenario = scenario_pb2.Scenario()
    feature = scenario.map_features.add()
    feature.id = 1
    for x, y in [(-2, -2), (2, -2), (2, 2), (-2, 2)]:
        point = feature.driveway.polygon.add()
        point.x = x
        point.y = y
    return scenario


class RoadTypeStatisticsTest(unittest.TestCase):
    def test_three_class_distance_thresholds(self):
        config = ThreeClassRoadConfig(
            approach_stop_line_distance_m=30.0,
            exit_junction_distance_m=15.0,
        )
        approach_near = classify_ego_frame_three_class(
            frame(
                RegionType.NEAR_INTERSECTION_APPROACH,
                side_lane_count=2,
                arm_count=4,
                distance_to_junction_m=29.9,
            ),
            config=config,
        )
        approach_far = classify_ego_frame_three_class(
            frame(
                RegionType.NEAR_INTERSECTION_APPROACH,
                side_lane_count=2,
                arm_count=4,
                distance_to_junction_m=30.1,
            ),
            config=config,
        )
        exit_near = classify_ego_frame_three_class(
            frame(
                RegionType.NEAR_INTERSECTION_EXIT,
                side_lane_count=1,
                arm_count=3,
                distance_to_junction_m=15.0,
            ),
            config=config,
        )
        exit_far = classify_ego_frame_three_class(
            frame(
                RegionType.NEAR_INTERSECTION_EXIT,
                side_lane_count=1,
                arm_count=3,
                distance_to_junction_m=15.1,
            ),
            config=config,
        )

        self.assertEqual(approach_near.category, RoadCategory.INTERSECTION.value)
        self.assertEqual(approach_near.subtype, "FOUR_ARM_INTERSECTION")
        self.assertEqual(approach_far.category, RoadCategory.ROAD_SEGMENT.value)
        self.assertEqual(approach_far.subtype, "LANE_COUNT_2")
        self.assertEqual(exit_near.category, RoadCategory.INTERSECTION.value)
        self.assertEqual(exit_near.subtype, "THREE_ARM_INTERSECTION")
        self.assertEqual(exit_far.category, RoadCategory.ROAD_SEGMENT.value)
        self.assertEqual(exit_far.subtype, "LANE_COUNT_1")

    def test_intersection_subtypes(self):
        current = classify_ego_frame(
            frame(RegionType.INTERSECTION, arm_count=4)
        )
        three_arm = classify_ego_frame(
            frame(RegionType.IN_INTERSECTION, arm_count=3)
        )
        four_arm = classify_ego_frame(
            frame(RegionType.IN_INTERSECTION, arm_count=4)
        )
        other = classify_ego_frame(
            frame(RegionType.IN_INTERSECTION, arm_count=5)
        )

        self.assertEqual(current.category, RoadCategory.INTERSECTION.value)
        self.assertEqual(current.subtype, "FOUR_ARM_INTERSECTION")
        self.assertEqual(three_arm.subtype, "THREE_ARM_INTERSECTION")
        self.assertEqual(four_arm.subtype, "FOUR_ARM_INTERSECTION")
        self.assertEqual(other.subtype, "OTHER_INTERSECTION")

    def test_road_and_near_intersection_lane_counts(self):
        road = classify_ego_frame(
            frame(RegionType.ROAD_SEGMENT, lane_count=3)
        )
        near = classify_ego_frame(
            frame(
                RegionType.NEAR_INTERSECTION_APPROACH,
                lane_count=4,
                side_lane_count=2,
                arm_count=4,
            )
        )

        self.assertEqual(road.category, RoadCategory.ROAD_SEGMENT.value)
        self.assertEqual(road.subtype, "LANE_COUNT_3")
        self.assertEqual(near.category, RoadCategory.NEAR_INTERSECTION.value)
        self.assertEqual(near.subtype, "LANE_COUNT_2")

    def test_driveway_polygon_is_parking_lot_proxy(self):
        index = DrivewayPolygonIndex(driveway_scenario())
        inside = classify_ego_frame(
            frame(RegionType.ROAD_SEGMENT, position=(0.0, 0.0), lane_count=1),
            index,
        )
        boundary = classify_ego_frame(
            frame(RegionType.ROAD_SEGMENT, position=(2.0, 0.0), lane_count=1),
            index,
        )
        outside = classify_ego_frame(
            frame(RegionType.ROAD_SEGMENT, position=(3.0, 0.0), lane_count=1),
            index,
        )

        self.assertEqual(inside.category, RoadCategory.PARKING_LOT_PROXY.value)
        self.assertEqual(boundary.category, RoadCategory.PARKING_LOT_PROXY.value)
        self.assertEqual(outside.category, RoadCategory.ROAD_SEGMENT.value)

    def test_scenario_summary_uses_current_frame_and_valid_frames(self):
        annotation = SimpleNamespace(
            current_time_index=1,
            ego_frames=(
                frame(RegionType.ROAD_SEGMENT, frame_index=0, lane_count=1),
                frame(
                    RegionType.NEAR_INTERSECTION_APPROACH,
                    frame_index=1,
                    side_lane_count=2,
                    arm_count=4,
                ),
                frame(
                    RegionType.IN_INTERSECTION,
                    frame_index=2,
                    arm_count=4,
                    valid=False,
                ),
            ),
        )

        summary = summarize_scenario_road_types(annotation)
        current = summary["current_label"]
        keys = {decode_count_key(key) for key in summary["frame_counts"]}

        self.assertEqual(current["category"], RoadCategory.NEAR_INTERSECTION.value)
        self.assertEqual(current["subtype"], "LANE_COUNT_2")
        self.assertEqual(summary["valid_frame_count"], 2)
        self.assertEqual(summary["invalid_frame_count"], 1)
        self.assertEqual(
            summary["occurrence_categories"],
            [
                RoadCategory.NEAR_INTERSECTION.value,
                RoadCategory.ROAD_SEGMENT.value,
            ],
        )
        self.assertEqual(
            keys,
            {
                (RoadCategory.ROAD_SEGMENT.value, "LANE_COUNT_1"),
                (RoadCategory.NEAR_INTERSECTION.value, "LANE_COUNT_2"),
            },
        )


if __name__ == "__main__":
    unittest.main()
