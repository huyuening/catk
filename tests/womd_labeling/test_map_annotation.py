import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.womd_labeling.map_annotation import (  # noqa: E402
    EgoMapAnnotator,
    JunctionKind,
    MapAnnotationConfig,
    MapMatch,
    RegionType,
    RoadEnvironment,
    RoadEnvironmentSubtype,
    _LaneGeometryIndex,
    _build_arms,
    _directional_branch_through_lane_ids,
    _merge_spatially_overlapping_groups,
    _prefer_complete_lane_cross_section,
    annotate_scenario,
)
from src.womd_labeling.map_annotation import ScenarioProcessor  # noqa: E402
from src.womd_labeling.proto import scenario_pb2  # noqa: E402


def add_lane(scenario, lane_id, points):
    feature = scenario.map_features.add()
    feature.id = lane_id
    feature.lane.type = 2
    feature.lane.speed_limit_mph = 30.0
    for x, y in points:
        point = feature.lane.polyline.add()
        point.x = float(x)
        point.y = float(y)
        point.z = 0.0
    return feature.lane


def add_road_edge(scenario, feature_id, points):
    feature = scenario.map_features.add()
    feature.id = feature_id
    feature.road_edge.type = 1
    for x, y in points:
        point = feature.road_edge.polyline.add()
        point.x = float(x)
        point.y = float(y)
        point.z = 0.0
    return feature.road_edge


def add_lane_neighbor_pair(right_lane, right_id, left_lane, left_id):
    right_neighbor = right_lane.left_neighbors.add()
    right_neighbor.feature_id = left_id
    right_neighbor.self_start_index = 0
    right_neighbor.self_end_index = len(right_lane.polyline) - 1
    right_neighbor.neighbor_start_index = 0
    right_neighbor.neighbor_end_index = len(left_lane.polyline) - 1

    left_neighbor = left_lane.right_neighbors.add()
    left_neighbor.feature_id = right_id
    left_neighbor.self_start_index = 0
    left_neighbor.self_end_index = len(left_lane.polyline) - 1
    left_neighbor.neighbor_start_index = 0
    left_neighbor.neighbor_end_index = len(right_lane.polyline) - 1


def build_four_arm_scenario():
    scenario = scenario_pb2.Scenario()
    scenario.scenario_id = "synthetic-four-arm"
    scenario.current_time_index = 3
    scenario.sdc_track_index = 0
    positions = [
        (0.0, 20.0),
        (0.0, 11.0),
        (0.0, 8.0),
        (0.0, 0.0),
        (0.0, -8.0),
        (0.0, -11.0),
        (0.0, -20.0),
    ]
    scenario.timestamps_seconds.extend(index * 0.1 for index in range(len(positions)))

    incoming = {
        "north": (10, [(0.0, 30.0), (0.0, 10.0)]),
        "east": (20, [(30.0, 0.0), (10.0, 0.0)]),
        "south": (30, [(0.0, -30.0), (0.0, -10.0)]),
        "west": (40, [(-30.0, 0.0), (-10.0, 0.0)]),
    }
    outgoing = {
        "north": (11, [(0.0, 10.0), (0.0, 30.0)]),
        "east": (21, [(10.0, 0.0), (30.0, 0.0)]),
        "south": (31, [(0.0, -10.0), (0.0, -30.0)]),
        "west": (41, [(-10.0, 0.0), (-30.0, 0.0)]),
    }
    lanes = {}
    for lane_id, points in incoming.values():
        lanes[lane_id] = add_lane(scenario, lane_id, points)
    for lane_id, points in outgoing.values():
        lanes[lane_id] = add_lane(scenario, lane_id, points)

    movements = [
        ("north", "south"),
        ("north", "west"),
        ("east", "west"),
        ("east", "north"),
        ("south", "north"),
        ("south", "east"),
        ("west", "east"),
        ("west", "south"),
    ]
    connector_ids = []
    outgoing_entries = {name: [] for name in outgoing}
    incoming_exits = {name: [] for name in incoming}
    for offset, (from_name, to_name) in enumerate(movements):
        connector_id = 100 + offset
        connector_ids.append(connector_id)
        start = incoming[from_name][1][-1]
        end = outgoing[to_name][1][0]
        connector = add_lane(scenario, connector_id, [start, (0.0, 0.0), end])
        connector.entry_lanes.append(incoming[from_name][0])
        connector.exit_lanes.append(outgoing[to_name][0])
        incoming_exits[from_name].append(connector_id)
        outgoing_entries[to_name].append(connector_id)
        lanes[connector_id] = connector

    for name, connector_lane_ids in incoming_exits.items():
        lanes[incoming[name][0]].exit_lanes.extend(connector_lane_ids)
    for name, connector_lane_ids in outgoing_entries.items():
        lanes[outgoing[name][0]].entry_lanes.extend(connector_lane_ids)

    for _ in scenario.timestamps_seconds:
        dynamic_state = scenario.dynamic_map_states.add()
        for connector_id in connector_ids:
            lane_state = dynamic_state.lane_states.add()
            lane_state.lane = connector_id
            lane_state.state = 6
            connector = lanes[connector_id]
            lane_state.stop_point.CopyFrom(connector.polyline[0])

    sdc = scenario.tracks.add()
    sdc.id = 1
    sdc.object_type = 1
    for x, y in positions:
        state = sdc.states.add()
        state.center_x = x
        state.center_y = y
        state.center_z = 0.0
        state.heading = -math.pi / 2.0
        state.velocity_y = -5.0
        state.length = 4.8
        state.width = 2.0
        state.height = 1.6
        state.valid = True
    return scenario


def build_roundabout_scenario():
    scenario = scenario_pb2.Scenario()
    scenario.scenario_id = "synthetic-roundabout"
    scenario.current_time_index = 0
    scenario.sdc_track_index = 0
    scenario.timestamps_seconds.append(0.0)

    rings = []
    lanes = {}
    for ring_index, radius in enumerate((10.0, 14.0)):
        ring_lane_ids = []
        for segment_index in range(4):
            lane_id = 100 + 10 * ring_index + segment_index
            start_angle_deg = 90.0 * segment_index
            points = [
                (
                    radius * math.cos(math.radians(start_angle_deg + offset)),
                    radius * math.sin(math.radians(start_angle_deg + offset)),
                )
                for offset in range(0, 91, 10)
            ]
            lanes[lane_id] = add_lane(scenario, lane_id, points)
            ring_lane_ids.append(lane_id)
        for segment_index, lane_id in enumerate(ring_lane_ids):
            previous_lane_id = ring_lane_ids[(segment_index - 1) % 4]
            next_lane_id = ring_lane_ids[(segment_index + 1) % 4]
            lanes[lane_id].entry_lanes.append(previous_lane_id)
            lanes[lane_id].exit_lanes.append(next_lane_id)
        rings.append(ring_lane_ids)

    outer_ring = rings[1]
    for arm_index, angle_deg in enumerate((0.0, 90.0, 180.0, 270.0)):
        angle = math.radians(angle_deg)
        gate = (14.0 * math.cos(angle), 14.0 * math.sin(angle))
        far = (35.0 * math.cos(angle), 35.0 * math.sin(angle))
        incoming_id = 200 + arm_index
        outgoing_id = 300 + arm_index
        incoming = add_lane(scenario, incoming_id, [far, gate])
        outgoing = add_lane(scenario, outgoing_id, [gate, far])
        current_ring_lane_id = outer_ring[arm_index]
        previous_ring_lane_id = outer_ring[(arm_index - 1) % 4]
        incoming.exit_lanes.append(current_ring_lane_id)
        lanes[current_ring_lane_id].entry_lanes.append(incoming_id)
        lanes[previous_ring_lane_id].exit_lanes.append(outgoing_id)
        outgoing.entry_lanes.append(previous_ring_lane_id)

    angle = math.radians(45.0)
    sdc = scenario.tracks.add()
    sdc.id = 1
    sdc.object_type = 1
    state = sdc.states.add()
    state.center_x = 14.0 * math.cos(angle)
    state.center_y = 14.0 * math.sin(angle)
    state.center_z = 0.0
    state.heading = angle + math.pi / 2.0
    state.velocity_x = 5.0 * math.cos(state.heading)
    state.velocity_y = 5.0 * math.sin(state.heading)
    state.length = 4.8
    state.width = 2.0
    state.height = 1.6
    state.valid = True
    return scenario


def build_road_environment_scenario(
    lane_type,
    speed_limit_mph,
    *,
    parked_vehicle_count=0,
    enclosed_by_road_edges=False,
):
    scenario = scenario_pb2.Scenario()
    scenario.scenario_id = "synthetic-road-environment"
    scenario.current_time_index = 0
    scenario.sdc_track_index = 0
    scenario.timestamps_seconds.append(0.0)
    lane = add_lane(
        scenario,
        1,
        [(float(x), 0.0) for x in range(-40, 41)],
    )
    lane.type = lane_type
    lane.speed_limit_mph = speed_limit_mph
    if enclosed_by_road_edges:
        add_road_edge(
            scenario,
            9001,
            [
                (-30.0, -12.0),
                (30.0, -12.0),
                (30.0, 12.0),
                (-30.0, 12.0),
                (-30.0, -12.0),
            ],
        )

    sdc = scenario.tracks.add()
    sdc.id = 1
    sdc.object_type = 1
    state = sdc.states.add()
    state.center_x = 0.0
    state.center_y = 0.0
    state.heading = 0.0
    state.length = 4.8
    state.width = 2.0
    state.height = 1.6
    state.valid = True

    for index in range(parked_vehicle_count):
        track = scenario.tracks.add()
        track.id = 100 + index
        track.object_type = 1
        parked_state = track.states.add()
        parked_state.center_x = -30.0 + 2.5 * (index % 25)
        parked_state.center_y = 8.0 if index % 2 == 0 else -8.0
        parked_state.heading = 0.0
        parked_state.length = 4.8
        parked_state.width = 2.0
        parked_state.height = 1.6
        parked_state.valid = True
    return scenario


def build_freeway_ramp_scenario():
    scenario = scenario_pb2.Scenario()
    scenario.scenario_id = "synthetic-freeway-ramp"
    scenario.current_time_index = 0
    scenario.sdc_track_index = 0
    scenario.timestamps_seconds.append(0.0)

    ramp_right = add_lane(
        scenario,
        1,
        [(float(x), 0.0) for x in range(-60, 1, 5)],
    )
    ramp_left = add_lane(
        scenario,
        2,
        [(float(x), 3.0) for x in range(-60, 1, 5)],
    )
    transition_right = add_lane(
        scenario,
        3,
        [(float(x), 0.0) for x in range(0, 21, 5)],
    )
    transition_left = add_lane(
        scenario,
        4,
        [(float(x), 3.0) for x in range(0, 21, 5)],
    )
    mainline_lanes = [
        add_lane(
            scenario,
            10 + lane_index,
            [
                (float(x), 1.5 + 3.0 * lane_index)
                for x in range(20, 81, 5)
            ],
        )
        for lane_index in range(4)
    ]
    all_lanes = [
        ramp_right,
        ramp_left,
        transition_right,
        transition_left,
        *mainline_lanes,
    ]
    for lane in all_lanes:
        lane.type = 1
        lane.speed_limit_mph = 65.0

    add_lane_neighbor_pair(ramp_right, 1, ramp_left, 2)
    add_lane_neighbor_pair(transition_right, 3, transition_left, 4)
    for lane_index in range(len(mainline_lanes) - 1):
        add_lane_neighbor_pair(
            mainline_lanes[lane_index],
            10 + lane_index,
            mainline_lanes[lane_index + 1],
            11 + lane_index,
        )

    ramp_right.exit_lanes.append(3)
    transition_right.entry_lanes.append(1)
    ramp_left.exit_lanes.append(4)
    transition_left.entry_lanes.append(2)
    transition_right.exit_lanes.append(10)
    transition_left.exit_lanes.append(10)
    mainline_lanes[0].entry_lanes.extend((3, 4))

    sdc = scenario.tracks.add()
    sdc.id = 1
    sdc.object_type = 1
    state = sdc.states.add()
    state.center_x = -30.0
    state.center_y = 0.0
    state.heading = 0.0
    state.velocity_x = 15.0
    state.length = 4.8
    state.width = 2.0
    state.height = 1.6
    state.valid = True
    return scenario


class MapAnnotationTest(unittest.TestCase):
    def test_signalized_intersection_and_ego_regions(self):
        annotation = annotate_scenario(build_four_arm_scenario())

        self.assertEqual(len(annotation.junctions), 1)
        junction = annotation.junctions[0]
        self.assertEqual(junction.kind.value, "signalized")
        self.assertEqual(junction.arm_count, 4)
        self.assertEqual(len(junction.incoming_lane_ids), 4)
        self.assertEqual(len(junction.outgoing_lane_ids), 4)
        self.assertTrue(junction.boundary_polygons_xy)
        self.assertTrue(all(arm.stop_line_xy is not None for arm in junction.arms))

        regions = [frame.region_type for frame in annotation.ego_frames]
        self.assertEqual(
            regions,
            [
                RegionType.INTERSECTION,
                RegionType.INTERSECTION,
                RegionType.INTERSECTION,
                RegionType.INTERSECTION,
                RegionType.INTERSECTION,
                RegionType.ROAD_SEGMENT,
                RegionType.ROAD_SEGMENT,
            ],
        )
        self.assertEqual(annotation.ego_frames[0].same_direction_lane_count, 1)
        self.assertEqual(annotation.ego_frames[0].same_direction_lane_ids, (10,))
        self.assertIsNone(annotation.ego_frames[2].same_direction_lane_count)
        self.assertIsNone(annotation.ego_frames[2].same_direction_lane_ids)
        self.assertEqual(annotation.ego_frames[4].junction_arm_count, 4)
        self.assertIsNone(annotation.ego_frames[-1].junction_arm_count)

    def test_invalid_sdc_frame_is_unknown(self):
        scenario = build_four_arm_scenario()
        scenario.tracks[0].states[3].valid = False
        annotation = annotate_scenario(scenario)

        frame = annotation.ego_frames[3]
        self.assertEqual(frame.region_type, RegionType.UNKNOWN)
        self.assertEqual(frame.reason, "invalid_sdc_state")
        self.assertEqual(frame.confidence, 0.0)

    def test_can_annotate_only_selected_frame(self):
        annotation = annotate_scenario(
            build_four_arm_scenario(),
            frame_indices=(3,),
        )

        self.assertEqual(len(annotation.ego_frames), 1)
        self.assertEqual(annotation.ego_frames[0].frame_index, 3)
        self.assertEqual(
            annotation.ego_frames[0].region_type,
            RegionType.INTERSECTION,
        )

    def test_annotation_payload_is_json_serializable(self):
        payload = annotate_scenario(
            build_four_arm_scenario(),
            scenario_index=7,
            source_file="sample.tfrecord",
        ).to_dict()

        encoded = json.dumps(payload)
        self.assertIn("ego-map-annotation-v7", encoded)
        self.assertEqual(payload["scenario_index"], 7)
        self.assertEqual(payload["source_file"], "sample.tfrecord")
        self.assertEqual(payload["statistics"]["signalized_junction_count"], 1)
        self.assertEqual(
            payload["statistics"]["road_environment_counts"],
            {"URBAN_STREET": 7},
        )
        self.assertEqual(
            payload["statistics"]["road_environment_subtype_counts"],
            {},
        )
        self.assertEqual(payload["ego_frames"][0]["same_direction_lane_ids"], [10])

    def test_native_freeway_lane_is_reported_with_lane_count(self):
        frame = annotate_scenario(
            build_road_environment_scenario(1, 65.0)
        ).ego_frames[0]

        self.assertEqual(frame.road_environment, RoadEnvironment.FREEWAY)
        self.assertEqual(
            frame.road_environment_subtype,
            RoadEnvironmentSubtype.FREEWAY_MAINLINE,
        )
        self.assertEqual(frame.road_environment_lane_count, 1)
        self.assertEqual(frame.matched_lane_type, "FREEWAY")
        self.assertEqual(
            frame.road_environment_reason,
            "womd_lane_type_freeway",
        )

    def test_narrow_freeway_branch_connected_to_wider_corridor_is_ramp(self):
        frame = annotate_scenario(
            build_freeway_ramp_scenario()
        ).ego_frames[0]

        self.assertEqual(frame.road_environment, RoadEnvironment.FREEWAY)
        self.assertEqual(
            frame.road_environment_subtype,
            RoadEnvironmentSubtype.FREEWAY_RAMP,
        )
        self.assertEqual(frame.road_environment_lane_count, 2)
        self.assertEqual(
            frame.road_environment_subtype_reason,
            "freeway_ramp_narrow_branch_to_wider_mainline",
        )

    def test_dense_parked_context_without_road_edge_enclosure_is_urban(self):
        frame = annotate_scenario(
            build_road_environment_scenario(
                2,
                15.0,
                parked_vehicle_count=30,
            )
        ).ego_frames[0]

        self.assertEqual(frame.road_environment, RoadEnvironment.URBAN_STREET)
        self.assertEqual(
            frame.road_environment_reason,
            "womd_lane_type_surface_street",
        )

    def test_road_edge_enclosed_dense_context_is_parking_lot(self):
        frame = annotate_scenario(
            build_road_environment_scenario(
                2,
                15.0,
                parked_vehicle_count=30,
                enclosed_by_road_edges=True,
            )
        ).ego_frames[0]

        self.assertEqual(frame.road_environment, RoadEnvironment.PARKING_LOT)
        self.assertEqual(frame.road_environment_lane_count, 1)
        self.assertEqual(frame.matched_lane_type, "SURFACE_STREET")
        self.assertEqual(
            frame.road_environment_reason,
            "compact_road_edge_enclosed_parking_area",
        )

    def test_config_rejects_invalid_arm_range(self):
        with self.assertRaises(ValueError):
            MapAnnotationConfig(min_junction_arms=5, max_junction_arms=4)

    def test_default_intersection_distance_is_40_metres(self):
        self.assertEqual(MapAnnotationConfig().near_distance_m, 40.0)

    def test_lane_count_uses_the_more_complete_approach_cross_section(self):
        self.assertEqual(
            _prefer_complete_lane_cross_section(
                (710,),
                (668, 669, 670, 671, 672, 710),
            ),
            (668, 669, 670, 671, 672, 710),
        )
        self.assertEqual(
            _prefer_complete_lane_cross_section(
                (188, 190, 191, 192),
                (190, 191, 192),
            ),
            (188, 190, 191, 192),
        )
        self.assertIsNone(
            _prefer_complete_lane_cross_section(
                None,
                (668, 669, 670, 671, 672, 710),
            )
        )

    def test_lane_count_deduplicates_overlapping_centerlines(self):
        scenario = scenario_pb2.Scenario()
        primary = add_lane(
            scenario,
            1,
            [(float(x), 0.0) for x in range(21)],
        )
        adjacent = add_lane(
            scenario,
            2,
            [(float(x), 3.0) for x in range(21)],
        )
        overlapping = add_lane(
            scenario,
            3,
            [(float(x), 3.2) for x in range(21)],
        )
        first_neighbor = primary.left_neighbors.add()
        first_neighbor.feature_id = 2
        first_neighbor.self_start_index = 0
        first_neighbor.self_end_index = 20
        first_neighbor.neighbor_start_index = 0
        first_neighbor.neighbor_end_index = 20
        duplicate_neighbor = adjacent.left_neighbors.add()
        duplicate_neighbor.feature_id = 3
        duplicate_neighbor.self_start_index = 0
        duplicate_neighbor.self_end_index = 20
        duplicate_neighbor.neighbor_start_index = 0
        duplicate_neighbor.neighbor_end_index = 20
        lanes = {
            1: SimpleNamespace(lane=primary),
            2: SimpleNamespace(lane=adjacent),
            3: SimpleNamespace(lane=overlapping),
        }

        lane_index = _LaneGeometryIndex(lanes, MapAnnotationConfig())

        self.assertEqual(lane_index.same_direction_lane_ids(1, 10), (1, 2))
        self.assertEqual(lane_index.same_direction_lane_count(1, 10), 2)

    def test_geometry_only_intersection_is_annotated_without_controls(self):
        scenario = build_four_arm_scenario()
        for dynamic_state in scenario.dynamic_map_states:
            dynamic_state.ClearField("lane_states")

        annotation = annotate_scenario(scenario)

        self.assertEqual(len(annotation.junctions), 1)
        self.assertEqual(annotation.junctions[0].kind, JunctionKind.GEOMETRIC)
        self.assertEqual(annotation.junctions[0].arm_count, 4)
        self.assertEqual(annotation.junctions[0].signal_lane_ids, ())
        self.assertEqual(
            [frame.region_type for frame in annotation.ego_frames],
            [
                RegionType.INTERSECTION,
                RegionType.INTERSECTION,
                RegionType.INTERSECTION,
                RegionType.INTERSECTION,
                RegionType.INTERSECTION,
                RegionType.ROAD_SEGMENT,
                RegionType.ROAD_SEGMENT,
            ],
        )

    def test_multilane_roundabout_is_annotated_as_intersection(self):
        annotation = annotate_scenario(build_roundabout_scenario())

        roundabouts = [
            junction
            for junction in annotation.junctions
            if junction.kind == JunctionKind.ROUNDABOUT
        ]
        self.assertEqual(len(roundabouts), 1)
        self.assertEqual(roundabouts[0].arm_count, 4)
        self.assertIn("directed_lane_cycle", roundabouts[0].evidence)
        self.assertEqual(
            annotation.ego_frames[0].region_type,
            RegionType.INTERSECTION,
        )
        self.assertEqual(
            annotation.ego_frames[0].junction_kind,
            JunctionKind.ROUNDABOUT,
        )
        self.assertEqual(
            annotation.to_dict()["statistics"]["roundabout_junction_count"],
            1,
        )

    def test_travel_heading_prevents_chained_radial_arm_merge(self):
        scenario = scenario_pb2.Scenario()
        lanes = {}

        def add_gate_lane(
            lane_id,
            gate_angle_deg,
            arm_angle_deg,
            incoming,
        ):
            gate_angle = math.radians(gate_angle_deg)
            arm_angle = math.radians(arm_angle_deg)
            gate = (
                20.0 * math.cos(gate_angle),
                20.0 * math.sin(gate_angle),
            )
            outward = (
                20.0 * math.cos(arm_angle),
                20.0 * math.sin(arm_angle),
            )
            if incoming:
                points = [
                    (gate[0] + outward[0], gate[1] + outward[1]),
                    gate,
                ]
            else:
                points = [
                    gate,
                    (gate[0] + outward[0], gate[1] + outward[1]),
                ]
            lane = add_lane(scenario, lane_id, points)
            lanes[lane_id] = SimpleNamespace(lane=lane)

        add_gate_lane(1, -10.0, 0.0, True)
        add_gate_lane(2, 10.0, 0.0, False)
        add_gate_lane(3, 29.0, 50.0, True)
        add_gate_lane(4, 37.0, 50.0, False)
        config = MapAnnotationConfig()

        radial_arms = _build_arms(
            {1, 3},
            {2, 4},
            lanes,
            (0.0, 0.0),
            config,
            use_radial_angles=True,
        )
        travel_heading_arms = _build_arms(
            {1, 3},
            {2, 4},
            lanes,
            (0.0, 0.0),
            config,
            use_radial_angles=False,
        )

        self.assertEqual(len(radial_arms), 1)
        self.assertEqual(len(travel_heading_arms), 2)

    def test_distance_threshold_emits_only_road_or_intersection(self):
        annotation = annotate_scenario(
            build_four_arm_scenario(),
            MapAnnotationConfig(near_distance_m=5.0),
        )

        self.assertEqual(
            [frame.region_type for frame in annotation.ego_frames],
            [
                RegionType.ROAD_SEGMENT,
                RegionType.INTERSECTION,
                RegionType.INTERSECTION,
                RegionType.INTERSECTION,
                RegionType.INTERSECTION,
                RegionType.ROAD_SEGMENT,
                RegionType.ROAD_SEGMENT,
            ],
        )

    def test_upcoming_junction_is_preferred_over_closer_passed_junction(self):
        passed = SimpleNamespace(
            core_lane_ids=(),
            _directional_branch_through_lane_ids=frozenset(),
            _to_core={},
            _from_core={1: (0.0, 2)},
            confidence=0.85,
        )
        upcoming = SimpleNamespace(
            core_lane_ids=(),
            _directional_branch_through_lane_ids=frozenset(),
            _to_core={1: (23.2, 0)},
            _from_core={},
            confidence=1.0,
        )
        annotator = EgoMapAnnotator.__new__(EgoMapAnnotator)
        annotator.config = MapAnnotationConfig(near_distance_m=40.0)
        annotator.junctions = (passed, upcoming)
        match = MapMatch(
            lane_id=1,
            point_index=0,
            lane_s_m=8.5,
            distance_m=0.0,
            heading_error_rad=0.0,
            confidence=1.0,
            confident=True,
        )

        region, junction, arm_index, distance = annotator._classify(match)

        self.assertEqual(region, RegionType.INTERSECTION)
        self.assertIs(junction, upcoming)
        self.assertEqual(arm_index, 0)
        self.assertAlmostEqual(distance, 14.7)

        annotator.junctions = (passed,)
        region, junction, arm_index, distance = annotator._classify(match)
        self.assertEqual(region, RegionType.ROAD_SEGMENT)
        self.assertIsNone(junction)
        self.assertIsNone(arm_index)
        self.assertIsNone(distance)

    def test_directional_branch_keeps_pure_mainline_lane_as_road_segment(self):
        def lane_view(*, entries=(), exits=()):
            return SimpleNamespace(
                lane=SimpleNamespace(
                    entry_lanes=list(entries),
                    exit_lanes=list(exits),
                )
            )

        lanes = {
            1: lane_view(exits=(10,)),
            2: lane_view(entries=(10, 12)),
            3: lane_view(exits=(10, 11)),
            4: lane_view(entries=(11,)),
            5: lane_view(exits=(12,)),
            10: lane_view(entries=(1, 3), exits=(2,)),
            11: lane_view(entries=(3,), exits=(4,)),
            12: lane_view(entries=(5,), exits=(2,)),
        }
        side_arm = SimpleNamespace(
            arm_index=0,
            angle_rad=math.pi / 2.0,
            incoming_lane_ids=(5,),
            outgoing_lane_ids=(4,),
        )
        outgoing_main_arm = SimpleNamespace(
            arm_index=1,
            angle_rad=math.pi,
            incoming_lane_ids=(),
            outgoing_lane_ids=(2,),
        )
        incoming_main_arm = SimpleNamespace(
            arm_index=2,
            angle_rad=0.0,
            incoming_lane_ids=(1, 3),
            outgoing_lane_ids=(),
        )
        junction = SimpleNamespace(
            kind=JunctionKind.STOP_CONTROLLED,
            arm_count=3,
            arms=(side_arm, outgoing_main_arm, incoming_main_arm),
            core_lane_ids=(10, 11, 12),
            _to_core={
                1: (20.0, 2),
                3: (20.0, 2),
                5: (20.0, 0),
            },
            _from_core={},
            confidence=0.9,
        )
        config = MapAnnotationConfig()
        through_lane_ids = _directional_branch_through_lane_ids(
            junction,
            lanes,
            config,
        )

        self.assertEqual(through_lane_ids, frozenset((1, 10)))

        junction._directional_branch_through_lane_ids = through_lane_ids
        annotator = EgoMapAnnotator.__new__(EgoMapAnnotator)
        annotator.config = config
        annotator.junctions = (junction,)

        def classify(lane_id):
            return annotator._classify(
                MapMatch(
                    lane_id=lane_id,
                    point_index=0,
                    lane_s_m=0.0,
                    distance_m=0.0,
                    heading_error_rad=0.0,
                    confidence=1.0,
                    confident=True,
                )
            )

        self.assertEqual(classify(1)[0], RegionType.ROAD_SEGMENT)
        self.assertEqual(classify(10)[0], RegionType.ROAD_SEGMENT)
        self.assertEqual(classify(3)[0], RegionType.INTERSECTION)
        self.assertEqual(classify(11)[0], RegionType.INTERSECTION)
        self.assertEqual(classify(5)[0], RegionType.INTERSECTION)

    def test_lane_count_extends_neighbor_range_for_turn_lane(self):
        scenario = scenario_pb2.Scenario()
        primary = add_lane(
            scenario,
            1,
            [(float(x), 0.0) for x in range(21)],
        )
        turn_lane = add_lane(
            scenario,
            2,
            [(float(x), -4.0) for x in range(21)],
        )
        neighbor = primary.right_neighbors.add()
        neighbor.feature_id = 2
        neighbor.self_start_index = 0
        neighbor.self_end_index = 10
        neighbor.neighbor_start_index = 0
        neighbor.neighbor_end_index = 10
        lanes = {
            1: SimpleNamespace(lane=primary),
            2: SimpleNamespace(lane=turn_lane),
        }

        lane_index = _LaneGeometryIndex(lanes, MapAnnotationConfig())

        self.assertEqual(lane_index.same_direction_lane_count(1, 15), 2)
        self.assertEqual(lane_index.same_direction_lane_ids(1, 15), (1, 2))

    def test_lane_count_includes_diverging_turn_lane_at_current_station(self):
        scenario = scenario_pb2.Scenario()
        primary = add_lane(
            scenario,
            1,
            [(float(x), 0.0) for x in range(21)],
        )
        adjacent = add_lane(
            scenario,
            2,
            [(float(x), -3.0) for x in range(21)],
        )
        turn_lane = add_lane(
            scenario,
            3,
            [
                (float(x), -3.0 - 0.2 * float(x))
                for x in range(21)
            ],
        )
        neighbor = primary.right_neighbors.add()
        neighbor.feature_id = 2
        neighbor.self_start_index = 0
        neighbor.self_end_index = 20
        neighbor.neighbor_start_index = 0
        neighbor.neighbor_end_index = 20

        def lane_view(lane, *, diverge_lanes=()):
            return SimpleNamespace(
                lane=SimpleNamespace(
                    polyline=lane.polyline,
                    entry_lanes=lane.entry_lanes,
                    exit_lanes=lane.exit_lanes,
                    left_neighbors=lane.left_neighbors,
                    right_neighbors=lane.right_neighbors,
                    diverge_lanes=set(diverge_lanes),
                    merge_lanes=set(),
                )
            )

        lanes = {
            1: lane_view(primary),
            2: lane_view(adjacent, diverge_lanes=(3,)),
            3: lane_view(turn_lane, diverge_lanes=(2,)),
        }

        lane_index = _LaneGeometryIndex(lanes, MapAnnotationConfig())

        self.assertEqual(
            lane_index.same_direction_lane_ids(1, 15),
            (1, 2, 3),
        )
        self.assertEqual(lane_index.same_direction_lane_count(1, 15), 3)

    def test_lane_count_deduplicates_overlapping_merge_transition(self):
        scenario = scenario_pb2.Scenario()
        primary = add_lane(
            scenario,
            1,
            [(float(x), 0.0) for x in range(21)],
        )
        adjacent = add_lane(
            scenario,
            2,
            [(float(x), -3.0) for x in range(21)],
        )
        merge_transition = add_lane(
            scenario,
            3,
            [
                (float(x), -6.0 + 0.15 * float(x))
                for x in range(21)
            ],
        )
        neighbor = primary.right_neighbors.add()
        neighbor.feature_id = 2
        neighbor.self_start_index = 0
        neighbor.self_end_index = 20
        neighbor.neighbor_start_index = 0
        neighbor.neighbor_end_index = 20

        def lane_view(lane, *, merge_lanes=()):
            return SimpleNamespace(
                lane=SimpleNamespace(
                    polyline=lane.polyline,
                    entry_lanes=lane.entry_lanes,
                    exit_lanes=lane.exit_lanes,
                    left_neighbors=lane.left_neighbors,
                    right_neighbors=lane.right_neighbors,
                    diverge_lanes=set(),
                    merge_lanes=set(merge_lanes),
                )
            )

        lanes = {
            1: lane_view(primary),
            2: lane_view(adjacent, merge_lanes=(3,)),
            3: lane_view(merge_transition, merge_lanes=(2,)),
        }

        lane_index = _LaneGeometryIndex(lanes, MapAnnotationConfig())

        self.assertEqual(
            lane_index.same_direction_lane_ids(1, 10),
            (1, 2),
        )
        self.assertEqual(lane_index.same_direction_lane_count(1, 10), 2)

    def test_lane_count_inherits_neighbors_across_segment_boundaries(self):
        scenario = scenario_pb2.Scenario()
        primary_predecessor = add_lane(
            scenario,
            10,
            [(float(x), 0.0) for x in range(-20, 1)],
        )
        turn_predecessor = add_lane(
            scenario,
            20,
            [(float(x), -3.0) for x in range(-20, 1)],
        )
        primary = add_lane(
            scenario,
            1,
            [(float(x), 0.0) for x in range(21)],
        )
        turn_lane = add_lane(
            scenario,
            2,
            [
                (float(x), -3.0 - 0.5 * float(x))
                for x in range(21)
            ],
        )
        primary.entry_lanes.append(10)
        primary_predecessor.exit_lanes.append(1)
        turn_lane.entry_lanes.append(20)
        turn_predecessor.exit_lanes.append(2)

        right_neighbor = primary_predecessor.right_neighbors.add()
        right_neighbor.feature_id = 20
        right_neighbor.self_start_index = 0
        right_neighbor.self_end_index = 20
        right_neighbor.neighbor_start_index = 0
        right_neighbor.neighbor_end_index = 20
        left_neighbor = turn_predecessor.left_neighbors.add()
        left_neighbor.feature_id = 10
        left_neighbor.self_start_index = 0
        left_neighbor.self_end_index = 20
        left_neighbor.neighbor_start_index = 0
        left_neighbor.neighbor_end_index = 20

        lanes = {
            1: SimpleNamespace(lane=primary),
            2: SimpleNamespace(lane=turn_lane),
            10: SimpleNamespace(lane=primary_predecessor),
            20: SimpleNamespace(lane=turn_predecessor),
        }

        lane_index = _LaneGeometryIndex(lanes, MapAnnotationConfig())

        self.assertEqual(lane_index.same_direction_lane_ids(1, 15), (1, 2))
        self.assertEqual(lane_index.same_direction_lane_count(1, 15), 2)

    def test_lane_count_does_not_extend_a_merged_predecessor(self):
        scenario = scenario_pb2.Scenario()
        matched = add_lane(
            scenario,
            1,
            [(float(x), 0.0) for x in range(11)],
        )
        adjacent = add_lane(
            scenario,
            2,
            [(float(x), -3.0) for x in range(11)],
        )
        predecessor = add_lane(
            scenario,
            3,
            [(float(x), 0.0) for x in range(-5, 1)],
        )
        matched.entry_lanes.append(3)
        predecessor.exit_lanes.append(1)

        matched_neighbor = matched.right_neighbors.add()
        matched_neighbor.feature_id = 2
        matched_neighbor.self_start_index = 0
        matched_neighbor.self_end_index = 10
        matched_neighbor.neighbor_start_index = 0
        matched_neighbor.neighbor_end_index = 10

        predecessor_neighbor = adjacent.left_neighbors.add()
        predecessor_neighbor.feature_id = 3
        predecessor_neighbor.self_start_index = 0
        predecessor_neighbor.self_end_index = 1
        predecessor_neighbor.neighbor_start_index = 4
        predecessor_neighbor.neighbor_end_index = 5

        lanes = {
            1: SimpleNamespace(lane=matched),
            2: SimpleNamespace(lane=adjacent),
            3: SimpleNamespace(lane=predecessor),
        }
        lane_index = _LaneGeometryIndex(lanes, MapAnnotationConfig())

        self.assertEqual(lane_index.same_direction_lane_count(1, 4), 2)
        self.assertEqual(lane_index.same_direction_lane_ids(1, 4), (1, 2))

    def test_lane_count_follows_neighbor_successor_at_current_station(self):
        scenario = scenario_pb2.Scenario()
        primary = add_lane(
            scenario,
            1,
            [(float(x), 0.0) for x in range(21)],
        )
        short_neighbor = add_lane(
            scenario,
            2,
            [(float(x), -4.0) for x in range(11)],
        )
        continued_neighbor = add_lane(
            scenario,
            3,
            [(float(x), -4.0) for x in range(10, 21)],
        )
        short_neighbor.exit_lanes.append(3)
        continued_neighbor.entry_lanes.append(2)

        neighbor = primary.right_neighbors.add()
        neighbor.feature_id = 2
        neighbor.self_start_index = 0
        neighbor.self_end_index = 10
        neighbor.neighbor_start_index = 0
        neighbor.neighbor_end_index = 10
        lanes = {
            1: SimpleNamespace(lane=primary),
            2: SimpleNamespace(lane=short_neighbor),
            3: SimpleNamespace(lane=continued_neighbor),
        }

        lane_index = _LaneGeometryIndex(lanes, MapAnnotationConfig())

        self.assertEqual(lane_index.same_direction_lane_ids(1, 15), (1, 3))
        self.assertEqual(lane_index.same_direction_lane_count(1, 15), 2)

    def test_spatially_overlapping_signalized_groups_are_merged(self):
        scenario = scenario_pb2.Scenario()
        add_lane(scenario, 1, [(-10.0, 0.0), (10.0, 0.0)])
        add_lane(scenario, 2, [(0.0, -10.0), (0.0, 10.0)])
        add_lane(scenario, 3, [(100.0, -10.0), (100.0, 10.0)])
        lanes = ScenarioProcessor(
            scenario,
            load_boundaries=False,
        ).lanecenters

        merged = _merge_spatially_overlapping_groups(
            [
                (JunctionKind.SIGNALIZED, {1}),
                (JunctionKind.SIGNALIZED, {2}),
                (JunctionKind.SIGNALIZED, {3}),
            ],
            lanes,
            MapAnnotationConfig(),
        )

        self.assertEqual(
            [(kind, frozenset(core)) for kind, core in merged],
            [
                (JunctionKind.SIGNALIZED, frozenset({1, 2})),
                (JunctionKind.SIGNALIZED, frozenset({3})),
            ],
        )


if __name__ == "__main__":
    unittest.main()
