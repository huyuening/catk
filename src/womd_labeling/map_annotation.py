from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
import heapq
from itertools import combinations
import math
from typing import Any, Iterable

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

try:
    from scipy.spatial import QhullError
except ImportError:
    from scipy.spatial.qhull import QhullError

try:
    from shapely.geometry import LineString, MultiPolygon, Polygon
    from shapely.ops import unary_union
except ImportError:
    LineString = MultiPolygon = Polygon = None
    unary_union = None

from ._compat.waymonizer import Waymonizer as ScenarioProcessor


MAP_ANNOTATION_SCHEMA_VERSION = "ego-map-annotation-v7"
_MAX_LANE_NEIGHBOR_TOPOLOGY_HOPS = 8
_GATE_LANE_DEDUP_TOLERANCE_M = 1.0
_ROUNDABOUT_MIN_SPAN_M = 8.0
_ROUNDABOUT_MAX_COMPONENT_LANES = 64
_ROUNDABOUT_MIN_ASPECT_RATIO = 0.65
_ROUNDABOUT_MAX_RADIAL_RATIO = 1.5
_ROUNDABOUT_MAX_MEDIAN_RADIAL_ALIGNMENT = 0.25
_ROUNDABOUT_MAX_P90_RADIAL_ALIGNMENT = 0.6
_ROUNDABOUT_MAX_ANGULAR_GAP_RAD = math.radians(45.0)
_WOMD_LANE_TYPE_NAMES = {
    0: "UNDEFINED",
    1: "FREEWAY",
    2: "SURFACE_STREET",
    3: "BIKE_LANE",
}


def _prefer_complete_lane_cross_section(
    local_lane_ids: tuple[int, ...] | None,
    junction_side_lane_ids: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    """Use the fuller cross-section without merging longitudinally distinct lanes."""
    if local_lane_ids is None:
        return None
    if (
        junction_side_lane_ids is not None
        and len(junction_side_lane_ids) > len(local_lane_ids)
    ):
        return junction_side_lane_ids
    return local_lane_ids


class RegionType(str, Enum):
    ROAD_SEGMENT = "ROAD_SEGMENT"
    INTERSECTION = "INTERSECTION"
    UNKNOWN = "UNKNOWN"
    # Retained so older annotation files remain readable.
    NEAR_INTERSECTION_APPROACH = "NEAR_INTERSECTION_APPROACH"
    IN_INTERSECTION = "IN_INTERSECTION"
    NEAR_INTERSECTION_EXIT = "NEAR_INTERSECTION_EXIT"


class RoadEnvironment(str, Enum):
    FREEWAY = "FREEWAY"
    URBAN_STREET = "URBAN_STREET"
    PARKING_LOT = "PARKING_LOT"
    UNKNOWN = "UNKNOWN"


class RoadEnvironmentSubtype(str, Enum):
    FREEWAY_MAINLINE = "FREEWAY_MAINLINE"
    FREEWAY_RAMP = "FREEWAY_RAMP"


class JunctionKind(str, Enum):
    SIGNALIZED = "signalized"
    STOP_CONTROLLED = "stop_controlled"
    ROUNDABOUT = "roundabout"
    GEOMETRIC = "geometric"


def _junction_kind_priority(kind: JunctionKind) -> int:
    return {
        JunctionKind.SIGNALIZED: 0,
        JunctionKind.STOP_CONTROLLED: 1,
        JunctionKind.ROUNDABOUT: 2,
        JunctionKind.GEOMETRIC: 3,
    }[kind]


@dataclass(frozen=True)
class MapAnnotationConfig:
    near_distance_m: float = 40.0
    lane_half_width_m: float = 2.0
    lane_neighbor_extension_m: float = 12.0
    max_lane_neighbor_distance_m: float = 8.0
    junction_merge_overlap_ratio: float = 0.15
    arm_angle_threshold_deg: float = 30.0
    max_map_match_distance_m: float = 8.0
    max_map_match_heading_error_deg: float = 60.0
    heading_weight_m_per_rad: float = 2.0
    continuity_penalty_m: float = 2.0
    include_stop_controlled: bool = True
    min_junction_arms: int = 3
    max_junction_arms: int = 8
    parking_max_speed_limit_mph: float = 15.0
    parking_context_radius_m: float = 60.0
    parking_stationary_speed_mps: float = 0.5
    parking_off_lane_distance_m: float = 4.0
    parking_min_off_lane_stationary_vehicles: int = 15
    parking_dense_off_lane_stationary_vehicles: int = 25
    parking_min_off_lane_stationary_ratio: float = 0.75
    parking_min_low_speed_lane_ratio: float = 0.85
    parking_intersection_min_low_speed_lane_ratio: float = 0.90
    parking_access_max_driveway_distance_m: float = 20.0
    parking_access_max_low_speed_lane_ratio: float = 0.70
    parking_internal_lane_radius_m: float = 30.0
    parking_internal_min_lane_count: int = 18
    parking_internal_min_branch_lane_count: int = 4
    parking_internal_max_lane_length_m: float = 90.0
    parking_compact_edge_distance_m: float = 40.0
    freeway_ramp_max_lane_count: int = 3
    freeway_ramp_min_lane_count_gain: int = 2
    freeway_ramp_topology_hops: int = 6

    def __post_init__(self) -> None:
        if self.near_distance_m <= 0:
            raise ValueError("near_distance_m must be positive")
        if self.lane_half_width_m <= 0:
            raise ValueError("lane_half_width_m must be positive")
        if self.lane_neighbor_extension_m < 0:
            raise ValueError("lane_neighbor_extension_m must be non-negative")
        if self.max_lane_neighbor_distance_m <= 0:
            raise ValueError("max_lane_neighbor_distance_m must be positive")
        if not 0 < self.junction_merge_overlap_ratio <= 1:
            raise ValueError(
                "junction_merge_overlap_ratio must be in the interval (0, 1]"
            )
        if not 0 < self.arm_angle_threshold_deg < 180:
            raise ValueError("arm_angle_threshold_deg must be between 0 and 180")
        if self.max_map_match_distance_m <= 0:
            raise ValueError("max_map_match_distance_m must be positive")
        if not 0 < self.max_map_match_heading_error_deg <= 180:
            raise ValueError(
                "max_map_match_heading_error_deg must be between 0 and 180"
            )
        if self.min_junction_arms < 2:
            raise ValueError("min_junction_arms must be at least 2")
        if self.max_junction_arms < self.min_junction_arms:
            raise ValueError(
                "max_junction_arms must be at least min_junction_arms"
            )
        if self.parking_max_speed_limit_mph <= 0:
            raise ValueError("parking_max_speed_limit_mph must be positive")
        if self.parking_context_radius_m <= 0:
            raise ValueError("parking_context_radius_m must be positive")
        if self.parking_stationary_speed_mps < 0:
            raise ValueError("parking_stationary_speed_mps must be non-negative")
        if self.parking_off_lane_distance_m <= 0:
            raise ValueError("parking_off_lane_distance_m must be positive")
        if self.parking_min_off_lane_stationary_vehicles < 1:
            raise ValueError(
                "parking_min_off_lane_stationary_vehicles must be positive"
            )
        if (
            self.parking_dense_off_lane_stationary_vehicles
            < self.parking_min_off_lane_stationary_vehicles
        ):
            raise ValueError(
                "parking_dense_off_lane_stationary_vehicles must be at least "
                "parking_min_off_lane_stationary_vehicles"
            )
        for field_name, value in (
            (
                "parking_min_off_lane_stationary_ratio",
                self.parking_min_off_lane_stationary_ratio,
            ),
            (
                "parking_min_low_speed_lane_ratio",
                self.parking_min_low_speed_lane_ratio,
            ),
            (
                "parking_intersection_min_low_speed_lane_ratio",
                self.parking_intersection_min_low_speed_lane_ratio,
            ),
            (
                "parking_access_max_low_speed_lane_ratio",
                self.parking_access_max_low_speed_lane_ratio,
            ),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be in the interval [0, 1]")
        if self.parking_access_max_driveway_distance_m <= 0:
            raise ValueError(
                "parking_access_max_driveway_distance_m must be positive"
            )
        if self.parking_internal_lane_radius_m <= 0:
            raise ValueError("parking_internal_lane_radius_m must be positive")
        if self.parking_internal_min_lane_count < 1:
            raise ValueError(
                "parking_internal_min_lane_count must be positive"
            )
        if self.parking_internal_min_branch_lane_count < 0:
            raise ValueError(
                "parking_internal_min_branch_lane_count must be non-negative"
            )
        if self.parking_internal_max_lane_length_m <= 0:
            raise ValueError(
                "parking_internal_max_lane_length_m must be positive"
            )
        if self.parking_compact_edge_distance_m <= 0:
            raise ValueError(
                "parking_compact_edge_distance_m must be positive"
            )
        if self.freeway_ramp_max_lane_count < 1:
            raise ValueError("freeway_ramp_max_lane_count must be positive")
        if self.freeway_ramp_min_lane_count_gain < 1:
            raise ValueError(
                "freeway_ramp_min_lane_count_gain must be positive"
            )
        if self.freeway_ramp_topology_hops < 1:
            raise ValueError("freeway_ramp_topology_hops must be positive")
        if self.max_junction_arms < self.min_junction_arms:
            raise ValueError("max_junction_arms must not be smaller than min_junction_arms")


@dataclass(frozen=True)
class JunctionArm:
    arm_index: int
    angle_rad: float
    incoming_lane_ids: tuple[int, ...]
    outgoing_lane_ids: tuple[int, ...]
    gate_points_xy: tuple[tuple[float, float], ...]
    stop_line_xy: tuple[tuple[float, float], tuple[float, float]] | None

    @property
    def incoming_lane_count(self) -> int:
        return len(self.incoming_lane_ids)

    @property
    def outgoing_lane_count(self) -> int:
        return len(self.outgoing_lane_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_index": self.arm_index,
            "angle_rad": _rounded(self.angle_rad, 6),
            "incoming_lane_ids": list(self.incoming_lane_ids),
            "outgoing_lane_ids": list(self.outgoing_lane_ids),
            "incoming_lane_count": self.incoming_lane_count,
            "outgoing_lane_count": self.outgoing_lane_count,
            "gate_points_xy": [_rounded_xy(point) for point in self.gate_points_xy],
            "stop_line_xy": (
                None
                if self.stop_line_xy is None
                else [_rounded_xy(point) for point in self.stop_line_xy]
            ),
        }


@dataclass
class JunctionAnnotation:
    junction_id: int
    kind: JunctionKind
    core_lane_ids: tuple[int, ...]
    incoming_lane_ids: tuple[int, ...]
    outgoing_lane_ids: tuple[int, ...]
    signal_lane_ids: tuple[int, ...]
    center_xy: tuple[float, float]
    arms: tuple[JunctionArm, ...]
    stop_points_xy: tuple[tuple[float, float], ...]
    boundary_polygons_xy: tuple[tuple[tuple[float, float], ...], ...]
    confidence: float
    evidence: tuple[str, ...]
    _geometry: Any | None = field(default=None, repr=False)
    _to_core: dict[int, tuple[float, int]] = field(default_factory=dict, repr=False)
    _from_core: dict[int, tuple[float, int]] = field(default_factory=dict, repr=False)
    _directional_branch_through_lane_ids: frozenset[int] = field(
        default_factory=frozenset,
        repr=False,
    )

    @property
    def arm_count(self) -> int:
        return len(self.arms)

    def arm(self, arm_index: int | None) -> JunctionArm | None:
        if arm_index is None:
            return None
        return next((arm for arm in self.arms if arm.arm_index == arm_index), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "junction_id": self.junction_id,
            "kind": self.kind.value,
            "confidence": _rounded(self.confidence, 4),
            "evidence": list(self.evidence),
            "center_xy": _rounded_xy(self.center_xy),
            "arm_count": self.arm_count,
            "core_lane_ids": list(self.core_lane_ids),
            "incoming_lane_ids": list(self.incoming_lane_ids),
            "outgoing_lane_ids": list(self.outgoing_lane_ids),
            "signal_lane_ids": list(self.signal_lane_ids),
            "stop_points_xy": [_rounded_xy(point) for point in self.stop_points_xy],
            "boundary_polygons_xy": [
                [_rounded_xy(point) for point in polygon]
                for polygon in self.boundary_polygons_xy
            ],
            "arms": [arm.to_dict() for arm in self.arms],
        }


@dataclass(frozen=True)
class MapMatch:
    lane_id: int
    point_index: int
    lane_s_m: float
    distance_m: float
    heading_error_rad: float
    confidence: float
    confident: bool


@dataclass(frozen=True)
class EgoFrameAnnotation:
    frame_index: int
    timestamp_seconds: float | None
    valid: bool
    region_type: RegionType
    position_xy: tuple[float, float] | None
    matched_lane_id: int | None
    matched_lane_s_m: float | None
    same_direction_lane_count: int | None
    junction_id: int | None
    junction_kind: JunctionKind | None
    junction_arm_count: int | None
    junction_arm_index: int | None
    junction_side_lane_count: int | None
    distance_to_junction_m: float | None
    map_match_distance_m: float | None
    map_match_heading_error_rad: float | None
    confidence: float
    reason: str | None = None
    same_direction_lane_ids: tuple[int, ...] | None = None
    road_environment: RoadEnvironment = RoadEnvironment.UNKNOWN
    road_environment_subtype: RoadEnvironmentSubtype | None = None
    road_environment_lane_count: int | None = None
    road_environment_confidence: float = 0.0
    road_environment_reason: str | None = None
    road_environment_subtype_reason: str | None = None
    matched_lane_type: str | None = None
    matched_lane_speed_limit_mph: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": _rounded_optional(self.timestamp_seconds, 4),
            "valid": self.valid,
            "region_type": self.region_type.value,
            "position_xy": (
                None if self.position_xy is None else _rounded_xy(self.position_xy)
            ),
            "matched_lane_id": self.matched_lane_id,
            "matched_lane_s_m": _rounded_optional(self.matched_lane_s_m, 3),
            "same_direction_lane_count": self.same_direction_lane_count,
            "same_direction_lane_ids": (
                None
                if self.same_direction_lane_ids is None
                else list(self.same_direction_lane_ids)
            ),
            "road_environment": self.road_environment.value,
            "road_environment_subtype": (
                None
                if self.road_environment_subtype is None
                else self.road_environment_subtype.value
            ),
            "road_environment_lane_count": self.road_environment_lane_count,
            "road_environment_confidence": _rounded(
                self.road_environment_confidence,
                4,
            ),
            "road_environment_reason": self.road_environment_reason,
            "road_environment_subtype_reason": (
                self.road_environment_subtype_reason
            ),
            "matched_lane_type": self.matched_lane_type,
            "matched_lane_speed_limit_mph": _rounded_optional(
                self.matched_lane_speed_limit_mph,
                1,
            ),
            "junction_id": self.junction_id,
            "junction_kind": (
                None if self.junction_kind is None else self.junction_kind.value
            ),
            "junction_arm_count": self.junction_arm_count,
            "junction_arm_index": self.junction_arm_index,
            "junction_side_lane_count": self.junction_side_lane_count,
            "distance_to_junction_m": _rounded_optional(
                self.distance_to_junction_m, 3
            ),
            "map_match_distance_m": _rounded_optional(
                self.map_match_distance_m, 3
            ),
            "map_match_heading_error_rad": _rounded_optional(
                self.map_match_heading_error_rad, 6
            ),
            "confidence": _rounded(self.confidence, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ScenarioMapAnnotation:
    scenario_id: str
    scenario_index: int | None
    source_file: str | None
    current_time_index: int
    junctions: tuple[JunctionAnnotation, ...]
    ego_frames: tuple[EgoFrameAnnotation, ...]

    def to_dict(self) -> dict[str, Any]:
        region_counts = Counter(frame.region_type.value for frame in self.ego_frames)
        road_environment_counts = Counter(
            frame.road_environment.value for frame in self.ego_frames
        )
        road_environment_subtype_counts = Counter(
            frame.road_environment_subtype.value
            for frame in self.ego_frames
            if frame.road_environment_subtype is not None
        )
        return {
            "schema_version": MAP_ANNOTATION_SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "scenario_index": self.scenario_index,
            "source_file": self.source_file,
            "current_time_index": self.current_time_index,
            "statistics": {
                "junction_count": len(self.junctions),
                "signalized_junction_count": sum(
                    junction.kind == JunctionKind.SIGNALIZED
                    for junction in self.junctions
                ),
                "stop_controlled_junction_count": sum(
                    junction.kind == JunctionKind.STOP_CONTROLLED
                    for junction in self.junctions
                ),
                "roundabout_junction_count": sum(
                    junction.kind == JunctionKind.ROUNDABOUT
                    for junction in self.junctions
                ),
                "geometric_junction_count": sum(
                    junction.kind == JunctionKind.GEOMETRIC
                    for junction in self.junctions
                ),
                "ego_frame_count": len(self.ego_frames),
                "ego_valid_frame_count": sum(frame.valid for frame in self.ego_frames),
                "region_counts": dict(sorted(region_counts.items())),
                "road_environment_counts": dict(
                    sorted(road_environment_counts.items())
                ),
                "road_environment_subtype_counts": dict(
                    sorted(road_environment_subtype_counts.items())
                ),
            },
            "junctions": [junction.to_dict() for junction in self.junctions],
            "ego_frames": [frame.to_dict() for frame in self.ego_frames],
        }


@dataclass(frozen=True)
class _LaneProjection:
    lane_id: int
    point_index: int
    lane_s_m: float
    distance_m: float
    heading_rad: float


@dataclass(frozen=True)
class _Gate:
    lane_id: int
    incoming: bool
    point_xy: tuple[float, float]
    angle_rad: float


@dataclass(frozen=True)
class _CircularLaneComponent:
    lane_ids: frozenset[int]
    center_xy: tuple[float, float]
    median_radius_m: float


def _rounded(value: float, digits: int) -> float:
    return round(float(value), digits)


def _rounded_optional(value: float | None, digits: int) -> float | None:
    return None if value is None else _rounded(value, digits)


def _rounded_xy(point: tuple[float, float]) -> list[float]:
    return [_rounded(point[0], 3), _rounded(point[1], 3)]


def _wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _polyline_xy(polyline) -> np.ndarray:
    return np.asarray([(point.x, point.y) for point in polyline], dtype=np.float64)


def _cumulative_distance(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points), dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def _circular_mean(angles: list[float]) -> float:
    x = sum(math.cos(angle) for angle in angles)
    y = sum(math.sin(angle) for angle in angles)
    return math.atan2(y, x) % (2.0 * math.pi)


class _LaneGeometryIndex:
    def __init__(self, lanes, config: MapAnnotationConfig) -> None:
        self.lanes = lanes
        self.config = config
        self.points_by_lane: dict[int, np.ndarray] = {}
        self.s_by_lane: dict[int, np.ndarray] = {}
        self.segment_vectors_by_lane: dict[int, np.ndarray] = {}
        self.segment_length_sq_by_lane: dict[int, np.ndarray] = {}
        self.segment_lengths_by_lane: dict[int, np.ndarray] = {}
        self.segment_headings_by_lane: dict[int, np.ndarray] = {}
        sample_points = []
        sample_lane_ids = []
        for lane_id, lane_center in lanes.items():
            points = _polyline_xy(lane_center.lane.polyline)
            if len(points) < 2:
                continue
            segment_vectors = points[1:] - points[:-1]
            segment_length_sq = np.einsum(
                "ij,ij->i", segment_vectors, segment_vectors
            )
            self.points_by_lane[lane_id] = points
            self.s_by_lane[lane_id] = _cumulative_distance(points)
            self.segment_vectors_by_lane[lane_id] = segment_vectors
            self.segment_length_sq_by_lane[lane_id] = segment_length_sq
            self.segment_lengths_by_lane[lane_id] = np.sqrt(segment_length_sq)
            self.segment_headings_by_lane[lane_id] = np.arctan2(
                segment_vectors[:, 1], segment_vectors[:, 0]
            )
            sample_points.append(points)
            sample_lane_ids.extend([lane_id] * len(points))
        if not sample_points:
            raise ValueError("Scenario does not contain usable vehicle lane polylines")
        self.sample_points = np.concatenate(sample_points, axis=0)
        self.sample_lane_ids = np.asarray(sample_lane_ids, dtype=np.int64)
        self.tree = cKDTree(self.sample_points)
        self._same_direction_lane_ids_cache: dict[
            tuple[int, int], tuple[int, ...]
        ] = {}

    def _project(self, lane_id: int, position: np.ndarray) -> _LaneProjection:
        points = self.points_by_lane[lane_id]
        segment_vectors = self.segment_vectors_by_lane[lane_id]
        segment_length_sq = self.segment_length_sq_by_lane[lane_id]
        valid = segment_length_sq > 1e-12
        fractions = np.zeros(len(segment_vectors), dtype=np.float64)
        fractions[valid] = np.clip(
            np.einsum("ij,ij->i", position - points[:-1], segment_vectors)[valid]
            / segment_length_sq[valid],
            0.0,
            1.0,
        )
        projections = points[:-1] + fractions[:, None] * segment_vectors
        distances = np.linalg.norm(projections - position, axis=1)
        segment_index = int(np.argmin(distances))
        segment_length = self.segment_lengths_by_lane[lane_id][segment_index]
        heading = self.segment_headings_by_lane[lane_id][segment_index]
        lane_s = float(
            self.s_by_lane[lane_id][segment_index]
            + fractions[segment_index] * segment_length
        )
        point_index = segment_index + int(fractions[segment_index] >= 0.5)
        return _LaneProjection(
            lane_id=lane_id,
            point_index=point_index,
            lane_s_m=lane_s,
            distance_m=float(distances[segment_index]),
            heading_rad=heading,
        )

    def _topologically_related(self, previous_lane_id: int, lane_id: int) -> bool:
        if previous_lane_id == lane_id:
            return True
        previous = self.lanes.get(previous_lane_id)
        current = self.lanes.get(lane_id)
        if previous is None or current is None:
            return False
        related = {
            *previous.lane.entry_lanes,
            *previous.lane.exit_lanes,
            *(neighbor.feature_id for neighbor in previous.lane.left_neighbors),
            *(neighbor.feature_id for neighbor in previous.lane.right_neighbors),
            *previous.lane.diverge_lanes,
            *previous.lane.merge_lanes,
        }
        if lane_id in related:
            return True
        return previous_lane_id in {
            *current.lane.entry_lanes,
            *current.lane.exit_lanes,
        }

    def _longitudinally_related(self, left_lane_id: int, right_lane_id: int) -> bool:
        left = self.lanes[left_lane_id].lane
        right = self.lanes[right_lane_id].lane
        return bool(
            right_lane_id in left.entry_lanes
            or right_lane_id in left.exit_lanes
            or left_lane_id in right.entry_lanes
            or left_lane_id in right.exit_lanes
        )

    def _branch_related(self, left_lane_id: int, right_lane_id: int) -> bool:
        left = self.lanes[left_lane_id].lane
        right = self.lanes[right_lane_id].lane
        return bool(
            right_lane_id
            in {
                *getattr(left, "merge_lanes", ()),
                *getattr(left, "diverge_lanes", ()),
            }
            or left_lane_id
            in {
                *getattr(right, "merge_lanes", ()),
                *getattr(right, "diverge_lanes", ()),
            }
        )

    def _heading_at_point(self, lane_id: int, point_index: int) -> float:
        headings = self.segment_headings_by_lane[lane_id]
        segment_index = min(max(0, point_index), len(headings) - 1)
        return float(headings[segment_index])

    def _longitudinal_lane_candidates(
        self,
        lane_id: int,
        *,
        forward: bool,
    ) -> tuple[int, ...]:
        candidates = []
        visited = {lane_id}
        queue = [(lane_id, 0)]
        while queue:
            current_id, hop_count = queue.pop(0)
            if hop_count >= _MAX_LANE_NEIGHBOR_TOPOLOGY_HOPS:
                continue
            current_lane = self.lanes[current_id].lane
            linked_lane_ids = (
                current_lane.exit_lanes
                if forward
                else current_lane.entry_lanes
            )
            for linked_lane_id in linked_lane_ids:
                linked_lane_id = int(linked_lane_id)
                if (
                    linked_lane_id in visited
                    or linked_lane_id not in self.points_by_lane
                ):
                    continue
                visited.add(linked_lane_id)
                candidates.append(linked_lane_id)
                queue.append((linked_lane_id, hop_count + 1))
        return tuple(candidates)

    def _unambiguous_longitudinal_chain(
        self,
        lane_id: int,
        *,
        forward: bool,
    ) -> tuple[int, ...]:
        """Follow one-to-one lane continuations without crossing a split or merge."""
        chain = []
        visited = {lane_id}
        current_id = lane_id
        for _ in range(_MAX_LANE_NEIGHBOR_TOPOLOGY_HOPS):
            current_lane = self.lanes[current_id].lane
            linked_lane_ids = [
                int(linked_lane_id)
                for linked_lane_id in (
                    current_lane.exit_lanes
                    if forward
                    else current_lane.entry_lanes
                )
                if int(linked_lane_id) in self.points_by_lane
            ]
            if len(linked_lane_ids) != 1:
                break
            linked_lane_id = linked_lane_ids[0]
            if linked_lane_id in visited:
                break
            linked_lane = self.lanes[linked_lane_id].lane
            reverse_lane_ids = [
                int(reverse_lane_id)
                for reverse_lane_id in (
                    linked_lane.entry_lanes
                    if forward
                    else linked_lane.exit_lanes
                )
                if int(reverse_lane_id) in self.points_by_lane
            ]
            if reverse_lane_ids != [current_id]:
                break
            chain.append(linked_lane_id)
            visited.add(linked_lane_id)
            current_id = linked_lane_id
        return tuple(chain)

    def _deduplicate_cross_section_lane_ids(
        self,
        lane_ids: set[int],
        reference_lane_id: int,
        reference_point_index: int,
    ) -> tuple[int, ...]:
        if len(lane_ids) <= 1:
            return tuple(sorted(lane_ids))

        reference_points = self.points_by_lane[reference_lane_id]
        reference_point_index = min(
            max(0, reference_point_index),
            len(reference_points) - 1,
        )
        reference_point = reference_points[reference_point_index]
        reference_heading = self._heading_at_point(
            reference_lane_id,
            reference_point_index,
        )
        tangent = np.asarray(
            [math.cos(reference_heading), math.sin(reference_heading)]
        )
        normal = np.asarray([-tangent[1], tangent[0]])
        candidates = []
        for candidate_lane_id in lane_ids:
            projection = self._project(candidate_lane_id, reference_point)
            candidate_points = self.points_by_lane[candidate_lane_id]
            candidate_index = min(
                max(0, projection.point_index),
                len(candidate_points) - 1,
            )
            delta = candidate_points[candidate_index] - reference_point
            lateral_offset = float(np.dot(delta, normal))
            longitudinal_error = abs(float(np.dot(delta, tangent)))
            heading_error = abs(
                float(_wrap_angle(projection.heading_rad - reference_heading))
            )
            lane_length = float(self.s_by_lane[candidate_lane_id][-1])
            representative_rank = (
                candidate_lane_id != reference_lane_id,
                longitudinal_error,
                heading_error,
                -lane_length,
                candidate_lane_id,
            )
            candidates.append(
                (
                    lateral_offset,
                    representative_rank,
                    candidate_lane_id,
                )
            )

        tolerance_m = min(1.0, 0.5 * self.config.lane_half_width_m)
        branch_transition_tolerance_m = max(
            tolerance_m,
            min(2.0, self.config.lane_half_width_m),
        )
        candidates.sort(key=lambda item: item[0])
        clusters: list[list[tuple[float, tuple, int]]] = []
        for candidate in candidates:
            if not clusters:
                clusters.append([candidate])
                continue
            cluster_center = float(
                np.mean([item[0] for item in clusters[-1]])
            )
            overlaps_branch_transition = any(
                abs(candidate[0] - cluster_item[0])
                <= branch_transition_tolerance_m
                and self._branch_related(candidate[2], cluster_item[2])
                for cluster_item in clusters[-1]
            )
            if (
                abs(candidate[0] - cluster_center) <= tolerance_m
                or overlaps_branch_transition
            ):
                clusters[-1].append(candidate)
            else:
                clusters.append([candidate])

        representatives = [
            min(cluster, key=lambda item: item[1])[2]
            for cluster in clusters
        ]
        return tuple(sorted(representatives))

    def match(self, state, previous_lane_id: int | None) -> MapMatch:
        position = np.asarray([state.center_x, state.center_y], dtype=np.float64)
        k = min(64, len(self.sample_points))
        _, sample_indices = self.tree.query(position, k=k)
        sample_indices = np.atleast_1d(sample_indices)
        candidate_lane_ids = []
        for sample_index in sample_indices:
            lane_id = int(self.sample_lane_ids[sample_index])
            if lane_id not in candidate_lane_ids:
                candidate_lane_ids.append(lane_id)
            if len(candidate_lane_ids) >= 16:
                break
        if previous_lane_id in self.points_by_lane:
            previous = self.lanes[previous_lane_id]
            candidate_lane_ids.extend(
                lane_id
                for lane_id in [
                    previous_lane_id,
                    *previous.lane.entry_lanes,
                    *previous.lane.exit_lanes,
                    *(n.feature_id for n in previous.lane.left_neighbors),
                    *(n.feature_id for n in previous.lane.right_neighbors),
                ]
                if lane_id in self.points_by_lane
            )
            candidate_lane_ids = list(dict.fromkeys(candidate_lane_ids))

        max_heading_error = math.radians(
            self.config.max_map_match_heading_error_deg
        )
        candidates = []
        for lane_id in candidate_lane_ids:
            projection = self._project(lane_id, position)
            heading_error = abs(float(_wrap_angle(projection.heading_rad - state.heading)))
            related = (
                previous_lane_id is None
                or self._topologically_related(previous_lane_id, lane_id)
            )
            continuity_penalty = 0.0 if related else self.config.continuity_penalty_m
            score = (
                projection.distance_m
                + self.config.heading_weight_m_per_rad * heading_error
                + continuity_penalty
            )
            candidates.append((score, projection, heading_error, related))

        compatible = [
            candidate
            for candidate in candidates
            if candidate[1].distance_m <= self.config.max_map_match_distance_m
            and candidate[2] <= max_heading_error
        ]
        score, projection, heading_error, related = min(
            compatible or candidates, key=lambda candidate: candidate[0]
        )
        confident = bool(compatible)
        if confident:
            distance_confidence = max(
                0.0,
                1.0
                - projection.distance_m / self.config.max_map_match_distance_m,
            )
            heading_confidence = max(0.0, 1.0 - heading_error / max_heading_error)
            continuity_confidence = 1.0 if related else 0.0
            confidence = (
                0.55 * distance_confidence
                + 0.35 * heading_confidence
                + 0.10 * continuity_confidence
            )
        else:
            confidence = 0.0
        return MapMatch(
            lane_id=projection.lane_id,
            point_index=projection.point_index,
            lane_s_m=projection.lane_s_m,
            distance_m=projection.distance_m,
            heading_error_rad=heading_error,
            confidence=confidence,
            confident=confident,
        )

    def same_direction_lane_ids(
        self,
        lane_id: int,
        point_index: int,
    ) -> tuple[int, ...]:
        cache_key = (lane_id, point_index)
        if cache_key in self._same_direction_lane_ids_cache:
            return self._same_direction_lane_ids_cache[cache_key]

        visited = {lane_id}
        queue = [(lane_id, point_index)]
        while queue:
            current_id, current_index = queue.pop()
            current_lane = self.lanes[current_id].lane
            current_points = self.points_by_lane[current_id]
            current_index = min(max(0, current_index), len(current_points) - 1)
            current_heading = self._heading_at_point(current_id, current_index)
            tangent = np.asarray(
                [math.cos(current_heading), math.sin(current_heading)]
            )
            current_point = current_points[current_index]

            for side_neighbors in (
                current_lane.left_neighbors,
                current_lane.right_neighbors,
            ):
                candidates = []
                for neighbor in side_neighbors:
                    neighbor_id = neighbor.feature_id
                    if neighbor_id in visited or neighbor_id not in self.points_by_lane:
                        continue
                    self_start = min(
                        neighbor.self_start_index, neighbor.self_end_index
                    )
                    self_end = max(neighbor.self_start_index, neighbor.self_end_index)
                    if self_start <= current_index <= self_end:
                        fraction = (current_index - self_start) / max(
                            1, self_end - self_start
                        )
                        neighbor_index = round(
                            neighbor.neighbor_start_index
                            + fraction
                            * (
                                neighbor.neighbor_end_index
                                - neighbor.neighbor_start_index
                            )
                        )
                    else:
                        boundary_index = (
                            self_start
                            if current_index < self_start
                            else self_end
                        )
                        extension_distance = abs(
                            float(
                                self.s_by_lane[current_id][current_index]
                                - self.s_by_lane[current_id][boundary_index]
                            )
                        )
                        candidate_lane_ids = list(
                            self._longitudinal_lane_candidates(
                                neighbor_id,
                                forward=current_index > self_end,
                            )
                        )
                        if (
                            extension_distance
                            <= self.config.lane_neighbor_extension_m
                        ):
                            candidate_lane_ids.insert(0, neighbor_id)

                        for candidate_lane_id in candidate_lane_ids:
                            if (
                                candidate_lane_id in visited
                                or any(
                                    self._longitudinally_related(
                                        candidate_lane_id,
                                        visited_lane_id,
                                    )
                                    for visited_lane_id in visited
                                )
                            ):
                                continue
                            projection = self._project(
                                candidate_lane_id, current_point
                            )
                            if (
                                projection.distance_m
                                > self.config.max_lane_neighbor_distance_m
                            ):
                                continue
                            candidate_length = self.s_by_lane[
                                candidate_lane_id
                            ][-1]
                            if (
                                projection.lane_s_m <= 1e-6
                                or candidate_length - projection.lane_s_m
                                <= 1e-6
                            ):
                                continue
                            candidate_index = projection.point_index
                            candidate_points = self.points_by_lane[
                                candidate_lane_id
                            ]
                            candidate_index = min(
                                max(0, candidate_index),
                                len(candidate_points) - 1,
                            )
                            candidate_heading = self._heading_at_point(
                                candidate_lane_id, candidate_index
                            )
                            heading_error = abs(
                                float(
                                    _wrap_angle(
                                        candidate_heading - current_heading
                                    )
                                )
                            )
                            if heading_error > math.radians(45.0):
                                continue
                            longitudinal_error = abs(
                                float(
                                    np.dot(
                                        candidate_points[candidate_index]
                                        - current_point,
                                        tangent,
                                    )
                                )
                            )
                            candidates.append(
                                (
                                    longitudinal_error,
                                    candidate_lane_id,
                                    candidate_index,
                                )
                            )
                        continue

                    neighbor_points = self.points_by_lane[neighbor_id]
                    neighbor_index = min(
                        max(0, neighbor_index), len(neighbor_points) - 1
                    )
                    neighbor_heading = self._heading_at_point(
                        neighbor_id, neighbor_index
                    )
                    heading_error = abs(
                        float(
                            _wrap_angle(
                                neighbor_heading - current_heading
                            )
                        )
                    )
                    if heading_error > math.radians(45.0):
                        continue
                    longitudinal_error = abs(
                        float(
                            np.dot(
                                neighbor_points[neighbor_index] - current_point,
                                tangent,
                            )
                        )
                    )
                    candidates.append(
                        (longitudinal_error, neighbor_id, neighbor_index)
                    )
                if candidates:
                    _, neighbor_id, neighbor_index = min(candidates)
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, neighbor_index))

            # Some dedicated turn lanes are encoded as diverging or merging
            # peers after neighbor cleanup, even while both lanes still cross
            # the ego's current longitudinal station.
            branch_lane_ids = {
                *getattr(current_lane, "diverge_lanes", ()),
                *getattr(current_lane, "merge_lanes", ()),
            }
            branch_candidates = []
            for candidate_lane_id in sorted(branch_lane_ids):
                if (
                    candidate_lane_id in visited
                    or candidate_lane_id not in self.points_by_lane
                    or any(
                        self._longitudinally_related(
                            candidate_lane_id,
                            visited_lane_id,
                        )
                        for visited_lane_id in visited
                    )
                ):
                    continue
                projection = self._project(candidate_lane_id, current_point)
                if (
                    projection.distance_m
                    > self.config.max_lane_neighbor_distance_m
                ):
                    continue
                candidate_length = self.s_by_lane[candidate_lane_id][-1]
                if (
                    projection.lane_s_m <= 1e-6
                    or candidate_length - projection.lane_s_m <= 1e-6
                ):
                    continue
                candidate_index = projection.point_index
                candidate_points = self.points_by_lane[candidate_lane_id]
                candidate_index = min(
                    max(0, candidate_index),
                    len(candidate_points) - 1,
                )
                candidate_heading = self._heading_at_point(
                    candidate_lane_id,
                    candidate_index,
                )
                heading_error = abs(
                    float(_wrap_angle(candidate_heading - current_heading))
                )
                if heading_error > math.radians(45.0):
                    continue
                longitudinal_error = abs(
                    float(
                        np.dot(
                            candidate_points[candidate_index] - current_point,
                            tangent,
                        )
                    )
                )
                branch_candidates.append(
                    (
                        longitudinal_error,
                        candidate_lane_id,
                        candidate_index,
                    )
                )

            for _, candidate_lane_id, candidate_index in sorted(
                branch_candidates
            ):
                if candidate_lane_id in visited:
                    continue
                visited.add(candidate_lane_id)
                queue.append((candidate_lane_id, candidate_index))

            # Neighbor metadata can stop at a lane-segment boundary. Inherit
            # adjacency from one-to-one predecessors/successors, then advance
            # the neighboring lane chain back to the current cross-section.
            inherited_candidates = {}
            for anchor_forward in (False, True):
                candidate_forward = not anchor_forward
                anchor_lane_ids = self._unambiguous_longitudinal_chain(
                    current_id,
                    forward=anchor_forward,
                )
                for anchor_lane_id in anchor_lane_ids:
                    anchor_lane = self.lanes[anchor_lane_id].lane
                    for side_neighbors in (
                        anchor_lane.left_neighbors,
                        anchor_lane.right_neighbors,
                    ):
                        candidates = []
                        for neighbor in side_neighbors:
                            candidate_lane_ids = (
                                int(neighbor.feature_id),
                                *self._longitudinal_lane_candidates(
                                    int(neighbor.feature_id),
                                    forward=candidate_forward,
                                ),
                            )
                            for candidate_lane_id in candidate_lane_ids:
                                if (
                                    candidate_lane_id in visited
                                    or candidate_lane_id
                                    not in self.points_by_lane
                                    or any(
                                        self._longitudinally_related(
                                            candidate_lane_id,
                                            visited_lane_id,
                                        )
                                        for visited_lane_id in visited
                                    )
                                ):
                                    continue
                                projection = self._project(
                                    candidate_lane_id,
                                    current_point,
                                )
                                if (
                                    projection.distance_m
                                    > 2.0
                                    * self.config.max_lane_neighbor_distance_m
                                ):
                                    continue
                                candidate_length = self.s_by_lane[
                                    candidate_lane_id
                                ][-1]
                                if (
                                    projection.lane_s_m <= 1e-6
                                    or candidate_length
                                    - projection.lane_s_m
                                    <= 1e-6
                                ):
                                    continue
                                candidate_index = projection.point_index
                                candidate_points = self.points_by_lane[
                                    candidate_lane_id
                                ]
                                candidate_index = min(
                                    max(0, candidate_index),
                                    len(candidate_points) - 1,
                                )
                                candidate_heading = self._heading_at_point(
                                    candidate_lane_id,
                                    candidate_index,
                                )
                                heading_error = abs(
                                    float(
                                        _wrap_angle(
                                            candidate_heading
                                            - current_heading
                                        )
                                    )
                                )
                                if heading_error > math.radians(45.0):
                                    continue
                                longitudinal_error = abs(
                                    float(
                                        np.dot(
                                            candidate_points[candidate_index]
                                            - current_point,
                                            tangent,
                                        )
                                    )
                                )
                                if (
                                    longitudinal_error
                                    > self.config.lane_neighbor_extension_m
                                ):
                                    continue
                                candidates.append(
                                    (
                                        projection.distance_m,
                                        longitudinal_error,
                                        candidate_lane_id,
                                        candidate_index,
                                    )
                                )
                        if candidates:
                            candidate = min(candidates)
                            previous = inherited_candidates.get(candidate[2])
                            if previous is None or candidate < previous:
                                inherited_candidates[candidate[2]] = candidate

            for (
                _,
                _,
                candidate_lane_id,
                candidate_index,
            ) in sorted(inherited_candidates.values()):
                if candidate_lane_id in visited:
                    continue
                visited.add(candidate_lane_id)
                queue.append((candidate_lane_id, candidate_index))

        lane_ids = self._deduplicate_cross_section_lane_ids(
            visited,
            lane_id,
            point_index,
        )
        self._same_direction_lane_ids_cache[cache_key] = lane_ids
        return lane_ids

    def same_direction_lane_count(self, lane_id: int, point_index: int) -> int:
        return len(self.same_direction_lane_ids(lane_id, point_index))


@dataclass(frozen=True)
class _RoadEnvironmentResult:
    environment: RoadEnvironment
    confidence: float
    reason: str
    matched_lane_type: str
    matched_lane_speed_limit_mph: float
    subtype: RoadEnvironmentSubtype | None = None
    subtype_reason: str | None = None


class _RoadEnvironmentClassifier:
    """Derive parking context while preserving native WOMD freeway labels."""

    def __init__(
        self,
        scenario,
        lanes,
        lane_index: _LaneGeometryIndex,
        config: MapAnnotationConfig,
    ) -> None:
        self.scenario = scenario
        self.lanes = lanes
        self.lane_index = lane_index
        self.config = config
        driveway_polygons = []
        road_edge_segment_starts = []
        road_edge_segment_ends = []
        for feature in scenario.map_features:
            feature_type = feature.WhichOneof("feature_data")
            if feature_type == "driveway":
                polygon = np.asarray(
                    [
                        (point.x, point.y)
                        for point in feature.driveway.polygon
                    ],
                    dtype=np.float64,
                )
                if len(polygon) >= 3 and np.all(np.isfinite(polygon)):
                    driveway_polygons.append(polygon)
            elif (
                feature_type == "road_edge"
                and int(feature.road_edge.type) == 1
            ):
                points = _polyline_xy(feature.road_edge.polyline)
                if len(points) < 2 or not np.all(np.isfinite(points)):
                    continue
                vectors = np.diff(points, axis=0)
                valid = np.einsum("ij,ij->i", vectors, vectors) > 1e-12
                road_edge_segment_starts.append(points[:-1][valid])
                road_edge_segment_ends.append(points[1:][valid])
        self._driveway_polygons = tuple(driveway_polygons)
        if road_edge_segment_starts:
            self._road_edge_segment_starts = np.concatenate(
                road_edge_segment_starts,
                axis=0,
            )
            self._road_edge_segment_ends = np.concatenate(
                road_edge_segment_ends,
                axis=0,
            )
        else:
            self._road_edge_segment_starts = np.empty(
                (0, 2),
                dtype=np.float64,
            )
            self._road_edge_segment_ends = np.empty(
                (0, 2),
                dtype=np.float64,
            )
        self._road_edge_segment_vectors = (
            self._road_edge_segment_ends - self._road_edge_segment_starts
        )
        self._road_edge_segment_length_sq = np.einsum(
            "ij,ij->i",
            self._road_edge_segment_vectors,
            self._road_edge_segment_vectors,
        )

        lane_graph = {
            lane_id: {
                int(linked_lane_id)
                for linked_lane_id in (
                    *lane_center.lane.entry_lanes,
                    *lane_center.lane.exit_lanes,
                )
                if int(linked_lane_id) in lanes
            }
            for lane_id, lane_center in lanes.items()
        }
        for lane_id, linked_lane_ids in tuple(lane_graph.items()):
            for linked_lane_id in linked_lane_ids:
                lane_graph[linked_lane_id].add(lane_id)
        self._lane_graph_degrees = {
            lane_id: len(linked_lane_ids)
            for lane_id, linked_lane_ids in lane_graph.items()
        }
        self._freeway_cross_section_count_cache: dict[int, int] = {}
        self._freeway_subtype_cache: dict[
            tuple[int, int], tuple[RoadEnvironmentSubtype, str]
        ] = {}

    def _freeway_cross_section_lane_count(
        self,
        lane_id: int,
        point_index: int | None = None,
    ) -> int:
        if point_index is not None:
            lane_ids = self.lane_index.same_direction_lane_ids(
                lane_id,
                point_index,
            )
            return sum(
                int(self.lanes[candidate_id].lane.type) == 1
                for candidate_id in lane_ids
            )

        cached = self._freeway_cross_section_count_cache.get(lane_id)
        if cached is not None:
            return cached
        points = self.lane_index.points_by_lane[lane_id]
        sample_indices = {0, len(points) // 2, len(points) - 1}
        count = max(
            self._freeway_cross_section_lane_count(lane_id, sample_index)
            for sample_index in sample_indices
        )
        self._freeway_cross_section_count_cache[lane_id] = count
        return count

    def _freeway_linked_lane_ids(self, lane_id: int) -> tuple[int, ...]:
        lane = self.lanes[lane_id].lane
        return tuple(
            dict.fromkeys(
                int(linked_lane_id)
                for linked_lane_id in (
                    *lane.entry_lanes,
                    *lane.exit_lanes,
                )
                if linked_lane_id in self.lanes
                and int(self.lanes[linked_lane_id].lane.type) == 1
            )
        )

    def _is_freeway_branch_transition(self, lane_id: int) -> bool:
        lane = self.lanes[lane_id].lane
        freeway_entry_lanes = [
            linked_lane_id
            for linked_lane_id in lane.entry_lanes
            if linked_lane_id in self.lanes
            and int(self.lanes[linked_lane_id].lane.type) == 1
        ]
        freeway_exit_lanes = [
            linked_lane_id
            for linked_lane_id in lane.exit_lanes
            if linked_lane_id in self.lanes
            and int(self.lanes[linked_lane_id].lane.type) == 1
        ]
        freeway_branch_lanes = [
            linked_lane_id
            for linked_lane_id in (
                *lane.merge_lanes,
                *lane.diverge_lanes,
            )
            if linked_lane_id in self.lanes
            and int(self.lanes[linked_lane_id].lane.type) == 1
        ]
        return bool(
            len(freeway_entry_lanes) > 1
            or len(freeway_exit_lanes) > 1
            or freeway_branch_lanes
        )

    def _classify_freeway_subtype(
        self,
        match: MapMatch,
    ) -> tuple[RoadEnvironmentSubtype, str]:
        current_lane_count = self._freeway_cross_section_lane_count(
            match.lane_id,
            match.point_index,
        )
        cache_key = (match.lane_id, current_lane_count)
        cached = self._freeway_subtype_cache.get(cache_key)
        if cached is not None:
            return cached

        result = (
            RoadEnvironmentSubtype.FREEWAY_MAINLINE,
            "freeway_mainline_no_narrow_branch_transition",
        )
        if current_lane_count <= self.config.freeway_ramp_max_lane_count:
            queue = deque([(match.lane_id, 0, False)])
            visited = {(match.lane_id, False)}
            required_lane_count = (
                current_lane_count
                + self.config.freeway_ramp_min_lane_count_gain
            )
            while queue:
                lane_id, hop_count, branch_seen = queue.popleft()
                branch_seen = (
                    branch_seen
                    or self._is_freeway_branch_transition(lane_id)
                )
                if (
                    hop_count > 0
                    and branch_seen
                    and self._freeway_cross_section_lane_count(lane_id)
                    >= required_lane_count
                ):
                    result = (
                        RoadEnvironmentSubtype.FREEWAY_RAMP,
                        "freeway_ramp_narrow_branch_to_wider_mainline",
                    )
                    break
                if hop_count >= self.config.freeway_ramp_topology_hops:
                    continue
                for linked_lane_id in self._freeway_linked_lane_ids(lane_id):
                    state = (linked_lane_id, branch_seen)
                    if state in visited:
                        continue
                    visited.add(state)
                    queue.append(
                        (linked_lane_id, hop_count + 1, branch_seen)
                    )

        self._freeway_subtype_cache[cache_key] = result
        return result

    @staticmethod
    def _distance_to_closed_polyline(
        point: np.ndarray,
        polygon: np.ndarray,
    ) -> float:
        starts = polygon
        ends = np.roll(polygon, -1, axis=0)
        vectors = ends - starts
        length_sq = np.einsum("ij,ij->i", vectors, vectors)
        fractions = np.zeros(len(vectors), dtype=np.float64)
        valid = length_sq > 1e-12
        fractions[valid] = np.clip(
            np.einsum("ij,ij->i", point - starts, vectors)[valid]
            / length_sq[valid],
            0.0,
            1.0,
        )
        projections = starts + fractions[:, None] * vectors
        return float(np.min(np.linalg.norm(projections - point, axis=1)))

    def _nearest_driveway_distance_m(self, position: np.ndarray) -> float:
        if not self._driveway_polygons:
            return math.inf
        return min(
            self._distance_to_closed_polyline(position, polygon)
            for polygon in self._driveway_polygons
        )

    def _local_lane_ids(
        self,
        position: np.ndarray,
        radius_m: float,
    ) -> tuple[int, ...]:
        sample_indices = self.lane_index.tree.query_ball_point(
            position,
            radius_m,
        )
        if not sample_indices:
            return ()
        lane_ids = np.unique(
            self.lane_index.sample_lane_ids[
                np.asarray(sample_indices, dtype=np.int64)
            ]
        )
        return tuple(int(lane_id) for lane_id in lane_ids)

    def _local_low_speed_lane_ratio(
        self,
        local_lane_ids: tuple[int, ...],
    ) -> float:
        known_speed_limits = [
            float(self.lanes[lane_id].lane.speed_limit_mph)
            for lane_id in local_lane_ids
            if float(self.lanes[lane_id].lane.speed_limit_mph) > 0.0
        ]
        if not known_speed_limits:
            return 0.0
        return float(
            np.mean(
                np.asarray(known_speed_limits)
                <= self.config.parking_max_speed_limit_mph
            )
        )

    def _nearest_road_edge_distance_m(self, position: np.ndarray) -> float:
        if len(self._road_edge_segment_starts) == 0:
            return math.inf
        vectors = self._road_edge_segment_vectors
        fractions = np.clip(
            np.einsum(
                "ij,ij->i",
                position - self._road_edge_segment_starts,
                vectors,
            )
            / self._road_edge_segment_length_sq,
            0.0,
            1.0,
        )
        projections = (
            self._road_edge_segment_starts + fractions[:, None] * vectors
        )
        return float(np.min(np.linalg.norm(projections - position, axis=1)))

    @staticmethod
    def _cross_2d(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]

    def _road_edge_ray_distance_m(
        self,
        position: np.ndarray,
        heading_rad: float,
        max_distance_m: float,
    ) -> float:
        if len(self._road_edge_segment_starts) == 0:
            return math.inf
        direction = np.asarray(
            [math.cos(heading_rad), math.sin(heading_rad)],
            dtype=np.float64,
        )
        directions = np.broadcast_to(
            direction,
            self._road_edge_segment_vectors.shape,
        )
        offsets = self._road_edge_segment_starts - position
        denominators = self._cross_2d(
            directions,
            self._road_edge_segment_vectors,
        )
        non_parallel = np.abs(denominators) > 1e-9
        ray_distances = np.full(len(denominators), math.inf)
        segment_fractions = np.full(len(denominators), math.inf)
        ray_distances[non_parallel] = (
            self._cross_2d(
                offsets[non_parallel],
                self._road_edge_segment_vectors[non_parallel],
            )
            / denominators[non_parallel]
        )
        segment_fractions[non_parallel] = (
            self._cross_2d(
                offsets[non_parallel],
                directions[non_parallel],
            )
            / denominators[non_parallel]
        )
        intersects = (
            (ray_distances > 0.25)
            & (ray_distances <= max_distance_m)
            & (segment_fractions >= -1e-6)
            & (segment_fractions <= 1.0 + 1e-6)
        )
        if not np.any(intersects):
            return math.inf
        return float(np.min(ray_distances[intersects]))

    def _stationary_vehicle_evidence(
        self,
        frame_index: int,
        position: np.ndarray,
    ) -> tuple[int, int, float]:
        stationary_positions = []
        for track_index, track in enumerate(self.scenario.tracks):
            if (
                track_index == self.scenario.sdc_track_index
                or int(track.object_type) != 1
                or frame_index >= len(track.states)
            ):
                continue
            state = track.states[frame_index]
            if not state.valid:
                continue
            vehicle_position = np.asarray(
                [state.center_x, state.center_y],
                dtype=np.float64,
            )
            if (
                not np.all(np.isfinite(vehicle_position))
                or np.linalg.norm(vehicle_position - position)
                > self.config.parking_context_radius_m
            ):
                continue
            speed_mps = math.hypot(state.velocity_x, state.velocity_y)
            if (
                not math.isfinite(speed_mps)
                or speed_mps > self.config.parking_stationary_speed_mps
            ):
                continue
            stationary_positions.append(vehicle_position)

        stationary_count = len(stationary_positions)
        if stationary_count == 0:
            return 0, 0, 0.0
        lane_distances, _ = self.lane_index.tree.query(
            np.asarray(stationary_positions),
            k=1,
        )
        off_lane_count = int(
            np.count_nonzero(
                np.asarray(lane_distances)
                > self.config.parking_off_lane_distance_m
            )
        )
        return (
            stationary_count,
            off_lane_count,
            off_lane_count / stationary_count,
        )

    def classify(
        self,
        match: MapMatch,
        state,
        frame_index: int,
        region_type: RegionType,
        junction_kind: JunctionKind | None,
        lane_count: int | None,
    ) -> _RoadEnvironmentResult:
        lane = self.lanes[match.lane_id].lane
        lane_type = int(lane.type)
        lane_type_name = _WOMD_LANE_TYPE_NAMES.get(
            lane_type,
            f"TYPE_{lane_type}",
        )
        speed_limit_mph = float(lane.speed_limit_mph)

        if lane_type == 1:
            freeway_subtype, freeway_subtype_reason = (
                self._classify_freeway_subtype(match)
            )
            return _RoadEnvironmentResult(
                environment=RoadEnvironment.FREEWAY,
                confidence=match.confidence,
                reason="womd_lane_type_freeway",
                matched_lane_type=lane_type_name,
                matched_lane_speed_limit_mph=speed_limit_mph,
                subtype=freeway_subtype,
                subtype_reason=freeway_subtype_reason,
            )
        if lane_type != 2:
            return _RoadEnvironmentResult(
                environment=RoadEnvironment.UNKNOWN,
                confidence=0.0,
                reason="unsupported_womd_lane_type",
                matched_lane_type=lane_type_name,
                matched_lane_speed_limit_mph=speed_limit_mph,
            )

        urban_result = _RoadEnvironmentResult(
            environment=RoadEnvironment.URBAN_STREET,
            confidence=match.confidence,
            reason="womd_lane_type_surface_street",
            matched_lane_type=lane_type_name,
            matched_lane_speed_limit_mph=speed_limit_mph,
        )
        if (
            speed_limit_mph <= 0.0
            or speed_limit_mph > self.config.parking_max_speed_limit_mph
            or junction_kind == JunctionKind.ROUNDABOUT
        ):
            return urban_result

        position = np.asarray(
            [state.center_x, state.center_y],
            dtype=np.float64,
        )
        (
            stationary_count,
            off_lane_count,
            off_lane_ratio,
        ) = self._stationary_vehicle_evidence(frame_index, position)
        if (
            stationary_count
            < self.config.parking_min_off_lane_stationary_vehicles
            or off_lane_count
            < self.config.parking_min_off_lane_stationary_vehicles
            or off_lane_ratio
            < self.config.parking_min_off_lane_stationary_ratio
        ):
            return urban_result

        context_lane_ids = self._local_lane_ids(
            position,
            self.config.parking_context_radius_m,
        )
        local_low_speed_lane_ratio = self._local_low_speed_lane_ratio(
            context_lane_ids
        )
        in_intersection = region_type in {
            RegionType.INTERSECTION,
            RegionType.IN_INTERSECTION,
            RegionType.NEAR_INTERSECTION_APPROACH,
            RegionType.NEAR_INTERSECTION_EXIT,
        }
        required_low_speed_ratio = (
            self.config.parking_intersection_min_low_speed_lane_ratio
            if in_intersection
            else self.config.parking_min_low_speed_lane_ratio
        )
        dense_low_speed_context = bool(
            off_lane_count
            >= self.config.parking_dense_off_lane_stationary_vehicles
            and local_low_speed_lane_ratio >= required_low_speed_ratio
        )

        lane_neighbor_count = (
            len(lane.left_neighbors) + len(lane.right_neighbors)
        )
        driveway_distance_m = self._nearest_driveway_distance_m(position)
        isolated_parking_access = bool(
            region_type == RegionType.ROAD_SEGMENT
            and lane_count == 1
            and lane_neighbor_count == 0
            and driveway_distance_m
            <= self.config.parking_access_max_driveway_distance_m
            and local_low_speed_lane_ratio
            <= self.config.parking_access_max_low_speed_lane_ratio
        )
        if not dense_low_speed_context and not isolated_parking_access:
            return urban_result

        internal_lane_ids = self._local_lane_ids(
            position,
            self.config.parking_internal_lane_radius_m,
        )
        internal_branch_lane_count = sum(
            self._lane_graph_degrees.get(lane_id, 0) >= 3
            for lane_id in internal_lane_ids
        )
        lane_length_m = float(
            self.lane_index.s_by_lane[match.lane_id][-1]
        )
        nearest_road_edge_distance_m = (
            self._nearest_road_edge_distance_m(position)
        )
        has_local_road_edge = (
            nearest_road_edge_distance_m
            <= self.config.parking_context_radius_m
        )
        effective_lane_count = lane_count
        if effective_lane_count is None:
            effective_lane_count = len(
                self.lane_index.same_direction_lane_ids(
                    match.lane_id,
                    match.point_index,
                )
            )
        single_direction_lane = effective_lane_count == 1

        internal_lane_network = bool(
            dense_low_speed_context
            and single_direction_lane
            and has_local_road_edge
            and len(internal_lane_ids)
            >= self.config.parking_internal_min_lane_count
            and internal_branch_lane_count
            >= self.config.parking_internal_min_branch_lane_count
            and lane_length_m
            <= self.config.parking_internal_max_lane_length_m
        )
        parking_access_network = bool(
            isolated_parking_access
            and single_direction_lane
            and has_local_road_edge
            and len(internal_lane_ids)
            >= 2 * self.config.parking_internal_min_lane_count
            and internal_branch_lane_count
            >= 2 * self.config.parking_internal_min_branch_lane_count
            and lane_length_m
            <= 0.75 * self.config.parking_internal_max_lane_length_m
        )

        compact_road_edge_enclosure = False
        if (
            dense_low_speed_context
            and single_direction_lane
            and region_type == RegionType.ROAD_SEGMENT
            and has_local_road_edge
        ):
            lane_heading_rad = self.lane_index._heading_at_point(
                match.lane_id,
                match.point_index,
            )
            forward_edge_distance_m = self._road_edge_ray_distance_m(
                position,
                lane_heading_rad,
                self.config.parking_compact_edge_distance_m,
            )
            backward_edge_distance_m = self._road_edge_ray_distance_m(
                position,
                lane_heading_rad + math.pi,
                self.config.parking_compact_edge_distance_m,
            )
            compact_road_edge_enclosure = bool(
                math.isfinite(forward_edge_distance_m)
                and math.isfinite(backward_edge_distance_m)
            )

        broad_road_edge_enclosure = bool(
            dense_low_speed_context
            and single_direction_lane
            and has_local_road_edge
            and nearest_road_edge_distance_m
            >= self.config.parking_off_lane_distance_m + 1.0
            and off_lane_count
            >= 3
            * self.config.parking_dense_off_lane_stationary_vehicles
            and len(context_lane_ids)
            >= self.config.parking_internal_min_lane_count
        )
        if not any(
            (
                internal_lane_network,
                parking_access_network,
                compact_road_edge_enclosure,
                broad_road_edge_enclosure,
            )
        ):
            return urban_result

        density_strength = min(
            1.0,
            off_lane_count
            / max(
                1,
                2
                * self.config.parking_dense_off_lane_stationary_vehicles,
            ),
        )
        ratio_strength = min(
            1.0,
            local_low_speed_lane_ratio
            / max(required_low_speed_ratio, 1e-6),
        )
        proxy_confidence = (
            0.70 + 0.20 * density_strength + 0.10 * ratio_strength
        )
        if compact_road_edge_enclosure:
            reason = "compact_road_edge_enclosed_parking_area"
            enclosure_confidence = 0.95
        elif parking_access_network:
            reason = "road_edge_bounded_parking_access_network"
            enclosure_confidence = 0.85
        elif internal_lane_network:
            reason = "road_edge_bounded_internal_parking_lane_network"
            enclosure_confidence = 0.90
        else:
            reason = "broad_road_edge_bounded_dense_parking_area"
            enclosure_confidence = 0.85

        return _RoadEnvironmentResult(
            environment=RoadEnvironment.PARKING_LOT,
            confidence=min(
                match.confidence,
                proxy_confidence,
                enclosure_confidence,
            ),
            reason=reason,
            matched_lane_type=lane_type_name,
            matched_lane_speed_limit_mph=speed_limit_mph,
        )


def _junction_geometry(core_lane_ids: set[int], lanes, half_width_m: float):
    if LineString is None:
        return None
    buffered = []
    for lane_id in core_lane_ids:
        points = _polyline_xy(lanes[lane_id].lane.polyline)
        if len(points) < 2:
            continue
        buffered.append(
            LineString(points).buffer(
                half_width_m,
                cap_style=2,
                join_style=2,
            )
        )
    if not buffered:
        return None
    geometry = unary_union(buffered).buffer(0)
    if geometry.is_empty:
        return None
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    polygonal = [
        item
        for item in getattr(geometry, "geoms", ())
        if isinstance(item, Polygon)
    ]
    return unary_union(polygonal) if polygonal else None


def _geometry_boundaries(geometry) -> tuple[tuple[tuple[float, float], ...], ...]:
    if geometry is None:
        return ()
    polygons = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    polygons.sort(key=lambda polygon: polygon.area, reverse=True)
    return tuple(
        tuple((float(x), float(y)) for x, y in polygon.exterior.coords)
        for polygon in polygons
    )


def _fallback_boundary(core_lane_ids: set[int], lanes, half_width_m: float):
    expanded_points = []
    for lane_id in core_lane_ids:
        points = _polyline_xy(lanes[lane_id].lane.polyline)
        if len(points) < 2:
            continue
        tangents = np.empty_like(points)
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        if len(points) > 2:
            tangents[1:-1] = points[2:] - points[:-2]
        norms = np.linalg.norm(tangents, axis=1)
        valid = norms > 1e-9
        normals = np.zeros_like(tangents)
        normals[valid, 0] = -tangents[valid, 1] / norms[valid]
        normals[valid, 1] = tangents[valid, 0] / norms[valid]
        expanded_points.extend((points + half_width_m * normals).tolist())
        expanded_points.extend((points - half_width_m * normals).tolist())
    points = np.asarray(expanded_points, dtype=np.float64)
    if len(points) < 3:
        center = tuple(np.mean(points, axis=0).tolist()) if len(points) else (0.0, 0.0)
        return center, ()
    try:
        hull = ConvexHull(points)
        polygon = points[hull.vertices]
    except QhullError:
        polygon = points
    polygon = np.concatenate((polygon, polygon[:1]), axis=0)
    center = tuple(np.mean(polygon[:-1], axis=0).tolist())
    boundary = tuple((float(x), float(y)) for x, y in polygon)
    return center, (boundary,)


def _cluster_gates(
    gates: list[_Gate], threshold_rad: float
) -> list[list[_Gate]]:
    if not gates:
        return []
    ordered = sorted(gates, key=lambda gate: gate.angle_rad)
    if len(ordered) == 1:
        return [ordered]
    gaps = [
        (
            (ordered[(index + 1) % len(ordered)].angle_rad - gate.angle_rad)
            % (2.0 * math.pi),
            index,
        )
        for index, gate in enumerate(ordered)
    ]
    largest_gap_index = max(gaps)[1]
    start = (largest_gap_index + 1) % len(ordered)
    clusters: list[list[_Gate]] = []
    current: list[_Gate] = []
    for offset in range(len(ordered)):
        index = (start + offset) % len(ordered)
        current.append(ordered[index])
        if offset == len(ordered) - 1:
            clusters.append(current)
            break
        next_index = (index + 1) % len(ordered)
        gap = (ordered[next_index].angle_rad - ordered[index].angle_rad) % (
            2.0 * math.pi
        )
        if gap > threshold_rad:
            clusters.append(current)
            current = []
    return clusters


def _stop_line(
    cluster: list[_Gate],
    center_xy: tuple[float, float],
    half_width_m: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    incoming_points = np.asarray(
        [gate.point_xy for gate in cluster if gate.incoming], dtype=np.float64
    )
    if not len(incoming_points):
        return None
    center = np.mean(incoming_points, axis=0)
    radial = center - np.asarray(center_xy)
    norm = float(np.linalg.norm(radial))
    if norm <= 1e-6:
        return None
    tangent = np.asarray([-radial[1], radial[0]]) / norm
    projections = (incoming_points - center) @ tangent
    low = float(np.min(projections) - half_width_m)
    high = float(np.max(projections) + half_width_m)
    left = center + low * tangent
    right = center + high * tangent
    return (float(left[0]), float(left[1])), (float(right[0]), float(right[1]))


def _build_arms(
    incoming_lane_ids: set[int],
    outgoing_lane_ids: set[int],
    lanes,
    center_xy: tuple[float, float],
    config: MapAnnotationConfig,
    *,
    use_radial_angles: bool = False,
) -> tuple[JunctionArm, ...]:
    gates: list[_Gate] = []
    for lane_id in sorted(incoming_lane_ids):
        points = _polyline_xy(lanes[lane_id].lane.polyline)
        point = points[-1]
        if use_radial_angles:
            vector = point - np.asarray(center_xy)
            angle = math.atan2(vector[1], vector[0]) % (2.0 * math.pi)
        else:
            vector = point - points[-min(50, len(points))]
            travel_heading = math.atan2(vector[1], vector[0])
            angle = (travel_heading + math.pi) % (2.0 * math.pi)
        gates.append(_Gate(lane_id, True, (float(point[0]), float(point[1])), angle))
    for lane_id in sorted(outgoing_lane_ids):
        points = _polyline_xy(lanes[lane_id].lane.polyline)
        point = points[0]
        if use_radial_angles:
            vector = point - np.asarray(center_xy)
            angle = math.atan2(vector[1], vector[0]) % (2.0 * math.pi)
        else:
            vector = points[min(49, len(points) - 1)] - point
            angle = math.atan2(vector[1], vector[0]) % (2.0 * math.pi)
        gates.append(_Gate(lane_id, False, (float(point[0]), float(point[1])), angle))

    clusters = _cluster_gates(
        gates, math.radians(config.arm_angle_threshold_deg)
    )
    cluster_data = [
        (_circular_mean([gate.angle_rad for gate in cluster]), cluster)
        for cluster in clusters
    ]
    cluster_data.sort(key=lambda item: item[0])
    arms = []
    for arm_index, (angle, cluster) in enumerate(cluster_data):
        selected_cluster = _deduplicate_gate_lanes(cluster, lanes)
        arms.append(
            JunctionArm(
                arm_index=arm_index,
                angle_rad=angle,
                incoming_lane_ids=tuple(
                    sorted(
                        gate.lane_id
                        for gate in selected_cluster
                        if gate.incoming
                    )
                ),
                outgoing_lane_ids=tuple(
                    sorted(
                        gate.lane_id
                        for gate in selected_cluster
                        if not gate.incoming
                    )
                ),
                gate_points_xy=tuple(
                    gate.point_xy for gate in selected_cluster
                ),
                stop_line_xy=_stop_line(
                    selected_cluster, center_xy, config.lane_half_width_m
                ),
            )
        )
    return tuple(arms)


def _deduplicate_gate_lanes(
    gates: list[_Gate],
    lanes,
) -> list[_Gate]:
    selected = []
    for incoming in (True, False):
        direction_gates = sorted(
            (gate for gate in gates if gate.incoming == incoming),
            key=lambda gate: gate.lane_id,
        )
        clusters: list[list[_Gate]] = []
        for gate in direction_gates:
            point = np.asarray(gate.point_xy)
            matching_cluster = next(
                (
                    cluster
                    for cluster in clusters
                    if np.linalg.norm(
                        point
                        - np.mean(
                            [item.point_xy for item in cluster],
                            axis=0,
                        )
                    )
                    <= _GATE_LANE_DEDUP_TOLERANCE_M
                ),
                None,
            )
            if matching_cluster is None:
                clusters.append([gate])
            else:
                matching_cluster.append(gate)
        for cluster in clusters:
            selected.append(
                min(
                    cluster,
                    key=lambda gate: (
                        -float(
                            _cumulative_distance(
                                _polyline_xy(
                                    lanes[gate.lane_id].lane.polyline
                                )
                            )[-1]
                        ),
                        gate.lane_id,
                    ),
                )
            )
    return sorted(selected, key=lambda gate: gate.angle_rad)


def _lane_lengths(lanes) -> dict[int, float]:
    return {
        lane_id: float(_cumulative_distance(_polyline_xy(lane.lane.polyline))[-1])
        for lane_id, lane in lanes.items()
    }


def _distances_to_core(
    junction: JunctionAnnotation, lanes, lengths: dict[int, float]
) -> dict[int, tuple[float, int]]:
    distances: dict[int, tuple[float, int]] = {}
    queue: list[tuple[float, int, int]] = []
    for arm in junction.arms:
        for lane_id in arm.incoming_lane_ids:
            candidate = (lengths[lane_id], arm.arm_index)
            if lane_id not in distances or candidate[0] < distances[lane_id][0]:
                distances[lane_id] = candidate
                heapq.heappush(queue, (candidate[0], lane_id, arm.arm_index))
    while queue:
        distance, lane_id, arm_index = heapq.heappop(queue)
        if distances.get(lane_id) != (distance, arm_index):
            continue
        for predecessor in lanes[lane_id].lane.entry_lanes:
            if predecessor not in lanes or predecessor in junction.core_lane_ids:
                continue
            candidate = distance + lengths[predecessor]
            if candidate < distances.get(predecessor, (math.inf, -1))[0]:
                distances[predecessor] = (candidate, arm_index)
                heapq.heappush(queue, (candidate, predecessor, arm_index))
    return distances


def _distances_from_core(
    junction: JunctionAnnotation, lanes, lengths: dict[int, float]
) -> dict[int, tuple[float, int]]:
    distances: dict[int, tuple[float, int]] = {}
    queue: list[tuple[float, int, int]] = []
    for arm in junction.arms:
        for lane_id in arm.outgoing_lane_ids:
            candidate = (0.0, arm.arm_index)
            if lane_id not in distances:
                distances[lane_id] = candidate
                heapq.heappush(queue, (0.0, lane_id, arm.arm_index))
    while queue:
        distance, lane_id, arm_index = heapq.heappop(queue)
        if distances.get(lane_id) != (distance, arm_index):
            continue
        for successor in lanes[lane_id].lane.exit_lanes:
            if successor not in lanes or successor in junction.core_lane_ids:
                continue
            candidate = distance + lengths[lane_id]
            if candidate < distances.get(successor, (math.inf, -1))[0]:
                distances[successor] = (candidate, arm_index)
                heapq.heappush(queue, (candidate, successor, arm_index))
    return distances


def _directional_branch_through_lane_ids(
    junction: JunctionAnnotation,
    lanes,
    config: MapAnnotationConfig,
) -> frozenset[int]:
    if junction.kind != JunctionKind.STOP_CONTROLLED or junction.arm_count != 3:
        return frozenset()

    bidirectional_arms = [
        arm
        for arm in junction.arms
        if arm.incoming_lane_ids and arm.outgoing_lane_ids
    ]
    incoming_only_arms = [
        arm
        for arm in junction.arms
        if arm.incoming_lane_ids and not arm.outgoing_lane_ids
    ]
    outgoing_only_arms = [
        arm
        for arm in junction.arms
        if arm.outgoing_lane_ids and not arm.incoming_lane_ids
    ]
    if not (
        len(bidirectional_arms)
        == len(incoming_only_arms)
        == len(outgoing_only_arms)
        == 1
    ):
        return frozenset()

    incoming_arm = incoming_only_arms[0]
    outgoing_arm = outgoing_only_arms[0]
    arm_separation = abs(
        float(_wrap_angle(incoming_arm.angle_rad - outgoing_arm.angle_rad))
    )
    opposition_error = abs(math.pi - arm_separation)
    if opposition_error > math.radians(config.arm_angle_threshold_deg):
        return frozenset()

    core_lane_ids = set(junction.core_lane_ids)
    approach_lane_ids = set(junction._to_core)

    source_arms: dict[int, set[int]] = {}
    source_queue: deque[int] = deque()
    for arm in junction.arms:
        for lane_id in arm.incoming_lane_ids:
            if lane_id not in lanes:
                continue
            source_arms.setdefault(lane_id, set()).add(arm.arm_index)
            source_queue.append(lane_id)
    while source_queue:
        lane_id = source_queue.popleft()
        for successor in lanes[lane_id].lane.exit_lanes:
            if successor not in core_lane_ids:
                continue
            previous_size = len(source_arms.get(successor, ()))
            source_arms.setdefault(successor, set()).update(
                source_arms[lane_id]
            )
            if len(source_arms[successor]) > previous_size:
                source_queue.append(successor)

    destination_arms: dict[int, set[int]] = {}
    destination_queue: deque[int] = deque()
    for arm in junction.arms:
        for lane_id in arm.outgoing_lane_ids:
            if lane_id not in lanes:
                continue
            destination_arms.setdefault(lane_id, set()).add(arm.arm_index)
            destination_queue.append(lane_id)
    reverse_domain = core_lane_ids | approach_lane_ids
    while destination_queue:
        lane_id = destination_queue.popleft()
        for predecessor in lanes[lane_id].lane.entry_lanes:
            if predecessor not in reverse_domain:
                continue
            if (
                predecessor in approach_lane_ids
                and lane_id in approach_lane_ids
                and junction._to_core[predecessor][0]
                <= junction._to_core[lane_id][0] + 1e-6
            ):
                continue
            previous_size = len(destination_arms.get(predecessor, ()))
            destination_arms.setdefault(predecessor, set()).update(
                destination_arms[lane_id]
            )
            if len(destination_arms[predecessor]) > previous_size:
                destination_queue.append(predecessor)

    incoming_arm_index = incoming_arm.arm_index
    outgoing_arm_index = outgoing_arm.arm_index
    through_lane_ids = {
        lane_id
        for lane_id, (_, arm_index) in junction._to_core.items()
        if arm_index == incoming_arm_index
        and destination_arms.get(lane_id) == {outgoing_arm_index}
    }
    through_lane_ids.update(
        lane_id
        for lane_id in core_lane_ids
        if source_arms.get(lane_id) == {incoming_arm_index}
        and destination_arms.get(lane_id) == {outgoing_arm_index}
    )
    return frozenset(through_lane_ids)


def _signal_stop_points(scenario) -> dict[int, tuple[float, float]]:
    stop_points = {}
    for dynamic_state in scenario.dynamic_map_states:
        for lane_state in dynamic_state.lane_states:
            stop_points.setdefault(
                int(lane_state.lane),
                (float(lane_state.stop_point.x), float(lane_state.stop_point.y)),
            )
    return stop_points


def _strongly_connected_lane_components(lanes) -> tuple[frozenset[int], ...]:
    adjacency = {
        lane_id: tuple(
            sorted(
                exit_id
                for exit_id in lane.lane.exit_lanes
                if exit_id in lanes
            )
        )
        for lane_id, lane in lanes.items()
    }
    reverse_adjacency = {lane_id: [] for lane_id in lanes}
    for lane_id, successors in adjacency.items():
        for successor in successors:
            reverse_adjacency[successor].append(lane_id)

    visited: set[int] = set()
    finish_order: list[int] = []
    for start in sorted(lanes):
        if start in visited:
            continue
        stack = [(start, False)]
        while stack:
            lane_id, expanded = stack.pop()
            if expanded:
                finish_order.append(lane_id)
                continue
            if lane_id in visited:
                continue
            visited.add(lane_id)
            stack.append((lane_id, True))
            stack.extend(
                (successor, False)
                for successor in reversed(adjacency[lane_id])
                if successor not in visited
            )

    assigned: set[int] = set()
    components = []
    for start in reversed(finish_order):
        if start in assigned:
            continue
        component = set()
        stack = [start]
        assigned.add(start)
        while stack:
            lane_id = stack.pop()
            component.add(lane_id)
            for predecessor in reverse_adjacency[lane_id]:
                if predecessor not in assigned:
                    assigned.add(predecessor)
                    stack.append(predecessor)
        if len(component) > 1 or start in adjacency[start]:
            components.append(frozenset(component))
    components.sort(key=lambda component: min(component))
    return tuple(components)


def _circular_lane_component(
    lane_ids: frozenset[int],
    lanes,
) -> _CircularLaneComponent | None:
    if len(lane_ids) > _ROUNDABOUT_MAX_COMPONENT_LANES:
        return None
    polylines = [
        _polyline_xy(lanes[lane_id].lane.polyline)
        for lane_id in lane_ids
        if lane_id in lanes and len(lanes[lane_id].lane.polyline) >= 2
    ]
    if not polylines:
        return None
    points = np.concatenate(polylines, axis=0)
    spans = np.ptp(points, axis=0)
    max_span = float(np.max(spans))
    min_span = float(np.min(spans))
    if (
        min_span < _ROUNDABOUT_MIN_SPAN_M
        or max_span <= 1e-6
        or min_span / max_span < _ROUNDABOUT_MIN_ASPECT_RATIO
    ):
        return None

    center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
    radii = np.linalg.norm(points - center, axis=1)
    radius_p10, radius_p90 = np.percentile(radii, [10.0, 90.0])
    if (
        radius_p10 < 3.0
        or radius_p90 / max(radius_p10, 1e-6)
        > _ROUNDABOUT_MAX_RADIAL_RATIO
    ):
        return None

    tangents = np.concatenate(
        [np.diff(polyline, axis=0) for polyline in polylines],
        axis=0,
    )
    midpoints = np.concatenate(
        [(polyline[:-1] + polyline[1:]) / 2.0 for polyline in polylines],
        axis=0,
    )
    tangent_norms = np.linalg.norm(tangents, axis=1)
    radial_vectors = midpoints - center
    radial_norms = np.linalg.norm(radial_vectors, axis=1)
    valid = (tangent_norms > 1e-6) & (radial_norms > 1e-6)
    if not np.any(valid):
        return None
    radial_alignment = np.abs(
        np.sum(
            tangents[valid]
            / tangent_norms[valid, None]
            * radial_vectors[valid]
            / radial_norms[valid, None],
            axis=1,
        )
    )
    if (
        float(np.median(radial_alignment))
        > _ROUNDABOUT_MAX_MEDIAN_RADIAL_ALIGNMENT
        or float(np.percentile(radial_alignment, 90.0))
        > _ROUNDABOUT_MAX_P90_RADIAL_ALIGNMENT
    ):
        return None

    point_angles = np.sort(
        np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        % (2.0 * math.pi)
    )
    angular_gaps = np.diff(
        np.concatenate((point_angles, point_angles[:1] + 2.0 * math.pi))
    )
    if (
        not len(angular_gaps)
        or float(np.max(angular_gaps)) > _ROUNDABOUT_MAX_ANGULAR_GAP_RAD
    ):
        return None
    return _CircularLaneComponent(
        lane_ids=lane_ids,
        center_xy=(float(center[0]), float(center[1])),
        median_radius_m=float(np.median(radii)),
    )


def _merge_concentric_lane_components(
    components: list[_CircularLaneComponent],
) -> list[tuple[set[int], int]]:
    parents = list(range(len(components)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left_index, left in enumerate(components):
        for right_index in range(left_index + 1, len(components)):
            right = components[right_index]
            center_distance = float(
                np.linalg.norm(
                    np.asarray(left.center_xy) - np.asarray(right.center_xy)
                )
            )
            threshold = max(
                _ROUNDABOUT_MIN_SPAN_M,
                0.25 * min(left.median_radius_m, right.median_radius_m),
            )
            if center_distance <= threshold:
                left_root = find(left_index)
                right_root = find(right_index)
                if left_root != right_root:
                    parents[right_root] = left_root

    merged: dict[int, tuple[set[int], int]] = {}
    for index, component in enumerate(components):
        root = find(index)
        if root not in merged:
            merged[root] = (set(), 0)
        merged[root][0].update(component.lane_ids)
        merged[root] = (merged[root][0], merged[root][1] + 1)
    return sorted(merged.values(), key=lambda item: min(item[0]))


def _external_lane_ids(
    core_lane_ids: set[int],
    lanes,
) -> tuple[set[int], set[int]]:
    incoming = {
        entry_id
        for lane_id in core_lane_ids
        for entry_id in lanes[lane_id].lane.entry_lanes
        if entry_id in lanes and entry_id not in core_lane_ids
    }
    outgoing = {
        exit_id
        for lane_id in core_lane_ids
        for exit_id in lanes[lane_id].lane.exit_lanes
        if exit_id in lanes and exit_id not in core_lane_ids
    }
    return incoming, outgoing


def _external_corridor_count(
    incoming_lane_ids: set[int],
    outgoing_lane_ids: set[int],
    lanes,
    center_xy: tuple[float, float],
    config: MapAnnotationConfig,
) -> int:
    center = np.asarray(center_xy)
    gates = []
    for incoming, lane_ids in (
        (True, incoming_lane_ids),
        (False, outgoing_lane_ids),
    ):
        for lane_id in lane_ids:
            points = _polyline_xy(lanes[lane_id].lane.polyline)
            near_point = points[-1] if incoming else points[0]
            far_point = points[0] if incoming else points[-1]
            vector = far_point - center
            angle = math.atan2(vector[1], vector[0]) % (2.0 * math.pi)
            gates.append(
                _Gate(
                    lane_id=lane_id,
                    incoming=incoming,
                    point_xy=(float(near_point[0]), float(near_point[1])),
                    angle_rad=angle,
                )
            )
    return len(
        _cluster_gates(
            gates,
            math.radians(config.arm_angle_threshold_deg),
        )
    )


def _roundabout_groups(processor, config: MapAnnotationConfig) -> list[set[int]]:
    lanes = processor.lanecenters
    circular_components = [
        component
        for lane_ids in _strongly_connected_lane_components(lanes)
        if (component := _circular_lane_component(lane_ids, lanes)) is not None
    ]
    groups = []
    for core, circular_component_count in _merge_concentric_lane_components(
        circular_components
    ):
        incoming, outgoing = _external_lane_ids(core, lanes)
        if not incoming or not outgoing:
            continue
        points = np.concatenate(
            [
                _polyline_xy(lanes[lane_id].lane.polyline)
                for lane_id in core
            ],
            axis=0,
        )
        center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2.0
        center_xy = (float(center[0]), float(center[1]))
        arms = _build_arms(
            incoming,
            outgoing,
            lanes,
            center_xy,
            config,
            use_radial_angles=True,
        )
        if not config.min_junction_arms <= len(arms) <= config.max_junction_arms:
            continue
        if (
            circular_component_count == 1
            and _external_corridor_count(
                incoming,
                outgoing,
                lanes,
                center_xy,
                config,
            )
            < config.min_junction_arms
        ):
            continue
        groups.append(core)
    return groups


def _deduplicated_groups(
    processor,
    include_stop_controlled: bool,
    config: MapAnnotationConfig,
):
    groups = [
        (JunctionKind.SIGNALIZED, set(group))
        for group in processor.signalized_intersections
    ]
    if include_stop_controlled:
        groups.extend(
            (JunctionKind.STOP_CONTROLLED, set(group))
            for group in processor.stop_intersections
        )
        groups.extend(
            (JunctionKind.ROUNDABOUT, set(group))
            for group in _roundabout_groups(processor, config)
        )
        groups.extend(
            (JunctionKind.GEOMETRIC, set(group))
            for group in processor._find_special_intersections(
                "geometric_intersection"
            )
        )
    groups.sort(
        key=lambda item: (
            _junction_kind_priority(item[0]),
            min(item[1]) if item[1] else math.inf,
        )
    )
    kept = []
    for kind, core in groups:
        if kind in (JunctionKind.ROUNDABOUT, JunctionKind.GEOMETRIC) and any(
            len(core & existing) / max(1, min(len(core), len(existing))) >= 0.8
            for _, existing in kept
        ):
            continue
        if any(
            len(core & existing) / max(1, len(core | existing)) >= 0.8
            for _, existing in kept
        ):
            continue
        kept.append((kind, core))
    return kept


def _geometry_overlap_ratio(left, right) -> float:
    if left is None or right is None or left.is_empty or right.is_empty:
        return 0.0
    smaller_area = min(float(left.area), float(right.area))
    if smaller_area <= 1e-9:
        return 0.0
    intersection = left.intersection(right)
    if intersection.is_empty:
        return 0.0
    return float(intersection.area) / smaller_area


def _merge_spatially_overlapping_groups(groups, lanes, config):
    if len(groups) < 2 or LineString is None:
        return groups

    geometries = [
        _junction_geometry(core, lanes, config.lane_half_width_m)
        for _, core in groups
    ]
    parents = list(range(len(groups)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, (left_kind, _) in enumerate(groups):
        for right_index in range(left_index + 1, len(groups)):
            right_kind, _ = groups[right_index]
            if left_kind != right_kind:
                continue
            overlap_ratio = _geometry_overlap_ratio(
                geometries[left_index],
                geometries[right_index],
            )
            if overlap_ratio >= config.junction_merge_overlap_ratio:
                union(left_index, right_index)

    merged: dict[int, tuple[JunctionKind, set[int]]] = {}
    for index, (kind, core) in enumerate(groups):
        root = find(index)
        if root not in merged:
            merged[root] = (kind, set())
        merged[root][1].update(core)

    result = list(merged.values())
    result.sort(
        key=lambda item: (
            _junction_kind_priority(item[0]),
            min(item[1]) if item[1] else math.inf,
        )
    )
    return result


def _is_geometric_junction_group(
    core_lane_ids: set[int],
    arms: tuple[JunctionArm, ...],
    lanes,
) -> bool:
    if LineString is None or len(core_lane_ids) < 3:
        return False
    bidirectional_arm_count = sum(
        bool(arm.incoming_lane_ids) and bool(arm.outgoing_lane_ids)
        for arm in arms
    )
    if bidirectional_arm_count < max(2, len(arms) - 1):
        return False
    centerlines = [
        LineString(_polyline_xy(lanes[lane_id].lane.polyline))
        for lane_id in core_lane_ids
        if len(lanes[lane_id].lane.polyline) >= 2
    ]
    return any(left.crosses(right) for left, right in combinations(centerlines, 2))


def _build_junctions(
    scenario,
    processor,
    config: MapAnnotationConfig,
) -> tuple[JunctionAnnotation, ...]:
    lanes = processor.lanecenters
    signal_stop_points = _signal_stop_points(scenario)
    junctions = []
    groups = _deduplicated_groups(
        processor,
        config.include_stop_controlled,
        config,
    )
    groups = _merge_spatially_overlapping_groups(groups, lanes, config)
    for kind, raw_core in groups:
        core = {lane_id for lane_id in raw_core if lane_id in lanes}
        if not core:
            continue
        incoming, outgoing = _external_lane_ids(core, lanes)
        geometry = _junction_geometry(core, lanes, config.lane_half_width_m)
        if geometry is not None:
            center_xy = (float(geometry.centroid.x), float(geometry.centroid.y))
            boundaries = _geometry_boundaries(geometry)
        else:
            center_xy, boundaries = _fallback_boundary(
                core, lanes, config.lane_half_width_m
            )
        validation_arms = _build_arms(
            incoming,
            outgoing,
            lanes,
            center_xy,
            config,
            use_radial_angles=kind
            in (JunctionKind.ROUNDABOUT, JunctionKind.GEOMETRIC),
        )
        if not (
            config.min_junction_arms
            <= len(validation_arms)
            <= config.max_junction_arms
        ):
            continue
        if (
            kind == JunctionKind.GEOMETRIC
            and not _is_geometric_junction_group(core, validation_arms, lanes)
        ):
            continue
        arms = validation_arms
        if kind == JunctionKind.GEOMETRIC:
            travel_heading_arms = _build_arms(
                incoming,
                outgoing,
                lanes,
                center_xy,
                config,
                use_radial_angles=False,
            )
            if (
                config.min_junction_arms
                <= len(travel_heading_arms)
                <= config.max_junction_arms
            ):
                arms = travel_heading_arms

        inferred_stop_points = {
            tuple(_polyline_xy(lanes[lane_id].lane.polyline)[0])
            for lane_id in core
            if any(
                entry_id not in core
                for entry_id in lanes[lane_id].lane.entry_lanes
            )
        }
        observed_stop_points = {
            signal_stop_points[lane_id]
            for lane_id in core
            if lane_id in signal_stop_points
        }
        stop_points = observed_stop_points or inferred_stop_points
        evidence = ["lane_topology"]
        if kind == JunctionKind.ROUNDABOUT:
            evidence.extend(
                ("directed_lane_cycle", "circular_lane_geometry")
            )
        else:
            evidence.append("diverge_merge_union_find")
        confidence = 0.70
        if kind == JunctionKind.SIGNALIZED:
            evidence.append("traffic_signal_lane")
            confidence += 0.20
        elif kind == JunctionKind.STOP_CONTROLLED:
            evidence.append("stop_sign_lane")
            confidence += 0.08
        elif kind == JunctionKind.ROUNDABOUT:
            confidence += 0.15
        else:
            evidence.append("lane_crossing_conflicts")
            confidence += 0.03
        if geometry is not None:
            evidence.append("lane_buffer_boundary")
            confidence += 0.05
        elif boundaries:
            evidence.append("lane_width_convex_hull_boundary")
            confidence += 0.03
        if len(validation_arms) in (3, 4):
            confidence += 0.05

        junctions.append(
            JunctionAnnotation(
                junction_id=-1,
                kind=kind,
                core_lane_ids=tuple(sorted(core)),
                incoming_lane_ids=tuple(sorted(incoming)),
                outgoing_lane_ids=tuple(sorted(outgoing)),
                signal_lane_ids=tuple(sorted(core & signal_stop_points.keys())),
                center_xy=center_xy,
                arms=arms,
                stop_points_xy=tuple(sorted(stop_points)),
                boundary_polygons_xy=boundaries,
                confidence=min(1.0, confidence),
                evidence=tuple(evidence),
                _geometry=geometry,
            )
        )

    junctions.sort(
        key=lambda junction: (
            _junction_kind_priority(junction.kind),
            junction.center_xy[0],
            junction.center_xy[1],
        )
    )
    lengths = _lane_lengths(lanes)
    for junction_id, junction in enumerate(junctions):
        junction.junction_id = junction_id
        junction._to_core = _distances_to_core(junction, lanes, lengths)
        junction._from_core = _distances_from_core(junction, lanes, lengths)
        junction._directional_branch_through_lane_ids = (
            _directional_branch_through_lane_ids(
                junction,
                lanes,
                config,
            )
        )
        if junction._directional_branch_through_lane_ids:
            junction.evidence += ("directional_branch_mainline",)
    return tuple(junctions)


class EgoMapAnnotator:
    def __init__(self, scenario, config: MapAnnotationConfig | None = None) -> None:
        self.scenario = scenario
        self.config = config or MapAnnotationConfig()
        self.processor = ScenarioProcessor(scenario, load_boundaries=False)
        self.lanes = self.processor.lanecenters
        self.lane_index = (
            _LaneGeometryIndex(self.lanes, self.config) if self.lanes else None
        )
        self.road_environment_classifier = (
            _RoadEnvironmentClassifier(
                scenario,
                self.lanes,
                self.lane_index,
                self.config,
            )
            if self.lane_index is not None
            else None
        )
        self.junctions = _build_junctions(
            scenario, self.processor, self.config
        )

    def _classify(
        self, match: MapMatch
    ) -> tuple[RegionType, JunctionAnnotation | None, int | None, float | None]:
        core_matches = [
            junction
            for junction in self.junctions
            if match.lane_id in junction.core_lane_ids
            and match.lane_id
            not in junction._directional_branch_through_lane_ids
        ]
        if core_matches:
            junction = max(core_matches, key=lambda item: item.confidence)
            return RegionType.INTERSECTION, junction, None, 0.0

        upcoming_candidates = []
        for junction in self.junctions:
            if match.lane_id in junction._directional_branch_through_lane_ids:
                continue
            if match.lane_id in junction._to_core:
                distance_at_start, arm_index = junction._to_core[match.lane_id]
                distance = max(0.0, distance_at_start - match.lane_s_m)
                upcoming_candidates.append(
                    (
                        distance,
                        junction,
                        arm_index,
                    )
                )
        if not upcoming_candidates:
            return RegionType.ROAD_SEGMENT, None, None, None
        distance, junction, arm_index = min(
            upcoming_candidates,
            key=lambda item: (
                item[0],
                -item[1].confidence,
            ),
        )
        if distance <= self.config.near_distance_m:
            return RegionType.INTERSECTION, junction, arm_index, distance
        return RegionType.ROAD_SEGMENT, None, None, distance

    def annotate_ego_frames(
        self,
        frame_indices: Iterable[int] | None = None,
    ) -> tuple[EgoFrameAnnotation, ...]:
        timestamps = list(self.scenario.timestamps_seconds)
        sdc_index = self.scenario.sdc_track_index
        sdc_is_valid = 0 <= sdc_index < len(self.scenario.tracks)
        states = self.scenario.tracks[sdc_index].states if sdc_is_valid else ()
        frame_count = max(len(timestamps), len(states))
        if frame_indices is None:
            selected_indices = tuple(range(frame_count))
        else:
            selected_indices = tuple(dict.fromkeys(int(index) for index in frame_indices))
            if any(index < 0 for index in selected_indices):
                raise ValueError("frame indices must be non-negative")
        if (
            self.lane_index is None
            or not sdc_is_valid
        ):
            return tuple(
                EgoFrameAnnotation(
                    frame_index=index,
                    timestamp_seconds=(timestamps[index] if index < len(timestamps) else None),
                    valid=False,
                    region_type=RegionType.UNKNOWN,
                    position_xy=None,
                    matched_lane_id=None,
                    matched_lane_s_m=None,
                    same_direction_lane_count=None,
                    junction_id=None,
                    junction_kind=None,
                    junction_arm_count=None,
                    junction_arm_index=None,
                    junction_side_lane_count=None,
                    distance_to_junction_m=None,
                    map_match_distance_m=None,
                    map_match_heading_error_rad=None,
                    confidence=0.0,
                    reason="missing_sdc_or_vehicle_lane_map",
                )
                for index in selected_indices
            )

        previous_lane_id = None
        annotations = []
        for frame_index in selected_indices:
            timestamp = timestamps[frame_index] if frame_index < len(timestamps) else None
            if frame_index >= len(states) or not states[frame_index].valid:
                annotations.append(
                    EgoFrameAnnotation(
                        frame_index=frame_index,
                        timestamp_seconds=timestamp,
                        valid=False,
                        region_type=RegionType.UNKNOWN,
                        position_xy=None,
                        matched_lane_id=None,
                        matched_lane_s_m=None,
                        same_direction_lane_count=None,
                        junction_id=None,
                        junction_kind=None,
                        junction_arm_count=None,
                        junction_arm_index=None,
                        junction_side_lane_count=None,
                        distance_to_junction_m=None,
                        map_match_distance_m=None,
                        map_match_heading_error_rad=None,
                        confidence=0.0,
                        reason="invalid_sdc_state",
                    )
                )
                continue

            state = states[frame_index]
            match = self.lane_index.match(state, previous_lane_id)
            position_xy = (float(state.center_x), float(state.center_y))
            if not match.confident:
                annotations.append(
                    EgoFrameAnnotation(
                        frame_index=frame_index,
                        timestamp_seconds=timestamp,
                        valid=True,
                        region_type=RegionType.UNKNOWN,
                        position_xy=position_xy,
                        matched_lane_id=match.lane_id,
                        matched_lane_s_m=match.lane_s_m,
                        same_direction_lane_count=None,
                        junction_id=None,
                        junction_kind=None,
                        junction_arm_count=None,
                        junction_arm_index=None,
                        junction_side_lane_count=None,
                        distance_to_junction_m=None,
                        map_match_distance_m=match.distance_m,
                        map_match_heading_error_rad=match.heading_error_rad,
                        confidence=0.0,
                        reason="low_confidence_map_match",
                    )
                )
                previous_lane_id = None
                continue

            previous_lane_id = match.lane_id
            region_type, junction, arm_index, distance = self._classify(match)
            inside_junction_core = bool(
                junction is not None
                and match.lane_id in junction.core_lane_ids
            )
            local_same_direction_lane_ids = (
                None
                if inside_junction_core
                else self.lane_index.same_direction_lane_ids(
                    match.lane_id, match.point_index
                )
            )
            arm = None if junction is None else junction.arm(arm_index)
            if arm is None:
                side_lane_ids = None
            elif (
                junction is not None
                and match.lane_id in junction._to_core
            ):
                side_lane_ids = arm.incoming_lane_ids
            elif (
                junction is not None
                and match.lane_id in junction._from_core
            ):
                side_lane_ids = arm.outgoing_lane_ids
            else:
                side_lane_ids = None
            side_lane_count = (
                None if side_lane_ids is None else len(side_lane_ids)
            )
            same_direction_lane_ids = _prefer_complete_lane_cross_section(
                local_same_direction_lane_ids,
                side_lane_ids,
            )
            lane_count = (
                None
                if same_direction_lane_ids is None
                else len(same_direction_lane_ids)
            )
            road_environment = self.road_environment_classifier.classify(
                match,
                state,
                frame_index,
                region_type,
                None if junction is None else junction.kind,
                lane_count,
            )
            road_environment_lane_count = lane_count
            if (
                road_environment_lane_count is None
                and road_environment.environment
                in {
                    RoadEnvironment.FREEWAY,
                    RoadEnvironment.PARKING_LOT,
                }
            ):
                road_environment_lane_count = len(
                    self.lane_index.same_direction_lane_ids(
                        match.lane_id,
                        match.point_index,
                    )
                )
            annotation_confidence = match.confidence * (
                1.0 if junction is None else junction.confidence
            )
            annotations.append(
                EgoFrameAnnotation(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp,
                    valid=True,
                    region_type=region_type,
                    position_xy=position_xy,
                    matched_lane_id=match.lane_id,
                    matched_lane_s_m=match.lane_s_m,
                    same_direction_lane_count=lane_count,
                    junction_id=None if junction is None else junction.junction_id,
                    junction_kind=None if junction is None else junction.kind,
                    junction_arm_count=None if junction is None else junction.arm_count,
                    junction_arm_index=arm_index,
                    junction_side_lane_count=side_lane_count,
                    distance_to_junction_m=distance,
                    map_match_distance_m=match.distance_m,
                    map_match_heading_error_rad=match.heading_error_rad,
                    confidence=annotation_confidence,
                    same_direction_lane_ids=same_direction_lane_ids,
                    road_environment=road_environment.environment,
                    road_environment_subtype=road_environment.subtype,
                    road_environment_lane_count=road_environment_lane_count,
                    road_environment_confidence=(
                        road_environment.confidence
                    ),
                    road_environment_reason=road_environment.reason,
                    road_environment_subtype_reason=(
                        road_environment.subtype_reason
                    ),
                    matched_lane_type=road_environment.matched_lane_type,
                    matched_lane_speed_limit_mph=(
                        road_environment.matched_lane_speed_limit_mph
                    ),
                )
            )
        return tuple(annotations)

    def annotate(
        self,
        *,
        scenario_index: int | None = None,
        source_file: str | None = None,
        frame_indices: Iterable[int] | None = None,
    ) -> ScenarioMapAnnotation:
        return ScenarioMapAnnotation(
            scenario_id=self.scenario.scenario_id,
            scenario_index=scenario_index,
            source_file=source_file,
            current_time_index=self.scenario.current_time_index,
            junctions=self.junctions,
            ego_frames=self.annotate_ego_frames(frame_indices),
        )


def annotate_scenario(
    scenario,
    config: MapAnnotationConfig | None = None,
    *,
    scenario_index: int | None = None,
    source_file: str | None = None,
    frame_indices: Iterable[int] | None = None,
) -> ScenarioMapAnnotation:
    return EgoMapAnnotator(scenario, config).annotate(
        scenario_index=scenario_index,
        source_file=source_file,
        frame_indices=frame_indices,
    )
