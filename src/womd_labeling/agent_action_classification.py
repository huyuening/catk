"""Current-frame agent action labels adapted from D2S ``get_action.py``."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .agent_size_classification import object_type_label_zh, object_type_name


ACTION_NAMES = {
    0: "UNSET",
    1: "STOP",
    2: "U_TURN",
    3: "LEFT_TURN",
    4: "LEFT_LANE_CHANGE",
    5: "DECELERATE",
    6: "KEEP_SPEED",
    7: "ACCELERATE",
    8: "RIGHT_LANE_CHANGE",
    9: "RIGHT_TURN",
}

ACTION_LABELS_ZH = {
    0: "未标注",
    1: "停车",
    2: "掉头",
    3: "左转",
    4: "向左变道",
    5: "减速",
    6: "保持速度",
    7: "加速",
    8: "向右变道",
    9: "右转",
}


@dataclass(frozen=True)
class AgentActionConfig:
    """Thresholds retained from the reference D2S action labeler."""

    stop_speed_mps: float = 0.2
    acceleration_threshold_mps2: float = 0.5
    deceleration_threshold_mps2: float = -0.5
    turn_heading_diff_rad: float = 0.25
    u_turn_heading_change_rad: float = math.radians(160.0)
    valid_lookaround_frames: int = 10
    u_turn_lookahead_frames: int = 30
    lane_change_monotonic_frames: int = 8
    lane_change_start_offset_m: float = 0.75
    lane_change_crossing_offset_m: float = 1.25
    lane_change_end_heading_rad: float = math.radians(5.0)
    lane_change_end_offset_m: float = 0.55

    def __post_init__(self) -> None:
        positive = (
            "stop_speed_mps",
            "acceleration_threshold_mps2",
            "turn_heading_diff_rad",
            "u_turn_heading_change_rad",
            "lane_change_start_offset_m",
            "lane_change_crossing_offset_m",
            "lane_change_end_heading_rad",
            "lane_change_end_offset_m",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if (
            not math.isfinite(self.deceleration_threshold_mps2)
            or self.deceleration_threshold_mps2 >= 0
        ):
            raise ValueError(
                "deceleration_threshold_mps2 must be a negative finite number"
            )
        for name in (
            "valid_lookaround_frames",
            "u_turn_lookahead_frames",
            "lane_change_monotonic_frames",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")


def normalize_angle(angle):
    """Normalize one angle or an array of angles to ``[-pi, pi)``."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def compute_signed_distance_to_lane_v1_fixed(
    trajectory: np.ndarray,
    polyline: np.ndarray,
) -> np.ndarray:
    """Mirror the nearest-vertex signed-distance helper in D2S."""
    trajectory = np.asarray(trajectory, dtype=float)
    polyline = np.asarray(polyline, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("trajectory must have shape (N, 2)")
    if polyline.ndim != 2 or polyline.shape[1] != 2 or len(polyline) < 2:
        raise ValueError("polyline must have shape (M, 2), M >= 2")

    distances = []
    for point in trajectory:
        differences = polyline - point
        squared_distances = np.sum(differences**2, axis=1)
        closest_index = int(np.argmin(squared_distances))
        distance = float(np.sqrt(squared_distances[closest_index]))
        if closest_index < len(polyline) - 1:
            start = polyline[closest_index]
            end = polyline[closest_index + 1]
        else:
            start = polyline[closest_index - 1]
            end = polyline[closest_index]
        lane_vector = end - start
        point_vector = point - start
        cross_product = (
            lane_vector[0] * point_vector[1]
            - lane_vector[1] * point_vector[0]
        )
        distances.append(distance if cross_product >= 0 else -distance)
    return np.asarray(distances, dtype=float)


class LaneVertexIndex:
    """Accelerate the reference nearest-lane-vertex calculation."""

    def __init__(self, scenario) -> None:
        self.polylines: dict[int, np.ndarray] = {}
        vertices = []
        lane_ids = []
        tangent_starts = []
        tangent_ends = []

        for feature in scenario.map_features:
            if feature.WhichOneof("feature_data") != "lane":
                continue
            polyline = np.asarray(
                [(point.x, point.y) for point in feature.lane.polyline],
                dtype=float,
            )
            if (
                polyline.shape != (len(feature.lane.polyline), 2)
                or len(polyline) < 2
                or not np.all(np.isfinite(polyline))
            ):
                continue
            lane_id = int(feature.id)
            self.polylines[lane_id] = polyline
            for point_index, point in enumerate(polyline):
                if point_index < len(polyline) - 1:
                    start = polyline[point_index]
                    end = polyline[point_index + 1]
                else:
                    start = polyline[point_index - 1]
                    end = polyline[point_index]
                vertices.append(point)
                lane_ids.append(lane_id)
                tangent_starts.append(start)
                tangent_ends.append(end)

        self.vertices = np.asarray(vertices, dtype=float).reshape(-1, 2)
        self.lane_ids = np.asarray(lane_ids, dtype=np.int64)
        self.tangent_starts = np.asarray(tangent_starts, dtype=float).reshape(-1, 2)
        self.tangent_ends = np.asarray(tangent_ends, dtype=float).reshape(-1, 2)
        self.tree = cKDTree(self.vertices) if len(self.vertices) else None

    @property
    def lane_count(self) -> int:
        return len(self.polylines)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    def _resolve_tied_vertex(
        self,
        point: np.ndarray,
        nearest_distance: float,
        nearest_index: int,
    ) -> int:
        tolerance = max(1e-10, abs(nearest_distance) * 1e-12)
        candidates = self.tree.query_ball_point(
            point,
            nearest_distance + tolerance,
        )
        if len(candidates) <= 1:
            return nearest_index
        candidate_indices = np.asarray(candidates, dtype=int)
        squared = np.sum(
            (self.vertices[candidate_indices] - point) ** 2,
            axis=1,
        )
        minimum = float(np.min(squared))
        tied = candidate_indices[
            np.isclose(squared, minimum, rtol=1e-12, atol=1e-12)
        ]
        return int(np.min(tied))

    def closest_signed_distances(
        self,
        trajectory: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the closest lane's signed distance and id per point."""
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError("trajectory must have shape (N, 2)")
        if self.tree is None:
            return (
                np.empty(len(trajectory), dtype=float),
                np.empty(len(trajectory), dtype=np.int64),
            )

        nearest_distances, nearest_indices = self.tree.query(trajectory, k=1)
        nearest_distances = np.atleast_1d(nearest_distances).astype(float)
        nearest_indices = np.atleast_1d(nearest_indices).astype(int)
        for row, point in enumerate(trajectory):
            nearest_indices[row] = self._resolve_tied_vertex(
                point,
                float(nearest_distances[row]),
                int(nearest_indices[row]),
            )

        starts = self.tangent_starts[nearest_indices]
        lane_vectors = self.tangent_ends[nearest_indices] - starts
        point_vectors = trajectory - starts
        cross_products = (
            lane_vectors[:, 0] * point_vectors[:, 1]
            - lane_vectors[:, 1] * point_vectors[:, 0]
        )
        exact_distances = np.linalg.norm(
            self.vertices[nearest_indices] - trajectory,
            axis=1,
        )
        signed_distances = np.where(
            cross_products >= 0,
            exact_distances,
            -exact_distances,
        )
        return signed_distances, self.lane_ids[nearest_indices]


@dataclass(frozen=True)
class _TrackKinematics:
    valid_indices: np.ndarray
    positions_xy: np.ndarray
    headings: np.ndarray
    longitudinal_velocity_mps: np.ndarray
    absolute_speed_mps: np.ndarray
    longitudinal_acceleration_mps2: np.ndarray
    acceleration_fallback: bool


def _track_kinematics(track, timestamps_seconds) -> _TrackKinematics:
    state_count = len(track.states)
    valid_indices = np.asarray(
        [index for index, state in enumerate(track.states) if state.valid],
        dtype=int,
    )
    positions = np.asarray(
        [
            (track.states[index].center_x, track.states[index].center_y)
            for index in valid_indices
        ],
        dtype=float,
    ).reshape(-1, 2)
    headings = np.asarray(
        [track.states[index].heading for index in valid_indices],
        dtype=float,
    )
    velocity_x = np.asarray(
        [track.states[index].velocity_x for index in valid_indices],
        dtype=float,
    )
    velocity_y = np.asarray(
        [track.states[index].velocity_y for index in valid_indices],
        dtype=float,
    )
    longitudinal_velocity = (
        velocity_x * np.cos(headings) + velocity_y * np.sin(headings)
    )
    absolute_speed = np.hypot(velocity_x, velocity_y)

    fallback = False
    if len(valid_indices) < 2:
        acceleration_x = np.zeros(len(valid_indices), dtype=float)
        acceleration_y = np.zeros(len(valid_indices), dtype=float)
        fallback = True
    else:
        timestamps = np.asarray(timestamps_seconds, dtype=float)
        if len(timestamps) >= state_count:
            valid_times = timestamps[valid_indices]
        else:
            valid_times = valid_indices.astype(float) * 0.1
            fallback = True
        if (
            not np.all(np.isfinite(valid_times))
            or np.any(np.diff(valid_times) <= 0)
        ):
            valid_times = valid_indices.astype(float) * 0.1
            fallback = True
        acceleration_x = np.gradient(velocity_x, valid_times)
        acceleration_y = np.gradient(velocity_y, valid_times)

    longitudinal_acceleration = (
        acceleration_x * np.cos(headings)
        + acceleration_y * np.sin(headings)
    )
    return _TrackKinematics(
        valid_indices=valid_indices,
        positions_xy=positions,
        headings=headings,
        longitudinal_velocity_mps=longitudinal_velocity,
        absolute_speed_mps=absolute_speed,
        longitudinal_acceleration_mps2=longitudinal_acceleration,
        acceleration_fallback=fallback,
    )


def _find_lane_change_end_index_local(
    crossing_index_local: int,
    kinematics: _TrackKinematics,
    closest_signed_distances: np.ndarray,
    closest_lane_ids: np.ndarray,
    lane_index: LaneVertexIndex,
    config: AgentActionConfig,
) -> int | None:
    for local_index in range(
        crossing_index_local + 1,
        len(kinematics.valid_indices),
    ):
        heading = kinematics.headings[local_index]
        vehicle_vector = np.asarray([np.cos(heading), np.sin(heading)])
        lane_id = int(closest_lane_ids[local_index])
        polyline = lane_index.polylines.get(lane_id)
        if polyline is None or len(polyline) < 2:
            continue

        position = kinematics.positions_xy[local_index]
        closest_index = int(np.argmin(np.linalg.norm(polyline - position, axis=1)))
        if closest_index >= len(polyline) - 1:
            closest_index = max(0, len(polyline) - 2)
        lane_vector = polyline[closest_index + 1] - polyline[closest_index]
        lane_vector = lane_vector / (np.linalg.norm(lane_vector) + 1e-6)
        angle_difference = np.arccos(
            np.clip(float(np.dot(vehicle_vector, lane_vector)), -1.0, 1.0)
        )
        lateral_offset = closest_signed_distances[local_index]
        if (
            angle_difference < config.lane_change_end_heading_rad
            and abs(lateral_offset) < config.lane_change_end_offset_m
        ):
            return local_index
    return None


def detect_vehicle_lane_changes(
    track,
    lane_index: LaneVertexIndex,
    timestamps_seconds=(),
    config: AgentActionConfig | None = None,
) -> list[tuple[int, int, str]]:
    """Detect global-frame lane-change intervals for one vehicle track."""
    config = config or AgentActionConfig()
    if int(track.object_type) != 1 or lane_index.tree is None:
        return []
    kinematics = _track_kinematics(track, timestamps_seconds)
    valid_indices = kinematics.valid_indices
    if len(valid_indices) < 2:
        return []

    closest_distances, closest_lane_ids = lane_index.closest_signed_distances(
        kinematics.positions_xy
    )
    differences = np.diff(closest_distances)
    extrema_indices = np.where(
        differences[:-1] * differences[1:] < 0
    )[0] + 1
    extrema_indices = np.insert(extrema_indices, 0, 0)
    events = []

    window = config.lane_change_monotonic_frames
    for candidate in extrema_indices:
        candidate = int(candidate)
        if candidate + window >= len(closest_distances):
            continue
        segment_differences = np.diff(
            closest_distances[candidate : candidate + window + 1]
        )
        direction = None
        end_of_segment = None
        future_differences = np.diff(closest_distances[candidate:])
        if np.all(segment_differences > 0):
            non_increasing = np.where(future_differences <= 0)[0]
            if len(non_increasing):
                end_of_segment = candidate + int(non_increasing[0])
                direction = "left"
        elif np.all(segment_differences < 0):
            non_decreasing = np.where(future_differences >= 0)[0]
            if len(non_decreasing):
                end_of_segment = candidate + int(non_decreasing[0])
                direction = "right"
        if direction is None or end_of_segment is None:
            continue
        if abs(closest_distances[candidate]) > config.lane_change_start_offset_m:
            continue
        if end_of_segment + 1 >= len(closest_distances):
            continue
        before_crossing = closest_distances[end_of_segment]
        after_crossing = closest_distances[end_of_segment + 1]
        if before_crossing * after_crossing >= 0:
            continue
        if (
            abs(before_crossing) < config.lane_change_crossing_offset_m
            or abs(after_crossing) < config.lane_change_crossing_offset_m
        ):
            continue

        crossing_index = end_of_segment + 1
        end_index = _find_lane_change_end_index_local(
            crossing_index,
            kinematics,
            closest_distances,
            closest_lane_ids,
            lane_index,
            config,
        )
        if end_index is not None:
            events.append(
                (
                    int(valid_indices[candidate]),
                    int(valid_indices[end_index]),
                    direction,
                )
            )
    return events


def _classify_current_action(
    object_type: int,
    frame_index: int,
    local_index: int,
    kinematics: _TrackKinematics,
    lane_change_intervals: list[tuple[int, int, str]],
    config: AgentActionConfig,
) -> tuple[int, str, tuple[int, int, str] | None, dict[str, int]]:
    speed = abs(float(kinematics.longitudinal_velocity_mps[local_index]))
    window_indices = {
        "past_valid_frame_index": int(
            kinematics.valid_indices[
                max(local_index - config.valid_lookaround_frames, 0)
            ]
        ),
        "future_valid_frame_index": int(
            kinematics.valid_indices[
                min(
                    local_index + config.valid_lookaround_frames,
                    len(kinematics.valid_indices) - 1,
                )
            ]
        ),
        "future_long_valid_frame_index": int(
            kinematics.valid_indices[
                min(
                    local_index + config.u_turn_lookahead_frames,
                    len(kinematics.valid_indices) - 1,
                )
            ]
        ),
    }

    if object_type in (1, 3) and speed < config.stop_speed_mps:
        return 1, "stop_speed_threshold", None, window_indices
    if object_type == 2 and speed < config.stop_speed_mps / 2.0:
        return 1, "pedestrian_stop_speed_threshold", None, window_indices

    if object_type == 1:
        for interval in lane_change_intervals:
            start, end, direction = interval
            if start <= frame_index <= end:
                action_id = 4 if direction == "left" else 8
                return action_id, f"{direction}_lane_change_interval", interval, window_indices

    if object_type in (1, 3):
        current_heading = kinematics.headings[local_index]
        past_heading = kinematics.headings[
            max(local_index - config.valid_lookaround_frames, 0)
        ]
        future_heading = kinematics.headings[
            min(
                local_index + config.valid_lookaround_frames,
                len(kinematics.valid_indices) - 1,
            )
        ]
        future_difference = float(normalize_angle(future_heading - current_heading))
        past_difference = float(normalize_angle(current_heading - past_heading))
        if object_type == 1:
            future_long_heading = kinematics.headings[
                min(
                    local_index + config.u_turn_lookahead_frames,
                    len(kinematics.valid_indices) - 1,
                )
            ]
            total_change = abs(
                float(normalize_angle(future_long_heading - past_heading))
            )
            if total_change > config.u_turn_heading_change_rad:
                return 2, "u_turn_heading_change", None, window_indices
        if (
            future_difference > config.turn_heading_diff_rad
            or past_difference > config.turn_heading_diff_rad
        ):
            return 3, "left_turn_heading_change", None, window_indices
        if (
            future_difference < -config.turn_heading_diff_rad
            or past_difference < -config.turn_heading_diff_rad
        ):
            return 9, "right_turn_heading_change", None, window_indices

    acceleration = float(
        kinematics.longitudinal_acceleration_mps2[local_index]
    )
    if object_type in (1, 3):
        if acceleration > config.acceleration_threshold_mps2:
            return 7, "longitudinal_acceleration_threshold", None, window_indices
        if acceleration < config.deceleration_threshold_mps2:
            return 5, "longitudinal_deceleration_threshold", None, window_indices
        return 6, "longitudinal_acceleration_deadband", None, window_indices
    if object_type == 2:
        if acceleration > config.acceleration_threshold_mps2 / 2.0:
            return 7, "pedestrian_acceleration_threshold", None, window_indices
        if acceleration < config.deceleration_threshold_mps2 / 2.0:
            return 5, "pedestrian_deceleration_threshold", None, window_indices
        return 6, "pedestrian_acceleration_deadband", None, window_indices
    return 0, "unsupported_object_type", None, window_indices


def label_scenario_actions(
    scenario,
    frame_index: int | None = None,
    config: AgentActionConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Label all valid agent frames, or one requested frame when specified."""
    if frame_index is not None and frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    config = config or AgentActionConfig()
    lane_index = LaneVertexIndex(scenario)
    diagnostics: Counter[str] = Counter(
        total_tracks=len(scenario.tracks),
        lane_center_count=lane_index.lane_count,
        lane_vertex_count=lane_index.vertex_count,
    )
    records = []

    for track_index, track in enumerate(scenario.tracks):
        type_name = object_type_name(track.object_type)
        kinematics = _track_kinematics(track, scenario.timestamps_seconds)
        if frame_index is None:
            local_indices = range(len(kinematics.valid_indices))
            diagnostics["invalid_state_frames"] += (
                len(track.states) - len(kinematics.valid_indices)
            )
            diagnostics[f"invalid_state_frames:{type_name}"] += (
                len(track.states) - len(kinematics.valid_indices)
            )
        else:
            if frame_index >= len(track.states):
                diagnostics["missing_state"] += 1
                diagnostics[f"missing_state:{type_name}"] += 1
                continue
            if not track.states[frame_index].valid:
                diagnostics["invalid_state"] += 1
                diagnostics[f"invalid_state:{type_name}"] += 1
                continue
            matches = np.flatnonzero(kinematics.valid_indices == frame_index)
            if len(matches) != 1:
                diagnostics["kinematics_lookup_failure"] += 1
                continue
            local_indices = (int(matches[0]),)

        if not len(kinematics.valid_indices):
            diagnostics["tracks_without_valid_states"] += 1
            continue
        diagnostics["tracks_with_valid_states"] += 1
        diagnostics[f"tracks_with_valid_states:{type_name}"] += 1
        if kinematics.acceleration_fallback:
            diagnostics["acceleration_fallback"] += 1

        lane_change_intervals = detect_vehicle_lane_changes(
            track,
            lane_index,
            scenario.timestamps_seconds,
            config,
        )
        diagnostics["lane_change_events_detected"] += len(lane_change_intervals)
        for local_index in local_indices:
            current_frame_index = int(kinematics.valid_indices[local_index])
            diagnostics["valid_state_frames"] += 1
            diagnostics[f"valid_state_frames:{type_name}"] += 1
            action_id, reason, matched_interval, window_indices = (
                _classify_current_action(
                    int(track.object_type),
                    current_frame_index,
                    local_index,
                    kinematics,
                    lane_change_intervals,
                    config,
                )
            )
            action_name = ACTION_NAMES[action_id]
            diagnostics[f"action_frames:{action_name}"] += 1
            diagnostics[
                "action_labeled_frames" if action_id else "action_unset_frames"
            ] += 1

            lane_change_start = None
            lane_change_end = None
            lane_change_direction = None
            if matched_interval is not None:
                lane_change_start, lane_change_end, lane_change_direction = (
                    matched_interval
                )
            records.append(
                {
                    "frame_number": current_frame_index + 1,
                    "frame_index": current_frame_index,
                    "track_index": track_index,
                    "track_id": int(track.id),
                    "is_sdc": track_index == scenario.sdc_track_index,
                    "object_type": type_name,
                    "object_type_value": int(track.object_type),
                    "object_type_zh": object_type_label_zh(track.object_type),
                    "action_id": action_id,
                    "action": action_name,
                    "action_zh": ACTION_LABELS_ZH[action_id],
                    "decision_reason": reason,
                    "longitudinal_velocity_mps": float(
                        kinematics.longitudinal_velocity_mps[local_index]
                    ),
                    "absolute_speed_mps": float(
                        kinematics.absolute_speed_mps[local_index]
                    ),
                    "longitudinal_acceleration_mps2": float(
                        kinematics.longitudinal_acceleration_mps2[local_index]
                    ),
                    "valid_track_frame_count": len(kinematics.valid_indices),
                    "lane_change_start_frame_index": lane_change_start,
                    "lane_change_end_frame_index": lane_change_end,
                    "lane_change_direction": lane_change_direction,
                    **window_indices,
                }
            )
    return records, dict(diagnostics)


def encode_agent_action_key(object_type: str, action_id: int) -> str:
    return f"{object_type}\t{int(action_id)}"


def decode_agent_action_key(key: str) -> tuple[str, int]:
    object_type, action_id = key.split("\t", 1)
    return object_type, int(action_id)


def encode_agent_action_frame_key(
    frame_index: int,
    object_type: str,
    action_id: int,
) -> str:
    return f"{int(frame_index)}\t{object_type}\t{int(action_id)}"


def decode_agent_action_frame_key(key: str) -> tuple[int, str, int]:
    frame_index, object_type, action_id = key.split("\t", 2)
    return int(frame_index), object_type, int(action_id)
