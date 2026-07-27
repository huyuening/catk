from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np


DEFAULT_MAP_FRAME_INDEX = 10


@dataclass(frozen=True)
class MapVisualizationConfig:
    x_min_m: float = -50.0
    x_max_m: float = 100.0
    y_min_m: float = -60.0
    y_max_m: float = 60.0
    dpi: int = 100
    show_agent_ids: bool = False

    def __post_init__(self) -> None:
        if self.x_min_m >= self.x_max_m:
            raise ValueError("x_min_m must be smaller than x_max_m")
        if self.y_min_m >= self.y_max_m:
            raise ValueError("y_min_m must be smaller than y_max_m")
        if self.dpi < 72:
            raise ValueError("dpi must be at least 72")


@dataclass(frozen=True)
class RenderedMapAnnotation:
    frame_index: int
    region_type: str
    summary: str
    output_path: Path


REGION_LABELS_ZH = {
    "ROAD_SEGMENT": "路段",
    "INTERSECTION": "路口",
    "NEAR_INTERSECTION_APPROACH": "路口附近（驶入）",
    "IN_INTERSECTION": "路口内",
    "NEAR_INTERSECTION_EXIT": "路口附近（驶离）",
    "UNKNOWN": "未知",
}

REGION_LABELS_EN = {
    "ROAD_SEGMENT": "Road segment",
    "INTERSECTION": "Intersection",
    "NEAR_INTERSECTION_APPROACH": "Near intersection (approach)",
    "IN_INTERSECTION": "Inside intersection",
    "NEAR_INTERSECTION_EXIT": "Near intersection (exit)",
    "UNKNOWN": "Unknown",
}

JUNCTION_CONTROL_LABELS_ZH = {
    "signalized": "信号灯控制",
    "stop_controlled": "停车标志控制",
    "roundabout": "无控制",
    "geometric": "未观测到控制信息",
}

JUNCTION_CONTROL_LABELS_EN = {
    "signalized": "Signalized",
    "stop_controlled": "Stop controlled",
    "roundabout": "Uncontrolled",
    "geometric": "No observed control",
}

ROAD_ENVIRONMENT_LABELS_ZH = {
    "FREEWAY": "高速公路",
    "URBAN_STREET": "城市道路",
    "PARKING_LOT": "停车场",
    "UNKNOWN": "未知",
}

ROAD_ENVIRONMENT_LABELS_EN = {
    "FREEWAY": "Freeway",
    "URBAN_STREET": "Urban street",
    "PARKING_LOT": "Parking lot",
    "UNKNOWN": "Unknown",
}

ROAD_ENVIRONMENT_SUBTYPE_LABELS_ZH = {
    "FREEWAY_MAINLINE": "主线",
    "FREEWAY_RAMP": "匝道",
}

ROAD_ENVIRONMENT_SUBTYPE_LABELS_EN = {
    "FREEWAY_MAINLINE": "mainline",
    "FREEWAY_RAMP": "ramp",
}


def world_to_ego(
    points_xy: np.ndarray | list[tuple[float, float]],
    origin_xy: tuple[float, float],
    ego_heading_rad: float,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.size == 0:
        return points.reshape((-1, 2))
    points = np.atleast_2d(points)
    delta = points - np.asarray(origin_xy, dtype=np.float64)
    cosine = math.cos(ego_heading_rad)
    sine = math.sin(ego_heading_rad)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]])
    return delta @ rotation.T


def _near_lane_count(frame: dict[str, Any]) -> int | None:
    same_direction_lane_count = frame.get("same_direction_lane_count")
    if same_direction_lane_count is not None:
        return same_direction_lane_count
    return frame.get("junction_side_lane_count")


def format_region_summary(frame: dict[str, Any], language: str = "zh") -> str:
    region = frame.get("region_type", "UNKNOWN")
    junction_kind = frame.get("junction_kind")
    lane_count = _near_lane_count(frame)
    arm_count = frame.get("junction_arm_count")
    if language == "zh":
        lane_text = "未知车道数" if lane_count is None else f"{lane_count} 车道"
        arm_text = "未知支路数" if arm_count is None else f"{arm_count} 支路口"
        if region == "ROAD_SEGMENT":
            return f"主车位置：路段（{lane_text}）"
        if region in {"INTERSECTION", "IN_INTERSECTION"}:
            if junction_kind == "roundabout":
                return "主车位置：路口（环形路口）"
            return f"主车位置：路口（{arm_text}）"
        if region in {
            "NEAR_INTERSECTION_APPROACH",
            "NEAR_INTERSECTION_EXIT",
        }:
            return f"主车位置：路口附近（{lane_text} + {arm_text}）"
        return "主车位置：未知"

    lane_text = "unknown lane count" if lane_count is None else f"{lane_count} lanes"
    arm_text = "unknown arm count" if arm_count is None else f"{arm_count}-arm"
    if region == "ROAD_SEGMENT":
        return f"Ego location: road segment ({lane_text})"
    if region in {"INTERSECTION", "IN_INTERSECTION"}:
        if junction_kind == "roundabout":
            return "Ego location: intersection (roundabout)"
        return f"Ego location: intersection ({arm_text})"
    if region in {
        "NEAR_INTERSECTION_APPROACH",
        "NEAR_INTERSECTION_EXIT",
    }:
        return f"Ego location: near intersection ({lane_text} + {arm_text})"
    return "Ego location: unknown"


def format_road_environment_summary(
    frame: dict[str, Any],
    language: str = "zh",
) -> str:
    environment = frame.get("road_environment", "UNKNOWN")
    subtype = frame.get("road_environment_subtype")
    lane_count = frame.get("road_environment_lane_count")
    if lane_count is None:
        lane_count = _near_lane_count(frame)
    if language == "zh":
        label = ROAD_ENVIRONMENT_LABELS_ZH.get(environment, "未知")
        lane_text = "未知车道数" if lane_count is None else f"{lane_count} 车道"
        subtype_label = ROAD_ENVIRONMENT_SUBTYPE_LABELS_ZH.get(subtype)
        if subtype_label is not None:
            return f"道路环境：{label}（{subtype_label}，{lane_text}）"
        return f"道路环境：{label}（{lane_text}）"
    label = ROAD_ENVIRONMENT_LABELS_EN.get(environment, "Unknown")
    lane_text = "unknown lane count" if lane_count is None else f"{lane_count} lanes"
    subtype_label = ROAD_ENVIRONMENT_SUBTYPE_LABELS_EN.get(subtype)
    if subtype_label is not None:
        return f"Road environment: {label} ({subtype_label}, {lane_text})"
    return f"Road environment: {label} ({lane_text})"


def select_render_frame(
    scenario,
    annotation: dict[str, Any],
    preferred_frame_index: int = DEFAULT_MAP_FRAME_INDEX,
) -> tuple[int, dict[str, Any]]:
    frames = {
        int(frame["frame_index"]): frame
        for frame in annotation.get("ego_frames", [])
    }
    if not frames:
        raise ValueError("Annotation does not contain ego_frames")
    if not 0 <= scenario.sdc_track_index < len(scenario.tracks):
        raise ValueError("Scenario does not contain a valid SDC track")

    states = scenario.tracks[scenario.sdc_track_index].states

    def usable(frame_index: int) -> bool:
        return (
            frame_index in frames
            and 0 <= frame_index < len(states)
            and states[frame_index].valid
        )

    if usable(preferred_frame_index):
        return preferred_frame_index, frames[preferred_frame_index]
    for frame_index in sorted(frames):
        if usable(frame_index):
            return frame_index, frames[frame_index]
    raise ValueError("Scenario does not contain a valid SDC state to render")


@lru_cache(maxsize=1)
def _cjk_font_name() -> str | None:
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in (
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Songti SC",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ):
        if candidate in available:
            return candidate
    return None


def _font_kwargs() -> dict[str, str]:
    font_name = _cjk_font_name()
    return {} if font_name is None else {"fontfamily": font_name}


def _points_xy(points) -> np.ndarray:
    return np.asarray([(point.x, point.y) for point in points], dtype=np.float64)


def _visible(points: np.ndarray, config: MapVisualizationConfig) -> bool:
    if len(points) == 0:
        return False
    margin = 5.0
    return not (
        np.max(points[:, 0]) < config.x_min_m - margin
        or np.min(points[:, 0]) > config.x_max_m + margin
        or np.max(points[:, 1]) < config.y_min_m - margin
        or np.min(points[:, 1]) > config.y_max_m + margin
    )


def _direction_arrow_vector(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 2:
        return None
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length < 1e-6:
        return None
    midpoint_distance = total_length * 0.5
    cumulative = np.cumsum(segment_lengths)
    segment_index = int(np.searchsorted(cumulative, midpoint_distance))
    segment_index = min(segment_index, len(points) - 2)
    start = points[segment_index]
    end = points[segment_index + 1]
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None
    direction /= length
    center = (start + end) * 0.5
    arrow_length = min(6.0, max(2.4, length * 0.8))
    return center, direction * arrow_length


def _lane_endpoint_direction(
    points: np.ndarray,
    *,
    at_end: bool,
) -> np.ndarray | None:
    differences = np.diff(points, axis=0)
    indices = (
        range(len(differences) - 1, -1, -1)
        if at_end
        else range(len(differences))
    )
    for index in indices:
        length = float(np.linalg.norm(differences[index]))
        if length > 1e-6:
            return differences[index] / length
    return None


def _lanes_form_continuation(
    upstream,
    downstream,
    *,
    max_join_distance_m: float = 8.0,
    max_heading_change_rad: float = math.radians(45.0),
) -> bool:
    upstream_points = _points_xy(upstream.lane.polyline)
    downstream_points = _points_xy(downstream.lane.polyline)
    if len(upstream_points) < 2 or len(downstream_points) < 2:
        return False
    if (
        float(np.linalg.norm(upstream_points[-1] - downstream_points[0]))
        > max_join_distance_m
    ):
        return False

    upstream_direction = _lane_endpoint_direction(upstream_points, at_end=True)
    downstream_direction = _lane_endpoint_direction(
        downstream_points, at_end=False
    )
    if upstream_direction is None or downstream_direction is None:
        return False
    cosine = float(
        np.clip(np.dot(upstream_direction, downstream_direction), -1.0, 1.0)
    )
    return math.acos(cosine) <= max_heading_change_rad


def _ego_lane_chain_ids(
    scenario,
    matched_lane_id: int | None,
) -> tuple[int, ...]:
    """Return the unambiguous topological lane chain containing the ego lane."""
    if matched_lane_id is None:
        return ()

    lanes = {
        int(feature.id): feature
        for feature in scenario.map_features
        if feature.WhichOneof("feature_data") == "lane"
    }
    matched_lane_id = int(matched_lane_id)
    if matched_lane_id not in lanes:
        return ()

    predecessors: list[int] = []
    visited = {matched_lane_id}
    current_id = matched_lane_id
    while True:
        entry_ids = [
            int(lane_id)
            for lane_id in lanes[current_id].lane.entry_lanes
            if int(lane_id) in lanes
        ]
        if len(entry_ids) != 1:
            break
        predecessor_id = entry_ids[0]
        predecessor = lanes[predecessor_id]
        predecessor_exits = [
            int(lane_id)
            for lane_id in predecessor.lane.exit_lanes
            if int(lane_id) in lanes
        ]
        if (
            predecessor_id in visited
            or predecessor_exits != [current_id]
            or not _lanes_form_continuation(predecessor, lanes[current_id])
        ):
            break
        predecessors.append(predecessor_id)
        visited.add(predecessor_id)
        current_id = predecessor_id

    successors: list[int] = []
    current_id = matched_lane_id
    while True:
        exit_ids = [
            int(lane_id)
            for lane_id in lanes[current_id].lane.exit_lanes
            if int(lane_id) in lanes
        ]
        if len(exit_ids) != 1:
            break
        successor_id = exit_ids[0]
        successor = lanes[successor_id]
        successor_entries = [
            int(lane_id)
            for lane_id in successor.lane.entry_lanes
            if int(lane_id) in lanes
        ]
        if (
            successor_id in visited
            or successor_entries != [current_id]
            or not _lanes_form_continuation(lanes[current_id], successor)
        ):
            break
        successors.append(successor_id)
        visited.add(successor_id)
        current_id = successor_id

    return tuple(reversed(predecessors)) + (matched_lane_id,) + tuple(successors)


def _draw_map_features(
    ax,
    scenario,
    origin_xy: tuple[float, float],
    ego_heading_rad: float,
    config: MapVisualizationConfig,
) -> dict[int, np.ndarray]:
    lanes: dict[int, np.ndarray] = {}
    line_groups: dict[str, list[np.ndarray]] = {
        "lanes": [],
        "bike_lanes": [],
        "road_edges": [],
        "white_solid": [],
        "white_broken": [],
        "yellow_solid": [],
        "yellow_broken": [],
    }
    for feature in scenario.map_features:
        feature_type = feature.WhichOneof("feature_data")
        if feature_type == "lane":
            points = world_to_ego(
                _points_xy(feature.lane.polyline), origin_xy, ego_heading_rad
            )
            if len(points) < 2 or not _visible(points, config):
                continue
            lanes[int(feature.id)] = points
            if feature.lane.type == 3:
                line_groups["bike_lanes"].append(points)
            else:
                line_groups["lanes"].append(points)
        elif feature_type == "road_edge":
            points = world_to_ego(
                _points_xy(feature.road_edge.polyline), origin_xy, ego_heading_rad
            )
            if len(points) >= 2 and _visible(points, config):
                line_groups["road_edges"].append(points)
        elif feature_type == "road_line":
            points = world_to_ego(
                _points_xy(feature.road_line.polyline), origin_xy, ego_heading_rad
            )
            if len(points) < 2 or not _visible(points, config):
                continue
            road_line_type = int(feature.road_line.type)
            color_group = "yellow" if road_line_type >= 4 else "white"
            style_group = "broken" if road_line_type in {1, 4, 5} else "solid"
            line_groups[f"{color_group}_{style_group}"].append(points)
        elif feature_type in {"crosswalk", "speed_bump", "driveway"}:
            polygon_points = getattr(feature, feature_type).polygon
            points = world_to_ego(
                _points_xy(polygon_points), origin_xy, ego_heading_rad
            )
            if len(points) < 3 or not _visible(points, config):
                continue
            if feature_type == "crosswalk":
                face_color, edge_color, alpha = "#D8E5E5", "#7D9698", 0.7
            elif feature_type == "speed_bump":
                face_color, edge_color, alpha = "#F2D58B", "#B78B27", 0.55
            else:
                face_color, edge_color, alpha = "#E7EBEB", "#B6BDBE", 0.35
            ax.add_patch(
                Polygon(
                    points,
                    closed=True,
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=0.8,
                    alpha=alpha,
                    zorder=1.5,
                )
            )

    collection_styles = {
        "lanes": ("#AEB8BA", 0.75, (0, (1.5, 3)), 3.0, 0.9),
        "bike_lanes": ("#79A88D", 0.9, (0, (3, 3)), 3.0, 0.9),
        "road_edges": ("#596365", 1.6, "solid", 2.8, 0.95),
        "white_solid": ("#7E898B", 1.0, "solid", 2.9, 0.95),
        "white_broken": ("#7E898B", 1.0, (0, (5, 5)), 2.9, 0.95),
        "yellow_solid": ("#C1922E", 1.0, "solid", 2.9, 0.95),
        "yellow_broken": ("#C1922E", 1.0, (0, (5, 5)), 2.9, 0.95),
    }
    for group_name, segments in line_groups.items():
        if not segments:
            continue
        color, width, linestyle, zorder, alpha = collection_styles[group_name]
        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                linewidths=width,
                linestyles=linestyle,
                alpha=alpha,
                zorder=zorder,
            )
        )
    return lanes


def _junction_by_id(
    annotation: dict[str, Any], junction_id: int | None
) -> dict[str, Any] | None:
    if junction_id is None:
        return None
    return next(
        (
            junction
            for junction in annotation.get("junctions", [])
            if int(junction["junction_id"]) == int(junction_id)
        ),
        None,
    )


def _additional_same_direction_lane_ids(
    frame: dict[str, Any],
    selected_junction: dict[str, Any] | None,
) -> tuple[int, ...]:
    if selected_junction is None:
        return ()
    same_direction_lane_ids = frame.get("same_direction_lane_ids")
    if not same_direction_lane_ids:
        return ()
    junction_lane_ids = {
        int(lane_id)
        for key in (
            "incoming_lane_ids",
            "outgoing_lane_ids",
            "core_lane_ids",
        )
        for lane_id in selected_junction.get(key, [])
    }
    return tuple(
        sorted(
            int(lane_id)
            for lane_id in same_direction_lane_ids
            if int(lane_id) not in junction_lane_ids
        )
    )


def _draw_junctions(
    ax,
    annotation: dict[str, Any],
    selected_junction: dict[str, Any] | None,
    origin_xy: tuple[float, float],
    ego_heading_rad: float,
    config: MapVisualizationConfig,
) -> None:
    selected_id = (
        None if selected_junction is None else selected_junction["junction_id"]
    )
    for junction in annotation.get("junctions", []):
        is_selected = junction["junction_id"] == selected_id
        is_signalized = junction.get("kind") == "signalized"
        face_color = "#52A675" if is_signalized else "#D4913B"
        for polygon_xy in junction.get("boundary_polygons_xy", []):
            points = world_to_ego(polygon_xy, origin_xy, ego_heading_rad)
            if len(points) < 3 or not _visible(points, config):
                continue
            ax.add_patch(
                Polygon(
                    points,
                    closed=True,
                    facecolor=face_color,
                    edgecolor="#176B4B" if is_signalized else "#9C5D0A",
                    linewidth=2.0 if is_selected else 0.9,
                    alpha=0.23 if is_selected else 0.08,
                    zorder=2,
                )
            )
        center = world_to_ego(
            [junction["center_xy"]], origin_xy, ego_heading_rad
        )[0]
        if (
            config.x_min_m <= center[0] <= config.x_max_m
            and config.y_min_m <= center[1] <= config.y_max_m
        ):
            ax.scatter(
                center[0],
                center[1],
                marker="+",
                s=85 if is_selected else 40,
                linewidths=1.8 if is_selected else 1.0,
                color="#176B4B" if is_signalized else "#9C5D0A",
                zorder=6,
            )


def _draw_highlighted_lanes(
    ax,
    lanes: dict[int, np.ndarray],
    selected_junction: dict[str, Any] | None,
    ego_lane_ids: tuple[int, ...],
    additional_same_direction_lane_ids: tuple[int, ...],
) -> None:
    lane_groups: list[
        tuple[list[int], str, float, float, str | tuple]
    ] = []
    if selected_junction is not None:
        lane_groups.extend(
            [
                (
                    selected_junction.get("incoming_lane_ids", []),
                    "#D45D4C",
                    1.8,
                    5.0,
                    "solid",
                ),
                (
                    selected_junction.get("outgoing_lane_ids", []),
                    "#447F9D",
                    1.8,
                    5.0,
                    "solid",
                ),
                (
                    selected_junction.get("core_lane_ids", []),
                    "#C58124",
                    2.4,
                    5.2,
                    "solid",
                ),
            ]
        )
    if additional_same_direction_lane_ids:
        lane_groups.append(
            (
                list(additional_same_direction_lane_ids),
                "#D45D4C",
                2.2,
                5.4,
                (0, (5, 3)),
            )
        )
    if ego_lane_ids:
        lane_groups.append(
            (list(ego_lane_ids), "#006D77", 3.4, 6.0, "solid")
        )

    for lane_ids, color, width, zorder, linestyle in lane_groups:
        segments = []
        arrow_origins = []
        arrow_vectors = []
        for lane_id in lane_ids:
            points = lanes.get(int(lane_id))
            if points is None:
                continue
            segments.append(points)
            arrow = _direction_arrow_vector(points)
            if arrow is not None:
                arrow_origins.append(arrow[0])
                arrow_vectors.append(arrow[1])
        if not segments:
            continue
        ax.add_collection(
            LineCollection(
                segments,
                colors=color,
                linewidths=width,
                linestyles=linestyle,
                alpha=0.92,
                capstyle="round",
                zorder=zorder,
            )
        )
        if arrow_origins:
            origins = np.asarray(arrow_origins)
            vectors = np.asarray(arrow_vectors)
            ax.quiver(
                origins[:, 0],
                origins[:, 1],
                vectors[:, 0],
                vectors[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                pivot="middle",
                color=color,
                width=0.003,
                headwidth=4.0,
                headlength=5.0,
                headaxislength=4.4,
                zorder=zorder + 0.1,
            )


def _signal_color(state: int) -> str:
    if state in {1, 4, 7}:
        return "#D63F3F"
    if state in {2, 5, 8}:
        return "#E2B93B"
    if state in {3, 6}:
        return "#2B9B62"
    return "#858D8F"


def _draw_controls(
    ax,
    scenario,
    frame_index: int,
    origin_xy: tuple[float, float],
    ego_heading_rad: float,
    config: MapVisualizationConfig,
) -> None:
    stop_points = []
    for feature in scenario.map_features:
        if feature.WhichOneof("feature_data") != "stop_sign":
            continue
        point = world_to_ego(
            [(feature.stop_sign.position.x, feature.stop_sign.position.y)],
            origin_xy,
            ego_heading_rad,
        )[0]
        if (
            config.x_min_m <= point[0] <= config.x_max_m
            and config.y_min_m <= point[1] <= config.y_max_m
        ):
            stop_points.append(point)
    if stop_points:
        points = np.asarray(stop_points)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            marker="D",
            s=36,
            facecolor="#D63F3F",
            edgecolor="white",
            linewidth=0.8,
            zorder=8,
        )

    if not 0 <= frame_index < len(scenario.dynamic_map_states):
        return
    signal_points: dict[str, list[np.ndarray]] = {}
    for lane_state in scenario.dynamic_map_states[frame_index].lane_states:
        point = world_to_ego(
            [(lane_state.stop_point.x, lane_state.stop_point.y)],
            origin_xy,
            ego_heading_rad,
        )[0]
        if not (
            config.x_min_m <= point[0] <= config.x_max_m
            and config.y_min_m <= point[1] <= config.y_max_m
        ):
            continue
        signal_points.setdefault(_signal_color(int(lane_state.state)), []).append(
            point
        )
    for color, color_points in signal_points.items():
        points = np.asarray(color_points)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            marker="o",
            s=42,
            facecolor=color,
            edgecolor="#263234",
            linewidth=0.8,
            zorder=8,
        )


def _agent_corners(state) -> np.ndarray:
    length = max(float(state.length), 0.8)
    width = max(float(state.width), 0.6)
    local = np.asarray(
        [
            [length * 0.5, width * 0.5],
            [length * 0.5, -width * 0.5],
            [-length * 0.5, -width * 0.5],
            [-length * 0.5, width * 0.5],
        ]
    )
    cosine = math.cos(float(state.heading))
    sine = math.sin(float(state.heading))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.asarray([state.center_x, state.center_y])


def _draw_agents(
    ax,
    scenario,
    frame_index: int,
    origin_xy: tuple[float, float],
    ego_heading_rad: float,
    config: MapVisualizationConfig,
) -> None:
    font_kwargs = _font_kwargs()
    agent_groups: dict[tuple[str, str, float, int], dict[str, list]] = {}
    for track_index, track in enumerate(scenario.tracks):
        if frame_index >= len(track.states) or not track.states[frame_index].valid:
            continue
        state = track.states[frame_index]
        center = world_to_ego(
            [(state.center_x, state.center_y)], origin_xy, ego_heading_rad
        )[0]
        if not (
            config.x_min_m - 5.0 <= center[0] <= config.x_max_m + 5.0
            and config.y_min_m - 5.0 <= center[1] <= config.y_max_m + 5.0
        ):
            continue
        is_ego = track_index == scenario.sdc_track_index
        if is_ego:
            face_color, edge_color, alpha, zorder = "#006D77", "#003F45", 1.0, 10
        elif track.object_type == 2:
            face_color, edge_color, alpha, zorder = "#8F6CB3", "#5D3E77", 0.75, 7
        elif track.object_type == 3:
            face_color, edge_color, alpha, zorder = "#D9913D", "#89551B", 0.75, 7
        else:
            face_color, edge_color, alpha, zorder = "#667173", "#30393B", 0.55, 7
        corners = world_to_ego(
            _agent_corners(state), origin_xy, ego_heading_rad
        )
        heading_local = float(state.heading) - ego_heading_rad
        front_length = max(1.0, float(state.length) * 0.52)
        front = center + front_length * np.asarray(
            [math.cos(heading_local), math.sin(heading_local)]
        )
        if is_ego:
            ax.add_patch(
                Polygon(
                    corners,
                    closed=True,
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=1.5,
                    alpha=alpha,
                    zorder=zorder,
                )
            )
            ax.plot(
                [center[0], front[0]],
                [center[1], front[1]],
                color="white",
                linewidth=1.4,
                zorder=zorder + 0.2,
            )
        else:
            group = agent_groups.setdefault(
                (face_color, edge_color, alpha, zorder),
                {"patches": [], "headings": []},
            )
            group["patches"].append(Polygon(corners, closed=True))
            group["headings"].append(np.asarray([center, front]))
        if is_ego or config.show_agent_ids:
            label = "主车" if is_ego and _cjk_font_name() else ("EGO" if is_ego else str(track.id))
            if config.show_agent_ids and is_ego:
                label = f"{label} {track.id}"
            ax.text(
                center[0],
                center[1] + max(2.2, float(state.width)),
                label,
                ha="center",
                va="bottom",
                fontsize=8.5 if is_ego else 6.5,
                color="#003F45" if is_ego else "#30393B",
                weight="bold" if is_ego else "normal",
                zorder=zorder + 0.5,
                **font_kwargs,
            )

    for (face_color, edge_color, alpha, zorder), group in agent_groups.items():
        ax.add_collection(
            PatchCollection(
                group["patches"],
                facecolor=face_color,
                edgecolor=edge_color,
                linewidth=0.7,
                alpha=alpha,
                zorder=zorder,
            )
        )
        ax.add_collection(
            LineCollection(
                group["headings"],
                colors=edge_color,
                linewidths=0.8,
                alpha=min(1.0, alpha + 0.15),
                zorder=zorder + 0.2,
            )
        )


def _format_optional(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def _panel_text(frame: dict[str, Any], language: str) -> str:
    region = frame.get("region_type", "UNKNOWN")
    kind = frame.get("junction_kind")
    matched_lane = frame.get("matched_lane_id")
    matched_lane_text = "-" if matched_lane is None else str(matched_lane)
    lane_count = _near_lane_count(frame)
    lane_count_text = "-" if lane_count is None else str(lane_count)
    if language == "zh":
        lines = [
            "主车地图标注",
            format_region_summary(frame, "zh"),
            format_road_environment_summary(frame, "zh"),
            f"细分类别：{REGION_LABELS_ZH.get(region, '未知')}",
            f"控制类型：{JUNCTION_CONTROL_LABELS_ZH.get(kind, '-')}",
            f"匹配车道：{matched_lane_text}",
            f"主车方向车道：{lane_count_text}",
            f"到路口距离：{_format_optional(frame.get('distance_to_junction_m'), 1, ' m')}",
            f"地图匹配误差：{_format_optional(frame.get('map_match_distance_m'), 2, ' m')}",
            f"置信度：{float(frame.get('confidence', 0.0)):.3f}",
        ]
    else:
        lines = [
            "Ego map annotation",
            format_region_summary(frame, "en"),
            format_road_environment_summary(frame, "en"),
            f"Region: {REGION_LABELS_EN.get(region, 'Unknown')}",
            f"Control: {JUNCTION_CONTROL_LABELS_EN.get(kind, '-')}",
            f"Matched lane: {matched_lane_text}",
            f"Ego-direction lanes: {lane_count_text}",
            f"Junction distance: {_format_optional(frame.get('distance_to_junction_m'), 1, ' m')}",
            f"Map-match error: {_format_optional(frame.get('map_match_distance_m'), 2, ' m')}",
            f"Confidence: {float(frame.get('confidence', 0.0)):.3f}",
        ]
    return "\n".join(lines)


def render_initial_frame_map(
    scenario,
    annotation: dict[str, Any],
    output_path: Path | str,
    *,
    preferred_frame_index: int = DEFAULT_MAP_FRAME_INDEX,
    config: MapVisualizationConfig | None = None,
) -> RenderedMapAnnotation:
    config = config or MapVisualizationConfig()
    output_path = Path(output_path)
    if annotation.get("scenario_id") != scenario.scenario_id:
        raise ValueError(
            "Scenario and annotation IDs differ: "
            f"{scenario.scenario_id!r} != {annotation.get('scenario_id')!r}"
        )

    frame_index, frame = select_render_frame(
        scenario, annotation, preferred_frame_index
    )
    ego_state = scenario.tracks[scenario.sdc_track_index].states[frame_index]
    origin_xy = (float(ego_state.center_x), float(ego_state.center_y))
    ego_heading_rad = float(ego_state.heading)
    selected_junction = _junction_by_id(annotation, frame.get("junction_id"))

    figure, ax = plt.subplots(figsize=(15.0, 9.2), facecolor="#F4F7F7")
    ax.set_facecolor("#F4F7F7")
    _draw_junctions(
        ax,
        annotation,
        selected_junction,
        origin_xy,
        ego_heading_rad,
        config,
    )
    lanes = _draw_map_features(
        ax, scenario, origin_xy, ego_heading_rad, config
    )
    ego_lane_ids = _ego_lane_chain_ids(scenario, frame.get("matched_lane_id"))
    additional_same_direction_lane_ids = _additional_same_direction_lane_ids(
        frame, selected_junction
    )
    _draw_highlighted_lanes(
        ax,
        lanes,
        selected_junction,
        ego_lane_ids,
        additional_same_direction_lane_ids,
    )
    _draw_controls(
        ax, scenario, frame_index, origin_xy, ego_heading_rad, config
    )
    _draw_agents(
        ax, scenario, frame_index, origin_xy, ego_heading_rad, config
    )

    ax.set_xlim(config.x_min_m, config.x_max_m)
    ax.set_ylim(config.y_min_m, config.y_max_m)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#D7DEDF", linewidth=0.65, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors="#4B5557")
    for spine in ax.spines.values():
        spine.set_color("#859092")
        spine.set_linewidth(0.8)

    font_kwargs = _font_kwargs()
    language = "zh" if _cjk_font_name() else "en"
    if language == "zh":
        title = (
            f"场景 {annotation.get('scenario_index', '-'):06d} | "
            f"{scenario.scenario_id} | 帧 {frame_index}"
        ) if isinstance(annotation.get("scenario_index"), int) else (
            f"场景 {scenario.scenario_id} | 帧 {frame_index}"
        )
        x_label, y_label = "主车纵向 x（米）", "主车横向 y（米）"
    else:
        title = (
            f"Scenario {annotation.get('scenario_index', '-')} | "
            f"{scenario.scenario_id} | frame {frame_index}"
        )
        x_label, y_label = "Ego longitudinal x (m)", "Ego lateral y (m)"
    ax.set_title(title, fontsize=12, color="#263234", pad=11, **font_kwargs)
    ax.set_xlabel(x_label, fontsize=9.5, color="#374244", **font_kwargs)
    ax.set_ylabel(y_label, fontsize=9.5, color="#374244", **font_kwargs)

    ax.text(
        0.985,
        0.975,
        _panel_text(frame, language),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color="#182326",
        linespacing=1.55,
        bbox={
            "boxstyle": "round,pad=0.65,rounding_size=0.18",
            "facecolor": "white",
            "edgecolor": "#536163",
            "linewidth": 0.9,
            "alpha": 0.94,
        },
        zorder=20,
        **font_kwargs,
    )

    junction_face_color = (
        "#52A675"
        if selected_junction is None
        or selected_junction.get("kind") == "signalized"
        else "#D4913B"
    )
    junction_edge_color = (
        "#176B4B"
        if selected_junction is None
        or selected_junction.get("kind") == "signalized"
        else "#9C5D0A"
    )
    legend_handles = [
        Line2D([0], [0], color="#006D77", lw=3.4, label="主车车道" if language == "zh" else "Ego lane"),
        Line2D([0], [0], color="#D45D4C", lw=2.0, label="驶入车道" if language == "zh" else "Incoming lanes"),
        Line2D(
            [0],
            [0],
            color="#D45D4C",
            lw=2.2,
            linestyle=(0, (5, 3)),
            label=(
                "附加同向车道"
                if language == "zh"
                else "Additional same-direction lane"
            ),
        ),
        Line2D([0], [0], color="#447F9D", lw=2.0, label="驶出车道" if language == "zh" else "Outgoing lanes"),
        Patch(
            facecolor=junction_face_color,
            edgecolor=junction_edge_color,
            alpha=0.25,
            label="路口区域" if language == "zh" else "Junction area",
        ),
        Patch(facecolor="#006D77", edgecolor="#003F45", label="主车" if language == "zh" else "Ego vehicle"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=7.5,
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#A4AEAF",
        framealpha=0.88,
        prop={"family": _cjk_font_name()} if _cjk_font_name() else None,
    )
    legend.set_zorder(20)

    figure.tight_layout(pad=1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")
    figure.savefig(
        partial_path,
        dpi=config.dpi,
        format="png",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    partial_path.replace(output_path)

    return RenderedMapAnnotation(
        frame_index=frame_index,
        region_type=frame.get("region_type", "UNKNOWN"),
        summary=format_region_summary(frame, "zh"),
        output_path=output_path,
    )
