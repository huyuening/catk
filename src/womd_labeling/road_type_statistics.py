"""Road-context taxonomy and aggregation helpers for WOMD ego trajectories."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Iterable

import numpy as np


class RoadCategory(str, Enum):
    INTERSECTION = "INTERSECTION"
    INTERSECTION_INTERIOR = "INTERSECTION_INTERIOR"
    ROAD_SEGMENT = "ROAD_SEGMENT"
    NEAR_INTERSECTION = "NEAR_INTERSECTION"
    PARKING_LOT_PROXY = "PARKING_LOT_PROXY"
    UNKNOWN = "UNKNOWN"


CATEGORY_LABELS_ZH = {
    RoadCategory.INTERSECTION.value: "路口",
    RoadCategory.INTERSECTION_INTERIOR.value: "路口内部",
    RoadCategory.ROAD_SEGMENT.value: "普通路段",
    RoadCategory.NEAR_INTERSECTION.value: "路口附近",
    RoadCategory.PARKING_LOT_PROXY.value: "停车场入口代理",
    RoadCategory.UNKNOWN.value: "未知",
}


@dataclass(frozen=True)
class ThreeClassRoadConfig:
    """Thresholds for the mutually exclusive intersection/road/parking taxonomy."""

    approach_stop_line_distance_m: float = 30.0
    exit_junction_distance_m: float = 15.0

    def __post_init__(self) -> None:
        if self.approach_stop_line_distance_m <= 0:
            raise ValueError("approach_stop_line_distance_m must be positive")
        if self.exit_junction_distance_m <= 0:
            raise ValueError("exit_junction_distance_m must be positive")


@dataclass(frozen=True)
class RoadTypeLabel:
    category: str
    subtype: str
    lane_count: int | None
    arm_count: int | None
    base_region_type: str
    driveway_polygon_match: bool
    confidence: float
    reason: str | None

    @property
    def key(self) -> str:
        return encode_count_key(self.category, self.subtype)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["category_zh"] = category_label_zh(self.category)
        payload["subtype_zh"] = subtype_label_zh(self.subtype)
        return payload


def category_label_zh(category: str) -> str:
    return CATEGORY_LABELS_ZH.get(category, category)


def subtype_label_zh(subtype: str) -> str:
    if subtype == "THREE_ARM_INTERSECTION":
        return "三支路口"
    if subtype == "FOUR_ARM_INTERSECTION":
        return "四支路口"
    if subtype == "OTHER_INTERSECTION":
        return "其他路口"
    if subtype == "DRIVEWAY_POLYGON_PROXY":
        return "Driveway 多边形（停车场入口代理）"
    if subtype == "UNKNOWN_LANE_COUNT":
        return "未知车道数"
    if subtype == "UNKNOWN":
        return "未知"
    if subtype.startswith("LANE_COUNT_"):
        return f"{int(subtype.removeprefix('LANE_COUNT_'))} 车道"
    return subtype


def encode_count_key(category: str, subtype: str) -> str:
    return f"{category}\t{subtype}"


def decode_count_key(key: str) -> tuple[str, str]:
    return tuple(key.split("\t", 1))  # type: ignore[return-value]


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    integer = int(value)
    return integer if integer > 0 else None


def _point_on_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    tolerance: float = 1e-7,
) -> bool:
    segment = end - start
    offset = point - start
    length = float(np.linalg.norm(segment))
    if length <= tolerance:
        return float(np.linalg.norm(offset)) <= tolerance
    cross = abs(float(segment[0] * offset[1] - segment[1] * offset[0]))
    if cross > tolerance * max(1.0, length):
        return False
    projection = float(np.dot(offset, segment))
    return -tolerance <= projection <= float(np.dot(segment, segment)) + tolerance


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Return whether a point is inside or on a simple polygon boundary."""
    if len(polygon) < 3:
        return False
    inside = False
    x, y = float(point[0]), float(point[1])
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


class DrivewayPolygonIndex:
    """Spatially index WOMD driveway polygons with inexpensive bounds checks."""

    def __init__(self, scenario) -> None:
        polygons = []
        for feature in scenario.map_features:
            if feature.WhichOneof("feature_data") != "driveway":
                continue
            polygon = np.asarray(
                [(point.x, point.y) for point in feature.driveway.polygon],
                dtype=np.float64,
            )
            if len(polygon) < 3 or not np.all(np.isfinite(polygon)):
                continue
            polygons.append(
                (
                    polygon,
                    (
                        float(np.min(polygon[:, 0])),
                        float(np.min(polygon[:, 1])),
                        float(np.max(polygon[:, 0])),
                        float(np.max(polygon[:, 1])),
                    ),
                )
            )
        self._polygons = tuple(polygons)

    @property
    def polygon_count(self) -> int:
        return len(self._polygons)

    def contains(self, position_xy: tuple[float, float] | None) -> bool:
        if position_xy is None:
            return False
        point = np.asarray(position_xy, dtype=np.float64)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            return False
        x, y = float(point[0]), float(point[1])
        for polygon, (x_min, y_min, x_max, y_max) in self._polygons:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                if _point_in_polygon(point, polygon):
                    return True
        return False


def classify_ego_frame(
    frame,
    driveway_index: DrivewayPolygonIndex | None = None,
) -> RoadTypeLabel:
    region = _enum_value(frame.region_type)
    confidence = float(frame.confidence) if math.isfinite(frame.confidence) else 0.0
    if not frame.valid:
        return RoadTypeLabel(
            category=RoadCategory.UNKNOWN.value,
            subtype="UNKNOWN",
            lane_count=None,
            arm_count=None,
            base_region_type=region,
            driveway_polygon_match=False,
            confidence=confidence,
            reason=frame.reason or "invalid_sdc_state",
        )

    arm_count = _positive_int(frame.junction_arm_count)
    if region in {"INTERSECTION", "IN_INTERSECTION"}:
        if arm_count == 3:
            subtype = "THREE_ARM_INTERSECTION"
        elif arm_count == 4:
            subtype = "FOUR_ARM_INTERSECTION"
        else:
            subtype = "OTHER_INTERSECTION"
        return RoadTypeLabel(
            category=(
                RoadCategory.INTERSECTION.value
                if region == "INTERSECTION"
                else RoadCategory.INTERSECTION_INTERIOR.value
            ),
            subtype=subtype,
            lane_count=None,
            arm_count=arm_count,
            base_region_type=region,
            driveway_polygon_match=False,
            confidence=confidence,
            reason=frame.reason,
        )

    in_driveway = bool(
        driveway_index is not None
        and driveway_index.contains(frame.position_xy)
    )
    if in_driveway:
        return RoadTypeLabel(
            category=RoadCategory.PARKING_LOT_PROXY.value,
            subtype="DRIVEWAY_POLYGON_PROXY",
            lane_count=None,
            arm_count=arm_count,
            base_region_type=region,
            driveway_polygon_match=True,
            confidence=confidence,
            reason="inside_womd_driveway_polygon",
        )

    if region in {"NEAR_INTERSECTION_APPROACH", "NEAR_INTERSECTION_EXIT"}:
        lane_count = _positive_int(frame.junction_side_lane_count)
        if lane_count is None:
            lane_count = _positive_int(frame.same_direction_lane_count)
        return RoadTypeLabel(
            category=RoadCategory.NEAR_INTERSECTION.value,
            subtype=(
                "UNKNOWN_LANE_COUNT"
                if lane_count is None
                else f"LANE_COUNT_{lane_count}"
            ),
            lane_count=lane_count,
            arm_count=arm_count,
            base_region_type=region,
            driveway_polygon_match=False,
            confidence=confidence,
            reason=frame.reason,
        )

    if region == "ROAD_SEGMENT":
        lane_count = _positive_int(frame.same_direction_lane_count)
        return RoadTypeLabel(
            category=RoadCategory.ROAD_SEGMENT.value,
            subtype=(
                "UNKNOWN_LANE_COUNT"
                if lane_count is None
                else f"LANE_COUNT_{lane_count}"
            ),
            lane_count=lane_count,
            arm_count=None,
            base_region_type=region,
            driveway_polygon_match=False,
            confidence=confidence,
            reason=frame.reason,
        )

    return RoadTypeLabel(
        category=RoadCategory.UNKNOWN.value,
        subtype="UNKNOWN",
        lane_count=None,
        arm_count=arm_count,
        base_region_type=region,
        driveway_polygon_match=False,
        confidence=confidence,
        reason=frame.reason or "unknown_map_context",
    )


def classify_ego_frame_three_class(
    frame,
    driveway_index: DrivewayPolygonIndex | None = None,
    config: ThreeClassRoadConfig | None = None,
) -> RoadTypeLabel:
    """Classify one ego frame as intersection, road segment, or parking proxy.

    The approach distance is measured along the matched lane to the inferred
    junction gate, which acts as a stop-line proxy. Exit distance is measured
    along the matched lane from the junction boundary.
    """
    config = config or ThreeClassRoadConfig()
    region = _enum_value(frame.region_type)
    confidence = float(frame.confidence) if math.isfinite(frame.confidence) else 0.0
    if not frame.valid:
        return RoadTypeLabel(
            category=RoadCategory.UNKNOWN.value,
            subtype="UNKNOWN",
            lane_count=None,
            arm_count=None,
            base_region_type=region,
            driveway_polygon_match=False,
            confidence=confidence,
            reason=frame.reason or "invalid_sdc_state",
        )

    arm_count = _positive_int(frame.junction_arm_count)
    in_driveway = bool(
        driveway_index is not None
        and driveway_index.contains(frame.position_xy)
    )
    if in_driveway:
        return RoadTypeLabel(
            category=RoadCategory.PARKING_LOT_PROXY.value,
            subtype="DRIVEWAY_POLYGON_PROXY",
            lane_count=None,
            arm_count=arm_count,
            base_region_type=region,
            driveway_polygon_match=True,
            confidence=confidence,
            reason="inside_womd_driveway_polygon",
        )

    raw_distance = getattr(frame, "distance_to_junction_m", None)
    distance = (
        float(raw_distance)
        if raw_distance is not None and math.isfinite(float(raw_distance))
        else None
    )
    in_intersection_zone = region in {"INTERSECTION", "IN_INTERSECTION"}
    if region == "NEAR_INTERSECTION_APPROACH" and distance is not None:
        in_intersection_zone = (
            distance <= config.approach_stop_line_distance_m
        )
    elif region == "NEAR_INTERSECTION_EXIT" and distance is not None:
        in_intersection_zone = distance <= config.exit_junction_distance_m

    if in_intersection_zone:
        if arm_count == 3:
            subtype = "THREE_ARM_INTERSECTION"
        elif arm_count == 4:
            subtype = "FOUR_ARM_INTERSECTION"
        else:
            subtype = "OTHER_INTERSECTION"
        if region == "NEAR_INTERSECTION_APPROACH":
            reason = (
                "within_approach_stop_line_distance_"
                f"{config.approach_stop_line_distance_m:g}m"
            )
        elif region == "NEAR_INTERSECTION_EXIT":
            reason = (
                "within_exit_junction_distance_"
                f"{config.exit_junction_distance_m:g}m"
            )
        else:
            reason = frame.reason
        return RoadTypeLabel(
            category=RoadCategory.INTERSECTION.value,
            subtype=subtype,
            lane_count=None,
            arm_count=arm_count,
            base_region_type=region,
            driveway_polygon_match=False,
            confidence=confidence,
            reason=reason,
        )

    if region in {
        "ROAD_SEGMENT",
        "NEAR_INTERSECTION_APPROACH",
        "NEAR_INTERSECTION_EXIT",
    }:
        if region == "ROAD_SEGMENT":
            lane_count = _positive_int(frame.same_direction_lane_count)
        else:
            lane_count = _positive_int(frame.junction_side_lane_count)
            if lane_count is None:
                lane_count = _positive_int(frame.same_direction_lane_count)
        return RoadTypeLabel(
            category=RoadCategory.ROAD_SEGMENT.value,
            subtype=(
                "UNKNOWN_LANE_COUNT"
                if lane_count is None
                else f"LANE_COUNT_{lane_count}"
            ),
            lane_count=lane_count,
            arm_count=arm_count,
            base_region_type=region,
            driveway_polygon_match=False,
            confidence=confidence,
            reason=frame.reason,
        )

    return RoadTypeLabel(
        category=RoadCategory.UNKNOWN.value,
        subtype="UNKNOWN",
        lane_count=None,
        arm_count=arm_count,
        base_region_type=region,
        driveway_polygon_match=False,
        confidence=confidence,
        reason=frame.reason or "unknown_map_context",
    )


def summarize_scenario_road_types(
    annotation,
    driveway_index: DrivewayPolygonIndex | None = None,
) -> dict[str, Any]:
    frames = tuple(annotation.ego_frames)
    current_frame = next(
        (
            frame
            for frame in frames
            if frame.frame_index == annotation.current_time_index
        ),
        None,
    )
    if current_frame is None:
        current_label = RoadTypeLabel(
            category=RoadCategory.UNKNOWN.value,
            subtype="UNKNOWN",
            lane_count=None,
            arm_count=None,
            base_region_type="UNKNOWN",
            driveway_polygon_match=False,
            confidence=0.0,
            reason="missing_current_time_frame",
        )
        current_frame_index = annotation.current_time_index
    else:
        current_label = classify_ego_frame(current_frame, driveway_index)
        current_frame_index = current_frame.frame_index

    frame_counts: Counter[str] = Counter()
    occurrence_keys = set()
    unknown_reason_counts: Counter[str] = Counter()
    near_direction_counts: Counter[str] = Counter()
    other_intersection_arm_counts: Counter[str] = Counter()
    valid_frame_count = 0
    invalid_frame_count = 0
    for frame in frames:
        if not frame.valid:
            invalid_frame_count += 1
            continue
        valid_frame_count += 1
        label = classify_ego_frame(frame, driveway_index)
        frame_counts[label.key] += 1
        occurrence_keys.add(label.key)
        if label.category == RoadCategory.UNKNOWN.value:
            unknown_reason_counts[label.reason or "unknown"] += 1
        if label.category == RoadCategory.NEAR_INTERSECTION.value:
            near_direction_counts[label.base_region_type] += 1
        if (
            label.subtype == "OTHER_INTERSECTION"
            and label.arm_count is not None
        ):
            other_intersection_arm_counts[str(label.arm_count)] += 1

    return {
        "current_frame_index": current_frame_index,
        "current_label": current_label.to_dict(),
        "total_frame_count": len(frames),
        "valid_frame_count": valid_frame_count,
        "invalid_frame_count": invalid_frame_count,
        "frame_counts": dict(frame_counts),
        "occurrence_keys": sorted(occurrence_keys),
        "occurrence_categories": sorted(
            {decode_count_key(key)[0] for key in occurrence_keys}
        ),
        "unknown_reason_counts": dict(unknown_reason_counts),
        "near_direction_counts": dict(near_direction_counts),
        "other_intersection_arm_counts": dict(other_intersection_arm_counts),
        "driveway_polygon_count": (
            0 if driveway_index is None else driveway_index.polygon_count
        ),
    }


def category_totals(counts: Iterable[tuple[str, int]]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for key, count in counts:
        category, _ = decode_count_key(key)
        totals[category] += int(count)
    return totals
