"""Current-frame agent subtype labels using explicit U.S. reference proxies."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any


METERS_PER_FOOT = 0.3048
METERS_PER_INCH = 0.0254

# These are reference proxies, not object-subtype ground truth. WOMD does not
# provide axles, motor power, pedals, age, or other attributes required by the
# corresponding U.S. definitions.
AASHTO_PASSENGER_CAR_LENGTH_M = 19.0 * METERS_PER_FOOT
FHWA_MOTORCYCLE_TYPICAL_MAX_WIDTH_M = 30.0 * METERS_PER_INCH
MOTORCYCLE_BOX_MAX_WIDTH_M = 1.20
MOTORCYCLE_BOX_MAX_LENGTH_M = 3.00
FHWA_BICYCLE_85TH_PERCENTILE_SPEED_MPS = 24.0 / 3.6
NHTSA_FOUR_FEET_NINE_INCHES_M = 4.0 * METERS_PER_FOOT + 9.0 * METERS_PER_INCH


OBJECT_TYPE_NAMES = {
    0: "TYPE_UNSET",
    1: "TYPE_VEHICLE",
    2: "TYPE_PEDESTRIAN",
    3: "TYPE_CYCLIST",
    4: "TYPE_OTHER",
}

OBJECT_TYPE_LABELS_ZH = {
    0: "未设置",
    1: "车辆",
    2: "行人",
    3: "骑车人",
    4: "其他",
}

SIZE_CLASS_LABELS_ZH = {
    "LARGE_VEHICLE_PROXY": "大型车（尺寸代理）",
    "SMALL_VEHICLE_PROXY": "小型车（尺寸代理）",
    "MOTORCYCLE_PROXY": "摩托车（尺寸代理）",
    "E_BIKE_PROXY": "电动车（速度代理）",
    "BICYCLE_PROXY": "自行车（速度代理）",
    "ADULT_PEDESTRIAN_PROXY": "成年人（身高代理）",
    "CHILD_PEDESTRIAN_PROXY": "儿童（身高代理）",
    "UNSUPPORTED_OBJECT_TYPE": "未分类对象类型",
    "INVALID_DIMENSIONS": "尺寸无效",
}


@dataclass(frozen=True)
class AgentSizeConfig:
    vehicle_large_length_m: float = AASHTO_PASSENGER_CAR_LENGTH_M
    vehicle_motorcycle_max_width_m: float = MOTORCYCLE_BOX_MAX_WIDTH_M
    vehicle_motorcycle_max_length_m: float = MOTORCYCLE_BOX_MAX_LENGTH_M
    cyclist_ebike_min_speed_mps: float = FHWA_BICYCLE_85TH_PERCENTILE_SPEED_MPS
    pedestrian_child_max_height_m: float = NHTSA_FOUR_FEET_NINE_INCHES_M

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class AgentSizeLabel:
    size_class: str
    size_class_zh: str
    rule_id: str
    classification_dimension: str | None
    threshold_value: float | None
    threshold_unit: str | None
    comparison: str | None
    secondary_classification_dimension: str | None
    secondary_threshold_value: float | None
    secondary_threshold_unit: str | None
    secondary_comparison: str | None
    supported: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def object_type_name(object_type: int) -> str:
    return OBJECT_TYPE_NAMES.get(int(object_type), f"TYPE_{int(object_type)}")


def object_type_label_zh(object_type: int) -> str:
    return OBJECT_TYPE_LABELS_ZH.get(int(object_type), f"类型 {int(object_type)}")


def _label(
    size_class: str,
    rule_id: str,
    dimension: str | None,
    threshold_value: float | None,
    threshold_unit: str | None,
    comparison: str | None,
    *,
    secondary_dimension: str | None = None,
    secondary_threshold_value: float | None = None,
    secondary_threshold_unit: str | None = None,
    secondary_comparison: str | None = None,
    supported: bool = True,
) -> AgentSizeLabel:
    return AgentSizeLabel(
        size_class=size_class,
        size_class_zh=SIZE_CLASS_LABELS_ZH[size_class],
        rule_id=rule_id,
        classification_dimension=dimension,
        threshold_value=threshold_value,
        threshold_unit=threshold_unit,
        comparison=comparison,
        secondary_classification_dimension=secondary_dimension,
        secondary_threshold_value=secondary_threshold_value,
        secondary_threshold_unit=secondary_threshold_unit,
        secondary_comparison=secondary_comparison,
        supported=supported,
    )


def classify_agent_dimensions(
    object_type: int,
    length_m: float,
    width_m: float,
    height_m: float,
    *,
    speed_mps: float = 0.0,
    config: AgentSizeConfig | None = None,
) -> AgentSizeLabel:
    """Classify one WOMD state with documented size and speed proxies."""
    config = config or AgentSizeConfig()
    dimensions = (float(length_m), float(width_m), float(height_m))
    if any(not math.isfinite(value) or value <= 0 for value in dimensions):
        return _label(
            "INVALID_DIMENSIONS",
            "invalid_nonpositive_or_nonfinite_dimensions",
            None,
            None,
            None,
            None,
            supported=False,
        )

    object_type = int(object_type)
    if object_type == 1:
        is_motorcycle = (
            length_m <= config.vehicle_motorcycle_max_length_m
            and width_m <= config.vehicle_motorcycle_max_width_m
        )
        if is_motorcycle:
            return _label(
                "MOTORCYCLE_PROXY",
                "us_fhwa_narrow_motorcycle_box_proxy",
                "width_m",
                config.vehicle_motorcycle_max_width_m,
                "m",
                "<=",
                secondary_dimension="length_m",
                secondary_threshold_value=config.vehicle_motorcycle_max_length_m,
                secondary_threshold_unit="m",
                secondary_comparison="<=",
            )
        is_large = length_m > config.vehicle_large_length_m
        return _label(
            "LARGE_VEHICLE_PROXY" if is_large else "SMALL_VEHICLE_PROXY",
            "us_aashto_passenger_car_19ft_length_proxy",
            "length_m",
            config.vehicle_large_length_m,
            "m",
            ">" if is_large else "<=",
        )
    if object_type == 3:
        speed_mps = float(speed_mps)
        if not math.isfinite(speed_mps) or speed_mps < 0.0:
            speed_mps = 0.0
        is_ebike = speed_mps >= config.cyclist_ebike_min_speed_mps
        return _label(
            "E_BIKE_PROXY" if is_ebike else "BICYCLE_PROXY",
            "us_fhwa_bicycle_85th_percentile_speed_proxy",
            "speed_mps",
            config.cyclist_ebike_min_speed_mps,
            "m/s",
            ">=" if is_ebike else "<",
        )
    if object_type == 2:
        is_child = height_m < config.pedestrian_child_max_height_m
        return _label(
            "CHILD_PEDESTRIAN_PROXY" if is_child else "ADULT_PEDESTRIAN_PROXY",
            "us_nhtsa_4ft9_height_proxy",
            "height_m",
            config.pedestrian_child_max_height_m,
            "m",
            "<" if is_child else ">=",
        )
    return _label(
        "UNSUPPORTED_OBJECT_TYPE",
        "womd_object_type_not_requested",
        None,
        None,
        None,
        None,
        supported=False,
    )


def extract_agent_size_records(
    scenario,
    frame_index: int,
    config: AgentSizeConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Extract valid track dimensions and labels at exactly one frame."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    config = config or AgentSizeConfig()
    records = []
    diagnostics: Counter[str] = Counter(total_tracks=len(scenario.tracks))
    for track_index, track in enumerate(scenario.tracks):
        type_name = object_type_name(track.object_type)
        if frame_index >= len(track.states):
            diagnostics["missing_state"] += 1
            diagnostics[f"missing_state:{type_name}"] += 1
            continue
        state = track.states[frame_index]
        if not state.valid:
            diagnostics["invalid_state"] += 1
            diagnostics[f"invalid_state:{type_name}"] += 1
            continue

        length_m = float(state.length)
        width_m = float(state.width)
        height_m = float(state.height)
        speed_mps = math.hypot(float(state.velocity_x), float(state.velocity_y))
        label = classify_agent_dimensions(
            track.object_type,
            length_m,
            width_m,
            height_m,
            speed_mps=speed_mps,
            config=config,
        )
        diagnostics["valid_state"] += 1
        diagnostics[f"valid_state:{type_name}"] += 1
        if label.size_class == "INVALID_DIMENSIONS":
            diagnostics["invalid_dimensions"] += 1
            diagnostics[f"invalid_dimensions:{type_name}"] += 1
        elif label.supported:
            diagnostics["classified_agent"] += 1
            diagnostics[f"classified_agent:{type_name}"] += 1
        else:
            diagnostics["unsupported_agent"] += 1
            diagnostics[f"unsupported_agent:{type_name}"] += 1

        records.append(
            {
                "frame_number": frame_index + 1,
                "frame_index": frame_index,
                "track_index": track_index,
                "track_id": int(track.id),
                "is_sdc": track_index == scenario.sdc_track_index,
                "object_type": type_name,
                "object_type_value": int(track.object_type),
                "object_type_zh": object_type_label_zh(track.object_type),
                "length_m": length_m,
                "width_m": width_m,
                "height_m": height_m,
                "speed_mps": speed_mps,
                **label.to_dict(),
            }
        )
    return records, dict(diagnostics)


def encode_agent_size_key(object_type: str, size_class: str) -> str:
    return f"{object_type}\t{size_class}"


def decode_agent_size_key(key: str) -> tuple[str, str]:
    object_type, size_class = key.split("\t", 1)
    return object_type, size_class
