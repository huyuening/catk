from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import traceback
from typing import Iterable, Sequence

from tqdm import tqdm

from .agent_action_classification import (
    ACTION_LABELS_ZH,
    ACTION_NAMES,
    AgentActionConfig,
    decode_agent_action_key,
    decode_agent_action_frame_key,
    encode_agent_action_key,
    encode_agent_action_frame_key,
    label_scenario_actions,
)
from .agent_size_classification import (
    AASHTO_PASSENGER_CAR_LENGTH_M,
    FHWA_BICYCLE_85TH_PERCENTILE_SPEED_MPS,
    FHWA_MOTORCYCLE_TYPICAL_MAX_WIDTH_M,
    MOTORCYCLE_BOX_MAX_LENGTH_M,
    MOTORCYCLE_BOX_MAX_WIDTH_M,
    NHTSA_FOUR_FEET_NINE_INCHES_M,
    AgentSizeConfig,
    SIZE_CLASS_LABELS_ZH,
    decode_agent_size_key,
    encode_agent_size_key,
    extract_agent_size_records,
)
from .map_annotation import MapAnnotationConfig, annotate_scenario
from .proto import scenario_pb2
from .road_type_statistics import (
    CATEGORY_LABELS_ZH,
    DrivewayPolygonIndex,
    RoadCategory,
    category_label_zh,
    category_totals,
    classify_ego_frame,
    decode_count_key,
    encode_count_key,
    subtype_label_zh,
)
from .tfrecord_io import (
    count_tfrecord_records,
    iter_tfrecord,
    resolve_tfrecord_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = str(
    PROJECT_ROOT / "dataset" / "training.tfrecord-00000-of-01000"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "road_type_statistics"
DEFAULT_FRAME_NUMBER = 11
SCHEMA_VERSION = "womd-current-frame-road-subtype-all-frame-action-statistics-v5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count road context and size/speed-based agent subtype proxies at "
            "one WOMD frame, plus motion actions at every valid agent frame."
        )
    )
    parser.add_argument(
        "--input-path",
        nargs="+",
        default=[DEFAULT_INPUT],
        help="TFRecord files, directories, or quoted glob patterns.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-number", type=int, default=DEFAULT_FRAME_NUMBER)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--near-distance-m", type=float, default=40.0)
    parser.add_argument("--lane-half-width-m", type=float, default=2.0)
    parser.add_argument("--arm-angle-threshold-deg", type=float, default=30.0)
    parser.add_argument("--max-map-match-distance-m", type=float, default=8.0)
    parser.add_argument(
        "--max-map-match-heading-error-deg", type=float, default=60.0
    )
    parser.add_argument("--signalized-only", action="store_true")
    parser.add_argument(
        "--vehicle-large-length-m",
        type=float,
        default=AASHTO_PASSENGER_CAR_LENGTH_M,
    )
    parser.add_argument(
        "--vehicle-motorcycle-max-width-m",
        type=float,
        default=MOTORCYCLE_BOX_MAX_WIDTH_M,
        help="Maximum WOMD box width for the motorcycle size proxy.",
    )
    parser.add_argument(
        "--vehicle-motorcycle-max-length-m",
        type=float,
        default=MOTORCYCLE_BOX_MAX_LENGTH_M,
        help="Maximum WOMD box length for the motorcycle size proxy.",
    )
    parser.add_argument(
        "--cyclist-ebike-min-speed-mps",
        type=float,
        default=FHWA_BICYCLE_85TH_PERCENTILE_SPEED_MPS,
        help="Current-frame speed threshold for the e-bike proxy.",
    )
    parser.add_argument(
        "--pedestrian-child-max-height-m",
        type=float,
        default=NHTSA_FOUR_FEET_NINE_INCHES_M,
    )
    parser.add_argument("--action-stop-speed-mps", type=float, default=0.2)
    parser.add_argument(
        "--action-acceleration-threshold-mps2",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--action-deceleration-threshold-mps2",
        type=float,
        default=-0.5,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def iter_selected_tasks(
    paths: Iterable[Path],
    start_index: int,
    max_scenarios: int | None,
):
    global_index = 0
    selected = 0
    for path in paths:
        for scenario_index, payload in iter_tfrecord(path):
            current_global_index = global_index
            global_index += 1
            if current_global_index < start_index:
                continue
            if max_scenarios is not None and selected >= max_scenarios:
                return
            selected += 1
            yield (
                str(path),
                path.name,
                scenario_index,
                current_global_index,
                payload,
            )


def selected_task_count(
    paths: Iterable[Path],
    start_index: int,
    max_scenarios: int | None,
) -> int:
    available = sum(count_tfrecord_records(path) for path in paths)
    remaining = max(0, available - start_index)
    return remaining if max_scenarios is None else min(remaining, max_scenarios)


def process_scenario(
    source_path: str,
    source_file: str,
    scenario_index: int,
    global_index: int,
    payload: bytes,
    map_config: MapAnnotationConfig,
    size_config: AgentSizeConfig,
    action_config: AgentActionConfig,
    frame_index: int,
) -> dict:
    scenario = scenario_pb2.Scenario()
    try:
        scenario.ParseFromString(payload)
        annotation = annotate_scenario(
            scenario,
            map_config,
            scenario_index=scenario_index,
            source_file=source_file,
            frame_indices=(frame_index,),
        )
        if len(annotation.ego_frames) != 1:
            raise RuntimeError("selected-frame annotation did not return one frame")
        driveway_index = DrivewayPolygonIndex(scenario)
        road_label = classify_ego_frame(
            annotation.ego_frames[0], driveway_index
        ).to_dict()
        agent_records, agent_diagnostics = extract_agent_size_records(
            scenario,
            frame_index,
            size_config,
        )
        action_records, action_diagnostics = label_scenario_actions(
            scenario,
            config=action_config,
        )
        size_track_indices = {record["track_index"] for record in agent_records}
        current_action_track_indices = {
            record["track_index"] for record in action_records
            if record["frame_index"] == frame_index
        }
        if size_track_indices != current_action_track_indices:
            raise RuntimeError(
                "agent size/action current-frame track sets do not match"
            )
        common = {
            "source_file": source_file,
            "scenario_index": scenario_index,
            "global_index": global_index,
            "scenario_id": scenario.scenario_id,
            "dataset_current_time_index": scenario.current_time_index,
        }
        return {
            "source_path": source_path,
            **common,
            "frame_number": frame_index + 1,
            "frame_index": frame_index,
            "current_index_matches": scenario.current_time_index == frame_index,
            "driveway_polygon_count": driveway_index.polygon_count,
            "road_label": road_label,
            "road_record": {
                **common,
                "frame_number": frame_index + 1,
                "frame_index": frame_index,
                **road_label,
            },
            "agent_records": [
                {**common, **record} for record in agent_records
            ],
            "agent_diagnostics": agent_diagnostics,
            "action_records": [
                {**common, **record} for record in action_records
            ],
            "action_diagnostics": action_diagnostics,
            "error": None,
        }
    except Exception as exc:
        return {
            "source_path": source_path,
            "source_file": source_file,
            "scenario_index": scenario_index,
            "global_index": global_index,
            "scenario_id": scenario.scenario_id or None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def new_accumulator() -> dict:
    return {
        "scenarios": 0,
        "errors": 0,
        "dataset_current_time_index_counts": Counter(),
        "current_index_mismatches": 0,
        "scenarios_with_driveway_polygons": 0,
        "driveway_polygons": 0,
        "road_counts": Counter(),
        "road_unknown_reason_counts": Counter(),
        "agent_diagnostics": Counter(),
        "agent_type_counts": Counter(),
        "agent_size_counts": Counter(),
        "agent_dimension_stats": {},
        "action_diagnostics": Counter(),
        "agent_action_type_counts": Counter(),
        "agent_action_counts": Counter(),
        "agent_action_frame_type_counts": Counter(),
        "agent_action_frame_counts": Counter(),
    }


def _update_dimension_stats(stats_by_key: dict, key: str, record: dict) -> None:
    dimensions = {
        "length_m": float(record["length_m"]),
        "width_m": float(record["width_m"]),
        "height_m": float(record["height_m"]),
    }
    if any(not math.isfinite(value) or value <= 0 for value in dimensions.values()):
        return
    stats = stats_by_key.setdefault(
        key,
        {
            "count": 0,
            "sum_length_m": 0.0,
            "sum_width_m": 0.0,
            "sum_height_m": 0.0,
            "min_length_m": math.inf,
            "min_width_m": math.inf,
            "min_height_m": math.inf,
            "max_length_m": -math.inf,
            "max_width_m": -math.inf,
            "max_height_m": -math.inf,
        },
    )
    stats["count"] += 1
    for dimension, value in dimensions.items():
        stats[f"sum_{dimension}"] += value
        stats[f"min_{dimension}"] = min(stats[f"min_{dimension}"], value)
        stats[f"max_{dimension}"] = max(stats[f"max_{dimension}"], value)


def update_accumulator(accumulator: dict, result: dict) -> None:
    accumulator["scenarios"] += 1
    if result["error"] is not None:
        accumulator["errors"] += 1
        return
    accumulator["dataset_current_time_index_counts"][
        str(result["dataset_current_time_index"])
    ] += 1
    accumulator["current_index_mismatches"] += int(
        not result["current_index_matches"]
    )
    driveway_count = result["driveway_polygon_count"]
    accumulator["driveway_polygons"] += driveway_count
    accumulator["scenarios_with_driveway_polygons"] += int(driveway_count > 0)

    road_label = result["road_label"]
    road_key = encode_count_key(road_label["category"], road_label["subtype"])
    accumulator["road_counts"][road_key] += 1
    if road_label["category"] == RoadCategory.UNKNOWN.value:
        accumulator["road_unknown_reason_counts"][
            road_label["reason"] or "unknown"
        ] += 1

    accumulator["agent_diagnostics"].update(result["agent_diagnostics"])
    for record in result["agent_records"]:
        object_type = record["object_type"]
        key = encode_agent_size_key(object_type, record["size_class"])
        accumulator["agent_type_counts"][object_type] += 1
        accumulator["agent_size_counts"][key] += 1
        _update_dimension_stats(accumulator["agent_dimension_stats"], key, record)

    accumulator["action_diagnostics"].update(result["action_diagnostics"])
    for record in result["action_records"]:
        object_type = record["object_type"]
        key = encode_agent_action_key(object_type, record["action_id"])
        frame_key = encode_agent_action_frame_key(
            record["frame_index"],
            object_type,
            record["action_id"],
        )
        accumulator["agent_action_type_counts"][object_type] += 1
        accumulator["agent_action_counts"][key] += 1
        accumulator["agent_action_frame_type_counts"][
            f'{record["frame_index"]}\t{object_type}'
        ] += 1
        accumulator["agent_action_frame_counts"][frame_key] += 1


CATEGORY_ORDER = {
    RoadCategory.INTERSECTION_INTERIOR.value: 0,
    RoadCategory.ROAD_SEGMENT.value: 1,
    RoadCategory.NEAR_INTERSECTION.value: 2,
    RoadCategory.PARKING_LOT_PROXY.value: 3,
    RoadCategory.UNKNOWN.value: 4,
}

EXPECTED_ROAD_SUBTYPES = {
    RoadCategory.INTERSECTION_INTERIOR.value: (
        "THREE_ARM_INTERSECTION",
        "FOUR_ARM_INTERSECTION",
        "OTHER_INTERSECTION",
    ),
    RoadCategory.PARKING_LOT_PROXY.value: ("DRIVEWAY_POLYGON_PROXY",),
    RoadCategory.UNKNOWN.value: ("UNKNOWN",),
}


def _road_subtype_sort_key(subtype: str):
    if subtype == "THREE_ARM_INTERSECTION":
        return (0, 3)
    if subtype == "FOUR_ARM_INTERSECTION":
        return (0, 4)
    if subtype == "OTHER_INTERSECTION":
        return (0, 99)
    if subtype.startswith("LANE_COUNT_"):
        return (1, int(subtype.removeprefix("LANE_COUNT_")))
    if subtype == "UNKNOWN_LANE_COUNT":
        return (2, 0)
    return (3, subtype)


def road_count_rows(
    accumulator: dict,
    source_file: str,
    frame_number: int,
) -> list[dict]:
    counts = accumulator["road_counts"]
    totals = category_totals(counts.items())
    denominator = accumulator["scenarios"] - accumulator["errors"]
    rows = []
    for category in sorted(CATEGORY_ORDER, key=CATEGORY_ORDER.get):
        total = totals.get(category, 0)
        rows.append(
            {
                "frame_number": frame_number,
                "frame_index": frame_number - 1,
                "source_file": source_file,
                "category": category,
                "category_zh": category_label_zh(category),
                "subtype": "ALL",
                "subtype_zh": "合计",
                "count": total,
                "denominator": denominator,
                "percentage": 100.0 * total / denominator if denominator else 0.0,
            }
        )
        actual = {
            decode_count_key(key)[1]: count
            for key, count in counts.items()
            if decode_count_key(key)[0] == category
        }
        subtypes = set(actual) | set(EXPECTED_ROAD_SUBTYPES.get(category, ()))
        for subtype in sorted(subtypes, key=_road_subtype_sort_key):
            count = actual.get(subtype, 0)
            rows.append(
                {
                    "frame_number": frame_number,
                    "frame_index": frame_number - 1,
                    "source_file": source_file,
                    "category": category,
                    "category_zh": category_label_zh(category),
                    "subtype": subtype,
                    "subtype_zh": subtype_label_zh(subtype),
                    "count": count,
                    "denominator": denominator,
                    "percentage": (
                        100.0 * count / denominator if denominator else 0.0
                    ),
                }
            )
    return rows


AGENT_TYPE_ORDER = {
    "TYPE_VEHICLE": 0,
    "TYPE_CYCLIST": 1,
    "TYPE_PEDESTRIAN": 2,
    "TYPE_OTHER": 3,
    "TYPE_UNSET": 4,
}

AGENT_TYPE_LABELS_ZH = {
    "TYPE_VEHICLE": "车辆",
    "TYPE_CYCLIST": "骑车人",
    "TYPE_PEDESTRIAN": "行人",
    "TYPE_OTHER": "其他",
    "TYPE_UNSET": "未设置",
    "ALL": "全部智能体",
}

EXPECTED_AGENT_CLASSES = {
    "TYPE_VEHICLE": (
        "LARGE_VEHICLE_PROXY",
        "SMALL_VEHICLE_PROXY",
        "MOTORCYCLE_PROXY",
    ),
    "TYPE_CYCLIST": ("E_BIKE_PROXY", "BICYCLE_PROXY"),
    "TYPE_PEDESTRIAN": (
        "ADULT_PEDESTRIAN_PROXY",
        "CHILD_PEDESTRIAN_PROXY",
    ),
    "TYPE_OTHER": ("UNSUPPORTED_OBJECT_TYPE",),
    "TYPE_UNSET": ("UNSUPPORTED_OBJECT_TYPE",),
}


def _finalized_dimension_stats(stats: dict | None) -> dict:
    if not stats or not stats["count"]:
        return {
            "mean_length_m": None,
            "mean_width_m": None,
            "mean_height_m": None,
            "min_length_m": None,
            "min_width_m": None,
            "min_height_m": None,
            "max_length_m": None,
            "max_width_m": None,
            "max_height_m": None,
        }
    count = stats["count"]
    result = {}
    for dimension in ("length_m", "width_m", "height_m"):
        result[f"mean_{dimension}"] = stats[f"sum_{dimension}"] / count
        result[f"min_{dimension}"] = stats[f"min_{dimension}"]
        result[f"max_{dimension}"] = stats[f"max_{dimension}"]
    return result


def agent_count_rows(
    accumulator: dict,
    source_file: str,
    frame_number: int,
) -> list[dict]:
    counts = accumulator["agent_size_counts"]
    observed_types = {
        decode_agent_size_key(key)[0] for key in counts
    }
    object_types = set(EXPECTED_AGENT_CLASSES) | observed_types
    rows = []
    for object_type in sorted(
        object_types, key=lambda value: AGENT_TYPE_ORDER.get(value, 99)
    ):
        denominator = accumulator["agent_type_counts"].get(object_type, 0)
        actual = {
            decode_agent_size_key(key)[1]: count
            for key, count in counts.items()
            if decode_agent_size_key(key)[0] == object_type
        }
        classes = list(EXPECTED_AGENT_CLASSES.get(object_type, ()))
        classes.extend(sorted(set(actual) - set(classes)))
        for size_class in classes:
            key = encode_agent_size_key(object_type, size_class)
            count = actual.get(size_class, 0)
            rows.append(
                {
                    "frame_number": frame_number,
                    "frame_index": frame_number - 1,
                    "source_file": source_file,
                    "object_type": object_type,
                    "size_class": size_class,
                    "size_class_zh": SIZE_CLASS_LABELS_ZH[size_class],
                    "count": count,
                    "type_denominator": denominator,
                    "percentage_within_type": (
                        100.0 * count / denominator if denominator else 0.0
                    ),
                    **_finalized_dimension_stats(
                        accumulator["agent_dimension_stats"].get(key)
                    ),
                }
            )
    return rows


def action_count_rows(
    accumulator: dict,
    source_file: str,
) -> list[dict]:
    counts = accumulator["agent_action_counts"]
    observed_types = {
        decode_agent_action_key(key)[0] for key in counts
    }
    object_types = set(EXPECTED_AGENT_CLASSES) | observed_types
    rows = []

    for object_type in sorted(
        object_types,
        key=lambda value: AGENT_TYPE_ORDER.get(value, 99),
    ):
        denominator = accumulator["agent_action_type_counts"].get(
            object_type,
            0,
        )
        for action_id, action_name in ACTION_NAMES.items():
            count = counts.get(
                encode_agent_action_key(object_type, action_id),
                0,
            )
            rows.append(
                {
                    "scope": "ALL_VALID_AGENT_FRAMES",
                    "frame_number": "ALL",
                    "frame_index": "ALL",
                    "source_file": source_file,
                    "object_type": object_type,
                    "object_type_zh": AGENT_TYPE_LABELS_ZH.get(
                        object_type,
                        object_type,
                    ),
                    "action_id": action_id,
                    "action": action_name,
                    "action_zh": ACTION_LABELS_ZH[action_id],
                    "agent_frame_count": count,
                    "agent_frame_denominator": denominator,
                    "percentage_of_type_agent_frames": (
                        100.0 * count / denominator if denominator else 0.0
                    ),
                }
            )

    total_denominator = sum(accumulator["agent_action_type_counts"].values())
    for action_id, action_name in ACTION_NAMES.items():
        count = sum(
            count
            for key, count in counts.items()
            if decode_agent_action_key(key)[1] == action_id
        )
        rows.append(
            {
                "scope": "ALL_VALID_AGENT_FRAMES",
                "frame_number": "ALL",
                "frame_index": "ALL",
                "source_file": source_file,
                "object_type": "ALL",
                "object_type_zh": AGENT_TYPE_LABELS_ZH["ALL"],
                "action_id": action_id,
                "action": action_name,
                "action_zh": ACTION_LABELS_ZH[action_id],
                "agent_frame_count": count,
                "agent_frame_denominator": total_denominator,
                "percentage_of_type_agent_frames": (
                    100.0 * count / total_denominator
                    if total_denominator
                    else 0.0
                ),
            }
        )
    return rows


def action_count_rows_by_frame(
    accumulator: dict,
    source_file: str,
) -> list[dict]:
    counts = accumulator["agent_action_frame_counts"]
    frame_type_totals = accumulator["agent_action_frame_type_counts"]
    observed_frames = sorted(
        {decode_agent_action_frame_key(key)[0] for key in counts}
    )
    rows = []

    for frame_index in observed_frames:
        observed_types = {
            object_type
            for key in counts
            for key_frame, object_type, _ in (
                decode_agent_action_frame_key(key),
            )
            if key_frame == frame_index
        }
        object_types = set(EXPECTED_AGENT_CLASSES) | observed_types
        for object_type in sorted(
            object_types,
            key=lambda value: AGENT_TYPE_ORDER.get(value, 99),
        ):
            denominator = frame_type_totals.get(
                f"{frame_index}\t{object_type}",
                0,
            )
            for action_id, action_name in ACTION_NAMES.items():
                count = counts.get(
                    encode_agent_action_frame_key(
                        frame_index,
                        object_type,
                        action_id,
                    ),
                    0,
                )
                rows.append(
                    {
                        "scope": "ONE_DATASET_FRAME",
                        "frame_number": frame_index + 1,
                        "frame_index": frame_index,
                        "source_file": source_file,
                        "object_type": object_type,
                        "object_type_zh": AGENT_TYPE_LABELS_ZH.get(
                            object_type,
                            object_type,
                        ),
                        "action_id": action_id,
                        "action": action_name,
                        "action_zh": ACTION_LABELS_ZH[action_id],
                        "agent_frame_count": count,
                        "agent_frame_denominator": denominator,
                        "percentage_of_type_agent_frames": (
                            100.0 * count / denominator
                            if denominator
                            else 0.0
                        ),
                    }
                )

        frame_denominator = sum(
            count
            for key, count in frame_type_totals.items()
            if int(key.split("\t", 1)[0]) == frame_index
        )
        for action_id, action_name in ACTION_NAMES.items():
            count = sum(
                count
                for key, count in counts.items()
                if (
                    decode_agent_action_frame_key(key)[0] == frame_index
                    and decode_agent_action_frame_key(key)[2] == action_id
                )
            )
            rows.append(
                {
                    "scope": "ONE_DATASET_FRAME",
                    "frame_number": frame_index + 1,
                    "frame_index": frame_index,
                    "source_file": source_file,
                    "object_type": "ALL",
                    "object_type_zh": AGENT_TYPE_LABELS_ZH["ALL"],
                    "action_id": action_id,
                    "action": action_name,
                    "action_zh": ACTION_LABELS_ZH[action_id],
                    "agent_frame_count": count,
                    "agent_frame_denominator": frame_denominator,
                    "percentage_of_type_agent_frames": (
                        100.0 * count / frame_denominator
                        if frame_denominator
                        else 0.0
                    ),
                }
            )
    return rows


def serializable_accumulator(accumulator: dict) -> dict:
    payload = {}
    for key, value in accumulator.items():
        if key == "agent_dimension_stats":
            payload["agent_dimension_statistics"] = {
                dimension_key: _finalized_dimension_stats(stats)
                for dimension_key, stats in sorted(value.items())
            }
        elif isinstance(value, Counter):
            payload[key] = dict(value)
        else:
            payload[key] = value
    return payload


ROAD_DETAIL_FIELDS = (
    "source_file",
    "scenario_index",
    "global_index",
    "scenario_id",
    "dataset_current_time_index",
    "frame_number",
    "frame_index",
    "category",
    "category_zh",
    "subtype",
    "subtype_zh",
    "lane_count",
    "arm_count",
    "base_region_type",
    "driveway_polygon_match",
    "confidence",
    "reason",
)

AGENT_DETAIL_FIELDS = (
    "source_file",
    "scenario_index",
    "global_index",
    "scenario_id",
    "dataset_current_time_index",
    "frame_number",
    "frame_index",
    "track_index",
    "track_id",
    "is_sdc",
    "object_type",
    "object_type_value",
    "object_type_zh",
    "length_m",
    "width_m",
    "height_m",
    "speed_mps",
    "size_class",
    "size_class_zh",
    "rule_id",
    "classification_dimension",
    "threshold_value",
    "threshold_unit",
    "comparison",
    "secondary_classification_dimension",
    "secondary_threshold_value",
    "secondary_threshold_unit",
    "secondary_comparison",
    "supported",
)

ACTION_DETAIL_FIELDS = (
    "source_file",
    "scenario_index",
    "global_index",
    "scenario_id",
    "dataset_current_time_index",
    "frame_number",
    "frame_index",
    "track_index",
    "track_id",
    "is_sdc",
    "object_type",
    "object_type_value",
    "object_type_zh",
    "action_id",
    "action",
    "action_zh",
    "decision_reason",
    "longitudinal_velocity_mps",
    "absolute_speed_mps",
    "longitudinal_acceleration_mps2",
    "valid_track_frame_count",
    "lane_change_start_frame_index",
    "lane_change_end_frame_index",
    "lane_change_direction",
    "past_valid_frame_index",
    "future_valid_frame_index",
    "future_long_valid_frame_index",
)

ROAD_COUNT_FIELDS = (
    "frame_number",
    "frame_index",
    "source_file",
    "category",
    "category_zh",
    "subtype",
    "subtype_zh",
    "count",
    "denominator",
    "percentage",
)

AGENT_COUNT_FIELDS = (
    "frame_number",
    "frame_index",
    "source_file",
    "object_type",
    "size_class",
    "size_class_zh",
    "count",
    "type_denominator",
    "percentage_within_type",
    "mean_length_m",
    "mean_width_m",
    "mean_height_m",
    "min_length_m",
    "min_width_m",
    "min_height_m",
    "max_length_m",
    "max_width_m",
    "max_height_m",
)

ACTION_COUNT_FIELDS = (
    "scope",
    "frame_number",
    "frame_index",
    "source_file",
    "object_type",
    "object_type_zh",
    "action_id",
    "action",
    "action_zh",
    "agent_frame_count",
    "agent_frame_denominator",
    "percentage_of_type_agent_frames",
)


def run_statistics(args: argparse.Namespace) -> dict:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.frame_number < 1:
        raise ValueError("--frame-number must be at least 1")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_scenarios is not None and args.max_scenarios < 1:
        raise ValueError("--max-scenarios must be positive")

    paths = resolve_tfrecord_paths(args.input_path)
    frame_index = args.frame_number - 1
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "road_details": output_dir / "current_frame_road_types.csv.gz",
        "agent_details": output_dir / "current_frame_agent_sizes.csv.gz",
        "action_details": output_dir / "agent_actions_by_frame.csv.gz",
        "road_counts": output_dir / "current_frame_road_type_counts.csv",
        "agent_counts": output_dir / "current_frame_agent_size_counts.csv",
        "action_counts": output_dir / "agent_action_counts.csv",
        "action_counts_by_frame": output_dir / "agent_action_counts_by_frame.csv",
        "errors": output_dir / "errors.jsonl",
        "summary": output_dir / "summary.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {existing[0]}. Use --overwrite to replace it."
        )
    working_paths = {
        key: path.with_name(path.name + ".partial")
        for key, path in output_paths.items()
    }
    for path in working_paths.values():
        path.unlink(missing_ok=True)

    map_config = MapAnnotationConfig(
        near_distance_m=args.near_distance_m,
        lane_half_width_m=args.lane_half_width_m,
        arm_angle_threshold_deg=args.arm_angle_threshold_deg,
        max_map_match_distance_m=args.max_map_match_distance_m,
        max_map_match_heading_error_deg=args.max_map_match_heading_error_deg,
        include_stop_controlled=not args.signalized_only,
    )
    size_config = AgentSizeConfig(
        vehicle_large_length_m=args.vehicle_large_length_m,
        vehicle_motorcycle_max_width_m=args.vehicle_motorcycle_max_width_m,
        vehicle_motorcycle_max_length_m=args.vehicle_motorcycle_max_length_m,
        cyclist_ebike_min_speed_mps=args.cyclist_ebike_min_speed_mps,
        pedestrian_child_max_height_m=args.pedestrian_child_max_height_m,
    )
    action_config = AgentActionConfig(
        stop_speed_mps=args.action_stop_speed_mps,
        acceleration_threshold_mps2=(
            args.action_acceleration_threshold_mps2
        ),
        deceleration_threshold_mps2=(
            args.action_deceleration_threshold_mps2
        ),
    )
    total = selected_task_count(paths, args.start_index, args.max_scenarios)
    aggregate = new_accumulator()
    by_source = {path.name: new_accumulator() for path in paths}

    with gzip.open(
        working_paths["road_details"], "wt", newline="", encoding="utf-8"
    ) as road_stream, gzip.open(
        working_paths["agent_details"], "wt", newline="", encoding="utf-8"
    ) as agent_stream, gzip.open(
        working_paths["action_details"], "wt", newline="", encoding="utf-8"
    ) as action_stream, working_paths["errors"].open(
        "w", encoding="utf-8"
    ) as error_stream:
        road_writer = csv.DictWriter(road_stream, fieldnames=ROAD_DETAIL_FIELDS)
        agent_writer = csv.DictWriter(agent_stream, fieldnames=AGENT_DETAIL_FIELDS)
        action_writer = csv.DictWriter(
            action_stream,
            fieldnames=ACTION_DETAIL_FIELDS,
        )
        road_writer.writeheader()
        agent_writer.writeheader()
        action_writer.writeheader()

        def consume(result: dict) -> None:
            update_accumulator(aggregate, result)
            update_accumulator(by_source[result["source_file"]], result)
            if result["error"] is None:
                road_writer.writerow(result["road_record"])
                agent_writer.writerows(result["agent_records"])
                action_writer.writerows(result["action_records"])
            else:
                error_stream.write(json.dumps(result, ensure_ascii=False) + "\n")

        tasks = iter_selected_tasks(paths, args.start_index, args.max_scenarios)
        description = (
            f"Frame {args.frame_number} road/size + all-frame actions"
        )
        with tqdm(total=total, desc=description, unit="scenario") as progress:
            if args.workers == 1:
                for task in tasks:
                    consume(
                        process_scenario(
                            *task,
                            map_config,
                            size_config,
                            action_config,
                            frame_index,
                        )
                    )
                    progress.update(1)
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    pending = {}
                    for task in tasks:
                        future = executor.submit(
                            process_scenario,
                            *task,
                            map_config,
                            size_config,
                            action_config,
                            frame_index,
                        )
                        pending[future] = None
                        if len(pending) >= args.workers * 3:
                            done, _ = wait(pending, return_when=FIRST_COMPLETED)
                            for completed in done:
                                pending.pop(completed)
                                consume(completed.result())
                                progress.update(1)
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for completed in done:
                            pending.pop(completed)
                            consume(completed.result())
                            progress.update(1)

    road_rows = road_count_rows(aggregate, "ALL", args.frame_number)
    agent_rows = agent_count_rows(aggregate, "ALL", args.frame_number)
    action_rows = action_count_rows(aggregate, "ALL")
    action_frame_rows = action_count_rows_by_frame(aggregate, "ALL")
    for source_file in sorted(by_source):
        road_rows.extend(
            road_count_rows(by_source[source_file], source_file, args.frame_number)
        )
        agent_rows.extend(
            agent_count_rows(by_source[source_file], source_file, args.frame_number)
        )
        action_rows.extend(
            action_count_rows(by_source[source_file], source_file)
        )
        action_frame_rows.extend(
            action_count_rows_by_frame(
                by_source[source_file],
                source_file,
            )
        )
    with working_paths["road_counts"].open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=ROAD_COUNT_FIELDS)
        writer.writeheader()
        writer.writerows(road_rows)
    with working_paths["agent_counts"].open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=AGENT_COUNT_FIELDS)
        writer.writeheader()
        writer.writerows(agent_rows)
    with working_paths["action_counts"].open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=ACTION_COUNT_FIELDS)
        writer.writeheader()
        writer.writerows(action_rows)
    with working_paths["action_counts_by_frame"].open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=ACTION_COUNT_FIELDS)
        writer.writeheader()
        writer.writerows(action_frame_rows)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "input_files": [str(path) for path in paths],
        "frame_number": args.frame_number,
        "frame_index": frame_index,
        "workers": args.workers,
        "start_index": args.start_index,
        "max_scenarios": args.max_scenarios,
        "map_config": asdict(map_config),
        "agent_size_config": asdict(size_config),
        "agent_action_config": asdict(action_config),
        "methodology": {
            "road_context": (
                "One mutually exclusive SDC road label at only the requested frame."
            ),
            "category_labels_zh": CATEGORY_LABELS_ZH,
            "intersection_scope": (
                "Junctions detected by the existing signalized/stop-controlled "
                "map annotator."
            ),
            "parking_lot_proxy": (
                "SDC center covered by a WOMD driveway polygon; not parking-lot "
                "ground truth."
            ),
            "agent_scope": (
                "Every track with a valid state at the requested frame, including SDC."
            ),
            "agent_size_warning": (
                "All six requested subtypes are dimension proxies. WOMD has no axle, "
                "motor-power, pedal, or age attributes needed for regulatory labels."
            ),
            "agent_action_scope": (
                "Every valid state of every track is labeled. Counts use agent-frame "
                "units; valid past/future states provide turn and lane-change context."
            ),
            "agent_action_reference": (
                "Action ids, priority, valid-frame windows, and lane-change rules "
                "follow Demand2Scenario/d2s/processor/get_action.py."
            ),
            "agent_action_kinematics": (
                "WOMD stores global velocity but no acceleration. Longitudinal "
                "velocity is obtained by rotating global velocity into the agent "
                "frame; global velocity is differentiated over valid timestamps "
                "and rotated to obtain longitudinal acceleration."
            ),
        },
        "standard_references": {
            "vehicle": {
                "rule": "length > 19 ft (5.7912 m) -> large-vehicle proxy",
                "source": "AASHTO passenger-car design length via FHWA",
                "url": (
                    "https://highways.dot.gov/safety/hsip/xings/"
                    "highway-rail-crossing-handbook-third-edition/"
                    "b-appendix-components-highway-rail"
                ),
            },
            "motorcycle": {
                "rule": (
                    "TYPE_VEHICLE box width <= 1.20 m and length <= 3.00 m "
                    "-> motorcycle proxy"
                ),
                "source": (
                    "FHWA typical motorcycle width, expanded for the rider and "
                    "WOMD full-object bounding box"
                ),
                "reference_typical_max_width_m": (
                    FHWA_MOTORCYCLE_TYPICAL_MAX_WIDTH_M
                ),
                "url": (
                    "https://www.fhwa.dot.gov/policyinformation/tmguide/"
                    "tmg_2022/Appendix-I-Motorcycle-Data-Collection-Methods.pdf"
                ),
            },
            "cyclist": {
                "rule": (
                    "current-frame speed >= 24 km/h (6.6667 m/s) -> e-bike "
                    "proxy; otherwise bicycle proxy"
                ),
                "source": "FHWA conventional-bicycle 85th-percentile speed",
                "url": (
                    "https://www.fhwa.dot.gov/publications/research/safety/"
                    "pedbike/05137/chapter2.cfm"
                ),
            },
            "pedestrian": {
                "rule": "height < 4 ft 9 in (1.4478 m) -> child-height proxy",
                "source": "NHTSA child-restraint height reference",
                "url": (
                    "https://www.nhtsa.gov/book/countermeasures-that-work/"
                    "seat-belts-and-child-restraints/countermeasures/other-strategies-4"
                ),
            },
            "regulatory_limit": {
                "source": (
                    "Motorcycle and e-bike definitions require wheel, motor-power, "
                    "pedal, or assisted-speed attributes unavailable in WOMD; all "
                    "reported subtypes are observational proxies."
                ),
                "fhwa_url": (
                    "https://www.fhwa.dot.gov/policyinformation/tmguide/"
                    "tmg_2013/vehicle-types.cfm"
                ),
                "cpsc_url": "https://www.cpsc.gov/FAQ/Bicycles",
            },
        },
        "aggregate": serializable_accumulator(aggregate),
        "table_row_counts": {
            "road_counts": len(road_rows),
            "agent_counts": len(agent_rows),
            "action_counts": len(action_rows),
            "action_counts_by_frame": len(action_frame_rows),
        },
        "output_files": {key: str(path) for key, path in output_paths.items()},
    }
    working_paths["summary"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for key in output_paths:
        if key != "summary":
            working_paths[key].replace(output_paths[key])
    working_paths["summary"].replace(output_paths["summary"])
    return payload


def main() -> None:
    payload = run_statistics(parse_args())
    aggregate = payload["aggregate"]
    print(
        json.dumps(
            {
                "scenarios": aggregate["scenarios"],
                "errors": aggregate["errors"],
                "frame_number": payload["frame_number"],
                "road_counts": aggregate["road_counts"],
                "agent_size_counts": aggregate["agent_size_counts"],
                "agent_action_counts": aggregate["agent_action_counts"],
                "agent_diagnostics": aggregate["agent_diagnostics"],
                "action_diagnostics": aggregate["action_diagnostics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for path in payload["output_files"].values():
        print(path)


if __name__ == "__main__":
    main()
