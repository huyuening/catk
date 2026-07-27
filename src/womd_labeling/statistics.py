from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from .artifacts import (
    artifact_identity_matches,
    artifact_identity_record,
    artifact_matches,
    artifact_record,
    file_identity,
    input_fingerprint,
    stable_fingerprint,
)
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
SCHEMA_VERSION = "womd-current-frame-road-subtype-all-frame-action-statistics-v6"


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
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Validate and reuse completed per-TFRecord statistics shards. "
            "Enabled by default; --overwrite takes precedence."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def statistics_run_configuration(args: argparse.Namespace) -> dict:
    return {
        "frame_number": args.frame_number,
        "start_index": args.start_index,
        "max_scenarios": args.max_scenarios,
        "map_config": asdict(
            MapAnnotationConfig(
                near_distance_m=args.near_distance_m,
                lane_half_width_m=args.lane_half_width_m,
                arm_angle_threshold_deg=args.arm_angle_threshold_deg,
                max_map_match_distance_m=args.max_map_match_distance_m,
                max_map_match_heading_error_deg=(
                    args.max_map_match_heading_error_deg
                ),
                include_stop_controlled=not args.signalized_only,
            )
        ),
        "agent_size_config": asdict(
            AgentSizeConfig(
                vehicle_large_length_m=args.vehicle_large_length_m,
                vehicle_motorcycle_max_width_m=(
                    args.vehicle_motorcycle_max_width_m
                ),
                vehicle_motorcycle_max_length_m=(
                    args.vehicle_motorcycle_max_length_m
                ),
                cyclist_ebike_min_speed_mps=(
                    args.cyclist_ebike_min_speed_mps
                ),
                pedestrian_child_max_height_m=(
                    args.pedestrian_child_max_height_m
                ),
            )
        ),
        "agent_action_config": asdict(
            AgentActionConfig(
                stop_speed_mps=args.action_stop_speed_mps,
                acceleration_threshold_mps2=(
                    args.action_acceleration_threshold_mps2
                ),
                deceleration_threshold_mps2=(
                    args.action_deceleration_threshold_mps2
                ),
            )
        ),
    }


def statistics_run_fingerprint(
    args: argparse.Namespace,
    paths: Iterable[Path],
) -> str:
    identity = input_fingerprint(paths)
    return stable_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "input_fingerprint": identity["fingerprint"],
            "configuration": statistics_run_configuration(args),
        }
    )


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
    decoded_counts = {}
    observed_types_by_frame = defaultdict(set)
    frame_action_totals = Counter()
    frame_denominators = Counter()
    for key, count in counts.items():
        frame_index, object_type, action_id = (
            decode_agent_action_frame_key(key)
        )
        decoded_counts[(frame_index, object_type, action_id)] = count
        observed_types_by_frame[frame_index].add(object_type)
        frame_action_totals[(frame_index, action_id)] += count
    for key, count in frame_type_totals.items():
        frame_text, object_type = key.split("\t", 1)
        frame_index = int(frame_text)
        observed_types_by_frame[frame_index].add(object_type)
        frame_denominators[frame_index] += count
    observed_frames = sorted(observed_types_by_frame)
    rows = []

    for frame_index in observed_frames:
        observed_types = observed_types_by_frame[frame_index]
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
                count = decoded_counts.get(
                    (frame_index, object_type, action_id), 0
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

        frame_denominator = frame_denominators[frame_index]
        for action_id, action_name in ACTION_NAMES.items():
            count = frame_action_totals[(frame_index, action_id)]
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


COUNTER_ACCUMULATOR_KEYS = (
    "dataset_current_time_index_counts",
    "road_counts",
    "road_unknown_reason_counts",
    "agent_diagnostics",
    "agent_type_counts",
    "agent_size_counts",
    "action_diagnostics",
    "agent_action_type_counts",
    "agent_action_counts",
    "agent_action_frame_type_counts",
    "agent_action_frame_counts",
)

SCALAR_ACCUMULATOR_KEYS = (
    "scenarios",
    "errors",
    "current_index_mismatches",
    "scenarios_with_driveway_polygons",
    "driveway_polygons",
)


def raw_accumulator_payload(accumulator: dict) -> dict:
    payload = {
        key: int(accumulator[key]) for key in SCALAR_ACCUMULATOR_KEYS
    }
    payload.update(
        {
            key: dict(accumulator[key])
            for key in COUNTER_ACCUMULATOR_KEYS
        }
    )
    payload["agent_dimension_stats"] = {
        key: dict(stats)
        for key, stats in accumulator["agent_dimension_stats"].items()
    }
    return payload


def accumulator_from_raw(payload: dict) -> dict:
    accumulator = new_accumulator()
    for key in SCALAR_ACCUMULATOR_KEYS:
        accumulator[key] = int(payload[key])
    for key in COUNTER_ACCUMULATOR_KEYS:
        accumulator[key].update(payload[key])
    accumulator["agent_dimension_stats"] = {
        key: dict(stats)
        for key, stats in payload["agent_dimension_stats"].items()
    }
    return accumulator


def merge_accumulators(target: dict, source: dict) -> None:
    for key in SCALAR_ACCUMULATOR_KEYS:
        target[key] += source[key]
    for key in COUNTER_ACCUMULATOR_KEYS:
        target[key].update(source[key])
    for dimension_key, source_stats in source[
        "agent_dimension_stats"
    ].items():
        target_stats = target["agent_dimension_stats"].setdefault(
            dimension_key,
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
        target_stats["count"] += source_stats["count"]
        for dimension in ("length_m", "width_m", "height_m"):
            target_stats[f"sum_{dimension}"] += source_stats[
                f"sum_{dimension}"
            ]
            target_stats[f"min_{dimension}"] = min(
                target_stats[f"min_{dimension}"],
                source_stats[f"min_{dimension}"],
            )
            target_stats[f"max_{dimension}"] = max(
                target_stats[f"max_{dimension}"],
                source_stats[f"max_{dimension}"],
            )


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


STATISTICS_SHARD_SCHEMA_VERSION = "catk-womd-statistics-shard-v1"


def _selected_record_range(
    *,
    shard_global_start: int,
    shard_record_count: int,
    selection_start: int,
    selection_count: int | None,
) -> range:
    shard_global_end = shard_global_start + shard_record_count
    selection_end = (
        None
        if selection_count is None
        else selection_start + selection_count
    )
    selected_global_start = max(shard_global_start, selection_start)
    selected_global_end = shard_global_end
    if selection_end is not None:
        selected_global_end = min(selected_global_end, selection_end)
    if selected_global_end <= selected_global_start:
        return range(0, 0)
    return range(
        selected_global_start - shard_global_start,
        selected_global_end - shard_global_start,
    )


def _statistics_shard_paths(
    shards_dir: Path,
    source_order: int,
    source_file: str,
) -> dict[str, Path]:
    prefix = shards_dir / f"{source_order:05d}-{source_file}"
    return {
        "road_details": prefix.with_name(
            prefix.name + ".current-frame-road-types.csv.gz"
        ),
        "agent_details": prefix.with_name(
            prefix.name + ".current-frame-agent-sizes.csv.gz"
        ),
        "action_details": prefix.with_name(
            prefix.name + ".agent-actions-by-frame.csv.gz"
        ),
        "road_counts": prefix.with_name(
            prefix.name + ".current-frame-road-type-counts.csv"
        ),
        "agent_counts": prefix.with_name(
            prefix.name + ".current-frame-agent-size-counts.csv"
        ),
        "action_counts": prefix.with_name(
            prefix.name + ".agent-action-counts.csv"
        ),
        "action_counts_by_frame": prefix.with_name(
            prefix.name + ".agent-action-counts-by-frame.csv"
        ),
        "errors": prefix.with_name(prefix.name + ".errors.jsonl"),
        "summary": prefix.with_name(prefix.name + ".summary.json"),
    }


def _global_statistics_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "road_counts": output_dir / "current_frame_road_type_counts.csv",
        "agent_counts": output_dir / "current_frame_agent_size_counts.csv",
        "action_counts": output_dir / "agent_action_counts.csv",
        "action_counts_by_frame": (
            output_dir / "agent_action_counts_by_frame.csv"
        ),
        "errors": output_dir / "errors.jsonl",
        "summary": output_dir / "summary.json",
    }


def _completed_shard(
    summary_path: Path,
    *,
    run_fingerprint: str,
) -> dict | None:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") != STATISTICS_SHARD_SCHEMA_VERSION
        or payload.get("run_fingerprint") != run_fingerprint
        or not isinstance(payload.get("raw_accumulator"), dict)
    ):
        return None
    artifacts = payload.get("output_artifacts")
    if (
        not isinstance(artifacts, dict)
        or not artifacts
        or not all(
            artifact_identity_matches(record)
            for record in artifacts.values()
        )
    ):
        return None
    return payload


def _completed_statistics_run(
    summary_path: Path,
    *,
    run_fingerprint: str,
) -> dict | None:
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("run_fingerprint") != run_fingerprint
    ):
        return None
    artifacts = payload.get("output_artifacts")
    shard_manifests = payload.get("shard_manifests")
    if (
        not isinstance(artifacts, dict)
        or not artifacts
        or not all(
            artifact_identity_matches(record)
            for record in artifacts.values()
        )
        or not isinstance(shard_manifests, list)
    ):
        return None
    for shard in shard_manifests:
        summary_record = shard.get("summary_artifact")
        if not artifact_matches(summary_record):
            return None
        completed = _completed_shard(
            Path(summary_record["path"]),
            run_fingerprint=shard["run_fingerprint"],
        )
        if completed is None:
            return None
    payload = dict(payload)
    payload["resumed"] = True
    return payload


def _write_counter_tables(
    accumulator: dict,
    source_file: str,
    frame_number: int,
    paths: dict[str, Path],
) -> dict[str, int]:
    row_groups = (
        (
            "road_counts",
            ROAD_COUNT_FIELDS,
            road_count_rows(accumulator, source_file, frame_number),
        ),
        (
            "agent_counts",
            AGENT_COUNT_FIELDS,
            agent_count_rows(accumulator, source_file, frame_number),
        ),
        (
            "action_counts",
            ACTION_COUNT_FIELDS,
            action_count_rows(accumulator, source_file),
        ),
        (
            "action_counts_by_frame",
            ACTION_COUNT_FIELDS,
            action_count_rows_by_frame(accumulator, source_file),
        ),
    )
    row_counts = {}
    for key, fields, rows in row_groups:
        with paths[key].open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        row_counts[key] = len(rows)
    return row_counts


def run_statistics(args: argparse.Namespace) -> dict:
    """Compute resumable, per-TFRecord statistics and bounded split summaries."""
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.frame_number < 1:
        raise ValueError("--frame-number must be at least 1")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.max_scenarios is not None and args.max_scenarios < 1:
        raise ValueError("--max-scenarios must be positive")

    paths = resolve_tfrecord_paths(args.input_path)
    input_identity = input_fingerprint(paths)
    output_dir = args.output_dir.expanduser().resolve()
    shards_dir = output_dir / "shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)
    global_paths = _global_statistics_paths(output_dir)
    frame_index = args.frame_number - 1

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
        vehicle_motorcycle_max_width_m=(
            args.vehicle_motorcycle_max_width_m
        ),
        vehicle_motorcycle_max_length_m=(
            args.vehicle_motorcycle_max_length_m
        ),
        cyclist_ebike_min_speed_mps=args.cyclist_ebike_min_speed_mps,
        pedestrian_child_max_height_m=(
            args.pedestrian_child_max_height_m
        ),
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
    run_configuration = statistics_run_configuration(args)
    run_fingerprint = statistics_run_fingerprint(args, paths)

    if args.resume and not args.overwrite:
        completed = _completed_statistics_run(
            global_paths["summary"],
            run_fingerprint=run_fingerprint,
        )
        if completed is not None:
            return completed
    if not args.resume and not args.overwrite:
        existing = [
            path for path in global_paths.values() if path.exists()
        ]
        existing.extend(shards_dir.glob("*"))
        if existing:
            raise FileExistsError(
                f"Output exists: {existing[0]}. Use --overwrite to replace "
                "it or --resume to validate and reuse it."
            )

    record_counts = {
        path: count_tfrecord_records(path) for path in paths
    }
    selection_by_path = {}
    shard_global_starts = {}
    next_global_index = 0
    for path in paths:
        shard_global_starts[path] = next_global_index
        selection_by_path[path] = _selected_record_range(
            shard_global_start=next_global_index,
            shard_record_count=record_counts[path],
            selection_start=args.start_index,
            selection_count=args.max_scenarios,
        )
        next_global_index += record_counts[path]
    total = sum(len(indices) for indices in selection_by_path.values())

    aggregate = new_accumulator()
    shard_manifests = []
    shard_error_paths = []
    table_row_counts = {
        "road_counts": 0,
        "agent_counts": 0,
        "action_counts": 0,
        "action_counts_by_frame": 0,
    }
    executor = (
        ProcessPoolExecutor(max_workers=args.workers)
        if args.workers > 1
        else None
    )
    try:
        with tqdm(
            total=total,
            desc=(
                f"Frame {args.frame_number} road/size + all-frame actions"
            ),
            unit="scenario",
        ) as progress:
            for source_order, path in enumerate(paths):
                selected_indices = selection_by_path[path]
                if not selected_indices:
                    continue
                source_identity = file_identity(path)
                shard_paths = _statistics_shard_paths(
                    shards_dir,
                    source_order,
                    path.name,
                )
                shard_fingerprint = stable_fingerprint(
                    {
                        "schema_version": STATISTICS_SHARD_SCHEMA_VERSION,
                        "source": source_identity,
                        "selection": {
                            "start": selected_indices.start,
                            "stop": selected_indices.stop,
                        },
                        "configuration": run_configuration,
                    }
                )
                completed_shard = None
                if args.resume and not args.overwrite:
                    completed_shard = _completed_shard(
                        shard_paths["summary"],
                        run_fingerprint=shard_fingerprint,
                    )
                if completed_shard is not None:
                    source_accumulator = accumulator_from_raw(
                        completed_shard["raw_accumulator"]
                    )
                    merge_accumulators(aggregate, source_accumulator)
                    for key, count in completed_shard[
                        "table_row_counts"
                    ].items():
                        table_row_counts[key] += int(count)
                    progress.update(len(selected_indices))
                else:
                    partial_paths = {
                        key: output_path.with_name(
                            output_path.name + ".partial"
                        )
                        for key, output_path in shard_paths.items()
                    }
                    for partial_path in partial_paths.values():
                        partial_path.unlink(missing_ok=True)
                    source_accumulator = new_accumulator()
                    with gzip.open(
                        partial_paths["road_details"],
                        "wt",
                        newline="",
                        encoding="utf-8",
                    ) as road_stream, gzip.open(
                        partial_paths["agent_details"],
                        "wt",
                        newline="",
                        encoding="utf-8",
                    ) as agent_stream, gzip.open(
                        partial_paths["action_details"],
                        "wt",
                        newline="",
                        encoding="utf-8",
                    ) as action_stream, partial_paths["errors"].open(
                        "w",
                        encoding="utf-8",
                    ) as error_stream:
                        road_writer = csv.DictWriter(
                            road_stream,
                            fieldnames=ROAD_DETAIL_FIELDS,
                        )
                        agent_writer = csv.DictWriter(
                            agent_stream,
                            fieldnames=AGENT_DETAIL_FIELDS,
                        )
                        action_writer = csv.DictWriter(
                            action_stream,
                            fieldnames=ACTION_DETAIL_FIELDS,
                        )
                        road_writer.writeheader()
                        agent_writer.writeheader()
                        action_writer.writeheader()

                        def consume(result: dict) -> None:
                            update_accumulator(
                                source_accumulator,
                                result,
                            )
                            if result["error"] is None:
                                road_writer.writerow(result["road_record"])
                                agent_writer.writerows(
                                    result["agent_records"]
                                )
                                action_writer.writerows(
                                    result["action_records"]
                                )
                            else:
                                error_stream.write(
                                    json.dumps(
                                        result,
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )

                        if executor is None:
                            for record_index, raw_payload in iter_tfrecord(
                                path
                            ):
                                if record_index < selected_indices.start:
                                    continue
                                if record_index >= selected_indices.stop:
                                    break
                                consume(
                                    process_scenario(
                                        str(path),
                                        path.name,
                                        record_index,
                                        (
                                            shard_global_starts[path]
                                            + record_index
                                        ),
                                        raw_payload,
                                        map_config,
                                        size_config,
                                        action_config,
                                        frame_index,
                                    )
                                )
                                progress.update(1)
                        else:
                            pending = {}
                            result_buffer = {}
                            next_result_index = selected_indices.start

                            def drain_completed() -> None:
                                nonlocal next_result_index
                                done, _ = wait(
                                    pending,
                                    return_when=FIRST_COMPLETED,
                                )
                                for future in done:
                                    record_index = pending.pop(future)
                                    result_buffer[record_index] = (
                                        future.result()
                                    )
                                    progress.update(1)
                                while next_result_index in result_buffer:
                                    consume(
                                        result_buffer.pop(
                                            next_result_index
                                        )
                                    )
                                    next_result_index += 1

                            for record_index, raw_payload in iter_tfrecord(
                                path
                            ):
                                if record_index < selected_indices.start:
                                    continue
                                if record_index >= selected_indices.stop:
                                    break
                                future = executor.submit(
                                    process_scenario,
                                    str(path),
                                    path.name,
                                    record_index,
                                    (
                                        shard_global_starts[path]
                                        + record_index
                                    ),
                                    raw_payload,
                                    map_config,
                                    size_config,
                                    action_config,
                                    frame_index,
                                )
                                pending[future] = record_index
                                if len(pending) >= args.workers * 3:
                                    drain_completed()
                            while pending:
                                drain_completed()
                            if result_buffer:
                                raise RuntimeError(
                                    "Could not restore scenario order for "
                                    f"{path}: {sorted(result_buffer)}"
                                )

                    shard_table_counts = _write_counter_tables(
                        source_accumulator,
                        path.name,
                        args.frame_number,
                        partial_paths,
                    )
                    if file_identity(path) != source_identity:
                        raise RuntimeError(
                            f"Source TFRecord changed while processing: {path}"
                        )
                    shard_payload = {
                        "schema_version": (
                            STATISTICS_SHARD_SCHEMA_VERSION
                        ),
                        "run_fingerprint": shard_fingerprint,
                        "source": source_identity,
                        "source_file": path.name,
                        "source_order": source_order,
                        "selection": {
                            "start": selected_indices.start,
                            "stop": selected_indices.stop,
                            "count": len(selected_indices),
                        },
                        "raw_accumulator": raw_accumulator_payload(
                            source_accumulator
                        ),
                        "aggregate": serializable_accumulator(
                            source_accumulator
                        ),
                        "table_row_counts": shard_table_counts,
                        "output_files": {
                            key: str(output_path)
                            for key, output_path in shard_paths.items()
                        },
                    }
                    shard_payload["output_artifacts"] = {
                        key: artifact_identity_record(
                            partial_paths[key],
                            logical_path=shard_paths[key],
                        )
                        for key in shard_paths
                        if key != "summary"
                    }
                    partial_paths["summary"].write_text(
                        json.dumps(
                            shard_payload,
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    for key, output_path in shard_paths.items():
                        if key != "summary":
                            partial_paths[key].replace(output_path)
                    partial_paths["summary"].replace(
                        shard_paths["summary"]
                    )
                    merge_accumulators(aggregate, source_accumulator)
                    for key, count in shard_table_counts.items():
                        table_row_counts[key] += count
                    completed_shard = shard_payload

                shard_error_paths.append(shard_paths["errors"])
                shard_manifests.append(
                    {
                        "source_file": path.name,
                        "source_order": source_order,
                        "run_fingerprint": shard_fingerprint,
                        "summary_artifact": artifact_record(
                            shard_paths["summary"]
                        ),
                    }
                )
                del source_accumulator
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    partial_global_paths = {
        key: path.with_name(path.name + ".partial")
        for key, path in global_paths.items()
    }
    for partial_path in partial_global_paths.values():
        partial_path.unlink(missing_ok=True)
    aggregate_table_counts = _write_counter_tables(
        aggregate,
        "ALL",
        args.frame_number,
        partial_global_paths,
    )
    for key, count in aggregate_table_counts.items():
        table_row_counts[key] += count
    with partial_global_paths["errors"].open(
        "w",
        encoding="utf-8",
    ) as output:
        for error_path in shard_error_paths:
            with error_path.open("r", encoding="utf-8") as source:
                for line in source:
                    output.write(line)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "input_files": [str(path) for path in paths],
        "input_identity": input_identity,
        "frame_number": args.frame_number,
        "frame_index": frame_index,
        "workers": args.workers,
        "start_index": args.start_index,
        "max_scenarios": args.max_scenarios,
        "resume": args.resume,
        "map_config": asdict(map_config),
        "agent_size_config": asdict(size_config),
        "agent_action_config": asdict(action_config),
        "methodology": {
            "road_context": (
                "One mutually exclusive SDC road label at the requested frame."
            ),
            "agent_scope": (
                "Every valid agent at the requested frame is size-labeled."
            ),
            "agent_action_scope": (
                "Every valid state of every track is action-labeled; counts "
                "use agent-frame units."
            ),
            "storage": (
                "Scenario details are committed per TFRecord shard; the split "
                "summary and count tables contain bounded aggregates."
            ),
        },
        "aggregate": serializable_accumulator(aggregate),
        "table_row_counts": table_row_counts,
        "shard_output_directory": str(shards_dir),
        "shard_manifests": shard_manifests,
        "output_files": {
            key: str(path) for key, path in global_paths.items()
        },
    }
    payload["output_artifacts"] = {
        key: artifact_identity_record(
            partial_global_paths[key],
            logical_path=global_paths[key],
        )
        for key in global_paths
        if key != "summary"
    }
    partial_global_paths["summary"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for key, output_path in global_paths.items():
        if key != "summary":
            partial_global_paths[key].replace(output_path)
    partial_global_paths["summary"].replace(global_paths["summary"])
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
