from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

from .visualize import resolve_annotation_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATISTICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "road_type_statistics"
    / "training"
)
DEFAULT_MAP_ANNOTATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "map_annotations"
    / "training.tfrecord-00000-of-01000.map-annotations.jsonl.gz"
)
DEFAULT_OUTPUT_PREFIX = DEFAULT_STATISTICS_DIR / "womd_statistics_mixed_three_panel_en"
DEFAULT_FIGURE_WIDTH_CM = 16.0
AGGREGATE_SCHEMA_VERSION = "catk-womd-aggregate-visualization-v1"
FIGURE_ASPECT_RATIO = 13.0 / 16.0
DEFAULT_FONT_SIZE_PT = 7.5
PANEL_LABEL_SIZE_PT = 9.0

matplotlib_cache = Path(tempfile.gettempdir()) / "catk-womd-matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import colors as mpl_colors  # noqa: E402
from matplotlib import ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, Wedge  # noqa: E402
import numpy as np  # noqa: E402

from .agent_action_classification import (  # noqa: E402
    decode_agent_action_frame_key,
    decode_agent_action_key,
)
from .agent_size_classification import decode_agent_size_key  # noqa: E402
from .artifacts import (  # noqa: E402
    artifact_record,
    input_fingerprint,
    stable_fingerprint,
)
from .statistics import SCHEMA_VERSION as STATISTICS_SCHEMA_VERSION  # noqa: E402


FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
)

COLORS = {
    "intersection": "#D55E00",
    "road": "#009E73",
    "near": "#56B4E9",
    "parking": "#E69F00",
    "vehicle": "#0072B2",
    "cyclist": "#E69F00",
    "pedestrian": "#CC79A7",
    "heat_low": "#EDF7F5",
    "heat_mid": "#5DB7AA",
    "heat_high": "#075C56",
    "missing": "#EEF1F3",
    "on_dark": "#F8FBFA",
    "grid": "#D7DCE0",
    "text": "#253238",
    "muted": "#68747B",
}

ROAD_SUNBURST_PALETTE = {
    "Freeway": "#4D78A8",
    "Mainline": "#7FA8D1",
    "Ramp": "#B7D0E8",
    "Urban road": "#4B9A7B",
    "Intersection": "#79BEA1",
    "Road segment": "#A8D5BF",
    "Parking lot": "#E4C45A",
}

AGENT_SUNBURST_PALETTE = {
    "Vehicle": "#C65F6E",
    "Large vehicle": "#E19AA3",
    "Small vehicle": "#D57682",
    "Motorcyclist": "#F0C5CA",
    "Cyclist": "#D6A12B",
    "Pedestrian": "#A94D91",
    "Adult": "#C985B8",
    "Child": "#E2B9D8",
}

SUNBURST_HOLE_RADIUS = 0.22
SUNBURST_PARENT_RADIUS = 0.84
SUNBURST_OUTER_RADIUS = 1.50
SUNBURST_INTERNAL_LABEL_MIN_DEG = 18.0
SUNBURST_PARENT_INTERNAL_LABEL_MIN_DEG = 45.0
SUNBURST_CALLOUT_GAP = 0.24


ROAD_INTERSECTION_ORDER = (
    ("FOUR_ARM_INTERSECTION", "Four-legged"),
    ("THREE_ARM_INTERSECTION", "Three-legged"),
    ("OTHER_INTERSECTION", "Other"),
)

SUNBURST_MIN_CHILD_BLEND = 0.18
SUNBURST_MAX_CHILD_BLEND = 0.80

SIZE_ORDER = (
    ("Vehicle", "LARGE_VEHICLE_PROXY", "Large vehicle", "vehicle"),
    ("Vehicle", "SMALL_VEHICLE_PROXY", "Small vehicle", "vehicle"),
    ("Vehicle", "MOTORCYCLE_PROXY", "Motorcyclist", "vehicle"),
    ("Cyclist", "E_BIKE_PROXY", "E-bike", "cyclist"),
    ("Cyclist", "BICYCLE_PROXY", "Bicycle", "cyclist"),
    ("Pedestrian", "ADULT_PEDESTRIAN_PROXY", "Adult", "pedestrian"),
    ("Pedestrian", "CHILD_PEDESTRIAN_PROXY", "Child", "pedestrian"),
)

ACTION_ORDER = {
    "TYPE_VEHICLE": (
        (2, "U-turn"),
        (3, "Left turn"),
        (4, "Left lane change"),
        (5, "Deceleration"),
        (6, "Keep"),
        (7, "Acceleration"),
        (8, "Right lane change"),
        (9, "Right turn"),
        (1, "Stop"),
    ),
    "TYPE_CYCLIST": (
        (3, "Left turn"),
        (5, "Deceleration"),
        (6, "Keep"),
        (7, "Acceleration"),
        (9, "Right turn"),
        (1, "Stop"),
    ),
    "TYPE_PEDESTRIAN": (
        (5, "Deceleration"),
        (6, "Keep"),
        (7, "Acceleration"),
        (1, "Stop"),
    ),
}

ACTION_GROUPS = (
    ("TYPE_VEHICLE", "Vehicle", "vehicle"),
    ("TYPE_CYCLIST", "Cyclist", "cyclist"),
    ("TYPE_PEDESTRIAN", "Pedestrian", "pedestrian"),
)

ACTION_COLUMNS = (
    (2, "U-turn", "U-turn"),
    (3, "Left turn", "Left turn"),
    (4, "Left lane change", "Left lane change"),
    (5, "Deceleration", "Deceleration"),
    (6, "Keep", "Keep"),
    (7, "Acceleration", "Acceleration"),
    (8, "Right lane change", "Right lane change"),
    (9, "Right turn", "Right turn"),
    (1, "Stop", "Stop"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recount WOMD road, agent-subtype, and all-frame action labels and "
            "draw one three-panel figure."
        )
    )
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        default=DEFAULT_STATISTICS_DIR,
    )
    parser.add_argument(
        "--map-annotation-path",
        nargs="+",
        default=[str(DEFAULT_MAP_ANNOTATION_PATH)],
        help=(
            "Map-annotation JSONL/JSONL.GZ files, directories, or globs "
            "used for panel a."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
    )
    parser.add_argument(
        "--html-fragment",
        type=Path,
        default=None,
        help="Optional Codex inline-visualization fragment containing the SVG.",
    )
    parser.add_argument(
        "--width-cm",
        type=float,
        default=DEFAULT_FIGURE_WIDTH_CM,
        help="Figure width in centimeters; height follows the fixed aspect ratio.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing complete aggregate visualization set.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _font_name() -> str:
    for path in FONT_CANDIDATES:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
    return "DejaVu Sans"


def _read_gzip_csv(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def recount_road_types(statistics_dir: Path) -> dict:
    intersection = Counter()
    lane_counts = Counter()
    parking = 0
    unknown = 0
    rows = 0

    path = statistics_dir / "current_frame_road_types_three_class.csv.gz"
    for record in _read_gzip_csv(path):
        rows += 1
        category = record["category"]
        subtype = record["subtype"]
        if category == "INTERSECTION":
            intersection[subtype] += 1
        elif category == "ROAD_SEGMENT" and subtype.startswith("LANE_COUNT_"):
            lane_counts[int(subtype.removeprefix("LANE_COUNT_"))] += 1
        elif category == "PARKING_LOT_PROXY":
            parking += 1
        else:
            unknown += 1
    return {
        "rows": rows,
        "intersection": intersection,
        "lane_counts": lane_counts,
        "parking": parking,
        "unknown": unknown,
    }


def recount_road_hierarchy(
    paths: Path | Iterable[Path],
    frame_index: int = 10,
) -> dict:
    top_counts = Counter()
    child_counts = {
        "Freeway": Counter(),
        "Urban road": Counter(),
    }
    rows = 0
    unknown = 0
    errors = 0
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    for path in paths:
        path = Path(path)
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                annotation = json.loads(line)
                if annotation.get("error"):
                    errors += 1
                    continue
                frame = next(
                    (
                        item
                        for item in annotation.get("ego_frames", ())
                        if int(item.get("frame_index", -1)) == frame_index
                    ),
                    None,
                )
                if frame is None:
                    unknown += 1
                    continue
                rows += 1
                environment = frame.get("road_environment")
                region_type = frame.get("region_type")
                subtype = frame.get("road_environment_subtype")
                if environment == "FREEWAY" and subtype in {
                    "FREEWAY_MAINLINE",
                    "FREEWAY_RAMP",
                }:
                    child = (
                        "Mainline"
                        if subtype == "FREEWAY_MAINLINE"
                        else "Ramp"
                    )
                    top_counts["Freeway"] += 1
                    child_counts["Freeway"][child] += 1
                elif environment == "URBAN_STREET" and region_type in {
                    "INTERSECTION",
                    "ROAD_SEGMENT",
                }:
                    child = (
                        "Intersection"
                        if region_type == "INTERSECTION"
                        else "Road segment"
                    )
                    top_counts["Urban road"] += 1
                    child_counts["Urban road"][child] += 1
                elif environment == "PARKING_LOT":
                    top_counts["Urban road"] += 1
                    child_counts["Urban road"]["Parking lot"] += 1
                else:
                    unknown += 1
    return {
        "rows": rows,
        "top_counts": top_counts,
        "child_counts": child_counts,
        "unknown": unknown,
        "errors": errors,
    }


def recount_agent_sizes(statistics_dir: Path) -> dict:
    counts = Counter()
    rows = 0
    path = statistics_dir / "current_frame_agent_sizes.csv.gz"
    for record in _read_gzip_csv(path):
        rows += 1
        counts[record["size_class"]] += 1
    return {"rows": rows, "counts": counts}


def build_agent_hierarchy(sizes: dict) -> tuple[dict, ...]:
    counts = sizes["counts"]
    vehicle_children = {
        "Large vehicle": counts.get("LARGE_VEHICLE_PROXY", 0),
        "Small vehicle": counts.get("SMALL_VEHICLE_PROXY", 0),
        "Motorcyclist": counts.get("MOTORCYCLE_PROXY", 0),
    }
    pedestrian_children = {
        "Adult": counts.get("ADULT_PEDESTRIAN_PROXY", 0),
        "Child": counts.get("CHILD_PEDESTRIAN_PROXY", 0),
    }
    cyclist_count = counts.get("E_BIKE_PROXY", 0) + counts.get(
        "BICYCLE_PROXY",
        0,
    )
    return (
        {
            "label": "Vehicle",
            "count": sum(vehicle_children.values()),
            "children": vehicle_children,
        },
        {
            "label": "Cyclist",
            "count": cyclist_count,
            "children": {},
        },
        {
            "label": "Pedestrian",
            "count": sum(pedestrian_children.values()),
            "children": pedestrian_children,
        },
    )


def recount_agent_actions(statistics_dir: Path) -> dict:
    counts = Counter()
    rows = 0
    frame_indices = set()
    path = statistics_dir / "agent_actions_by_frame.csv.gz"
    for record in _read_gzip_csv(path):
        rows += 1
        counts[(record["object_type"], int(record["action_id"]))] += 1
        frame_indices.add(int(record["frame_index"]))
    return {"rows": rows, "counts": counts, "frame_indices": frame_indices}


def agent_sizes_from_summary(summary: dict) -> dict:
    aggregate = summary["aggregate"]
    counts = Counter()
    for key, count in aggregate["agent_size_counts"].items():
        _, size_class = decode_agent_size_key(key)
        counts[size_class] += int(count)
    return {
        "rows": sum(counts.values()),
        "counts": counts,
    }


def agent_actions_from_summary(summary: dict) -> dict:
    aggregate = summary["aggregate"]
    counts = Counter()
    for key, count in aggregate["agent_action_counts"].items():
        object_type, action_id = decode_agent_action_key(key)
        counts[(object_type, action_id)] += int(count)
    frame_indices = {
        decode_agent_action_frame_key(key)[0]
        for key in aggregate["agent_action_frame_counts"]
    }
    return {
        "rows": int(
            aggregate["action_diagnostics"]["valid_state_frames"]
        ),
        "counts": counts,
        "frame_indices": frame_indices,
    }


def aggregate_dependency_record(
    statistics_dir: Path,
    annotation_paths: Iterable[Path],
) -> dict:
    statistics_summary = statistics_dir / "summary.json"
    annotation_identity = input_fingerprint(annotation_paths)
    dependency_payload = {
        "statistics_summary": artifact_record(statistics_summary),
        "annotation_identity": annotation_identity,
    }
    return {
        **dependency_payload,
        "fingerprint": stable_fingerprint(dependency_payload),
    }


def _format_count(value: int) -> str:
    return f"{int(value):,}"


def _format_compact_count(value: float) -> str:
    for scale, suffix in (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if abs(value) >= scale:
            return f"{value / scale:.1f} {suffix}"
    return f"{value:.0f}"


def _add_panel_label(axis, label: str) -> None:
    axis.text(
        0.0,
        1.035,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE_PT,
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
    )


def _blend_with_white(color: str, amount: float) -> str:
    rgb = np.asarray(mpl_colors.to_rgb(color))
    mixed = rgb * (1.0 - amount) + np.ones(3) * amount
    return mpl_colors.to_hex(mixed)


def _sunburst_child_color(
    color: str,
    value: int,
    largest_sibling: int,
) -> str:
    if largest_sibling <= 0:
        return _blend_with_white(color, SUNBURST_MAX_CHILD_BLEND)
    relative_count = np.clip(value / largest_sibling, 0.0, 1.0)
    blend = SUNBURST_MAX_CHILD_BLEND - (
        (SUNBURST_MAX_CHILD_BLEND - SUNBURST_MIN_CHILD_BLEND)
        * relative_count
    )
    return _blend_with_white(color, round(float(blend), 2))


def _clockwise_spans(
    start_angle: float,
    end_angle: float,
    values: list[int],
) -> list[tuple[float, float]]:
    total = sum(values)
    if total <= 0:
        return [(start_angle, start_angle) for _ in values]
    available = start_angle - end_angle
    current = start_angle
    spans = []
    for value in values:
        next_angle = current - available * value / total
        spans.append((current, next_angle))
        current = next_angle
    return spans


def _build_sunburst_nodes(
    groups: tuple[dict, ...] | list[dict],
    start_angle: float = -90.0,
    end_angle: float = -450.0,
) -> tuple[dict, ...]:
    parent_spans = _clockwise_spans(
        start_angle,
        end_angle,
        [group["count"] for group in groups],
    )
    nodes = []
    for group, parent_span in zip(groups, parent_spans):
        children = group["children"]
        nodes.append(
            {
                "label": group["label"],
                "count": group["count"],
                "depth": 0,
                "start_angle": parent_span[0],
                "end_angle": parent_span[1],
                "terminal": not children,
                "parent": None,
            }
        )
        if not children:
            continue
        child_spans = _clockwise_spans(
            parent_span[0],
            parent_span[1],
            list(children.values()),
        )
        for (label, count), child_span in zip(
            children.items(),
            child_spans,
        ):
            nodes.append(
                {
                    "label": label,
                    "count": count,
                    "depth": 1,
                    "start_angle": child_span[0],
                    "end_angle": child_span[1],
                    "terminal": True,
                    "parent": group["label"],
                }
            )
    return tuple(nodes)


def _add_wedge(
    axis,
    span: tuple[float, float],
    radius: float,
    width: float,
    color: str,
    center: tuple[float, float] = (0.0, 0.0),
) -> None:
    start_angle, end_angle = span
    if start_angle <= end_angle:
        return
    axis.add_patch(
        Wedge(
            center,
            radius,
            end_angle,
            start_angle,
            width=width,
            facecolor=color,
            edgecolor="white",
            linewidth=1.2,
        )
    )


def _polar_point(
    angle: float,
    radius: float,
    center: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    radians = np.deg2rad(angle)
    return (
        center[0] + radius * np.cos(radians),
        center[1] + radius * np.sin(radians),
    )


def _spread_callout_positions(
    targets: list[float],
    lower: float = -1.48,
    upper: float = 1.48,
    gap: float = SUNBURST_CALLOUT_GAP,
) -> list[float]:
    if not targets:
        return []
    positions = [max(lower, min(upper, value)) for value in targets]
    for index in range(1, len(positions)):
        positions[index] = max(
            positions[index],
            positions[index - 1] + gap,
        )
    if positions[-1] > upper:
        shift = positions[-1] - upper
        positions = [value - shift for value in positions]
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(
            positions[index],
            positions[index + 1] - gap,
        )
    if positions[0] < lower:
        shift = lower - positions[0]
        positions = [value + shift for value in positions]
    return positions


def _callout_gap_data_units(axis) -> float:
    axis.figure.canvas.draw()
    axis_height_pixels = axis.get_window_extent().height
    if axis_height_pixels <= 0:
        return SUNBURST_CALLOUT_GAP
    text_height_pixels = (
        DEFAULT_FONT_SIZE_PT * 2.5 * axis.figure.dpi / 72.0
    )
    data_height = axis.get_ylim()[1] - axis.get_ylim()[0]
    return max(
        SUNBURST_CALLOUT_GAP,
        text_height_pixels / axis_height_pixels * data_height,
    )


def draw_sunburst_panel(
    axis,
    groups: tuple[dict, ...] | list[dict],
    palette: dict[str, str],
    *,
    panel_label: str,
    panel_key: str,
    unit: str,
    start_angle: float = 180.0,
    forced_label_positions: dict[str, tuple[float, float]] | None = None,
) -> list[dict]:
    forced_label_positions = forced_label_positions or {}
    axis.set_xlim(-1.95, 1.95)
    axis.set_ylim(-1.55, 1.55)
    axis.set_aspect("equal")
    nodes = _build_sunburst_nodes(
        groups,
        start_angle=start_angle,
        end_angle=start_angle - 360.0,
    )
    callouts = {-1: [], 1: []}
    plotted = []
    for node in nodes:
        span = (node["start_angle"], node["end_angle"])
        if node["terminal"] and node["depth"] == 0:
            radius = SUNBURST_OUTER_RADIUS
            width = SUNBURST_OUTER_RADIUS - SUNBURST_HOLE_RADIUS
            label_radius = (
                SUNBURST_HOLE_RADIUS + SUNBURST_OUTER_RADIUS
            ) / 2.0
            anchor_radius = SUNBURST_OUTER_RADIUS
        elif node["depth"] == 0:
            radius = SUNBURST_PARENT_RADIUS
            width = SUNBURST_PARENT_RADIUS - SUNBURST_HOLE_RADIUS
            label_radius = (
                SUNBURST_HOLE_RADIUS + SUNBURST_PARENT_RADIUS
            ) / 2.0
            anchor_radius = SUNBURST_OUTER_RADIUS
        else:
            radius = SUNBURST_OUTER_RADIUS
            width = SUNBURST_OUTER_RADIUS - SUNBURST_PARENT_RADIUS
            label_radius = (
                SUNBURST_PARENT_RADIUS + SUNBURST_OUTER_RADIUS
            ) / 2.0
            anchor_radius = SUNBURST_OUTER_RADIUS
        _add_wedge(
            axis,
            span,
            radius=radius,
            width=width,
            color=palette[node["label"]],
        )
        midpoint = sum(span) / 2.0
        angle_size = span[0] - span[1]
        force_internal_label = node["label"] in forced_label_positions
        text = (
            f"{node['label']}\n"
            f"{_format_compact_count(node['count'])}"
        )
        minimum_internal_angle = (
            SUNBURST_PARENT_INTERNAL_LABEL_MIN_DEG
            if node["depth"] == 0 and not node["terminal"]
            else SUNBURST_INTERNAL_LABEL_MIN_DEG
        )
        if force_internal_label or angle_size >= minimum_internal_angle:
            x, y = forced_label_positions.get(
                node["label"],
                _polar_point(midpoint, label_radius),
            )
            axis.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=DEFAULT_FONT_SIZE_PT,
                color="black",
                multialignment="center",
            )
        else:
            anchor_x, anchor_y = _polar_point(midpoint, anchor_radius)
            side = 1 if anchor_x >= 0 else -1
            callouts[side].append(
                {
                    "anchor": (anchor_x, anchor_y),
                    "target_y": anchor_y,
                    "text": text,
                }
            )
        plotted.append(
            {
                "panel": panel_key,
                "group": (
                    node["label"]
                    if node["parent"] is None
                    else node["parent"]
                ),
                "subgroup": (
                    "" if node["parent"] is None else node["label"]
                ),
                "label": node["label"],
                "count": node["count"],
                "unit": unit,
            }
        )

    for side, entries in callouts.items():
        entries.sort(key=lambda entry: entry["target_y"])
        positions = _spread_callout_positions(
            [entry["target_y"] for entry in entries],
            gap=_callout_gap_data_units(axis),
        )
        elbow_x = side * 1.66
        text_x = side * 1.78
        for entry, target_y in zip(entries, positions):
            anchor_x, anchor_y = entry["anchor"]
            axis.plot(
                [anchor_x, elbow_x, text_x - side * 0.035],
                [anchor_y, target_y, target_y],
                color=COLORS["muted"],
                linewidth=0.65,
                solid_capstyle="round",
            )
            axis.text(
                text_x,
                target_y,
                entry["text"],
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=DEFAULT_FONT_SIZE_PT,
                color="black",
                multialignment="center",
            )

    axis.add_patch(
        Circle(
            (0.0, 0.0),
            SUNBURST_HOLE_RADIUS,
            facecolor="white",
            edgecolor=COLORS["grid"],
            linewidth=0.8,
        )
    )
    axis.set_axis_off()
    _add_panel_label(axis, panel_label)
    return plotted


def draw_road_panel(axis, road: dict) -> list[dict]:
    top_counts = road["top_counts"]
    child_counts = road["child_counts"]
    groups = (
        {
            "label": "Freeway",
            "count": top_counts.get("Freeway", 0),
            "children": {
                "Mainline": child_counts["Freeway"].get("Mainline", 0),
                "Ramp": child_counts["Freeway"].get("Ramp", 0),
            },
        },
        {
            "label": "Urban road",
            "count": top_counts.get("Urban road", 0),
            "children": {
                "Intersection": child_counts["Urban road"].get(
                    "Intersection",
                    0,
                ),
                "Road segment": child_counts["Urban road"].get(
                    "Road segment",
                    0,
                ),
                "Parking lot": child_counts["Urban road"].get(
                    "Parking lot",
                    0,
                ),
            },
        },
    )
    return draw_sunburst_panel(
        axis,
        groups,
        ROAD_SUNBURST_PALETTE,
        panel_label="a",
        panel_key="a",
        unit="scene",
        start_angle=182.2,
    )


def draw_size_panel(axis, sizes: dict) -> list[dict]:
    return draw_sunburst_panel(
        axis,
        build_agent_hierarchy(sizes),
        AGENT_SUNBURST_PALETTE,
        panel_label="b",
        panel_key="b",
        unit="agent",
        start_angle=146.5,
        forced_label_positions={
            "Cyclist": (-0.51, -0.18),
            "Pedestrian": (-0.52, 0.28),
            "Adult": (-1.30, 0.45),
            "Small vehicle": (1.213, -0.523),
        },
    )


def draw_action_panel(axis, actions: dict) -> list[dict]:
    matrix = np.full((len(ACTION_GROUPS), len(ACTION_COLUMNS)), np.nan)
    plotted = []
    row_labels = []
    for row_index, (object_type, group_label, _) in enumerate(ACTION_GROUPS):
        applicable = {action_id for action_id, _ in ACTION_ORDER[object_type]}
        row_total = 0
        for column_index, (action_id, _display_label, action_label) in enumerate(
            ACTION_COLUMNS
        ):
            if action_id not in applicable:
                continue
            count = actions["counts"].get((object_type, action_id), 0)
            matrix[row_index, column_index] = count
            row_total += count
            plotted.append(
                {
                    "panel": "c",
                    "group": group_label,
                    "subgroup": str(action_id),
                    "label": action_label,
                    "count": count,
                    "unit": "agent-frame",
                }
            )
        row_labels.append(f"{group_label}\n{_format_compact_count(row_total)}")

    positive_values = matrix[np.isfinite(matrix) & (matrix > 0)]
    if positive_values.size == 0:
        raise RuntimeError("no positive action counts were found")
    value_min = float(positive_values.min())
    value_max = float(positive_values.max())
    if value_min == value_max:
        value_max = value_min * 10.0
    norm = mpl_colors.LogNorm(vmin=value_min, vmax=value_max)
    color_matrix = matrix.copy()
    color_matrix[np.isfinite(color_matrix) & (color_matrix <= 0)] = norm.vmin
    color_map = mpl_colors.LinearSegmentedColormap.from_list(
        "womd_action_counts",
        [COLORS["heat_low"], COLORS["heat_mid"], COLORS["heat_high"]],
    )
    color_map.set_bad(COLORS["missing"])
    image = axis.imshow(
        np.ma.masked_invalid(color_matrix),
        cmap=color_map,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                label = "N/A"
                color = COLORS["muted"]
            else:
                label = _format_compact_count(value)
                rgba = color_map(norm(max(value, norm.vmin)))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                color = COLORS["on_dark"] if luminance < 0.52 else COLORS["text"]
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=DEFAULT_FONT_SIZE_PT,
                color=color,
            )

    axis.set_xticks(
        np.arange(len(ACTION_COLUMNS)),
        [display_label for _, display_label, _ in ACTION_COLUMNS],
    )
    axis.set_yticks(np.arange(len(ACTION_GROUPS)), row_labels)
    axis.set_xticks(np.arange(-0.5, len(ACTION_COLUMNS), 1.0), minor=True)
    axis.set_yticks(np.arange(-0.5, len(ACTION_GROUPS), 1.0), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.9)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(
        axis="x",
        labelsize=DEFAULT_FONT_SIZE_PT,
        colors=COLORS["text"],
        pad=4,
    )
    axis.tick_params(
        axis="y",
        labelsize=DEFAULT_FONT_SIZE_PT,
        colors=COLORS["text"],
        length=0,
    )
    for tick_label in axis.get_xticklabels():
        tick_label.set_rotation(45)
        tick_label.set_rotation_mode("anchor")
        tick_label.set_ha("right")
        tick_label.set_va("top")
        tick_label.set_multialignment("center")
    for tick_label in axis.get_yticklabels():
        tick_label.set_ha("right")
        tick_label.set_multialignment("right")
    for spine in axis.spines.values():
        spine.set_visible(False)
    _add_panel_label(axis, "c")
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.018, pad=0.012)
    colorbar.ax.tick_params(labelsize=DEFAULT_FONT_SIZE_PT, colors=COLORS["text"])
    colorbar.ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda value, _position: _format_compact_count(value))
    )
    colorbar.outline.set_edgecolor(COLORS["grid"])
    return plotted


def save_plotted_counts(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("panel", "group", "subgroup", "label", "count", "unit"),
        )
        writer.writeheader()
        writer.writerows(
            {**row, "label": str(row["label"]).replace("\n", " ")}
            for row in rows
        )


def save_html_fragment(svg_path: Path, html_path: Path) -> None:
    svg = svg_path.read_text(encoding="utf-8")
    svg_start = svg.find("<svg")
    if svg_start < 0:
        raise RuntimeError("generated SVG does not contain an <svg> element")
    svg = svg[svg_start:]
    theme_colors = {
        "#ffffff": "var(--background)",
        "#000000": "var(--foreground)",
        COLORS["text"].lower(): "var(--foreground)",
        COLORS["muted"].lower(): "var(--muted-foreground)",
        COLORS["grid"].lower(): "var(--border)",
        COLORS["missing"].lower(): "var(--muted)",
        COLORS["on_dark"].lower(): "var(--primary-foreground)",
        COLORS["vehicle"].lower(): "var(--viz-series-1)",
        COLORS["road"].lower(): "var(--viz-series-2)",
        COLORS["near"].lower(): "var(--viz-series-3)",
        COLORS["pedestrian"].lower(): "var(--viz-series-4)",
        COLORS["cyclist"].lower(): "var(--viz-series-5)",
        COLORS["parking"].lower(): "var(--viz-series-5)",
        COLORS["intersection"].lower(): "var(--viz-series-6)",
        COLORS["heat_low"].lower(): (
            "color-mix(in srgb, var(--viz-series-2) 12%, var(--background))"
        ),
        COLORS["heat_mid"].lower(): (
            "color-mix(in srgb, var(--viz-series-2) 58%, var(--background))"
        ),
        COLORS["heat_high"].lower(): "var(--viz-series-2)",
    }
    sunburst_blended_colors = tuple(
        (color_key, percentage / 100.0, variable)
        for color_key, variable in (
            ("intersection", "--viz-series-6"),
            ("road", "--viz-series-2"),
            ("parking", "--viz-series-5"),
        )
        for percentage in range(
            round(SUNBURST_MIN_CHILD_BLEND * 100),
            round(SUNBURST_MAX_CHILD_BLEND * 100) + 1,
        )
    )
    blended_colors = sunburst_blended_colors + (
        ("near", 0.18, "--viz-series-3"),
        ("near", 0.235, "--viz-series-3"),
        ("near", 0.29, "--viz-series-3"),
        ("vehicle", 0.72, "--viz-series-1"),
        ("cyclist", 0.72, "--viz-series-5"),
        ("pedestrian", 0.72, "--viz-series-4"),
    )
    for color_key, amount, variable in blended_colors:
        theme_colors[_blend_with_white(COLORS[color_key], amount)] = (
            f"color-mix(in srgb, var({variable}) {round((1.0 - amount) * 100)}%, "
            "var(--background))"
        )
    for source, replacement in theme_colors.items():
        svg = svg.replace(source, replacement)
    fragment = (
        '<div id="womd-statistics-mixed-three-panel-en" role="img" '
        'aria-label="WOMD road, agent-size, and frame-level action statistics">\n'
        f"{svg}\n"
        "</div>\n"
        "<style>\n"
        "#womd-statistics-mixed-three-panel-en { width: 100%; color: var(--foreground); }\n"
        "#womd-statistics-mixed-three-panel-en svg { width: 100%; height: auto; display: block; }\n"
        "</style>\n"
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(fragment, encoding="utf-8")


def _fit_tight_output_size(
    figure,
    target_width_inches: float,
    target_height_inches: float,
) -> None:
    for _ in range(5):
        figure.canvas.draw()
        tight_bbox = figure.get_tightbbox(figure.canvas.get_renderer())
        width_scale = target_width_inches / tight_bbox.width
        height_scale = target_height_inches / tight_bbox.height
        if abs(width_scale - 1.0) < 1e-4 and abs(height_scale - 1.0) < 1e-4:
            break
        current_width, current_height = figure.get_size_inches()
        figure.set_size_inches(
            current_width * width_scale,
            current_height * height_scale,
            forward=True,
        )


def plot_statistics(args: argparse.Namespace) -> dict:
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    if args.width_cm <= 0:
        raise ValueError("--width-cm must be positive")
    statistics_dir = args.statistics_dir.expanduser().resolve()
    map_annotation_paths = resolve_annotation_paths(args.map_annotation_path)
    output_prefix = args.output_prefix.expanduser().resolve()

    summary = json.loads(
        (statistics_dir / "summary.json").read_text(encoding="utf-8")
    )
    if summary.get("schema_version") != STATISTICS_SCHEMA_VERSION:
        raise ValueError(
            "Statistics summary schema is incompatible with this plotter"
        )
    dependencies = aggregate_dependency_record(
        statistics_dir,
        map_annotation_paths,
    )
    road = recount_road_hierarchy(
        map_annotation_paths,
        frame_index=int(summary["frame_index"]),
    )
    sizes = agent_sizes_from_summary(summary)
    actions = agent_actions_from_summary(summary)
    aggregate = summary["aggregate"]
    expected_road_rows = aggregate["scenarios"] - aggregate["errors"]
    expected_size_rows = sum(aggregate["agent_size_counts"].values())
    expected_action_rows = aggregate["action_diagnostics"]["valid_state_frames"]
    if road["rows"] != expected_road_rows:
        raise RuntimeError(
            f"expected {expected_road_rows:,} road rows, found {road['rows']:,}"
        )
    if sum(road["top_counts"].values()) + road["unknown"] != expected_road_rows:
        raise RuntimeError(
            "road top-level and unknown counts do not sum to successful scenarios"
        )
    for group_label, group_count in road["top_counts"].items():
        if sum(road["child_counts"][group_label].values()) != group_count:
            raise RuntimeError(
                f"{group_label} child counts do not sum to the group total"
            )
    if sizes["rows"] != expected_size_rows:
        raise RuntimeError(
            f"expected {expected_size_rows:,} size rows, found {sizes['rows']:,}"
        )
    agent_groups = build_agent_hierarchy(sizes)
    if sum(group["count"] for group in agent_groups) != expected_size_rows:
        raise RuntimeError("agent hierarchy counts do not sum to all agents")
    for group in agent_groups:
        if (
            group["children"]
            and sum(group["children"].values()) != group["count"]
        ):
            raise RuntimeError(
                f"{group['label']} child counts do not sum to the group total"
            )
    if actions["rows"] != expected_action_rows:
        raise RuntimeError(
            f"expected {expected_action_rows:,} action rows, found {actions['rows']:,}"
        )
    if not actions["frame_indices"]:
        raise RuntimeError("no valid action frames were found")
    expected_frames = set(
        range(min(actions["frame_indices"]), max(actions["frame_indices"]) + 1)
    )
    if actions["frame_indices"] != expected_frames:
        raise RuntimeError("action frame indices are not contiguous")

    font_name = _font_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": DEFAULT_FONT_SIZE_PT,
            "axes.titlesize": DEFAULT_FONT_SIZE_PT,
            "axes.labelsize": DEFAULT_FONT_SIZE_PT,
            "xtick.labelsize": DEFAULT_FONT_SIZE_PT,
            "ytick.labelsize": DEFAULT_FONT_SIZE_PT,
            "legend.fontsize": DEFAULT_FONT_SIZE_PT,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
        }
    )

    figure_width_inches = args.width_cm / 2.54
    figure_height_inches = figure_width_inches * FIGURE_ASPECT_RATIO
    figure = plt.figure(
        figsize=(figure_width_inches, figure_height_inches),
        facecolor="white",
    )
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.28, 0.78),
        hspace=-0.03,
        wspace=0.08,
        left=0.13,
        right=0.94,
        top=0.95,
        bottom=0.12,
    )
    road_axis = figure.add_subplot(grid[0, 0])
    size_axis = figure.add_subplot(grid[0, 1])
    action_axis = figure.add_subplot(grid[1, :])

    plotted_rows = []
    plotted_rows.extend(draw_road_panel(road_axis, road))
    plotted_rows.extend(draw_size_panel(size_axis, sizes))
    plotted_rows.extend(draw_action_panel(action_axis, actions))

    _fit_tight_output_size(
        figure,
        figure_width_inches,
        figure_height_inches,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    svg_path = output_prefix.with_suffix(".svg")
    csv_path = output_prefix.with_name(output_prefix.name + "_counts.csv")
    output_paths = {
        "png": png_path,
        "pdf": pdf_path,
        "svg": svg_path,
        "counts": csv_path,
        "summary": output_prefix.with_name(
            output_prefix.name + ".summary.json"
        ),
    }
    if args.html_fragment is not None:
        output_paths["html"] = args.html_fragment.expanduser().resolve()
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        plt.close(figure)
        raise FileExistsError(
            f"Output exists: {existing[0]}. Use --overwrite to replace it."
        )
    partial_paths = {
        key: path.with_name(path.name + ".partial")
        for key, path in output_paths.items()
    }
    for path in partial_paths.values():
        path.unlink(missing_ok=True)
    savefig_options = {
        "facecolor": "white",
        "bbox_inches": "tight",
        "pad_inches": 0.0,
    }
    try:
        figure.savefig(
            partial_paths["png"],
            format="png",
            dpi=args.dpi,
            **savefig_options,
        )
        figure.savefig(
            partial_paths["pdf"],
            format="pdf",
            **savefig_options,
        )
        figure.savefig(
            partial_paths["svg"],
            format="svg",
            **savefig_options,
        )
    finally:
        plt.close(figure)
    save_plotted_counts(partial_paths["counts"], plotted_rows)
    if "html" in output_paths:
        save_html_fragment(
            partial_paths["svg"],
            partial_paths["html"],
        )
    result = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "statistics_dir": str(statistics_dir),
        "annotation_files": [str(path) for path in map_annotation_paths],
        "dependencies": dependencies,
        "configuration": {
            "dpi": args.dpi,
            "width_cm": args.width_cm,
            "html_fragment": (
                None
                if args.html_fragment is None
                else str(args.html_fragment.expanduser().resolve())
            ),
        },
        "road_scenarios": road["rows"],
        "road_unknown": road["unknown"],
        "annotation_errors": road["errors"],
        "agent_count": sizes["rows"],
        "agent_frame_count": actions["rows"],
        "output_files": {
            key: str(path) for key, path in output_paths.items()
        },
    }
    result["output_artifacts"] = {
        key: artifact_record(
            partial_paths[key],
            logical_path=output_paths[key],
        )
        for key in output_paths
        if key != "summary"
    }
    partial_paths["summary"].write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for key, path in output_paths.items():
        if key != "summary":
            partial_paths[key].replace(path)
    partial_paths["summary"].replace(output_paths["summary"])
    return result


def main() -> None:
    result = plot_statistics(parse_args())
    for path in result["output_files"].values():
        print(path)


if __name__ == "__main__":
    main()
