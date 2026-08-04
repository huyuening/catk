from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import sys
import time
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, TextIO, Tuple


DIRECTION_ACTIONS = {
    "LEFT_TURN": "LeftTurn",
    "RIGHT_TURN": "RightTurn",
    "LEFT_LANE_CHANGE": "LeftLaneChange",
    "RIGHT_LANE_CHANGE": "RightLaneChange",
}
TAG_PRIORITY = {
    "LeftTurn": 1,
    "RightTurn": 1,
    "LeftLaneChange": 1,
    "RightLaneChange": 1,
    "Straight": 2,
    "Accelerate": 3,
    "Decelerate": 3,
    "Stopping": 3,
    "KeepSpeed": 3,
    "Parked": 4,
}
REQUIRED_FIELDS = {
    "scenario_id",
    "global_index",
    "dataset_current_time_index",
    "frame_index",
    "track_id",
    "action",
    "absolute_speed_mps",
    "longitudinal_acceleration_mps2",
}


def _float_field(row: Mapping[str, str], field: str) -> float:
    value = row.get(field)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}={value!r}") from exc


def _int_field(row: Mapping[str, str], field: str) -> int:
    value = row.get(field)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}={value!r}") from exc


def derive_frame_tags(
    row: Mapping[str, str],
    *,
    stop_speed_mps: float = 0.2,
    accel_mps2: float = 0.5,
) -> Set[str]:
    """Derive independent direction and longitudinal ECoSim tags for one row."""

    if stop_speed_mps < 0:
        raise ValueError("stop_speed_mps must be non-negative")
    if accel_mps2 <= 0:
        raise ValueError("accel_mps2 must be positive")
    action = str(row.get("action", ""))
    speed = abs(_float_field(row, "absolute_speed_mps"))
    acceleration = _float_field(row, "longitudinal_acceleration_mps2")

    tags: Set[str] = set()
    direction = DIRECTION_ACTIONS.get(action)
    if direction is not None:
        tags.add(direction)
    elif action != "U_TURN" and speed > stop_speed_mps:
        tags.add("Straight")

    if speed <= stop_speed_mps:
        tags.add("Parked")
    elif acceleration > accel_mps2:
        tags.add("Accelerate")
    elif acceleration < -accel_mps2:
        tags.add("Decelerate")
    else:
        tags.add("KeepSpeed")
    return tags


def _track_sort_key(track_id: str) -> Tuple[int, object]:
    try:
        return 0, int(track_id)
    except ValueError:
        return 1, track_id


def _contiguous_runs(frames: Iterable[int]) -> Iterator[Tuple[int, int]]:
    ordered = sorted(set(frames))
    if not ordered:
        return
    start = ordered[0]
    previous = ordered[0]
    for frame in ordered[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        yield start, previous + 1
        start = frame
        previous = frame
    yield start, previous + 1


def build_intervals(
    rows: Iterable[Mapping[str, str]],
    *,
    stop_speed_mps: float = 0.2,
    accel_mps2: float = 0.5,
) -> List[str]:
    """Build sorted half-open ECoSim intervals for one scenario."""

    frame_groups: Dict[Tuple[str, str], Set[int]] = {}
    for row in rows:
        frame = _int_field(row, "frame_index")
        current = _int_field(row, "dataset_current_time_index")
        if frame < current + 1 or frame > current + 80:
            continue
        track_id = str(row.get("track_id", ""))
        if not track_id:
            raise ValueError("track_id must be non-empty")
        for tag in derive_frame_tags(
            row,
            stop_speed_mps=stop_speed_mps,
            accel_mps2=accel_mps2,
        ):
            frame_groups.setdefault((track_id, tag), set()).add(frame)

    intervals: List[Tuple[str, str, int, int]] = []
    for (track_id, tag), frames in frame_groups.items():
        for start, end in _contiguous_runs(frames):
            intervals.append((track_id, tag, start, end))

    parked_starts: Dict[str, List[int]] = {}
    for track_id, tag, start, _ in intervals:
        if tag == "Parked":
            parked_starts.setdefault(track_id, []).append(start)

    renamed: List[Tuple[str, str, int, int]] = []
    for track_id, tag, start, end in intervals:
        if tag == "Decelerate" and any(
            0 <= parked_start - end <= 5
            for parked_start in parked_starts.get(track_id, [])
        ):
            tag = "Stopping"
        renamed.append((track_id, tag, start, end))

    renamed.sort(
        key=lambda value: (
            _track_sort_key(value[0]),
            value[2],
            TAG_PRIORITY.get(value[1], 99),
            value[1],
            value[3],
        )
    )
    return [
        f"{tag}({track_id} at {start}-{end})"
        for track_id, tag, start, end in renamed
    ]


def _open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _iter_rows(paths: Sequence[Path]) -> Iterator[Mapping[str, str]]:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"action row file does not exist: {path}")
        with _open_csv(path) as stream:
            reader = csv.DictReader(stream)
            missing = REQUIRED_FIELDS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing required columns {sorted(missing)}"
                )
            for row_number, row in enumerate(reader, start=2):
                if not row.get("scenario_id"):
                    raise ValueError(f"{path}:{row_number}: empty scenario_id")
                yield row


def _iter_scenarios(
    rows: Iterable[Mapping[str, str]],
) -> Iterator[Tuple[str, List[Mapping[str, str]]]]:
    current_id: Optional[str] = None
    current_rows: List[Mapping[str, str]] = []
    completed: Set[str] = set()
    for row in rows:
        scenario_id = str(row["scenario_id"])
        if current_id is None:
            current_id = scenario_id
        if scenario_id != current_id:
            completed.add(current_id)
            yield current_id, current_rows
            if scenario_id in completed:
                raise ValueError(
                    f"scenario rows must be contiguous; {scenario_id!r} reappeared"
                )
            current_id = scenario_id
            current_rows = []
        current_rows.append(row)
    if current_id is not None:
        yield current_id, current_rows


def _single_int(rows: Sequence[Mapping[str, str]], field: str) -> int:
    values = {_int_field(row, field) for row in rows}
    if len(values) != 1:
        raise ValueError(f"scenario has inconsistent {field}: {sorted(values)}")
    return next(iter(values))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _ProgressReporter:
    def __init__(self, *, split, workers, every, input_count, stream):
        self.split = split
        self.workers = workers
        self.every = every
        self.input_count = input_count
        self.stream = stream
        self.started_at = time.monotonic()
        self.submitted_count = 0
        self.completed_count = 0
        self.row_count = 0
        self.tag_count = 0

    def start(self):
        self._emit("start")

    def submitted(self):
        self.submitted_count += 1

    def completed(self, row_count, tag_count):
        self.completed_count += 1
        self.row_count += row_count
        self.tag_count += tag_count
        if self.completed_count % self.every == 0:
            self._emit("running")

    def finish(self):
        self._emit("complete")

    def _emit(self, status):
        elapsed = max(time.monotonic() - self.started_at, 0.0)
        rate = self.completed_count / elapsed if elapsed else 0.0
        pending = self.submitted_count - self.completed_count
        print(
            f"[text-tags {self.split}] status={status} workers={self.workers} "
            f"inputs={self.input_count} completed={self.completed_count} "
            f"submitted={self.submitted_count} rows={self.row_count} "
            f"tags={self.tag_count} pending={pending} "
            f"rate={rate:.1f} scenes/s elapsed={_format_elapsed(elapsed)}",
            file=self.stream,
            flush=True,
        )


def _write_scene_tags(
    rows: Sequence[Mapping[str, str]],
    tag_path: str | Path,
    stop_speed_mps: float,
    acceleration_threshold: float,
) -> Tuple[int, int]:
    tags = build_intervals(
        rows,
        stop_speed_mps=stop_speed_mps,
        accel_mps2=acceleration_threshold,
    )
    _write_json(Path(tag_path), tags)
    return len(rows), len(tags)


def convert_action_rows(
    *,
    input_paths: Sequence[str | Path],
    output_root: str | Path,
    split: str,
    mapping_output: str | Path,
    stop_speed_mps: float = 0.2,
    acceleration_threshold: float = 0.5,
    workers: int = 80,
    progress_every: int = 1000,
    progress_stream: TextIO | None = None,
) -> Dict[str, str]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1")
    split = str(split).lower()
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("at least one input path is required")

    output_root = Path(output_root)
    tag_subdir = "waymo_train_v_action" if split == "train" else "waymo_val_v_action"
    mapping: Dict[str, str] = {}
    scene_owners: Dict[str, str] = {}
    reporter = _ProgressReporter(
        split=split,
        workers=workers,
        every=progress_every,
        input_count=len(paths),
        stream=sys.stderr if progress_stream is None else progress_stream,
    )
    reporter.start()

    if workers == 1:
        for scenario_id, rows in _iter_scenarios(_iter_rows(paths)):
            global_index = _single_int(rows, "global_index")
            _single_int(rows, "dataset_current_time_index")
            scene_id = f"scene_{global_index}"
            previous_owner = scene_owners.get(scene_id)
            if previous_owner is not None and previous_owner != scenario_id:
                raise ValueError(
                    f"{scene_id} maps to both {previous_owner!r} and {scenario_id!r}"
                )
            scene_owners[scene_id] = scenario_id
            mapping[scenario_id] = scene_id
            tag_path = (
                output_root
                / "tag_prompts"
                / tag_subdir
                / "tags"
                / str(global_index % 100)
                / f"{scene_id}.json"
            )
            reporter.submitted()
            row_count, tag_count = _write_scene_tags(
                rows,
                tag_path,
                stop_speed_mps,
                acceleration_threshold,
            )
            reporter.completed(row_count, tag_count)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        pending = set()
        max_pending = workers * 2
        try:
            for scenario_id, rows in _iter_scenarios(_iter_rows(paths)):
                global_index = _single_int(rows, "global_index")
                _single_int(rows, "dataset_current_time_index")
                scene_id = f"scene_{global_index}"
                previous_owner = scene_owners.get(scene_id)
                if previous_owner is not None and previous_owner != scenario_id:
                    raise ValueError(
                        f"{scene_id} maps to both {previous_owner!r} and {scenario_id!r}"
                    )
                scene_owners[scene_id] = scenario_id
                mapping[scenario_id] = scene_id
                tag_path = (
                    output_root
                    / "tag_prompts"
                    / tag_subdir
                    / "tags"
                    / str(global_index % 100)
                    / f"{scene_id}.json"
                )
                future = executor.submit(
                    _write_scene_tags,
                    rows,
                    tag_path,
                    stop_speed_mps,
                    acceleration_threshold,
                )
                pending.add(future)
                reporter.submitted()
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for completed_future in done:
                        row_count, tag_count = completed_future.result()
                        reporter.completed(row_count, tag_count)

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for completed_future in done:
                    row_count, tag_count = completed_future.result()
                    reporter.completed(row_count, tag_count)
        except BaseException:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    sorted_mapping = {key: mapping[key] for key in sorted(mapping)}
    _write_json(Path(mapping_output), sorted_mapping)
    reporter.finish()
    return sorted_mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert WOMD all-frame action rows to ECoSim tag prompts."
    )
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--mapping-output", required=True, type=Path)
    parser.add_argument("--stop-speed-mps", type=float, default=0.2)
    parser.add_argument("--acceleration-threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    convert_action_rows(
        input_paths=args.input,
        output_root=args.output_root,
        split=args.split,
        mapping_output=args.mapping_output,
        stop_speed_mps=args.stop_speed_mps,
        acceleration_threshold=args.acceleration_threshold,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
