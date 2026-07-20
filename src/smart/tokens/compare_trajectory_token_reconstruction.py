#!/usr/bin/env python3
# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Compare CatK trajectory vocabularies before and after WOMD reconstruction.

The tool deliberately creates *agent-only* CatK caches.  Map preprocessing is
identical between the two branches and is not consumed by trajectory-vocabulary
clustering, so omitting it keeps the experiment focused and substantially
reduces disk use.  The agent dictionaries use the same keys, dtypes, history
selection, interpolation, and 0.5 s segmentation rules as CatK.

The original branch reproduces CatK's legacy linear gap interpolation and
heading cleanup.  The reconstructed branch calls CatK's bundled geometric
filter once on the complete 91-frame training trajectory and skips those legacy
repairs.  These caches are vocabulary sources only: they are never substituted
for CatK model inputs or training labels.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib
import json
import math
import os
import pickle
import re
import shutil
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

# WOMD protobufs are loaded with the Python protobuf runtime for compatibility.
# Set this before importing TensorFlow or protobuf modules.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
TYPE_KEYS = ("veh", "ped", "cyc")
TYPE_LABELS = {"veh": "Vehicle", "ped": "Pedestrian", "cyc": "Cyclist"}
TYPE_INDEX = {"veh": 0, "ped": 1, "cyc": 2}
CANONICAL_WIDTH_LENGTH = {
    "veh": (2.0, 4.8),
    "ped": (1.0, 1.0),
    "cyc": (1.0, 2.0),
}


@dataclass(frozen=True)
class WorkerConfig:
    reconstruction_root: str | None
    method: str
    filter_strength: str
    max_gap_frames: int
    batch_linear_jerk_weight: float
    batch_angular_jerk_weight: float
    serialize_reconstructed: bool
    original_cache_dir: str
    reconstructed_cache_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create raw/reconstructed CatK agent caches, learn matched K-disk "
            "trajectory vocabularies, and visualize their differences."
        )
    )
    parser.add_argument(
        "--input-tfrecord",
        "--input-path",
        dest="input_tfrecord",
        required=True,
        help="One WOMD TFRecord shard or a directory containing training shards.",
    )
    parser.add_argument(
        "--reconstruction-root",
        default=None,
        help=(
            "Optional WOMD-Traffic-Signal-Data-Improvement checkout. The "
            "default filter method is bundled with CatK; this is required "
            "only for batch or optimizer."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/trajectory_token_reconstruction_comparison",
    )
    parser.add_argument(
        "--vocab-output-dir",
        default="src/smart/tokens",
        help="Directory receiving the final CatK-compatible reconstructed vocabulary.",
    )
    parser.add_argument(
        "--vocab-output-name",
        default="agent_vocab_reconstructed.pkl",
    )
    parser.add_argument(
        "--stage", choices=("all", "preprocess", "cluster"), default="all"
    )
    parser.add_argument(
        "--method", choices=("filter", "batch", "optimizer"), default="filter"
    )
    parser.add_argument(
        "--filter-strength",
        choices=("light", "balanced", "strong"),
        default="strong",
    )
    parser.add_argument("--max-gap-frames", type=int, default=-1)
    parser.add_argument("--batch-linear-jerk-weight", type=float, default=1.0)
    parser.add_argument("--batch-angular-jerk-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--worker-backend",
        choices=("thread", "process"),
        default="thread",
        help=(
            "Parallel backend used when --num-workers is greater than one. "
            "Threads work in restricted macOS sandboxes; processes can be "
            "faster on unrestricted Linux hosts."
        ),
    )
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--num-clusters", type=int, default=2048)
    parser.add_argument("--max-trajectories-per-class", type=int, default=204800)
    parser.add_argument("--cluster-tolerance-m", type=float, default=0.05)
    parser.add_argument("--metric-sample-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--write-reconstructed-tfrecord",
        action="store_true",
        help=(
            "Also serialize the full-trajectory reconstruction for auditing. "
            "It is not a CatK input or label dataset."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def _write_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _canonical_cache_paths(cache_dir: Path) -> list[Path]:
    """Return scenario caches, excluding macOS/iCloud conflict copies."""

    return sorted(
        path
        for path in cache_dir.glob("*.pkl")
        if " " not in path.stem
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return value


def _load_scenario_class():
    try:
        from waymo_open_dataset.protos import scenario_pb2
    except ModuleNotFoundError:
        pb2_root = REPO_ROOT / "src" / "smart" / "tokens" / "womd_proto" / "pb2"
        pb2_root_string = str(pb2_root)
        if pb2_root_string not in sys.path:
            sys.path.insert(0, pb2_root_string)
        scenario_pb2 = importlib.import_module("scenario_pb2")

    return scenario_pb2.Scenario


def _load_catk_reconstruction_bridge():
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return importlib.import_module("src.smart.tokens.womd_trajectory_reconstruction")


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    value = np.asarray(angle)
    return -np.pi + (value + np.pi) % (2.0 * np.pi)


def _decode_tracks(scenario: Any) -> Dict[str, np.ndarray]:
    prediction_indices = {item.track_index for item in scenario.tracks_to_predict}
    objects_of_interest = set(scenario.objects_of_interest)
    object_ids = []
    object_types = []
    states = []
    valid = []
    roles = []
    for index, track in enumerate(scenario.tracks):
        object_ids.append(track.id)
        object_types.append(track.object_type - 1)
        states.append(
            np.asarray(
                [
                    [
                        state.center_x,
                        state.center_y,
                        state.center_z,
                        state.length,
                        state.width,
                        state.height,
                        state.heading,
                        state.velocity_x,
                        state.velocity_y,
                    ]
                    for state in track.states
                ],
                dtype=np.float32,
            )
        )
        valid.append(np.asarray([state.valid for state in track.states], dtype=bool))
        roles.append(
            [
                index == scenario.sdc_track_index,
                track.id in objects_of_interest,
                index in prediction_indices,
            ]
        )
    return {
        "object_id": np.asarray(object_ids, dtype=np.int64),
        "object_type": np.asarray(object_types, dtype=np.uint8),
        "states": np.asarray(states, dtype=np.float32),
        "valid": np.asarray(valid, dtype=bool),
        "role": np.asarray(roles, dtype=bool),
    }


def _interpolate_columns(
    values: np.ndarray, valid_steps: np.ndarray, target_steps: np.ndarray
) -> np.ndarray:
    result = np.empty((len(target_steps), values.shape[1]), dtype=np.float32)
    for column in range(values.shape[1]):
        result[:, column] = np.interp(
            target_steps,
            valid_steps,
            values[valid_steps, column],
        )
    return result


def _agent_cache(
    scenario: Any,
    trajectories_reconstructed: bool,
    selected_object_ids: Sequence[int] | None = None,
) -> Dict[str, Any]:
    import torch

    track_infos = _decode_tracks(scenario)
    current_index = int(scenario.current_time_index)
    num_steps = int(track_infos["states"].shape[1])
    if selected_object_ids is None:
        selected = np.flatnonzero(track_infos["valid"][:, current_index])
    else:
        index_by_id = {
            int(object_id): index
            for index, object_id in enumerate(track_infos["object_id"])
        }
        missing = [
            int(object_id)
            for object_id in selected_object_ids
            if int(object_id) not in index_by_id
        ]
        if missing:
            raise ValueError(
                "Full-trajectory reconstruction removed CatK agents: "
                f"{missing[:5]}"
            )
        selected = np.asarray(
            [index_by_id[int(object_id)] for object_id in selected_object_ids],
            dtype=np.int64,
        )
    num_agents = len(selected)

    valid_out = np.zeros((num_agents, num_steps), dtype=bool)
    position_out = np.zeros((num_agents, num_steps, 3), dtype=np.float32)
    heading_out = np.zeros((num_agents, num_steps), dtype=np.float32)
    velocity_out = np.zeros((num_agents, num_steps, 2), dtype=np.float32)
    shape_out = np.zeros((num_agents, 3), dtype=np.float32)

    for output_index, track_index in enumerate(selected):
        valid = track_infos["valid"][track_index]
        valid_steps = np.flatnonzero(valid)
        states = track_infos["states"][track_index]
        # Match the updated CatK feature contract: the last history frame.
        shape_out[output_index] = states[current_index, 3:6]
        if trajectories_reconstructed:
            valid_out[output_index, valid_steps] = True
            position_out[output_index, valid_steps] = states[valid_steps, :3]
            heading_out[output_index, valid_steps] = states[valid_steps, 6]
            velocity_out[output_index, valid_steps] = states[valid_steps, 7:9]
        elif len(valid_steps) > 1:
            target_steps = np.arange(valid_steps[0], valid_steps[-1] + 1)
            valid_out[output_index, target_steps] = True
            position_out[output_index, target_steps] = _interpolate_columns(
                states[:, :3], valid_steps, target_steps
            )
            velocity_out[output_index, target_steps] = _interpolate_columns(
                states[:, 7:9], valid_steps, target_steps
            )
            unwrapped = np.unwrap(states[valid_steps, 6])
            heading_out[output_index, target_steps] = np.interp(
                target_steps, valid_steps, unwrapped
            ).astype(np.float32)
        elif len(valid_steps) == 1:
            step = int(valid_steps[0])
            valid_out[output_index, step] = True
            position_out[output_index, step] = states[step, :3]
            heading_out[output_index, step] = states[step, 6]
            velocity_out[output_index, step] = states[step, 7:9]

    agent = {
        "num_nodes": num_agents,
        "valid_mask": torch.from_numpy(valid_out),
        "role": torch.from_numpy(track_infos["role"][selected]),
        "id": torch.from_numpy(track_infos["object_id"][selected]),
        "type": torch.from_numpy(track_infos["object_type"][selected]),
        "position": torch.from_numpy(position_out),
        "heading": torch.from_numpy(heading_out),
        "velocity": torch.from_numpy(velocity_out),
        "shape": torch.from_numpy(shape_out),
        "trajectory_reconstructed": torch.full(
            (num_agents,), trajectories_reconstructed, dtype=torch.bool
        ),
    }
    return {
        "agent": agent,
        "scenario_id": str(scenario.scenario_id),
        "current_time_index": current_index,
        "comparison_scope": "trajectory_vocabulary_source_only",
    }


def _scenario_type_counts(cache: Mapping[str, Any]) -> Dict[str, int]:
    agent_types = np.asarray(cache["agent"]["type"])
    return {
        key: int(np.sum(agent_types == TYPE_INDEX[key]))
        for key in TYPE_KEYS
    }


def _process_scenario_task(task: tuple[int, bytes, WorkerConfig]) -> Dict[str, Any]:
    index, record_bytes, config = task
    scenario_class = _load_scenario_class()
    scenario = scenario_class()
    scenario.ParseFromString(record_bytes)

    original_cache = _agent_cache(scenario, trajectories_reconstructed=False)
    scenario_id = str(scenario.scenario_id)
    original_path = Path(config.original_cache_dir) / f"{scenario_id}.pkl"
    reconstructed_path = Path(config.reconstructed_cache_dir) / f"{scenario_id}.pkl"
    _write_pickle(original_path, original_cache)

    bridge = _load_catk_reconstruction_bridge()
    reconstruction_config = bridge.TrajectoryReconstructionConfig(
        method=config.method,
        project_root=config.reconstruction_root,
        filter_strength=config.filter_strength,
        max_gap_frames=config.max_gap_frames,
        batch_linear_jerk_weight=config.batch_linear_jerk_weight,
        batch_angular_jerk_weight=config.batch_angular_jerk_weight,
    )
    started = time.perf_counter()
    reconstructed, reconstruction_stats = bridge.reconstruct_scenario_for_vocabulary(
        scenario, reconstruction_config
    )
    elapsed = time.perf_counter() - started
    reconstructed_cache = _agent_cache(
        reconstructed,
        trajectories_reconstructed=True,
        selected_object_ids=np.asarray(original_cache["agent"]["id"], dtype=np.int64),
    )
    _write_pickle(reconstructed_path, reconstructed_cache)

    return {
        "index": index,
        "scenario_id": scenario_id,
        "raw_track_count": len(scenario.tracks),
        "original_agent_count": int(original_cache["agent"]["num_nodes"]),
        "reconstructed_agent_count": int(
            reconstructed_cache["agent"]["num_nodes"]
        ),
        "original_type_counts": _scenario_type_counts(original_cache),
        "reconstructed_type_counts": _scenario_type_counts(reconstructed_cache),
        "reconstruction_seconds": elapsed,
        "reconstruction_stats": _jsonable(reconstruction_stats),
        "reconstructed_record": (
            reconstructed.SerializeToString()
            if config.serialize_reconstructed
            else None
        ),
    }


def _prepare_output_for_preprocessing(output_dir: Path, force: bool) -> None:
    watched = [
        output_dir / "datasets" / "original" / "training",
        output_dir / "datasets" / "reconstructed" / "training",
        output_dir / "manifest.csv",
        output_dir / "tfrecords",
    ]
    existing = [path for path in watched if path.exists()]
    if existing and not force:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Comparison preprocessing output already exists. Use --stage cluster "
            f"to reuse it or --force to replace it:\n  {joined}"
        )
    if force:
        for path in watched:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


_TFRECORD_SHARD_PATTERN = re.compile(
    r"(?P<prefix>.*\.tfrecord)-(?P<index>\d+)-of-(?P<count>\d+)$"
)


def _resolve_input_tfrecords(input_path: Path) -> list[Path]:
    """Resolve one shard or a sorted training directory without audit copies."""

    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)

    paths = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file()
        and (
            path.name.endswith(".tfrecord")
            or _TFRECORD_SHARD_PATTERN.fullmatch(path.name) is not None
        )
    )
    shard_groups: Dict[str, list[Path]] = {}
    for path in paths:
        match = _TFRECORD_SHARD_PATTERN.fullmatch(path.name)
        if match is not None:
            shard_groups.setdefault(match.group("prefix"), []).append(path)
    preferred_prefix = f"{input_path.name}.tfrecord"
    if preferred_prefix in shard_groups:
        paths = shard_groups[preferred_prefix]
    elif shard_groups:
        _, paths = max(
            sorted(shard_groups.items()),
            key=lambda item: len(item[1]),
        )
    if not paths:
        raise FileNotFoundError(
            f"No TFRecord shards found directly under {input_path}"
        )
    return paths


def _record_iterator(
    paths: Sequence[Path], max_scenarios: int | None
) -> Iterator[tuple[int, bytes]]:
    import tensorflow as tf

    dataset = tf.data.TFRecordDataset(
        [str(path) for path in paths], compression_type="", num_parallel_reads=1
    )
    for index, value in enumerate(dataset):
        if max_scenarios is not None and index >= max_scenarios:
            break
        yield index, bytes(value.numpy())


def _reconstructed_tfrecord_path(output_dir: Path, input_path: Path) -> Path:
    name = (
        input_path.name
        if input_path.is_file()
        else f"{input_path.name}_reconstructed.tfrecord"
    )
    return output_dir / "tfrecords" / "reconstructed" / name


def _record_original_inputs(
    input_path: Path,
    input_tfrecords: Sequence[Path],
    output_dir: Path,
) -> str:
    """Record inputs without duplicating a complete multi-shard dataset."""

    original_dir = output_dir / "tfrecords" / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    if len(input_tfrecords) == 1:
        return _link_or_copy(
            input_tfrecords[0], original_dir / input_tfrecords[0].name
        )

    _write_json(
        original_dir / "source_files.json",
        {
            "input_directory": str(input_path),
            "shard_count": len(input_tfrecords),
            "shards": [str(path) for path in input_tfrecords],
        },
    )
    return "referenced"


def _bounded_parallel_results(
    records: Iterable[tuple[int, bytes]],
    worker_config: WorkerConfig,
    num_workers: int,
    worker_backend: str,
) -> Iterator[Dict[str, Any]]:
    if num_workers <= 1:
        for index, record in records:
            yield _process_scenario_task((index, record, worker_config))
        return

    record_iterator = iter(records)
    executor_class = (
        ThreadPoolExecutor if worker_backend == "thread" else ProcessPoolExecutor
    )
    with executor_class(max_workers=num_workers) as executor:
        pending: MutableMapping[Any, int] = {}

        def submit_one() -> bool:
            try:
                index, record = next(record_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                _process_scenario_task, (index, record, worker_config)
            )
            pending[future] = index
            return True

        for _ in range(max(1, num_workers * 2)):
            if not submit_one():
                break
        completed_buffer: Dict[int, Dict[str, Any]] = {}
        next_index = 0
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                result = future.result()
                completed_buffer[int(result["index"])] = result
                submit_one()
            while next_index in completed_buffer:
                yield completed_buffer.pop(next_index)
                next_index += 1


def preprocess_dataset(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    from tqdm import tqdm

    input_path = Path(args.input_tfrecord).expanduser().resolve()
    input_tfrecords = _resolve_input_tfrecords(input_path)
    reconstruction_root = (
        str(Path(args.reconstruction_root).expanduser().resolve())
        if args.reconstruction_root
        else None
    )
    bridge = _load_catk_reconstruction_bridge()
    bridge.TrajectoryReconstructionConfig(
        method=args.method,
        project_root=reconstruction_root,
        filter_strength=args.filter_strength,
        max_gap_frames=args.max_gap_frames,
        batch_linear_jerk_weight=args.batch_linear_jerk_weight,
        batch_angular_jerk_weight=args.batch_angular_jerk_weight,
    )
    _prepare_output_for_preprocessing(output_dir, args.force)

    original_cache_dir = output_dir / "datasets" / "original" / "training"
    reconstructed_cache_dir = output_dir / "datasets" / "reconstructed" / "training"
    original_cache_dir.mkdir(parents=True, exist_ok=True)
    reconstructed_cache_dir.mkdir(parents=True, exist_ok=True)

    original_copy_mode = _record_original_inputs(
        input_path, input_tfrecords, output_dir
    )
    reconstructed_tfrecord = _reconstructed_tfrecord_path(output_dir, input_path)
    reconstructed_partial = reconstructed_tfrecord.with_suffix(
        reconstructed_tfrecord.suffix + ".partial"
    )
    reconstructed_tfrecord.parent.mkdir(parents=True, exist_ok=True)
    if reconstructed_partial.exists():
        reconstructed_partial.unlink()

    worker_config = WorkerConfig(
        reconstruction_root=reconstruction_root,
        method=args.method,
        filter_strength=args.filter_strength,
        max_gap_frames=args.max_gap_frames,
        batch_linear_jerk_weight=args.batch_linear_jerk_weight,
        batch_angular_jerk_weight=args.batch_angular_jerk_weight,
        serialize_reconstructed=args.write_reconstructed_tfrecord,
        original_cache_dir=str(original_cache_dir),
        reconstructed_cache_dir=str(reconstructed_cache_dir),
    )

    if args.num_workers > 1 and args.worker_backend == "thread":
        # Avoid racing first-time imports across worker threads.
        _load_scenario_class()
        if reconstruction_root:
            bridge._load_reconstruction_entrypoint(reconstruction_root)
        else:
            importlib.import_module(
                "src.smart.tokens.trajectory_filter_reconstructor"
            )

    writer = None
    if args.write_reconstructed_tfrecord:
        import tensorflow as tf

        writer = tf.io.TFRecordWriter(str(reconstructed_partial))

    records = _record_iterator(input_tfrecords, args.max_scenarios)
    expected = args.max_scenarios
    manifest_rows = []
    started = time.perf_counter()
    try:
        results = _bounded_parallel_results(
            records,
            worker_config,
            args.num_workers,
            args.worker_backend,
        )
        for result in tqdm(results, total=expected, desc="Reconstruct scenarios"):
            reconstructed_record = result.pop("reconstructed_record")
            if writer is not None:
                if reconstructed_record is None:
                    raise RuntimeError("Reconstructed TFRecord serialization was disabled")
                writer.write(reconstructed_record)
            manifest_rows.append(result)
    finally:
        if writer is not None:
            writer.close()
    if writer is not None:
        reconstructed_partial.replace(reconstructed_tfrecord)

    manifest_path = output_dir / "manifest.csv"
    flat_rows = []
    for row in manifest_rows:
        flat_rows.append(
            {
                "index": row["index"],
                "scenario_id": row["scenario_id"],
                "raw_track_count": row["raw_track_count"],
                "original_agent_count": row["original_agent_count"],
                "reconstructed_agent_count": row["reconstructed_agent_count"],
                "original_vehicle_count": row["original_type_counts"]["veh"],
                "original_pedestrian_count": row["original_type_counts"]["ped"],
                "original_cyclist_count": row["original_type_counts"]["cyc"],
                "reconstructed_vehicle_count": row["reconstructed_type_counts"]["veh"],
                "reconstructed_pedestrian_count": row["reconstructed_type_counts"]["ped"],
                "reconstructed_cyclist_count": row["reconstructed_type_counts"]["cyc"],
                "reconstruction_seconds": f"{row['reconstruction_seconds']:.6f}",
                "reconstruction_stats_json": json.dumps(
                    row["reconstruction_stats"], ensure_ascii=False, sort_keys=True
                ),
            }
        )
    if flat_rows:
        with manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer_csv = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
            writer_csv.writeheader()
            writer_csv.writerows(flat_rows)

    return _refresh_preprocessing_summary(
        args,
        output_dir,
        elapsed_seconds=time.perf_counter() - started,
        original_tfrecord_storage=original_copy_mode,
    )


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_preprocessing_summary(
    args: argparse.Namespace,
    output_dir: Path,
    elapsed_seconds: float | None = None,
    original_tfrecord_storage: str = "existing",
) -> Dict[str, Any]:
    """Build preprocessing metadata from the durable manifest and cache files."""

    input_path = Path(args.input_tfrecord).expanduser().resolve()
    input_tfrecords = _resolve_input_tfrecords(input_path)
    manifest_path = output_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    original_cache_dir = output_dir / "datasets" / "original" / "training"
    reconstructed_cache_dir = output_dir / "datasets" / "reconstructed" / "training"
    cache_validation = validate_cache_pairs(
        original_cache_dir, reconstructed_cache_dir
    )
    _write_json(output_dir / "cache_validation.json", cache_validation)

    reconstructed_tfrecord = _reconstructed_tfrecord_path(output_dir, input_path)
    original_reference = (
        output_dir / "tfrecords" / "original" / input_tfrecords[0].name
        if len(input_tfrecords) == 1
        else output_dir / "tfrecords" / "original" / "source_files.json"
    )
    summary = {
        "input_path": str(input_path),
        "input_tfrecord_count": len(input_tfrecords),
        "input_tfrecords": [str(path) for path in input_tfrecords],
        "input_size_bytes": int(sum(path.stat().st_size for path in input_tfrecords)),
        "input_sha256": (
            _sha256(input_tfrecords[0]) if len(input_tfrecords) == 1 else None
        ),
        "original_tfrecord_reference": str(original_reference),
        "original_tfrecord_storage": original_tfrecord_storage,
        "reconstructed_tfrecord": (
            str(reconstructed_tfrecord)
            if reconstructed_tfrecord.is_file()
            else None
        ),
        "scenario_count": len(rows),
        "reconstruction": {
            "method": args.method,
            "filter_strength": args.filter_strength,
            "max_gap_frames": args.max_gap_frames,
            "implementation": (
                "catk_bundled_filter"
                if args.method == "filter" and not args.reconstruction_root
                else "external"
            ),
            "scope": "vocabulary_only",
            "uses_complete_training_trajectory": True,
            "model_inputs_reconstructed": False,
            "training_labels_reconstructed": False,
        },
        "elapsed_seconds": elapsed_seconds,
        "mean_reconstruction_seconds": (
            float(np.mean([float(row["reconstruction_seconds"]) for row in rows]))
            if rows
            else 0.0
        ),
        "agent_counts": {
            dataset: {
                key: int(
                    sum(
                        int(row[f"{dataset}_{TYPE_LABELS[key].lower()}_count"])
                        for row in rows
                    )
                )
                for key in TYPE_KEYS
            }
            for dataset in ("original", "reconstructed")
        },
        "cache_validation": cache_validation,
    }
    _write_json(output_dir / "preprocessing_summary.json", summary)
    return summary


def validate_cache_pairs(
    original_dir: Path,
    reconstructed_dir: Path,
) -> Dict[str, Any]:
    """Audit the invariants that must remain unchanged by reconstruction."""

    original_paths = {
        path.name: path for path in _canonical_cache_paths(original_dir)
    }
    reconstructed_paths = {
        path.name: path for path in _canonical_cache_paths(reconstructed_dir)
    }
    if set(original_paths) != set(reconstructed_paths):
        only_original = sorted(set(original_paths) - set(reconstructed_paths))
        only_reconstructed = sorted(set(reconstructed_paths) - set(original_paths))
        raise AssertionError(
            "Cache file sets differ: "
            f"only_original={only_original[:5]}, "
            f"only_reconstructed={only_reconstructed[:5]}"
        )

    total_agents = 0
    shape_minimum = np.full(3, np.inf, dtype=np.float64)
    shape_maximum = np.full(3, -np.inf, dtype=np.float64)
    for name in sorted(original_paths):
        with original_paths[name].open("rb") as stream:
            original = pickle.load(stream)
        with reconstructed_paths[name].open("rb") as stream:
            reconstructed = pickle.load(stream)
        if original["scenario_id"] != reconstructed["scenario_id"]:
            raise AssertionError(f"Scenario ID differs in {name}")
        if original["current_time_index"] != reconstructed["current_time_index"]:
            raise AssertionError(f"Current time index differs in {name}")

        current_index = int(original["current_time_index"])
        original_agent = original["agent"]
        reconstructed_agent = reconstructed["agent"]
        for key in ("id", "type", "role", "shape"):
            original_value = np.asarray(original_agent[key])
            reconstructed_value = np.asarray(reconstructed_agent[key])
            if not np.array_equal(original_value, reconstructed_value):
                raise AssertionError(f"Agent field '{key}' differs in {name}")
        if int(original_agent["num_nodes"]) != int(
            reconstructed_agent["num_nodes"]
        ):
            raise AssertionError(f"Agent count differs in {name}")

        original_marker = np.asarray(
            original_agent["trajectory_reconstructed"], dtype=bool
        )
        reconstructed_marker = np.asarray(
            reconstructed_agent["trajectory_reconstructed"], dtype=bool
        )
        if original_marker.any() or not reconstructed_marker.all():
            raise AssertionError(f"Reconstruction marker is invalid in {name}")
        original_current = np.asarray(
            original_agent["valid_mask"], dtype=bool
        )[:, current_index]
        reconstructed_current = np.asarray(
            reconstructed_agent["valid_mask"], dtype=bool
        )[:, current_index]
        if not original_current.all() or not reconstructed_current.all():
            raise AssertionError(f"Current-frame validity invariant failed in {name}")

        shape = np.asarray(original_agent["shape"], dtype=np.float64)
        if len(shape):
            if not np.isfinite(shape).all() or (shape <= 0).any():
                raise AssertionError(f"Invalid last-history shape in {name}")
            shape_minimum = np.minimum(shape_minimum, shape.min(axis=0))
            shape_maximum = np.maximum(shape_maximum, shape.max(axis=0))
        total_agents += int(original_agent["num_nodes"])

    return {
        "scenario_count": len(original_paths),
        "agent_count": total_agents,
        "matching_scenario_ids": True,
        "matching_agent_ids_types_roles_shapes": True,
        "current_frame_agent_set_preserved": True,
        "reconstruction_markers_valid": True,
        "last_history_shape_min_length_width_height_m": shape_minimum.tolist(),
        "last_history_shape_max_length_width_height_m": shape_maximum.tolist(),
    }


def _clean_heading(valid: np.ndarray, heading: np.ndarray) -> np.ndarray:
    cleaned = heading.copy()
    for step in range(cleaned.shape[0] - 1):
        if valid[step] and valid[step + 1]:
            difference = abs(float(_wrap_angle(cleaned[step] - cleaned[step + 1])))
            if difference > 1.5:
                cleaned[step + 1] = cleaned[step]
    return cleaned


def _local_segment(position: np.ndarray, heading: np.ndarray) -> np.ndarray:
    relative = position - position[[0]]
    cosine = math.cos(float(heading[0]))
    sine = math.sin(float(heading[0]))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    local_position = relative @ rotation
    local_heading = _wrap_angle(heading - heading[0]).astype(np.float32)
    return np.concatenate([local_position, local_heading[:, None]], axis=-1).astype(
        np.float32
    )


def _segment_key(segment: np.ndarray) -> bytes:
    # CatK treats L1 differences below 1e-2 as duplicates. Millimetre rounding
    # provides an efficient deterministic approximation without O(n^2) scans.
    return np.round(segment, decimals=3).astype(np.float32).tobytes()


def collect_segments(
    cache_dir: Path,
    max_per_class: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, int]]:
    trajectories: Dict[str, list[np.ndarray]] = {
        key: [np.zeros((6, 3), dtype=np.float32)] for key in TYPE_KEYS
    }
    seen = {key: {_segment_key(trajectories[key][0])} for key in TYPE_KEYS}
    candidate_counts = {key: 0 for key in TYPE_KEYS}
    for cache_path in _canonical_cache_paths(cache_dir):
        with cache_path.open("rb") as stream:
            cache = pickle.load(stream)
        agent = cache["agent"]
        valid_all = np.asarray(agent["valid_mask"], dtype=bool)
        position_all = np.asarray(agent["position"], dtype=np.float32)[..., :2]
        heading_all = np.asarray(agent["heading"], dtype=np.float32)
        type_all = np.asarray(agent["type"], dtype=np.int64)
        reconstructed_all = np.asarray(
            agent.get(
                "trajectory_reconstructed",
                np.zeros(len(type_all), dtype=bool),
            ),
            dtype=bool,
        )
        for agent_index, agent_type in enumerate(type_all):
            if agent_type < 0 or agent_type > 2:
                continue
            key = TYPE_KEYS[int(agent_type)]
            if len(trajectories[key]) >= max_per_class:
                continue
            valid = valid_all[agent_index]
            if int(valid.sum()) < 30:
                continue
            heading = heading_all[agent_index]
            if not reconstructed_all[agent_index]:
                heading = _clean_heading(valid, heading)
            position = position_all[agent_index]
            for start in range(0, position.shape[0] - 5, 5):
                if not (valid[start] and valid[start + 5]):
                    continue
                candidate_counts[key] += 1
                segment = _local_segment(
                    position[start : start + 6], heading[start : start + 6]
                )
                identity = _segment_key(segment)
                if identity in seen[key]:
                    continue
                seen[key].add(identity)
                trajectories[key].append(segment)
                if len(trajectories[key]) >= max_per_class:
                    break
    return (
        {
            key: np.stack(trajectories[key], axis=0).astype(np.float32)
            for key in TYPE_KEYS
        },
        candidate_counts,
    )


def polygon_contours(
    position: np.ndarray,
    heading: np.ndarray,
    width_length: Sequence[float],
) -> np.ndarray:
    width, length = width_length
    half_cos = 0.5 * np.cos(heading)
    half_sin = 0.5 * np.sin(heading)
    length_cos = length * half_cos
    length_sin = length * half_sin
    width_cos = width * half_cos
    width_sin = width * half_sin
    x = position[..., 0]
    y = position[..., 1]
    return np.stack(
        [
            np.stack([x + length_cos - width_sin, y + length_sin + width_cos], axis=-1),
            np.stack([x + length_cos + width_sin, y + length_sin - width_cos], axis=-1),
            np.stack([x - length_cos + width_sin, y - length_sin - width_cos], axis=-1),
            np.stack([x - length_cos - width_sin, y - length_sin + width_cos], axis=-1),
        ],
        axis=-2,
    ).astype(np.float32)


def kdisk_cluster(
    trajectories: np.ndarray,
    num_clusters: int,
    tolerance_m: float,
    width_length: Sequence[float],
    seed: int,
) -> np.ndarray:
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if len(trajectories) == 0:
        raise ValueError("Cannot cluster an empty trajectory array")
    endpoints = polygon_contours(
        trajectories[:, -1, :2], trajectories[:, -1, 2], width_length
    )
    remaining_trajectories = trajectories.copy()
    remaining_endpoints = endpoints
    rng = np.random.default_rng(seed)
    clusters = []
    for cluster_index in range(num_clusters):
        if len(remaining_trajectories) == 0:
            break
        choice_index = 0 if cluster_index == 0 else int(
            rng.integers(0, len(remaining_trajectories))
        )
        pivot = remaining_endpoints[choice_index]
        distances = np.linalg.norm(remaining_endpoints - pivot, axis=-1).mean(axis=-1)
        inside = distances <= tolerance_m
        clusters.append(remaining_trajectories[inside].mean(axis=0))
        keep = ~inside
        remaining_trajectories = remaining_trajectories[keep]
        remaining_endpoints = remaining_endpoints[keep]
        if (cluster_index + 1) % 128 == 0 or cluster_index + 1 == num_clusters:
            print(
                f"  K-disk {cluster_index + 1}/{num_clusters}: "
                f"{len(remaining_trajectories)} trajectories remain",
                flush=True,
            )
    result = np.stack(clusters, axis=0).astype(np.float32)
    result[:, :, 2] = _wrap_angle(result[:, :, 2])
    return result


def _third_difference_metric(values: np.ndarray, dt: float = 0.1) -> np.ndarray:
    third = np.diff(values, n=3, axis=1) / (dt**3)
    if third.ndim == 3:
        return np.linalg.norm(third, axis=-1).mean(axis=-1)
    return np.abs(third).mean(axis=-1)


def _quantization_errors(
    source: np.ndarray,
    tokens: np.ndarray,
    width_length: Sequence[float],
    sample_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(source) > sample_size:
        source = source[rng.choice(len(source), size=sample_size, replace=False)]
    source_contour = polygon_contours(
        source[:, -1, :2], source[:, -1, 2], width_length
    )
    token_contour = polygon_contours(
        tokens[:, -1, :2], tokens[:, -1, 2], width_length
    )
    errors = []
    for start in range(0, len(source_contour), 128):
        current = source_contour[start : start + 128]
        distances = np.linalg.norm(
            current[:, None, :, :] - token_contour[None, :, :, :], axis=-1
        ).mean(axis=-1)
        errors.append(distances.min(axis=1))
    return np.concatenate(errors, axis=0)


def _metric_row(
    dataset: str,
    key: str,
    source: np.ndarray,
    tokens: np.ndarray,
    candidate_count: int,
    sample_size: int,
    seed: int,
) -> Dict[str, Any]:
    linear_jerk = _third_difference_metric(tokens[:, :, :2])
    unwrapped_heading = np.unwrap(tokens[:, :, 2], axis=1)
    angular_jerk = _third_difference_metric(unwrapped_heading)
    errors = _quantization_errors(
        source,
        tokens,
        CANONICAL_WIDTH_LENGTH[key],
        sample_size,
        seed,
    )
    displacement = np.linalg.norm(tokens[:, -1, :2], axis=-1)
    return {
        "dataset": dataset,
        "class": key,
        "candidate_segment_count": int(candidate_count),
        "unique_segment_count": int(len(source)),
        "token_count": int(len(tokens)),
        "endpoint_displacement_mean_m": float(np.mean(displacement)),
        "endpoint_displacement_p95_m": float(np.percentile(displacement, 95)),
        "linear_jerk_median_mps3": float(np.median(linear_jerk)),
        "linear_jerk_p95_mps3": float(np.percentile(linear_jerk, 95)),
        "angular_jerk_median_radps3": float(np.median(angular_jerk)),
        "angular_jerk_p95_radps3": float(np.percentile(angular_jerk, 95)),
        "quantization_error_mean_m": float(np.mean(errors)),
        "quantization_error_p95_m": float(np.percentile(errors, 95)),
    }


def _save_vocab(
    path: Path,
    tokens: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> None:
    token_contours = {
        key: polygon_contours(
            token[:, :, :2], token[:, :, 2], CANONICAL_WIDTH_LENGTH[key]
        )
        for key, token in tokens.items()
    }
    _write_pickle(
        path,
        {
            "token_all": token_contours,
            "token_center": dict(tokens),
            "metadata": dict(metadata),
        },
    )


def _vocab_export_path(args: argparse.Namespace) -> Path:
    output_dir = Path(args.vocab_output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_name = Path(args.vocab_output_name)
    if output_name.name != str(output_name) or output_name.suffix != ".pkl":
        raise ValueError("--vocab-output-name must be a .pkl file name")
    return output_dir.resolve() / output_name.name


def _axis_limits(arrays: Sequence[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.concatenate([array[:, :, :2].reshape(-1, 2) for array in arrays], axis=0)
    minimum = np.minimum(points.min(axis=0), 0.0)
    maximum = np.maximum(points.max(axis=0), 0.0)
    span = np.maximum(maximum - minimum, 1.0)
    padding = np.maximum(span * 0.06, 0.25)
    return (
        (float(minimum[0] - padding[0]), float(maximum[0] + padding[0])),
        (float(minimum[1] - padding[1]), float(maximum[1] + padding[1])),
    )


def _plot_tokens(
    tokens_by_dataset: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"veh": "#0072B2", "ped": "#009E73", "cyc": "#D55E00"}
    dataset_titles = {"original": "Original CatK", "reconstructed": "Reconstructed"}
    rows = (("all", "All classes"),) + tuple(
        (key, TYPE_LABELS[key]) for key in TYPE_KEYS
    )
    figure, axes = plt.subplots(4, 2, figsize=(14, 15), constrained_layout=True)
    for row_index, (key, label) in enumerate(rows):
        selected_keys = TYPE_KEYS if key == "all" else (key,)
        arrays = [
            tokens_by_dataset[dataset][selected_key]
            for dataset in ("original", "reconstructed")
            for selected_key in selected_keys
        ]
        xlim, ylim = _axis_limits(arrays)
        for column_index, dataset in enumerate(("original", "reconstructed")):
            axis = axes[row_index, column_index]
            for selected_key in selected_keys:
                values = tokens_by_dataset[dataset][selected_key]
                for trajectory in values:
                    axis.plot(
                        trajectory[:, 0],
                        trajectory[:, 1],
                        color=colors[selected_key],
                        alpha=0.32 if key == "all" else 0.42,
                        linewidth=0.45,
                    )
            axis.scatter([0.0], [0.0], s=9, color="#222222", zorder=4)
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            # Match the wide, shallow TrajTok-style view while retaining exact
            # numeric axes for metric interpretation.
            axis.set_aspect("auto")
            axis.grid(True, linewidth=0.35, alpha=0.28)
            axis.set_xlabel("longitudinal x (m)")
            axis.set_ylabel("lateral y (m)")
            axis.set_title(f"{label} · {dataset_titles[dataset]}")
    figure.savefig(output_dir / "token_vocab_comparison.png", dpi=220)
    plt.close(figure)

    for key in ("all", *TYPE_KEYS):
        selected_keys = TYPE_KEYS if key == "all" else (key,)
        arrays = [
            tokens_by_dataset[dataset][selected_key]
            for dataset in ("original", "reconstructed")
            for selected_key in selected_keys
        ]
        xlim, ylim = _axis_limits(arrays)
        figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        for column_index, dataset in enumerate(("original", "reconstructed")):
            axis = axes[column_index]
            for selected_key in selected_keys:
                for trajectory in tokens_by_dataset[dataset][selected_key]:
                    axis.plot(
                        trajectory[:, 0],
                        trajectory[:, 1],
                        color=colors[selected_key],
                        alpha=0.32 if key == "all" else 0.45,
                        linewidth=0.5,
                    )
            axis.scatter([0.0], [0.0], s=10, color="#222222", zorder=4)
            axis.set_xlim(*xlim)
            axis.set_ylim(*ylim)
            axis.set_aspect("auto")
            axis.grid(True, linewidth=0.35, alpha=0.28)
            axis.set_xlabel("longitudinal x (m)")
            axis.set_ylabel("lateral y (m)")
            axis.set_title(dataset_titles[dataset])
        label = "all" if key == "all" else key
        figure.savefig(output_dir / f"{label}_tokens_comparison.png", dpi=220)
        plt.close(figure)


def cluster_and_visualize(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    original_dir = output_dir / "datasets" / "original" / "training"
    reconstructed_dir = output_dir / "datasets" / "reconstructed" / "training"
    if not original_dir.is_dir() or not reconstructed_dir.is_dir():
        raise FileNotFoundError(
            "Both original and reconstructed cache directories are required; "
            "run --stage preprocess first."
        )

    segments_by_dataset: Dict[str, Dict[str, np.ndarray]] = {}
    candidates_by_dataset: Dict[str, Dict[str, int]] = {}
    for dataset, cache_dir in (
        ("original", original_dir),
        ("reconstructed", reconstructed_dir),
    ):
        print(f"Collecting {dataset} 0.5 s segments from {cache_dir}", flush=True)
        segments, candidate_counts = collect_segments(
            cache_dir, args.max_trajectories_per_class
        )
        segments_by_dataset[dataset] = segments
        candidates_by_dataset[dataset] = candidate_counts
        for key in TYPE_KEYS:
            print(
                f"  {key}: {len(segments[key])} unique / "
                f"{candidate_counts[key]} candidates",
                flush=True,
            )

    segment_dir = output_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        segment_dir / "trajectory_segments.npz",
        **{
            f"{dataset}_{key}": segments_by_dataset[dataset][key]
            for dataset in ("original", "reconstructed")
            for key in TYPE_KEYS
        },
    )

    target_clusters_by_class = {
        key: min(
            args.num_clusters,
            len(segments_by_dataset["original"][key]),
            len(segments_by_dataset["reconstructed"][key]),
        )
        for key in TYPE_KEYS
    }
    if min(target_clusters_by_class.values()) < 1:
        raise RuntimeError("No supported trajectory segments were found")
    print(
        f"Requested/per-class K-disk targets: {target_clusters_by_class}",
        flush=True,
    )

    clustered: Dict[str, Dict[str, np.ndarray]] = {
        "original": {},
        "reconstructed": {},
    }
    for dataset in ("original", "reconstructed"):
        for class_index, key in enumerate(TYPE_KEYS):
            print(f"Clustering {dataset} {key}", flush=True)
            clustered[dataset][key] = kdisk_cluster(
                segments_by_dataset[dataset][key],
                target_clusters_by_class[key],
                args.cluster_tolerance_m,
                CANONICAL_WIDTH_LENGTH[key],
                args.seed + class_index,
            )
    common_clusters_by_class = {
        key: min(
            len(clustered["original"][key]),
            len(clustered["reconstructed"][key]),
        )
        for key in TYPE_KEYS
    }
    for key in TYPE_KEYS:
        if common_clusters_by_class[key] < target_clusters_by_class[key]:
            print(
                f"K-disk exhausted {key}; trimming both comparison branches "
                f"to {common_clusters_by_class[key]} tokens",
                flush=True,
            )
        for dataset in clustered:
            clustered[dataset][key] = clustered[dataset][key][
                : common_clusters_by_class[key]
            ]
    catk_common_clusters = min(common_clusters_by_class.values())
    catk_compatible = {
        dataset: {
            key: clustered[dataset][key][:catk_common_clusters]
            for key in TYPE_KEYS
        }
        for dataset in ("original", "reconstructed")
    }

    vocab_dir = output_dir / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_metadata = {
        "algorithm": "CatK K-disk",
        "seed": args.seed,
        "requested_num_clusters": args.num_clusters,
        "target_num_clusters_by_class": target_clusters_by_class,
        "analysis_num_clusters_by_class": common_clusters_by_class,
        "catk_compatible_common_num_clusters": catk_common_clusters,
        "tolerance_m": args.cluster_tolerance_m,
        "shift_frames": 5,
        "segment_frames": 6,
        "sample_hz": 10,
        "canonical_width_length_m": CANONICAL_WIDTH_LENGTH,
    }
    for dataset in ("original", "reconstructed"):
        _save_vocab(
            vocab_dir / f"{dataset}_agent_vocab.pkl",
            catk_compatible[dataset],
            {
                **vocab_metadata,
                "dataset": dataset,
                "vocabulary_scope": "catk_compatible_common_size",
            },
        )
        _save_vocab(
            vocab_dir / f"{dataset}_analysis_vocab.pkl",
            clustered[dataset],
            {
                **vocab_metadata,
                "dataset": dataset,
                "vocabulary_scope": "per_class_maximum_for_analysis",
            },
        )

    vocab_export_path = _vocab_export_path(args)
    _save_vocab(
        vocab_export_path,
        catk_compatible["reconstructed"],
        {
            **vocab_metadata,
            "dataset": "reconstructed",
            "vocabulary_scope": "catk_compatible_common_size",
            "exported_for_catk": True,
        },
    )

    metrics = []
    for dataset in ("original", "reconstructed"):
        for class_index, key in enumerate(TYPE_KEYS):
            metrics.append(
                _metric_row(
                    dataset,
                    key,
                    segments_by_dataset[dataset][key],
                    clustered[dataset][key],
                    candidates_by_dataset[dataset][key],
                    args.metric_sample_size,
                    args.seed + class_index,
                )
            )
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    _plot_tokens(clustered, output_dir)
    summary = {
        **vocab_metadata,
        "candidate_segment_counts": candidates_by_dataset,
        "unique_segment_counts": {
            dataset: {
                key: int(len(segments_by_dataset[dataset][key])) for key in TYPE_KEYS
            }
            for dataset in ("original", "reconstructed")
        },
        "metrics": metrics,
        "files": {
            "original_vocab": str(vocab_dir / "original_agent_vocab.pkl"),
            "reconstructed_vocab": str(
                vocab_dir / "reconstructed_agent_vocab.pkl"
            ),
            "original_analysis_vocab": str(
                vocab_dir / "original_analysis_vocab.pkl"
            ),
            "reconstructed_analysis_vocab": str(
                vocab_dir / "reconstructed_analysis_vocab.pkl"
            ),
            "catk_vocab_export": str(vocab_export_path),
            "comparison_plot": str(output_dir / "token_vocab_comparison.png"),
        },
    }
    _write_json(output_dir / "comparison_summary.json", summary)
    return summary


def _write_output_readme(output_dir: Path, args: argparse.Namespace) -> None:
    readme = output_dir / "README.md"
    num_cluster_line = f"  --num-clusters {args.num_clusters}"
    if args.write_reconstructed_tfrecord:
        num_cluster_line += " \\"
    reconstructed_tfrecord_line = (
        (
            "- `tfrecords/reconstructed`: optional full-trajectory audit shard; "
            "never use it as CatK input or labels."
        )
        if args.write_reconstructed_tfrecord
        else (
            "- Reconstructed TFRecord serialization was not requested; "
            "the agent-only vocabulary source is under `datasets/reconstructed`."
        )
    )
    lines = [
        "# CatK trajectory-token reconstruction comparison",
        "",
        "This directory contains a matched comparison built from real WOMD training shards.",
        "",
        "- `datasets/original/training`: agent-only CatK caches using legacy interpolation.",
        (
            "- `datasets/reconstructed/training`: agent-only, full-trajectory WOMD "
            "reconstruction used exclusively for vocabulary construction."
        ),
        "- `tfrecords/original`: source shard or a manifest of referenced shards.",
        reconstructed_tfrecord_line,
        "- `segments/trajectory_segments.npz`: local 0.5 s trajectories supplied to K-disk.",
        "- `vocab`: per-class analysis vocabularies plus common-size CatK-compatible vocabularies.",
        f"- Final CatK vocabulary: `{_vocab_export_path(args)}`.",
        "- `metrics.csv`: coverage and smoothness metrics by agent class.",
        "- `*_tokens_comparison.png`: matched-scale trajectory-token plots.",
        "",
        "The caches intentionally omit map tensors because maps are unchanged and are not read by",
        "`traj_clustering.py`. Shape uses the last history frame in both branches. The reconstructed",
        "cache is never passed to CatK as history input or future labels.",
        "",
        "## Reproduction",
        "",
        "```bash",
        (
            "conda run -n womd_tls python -m "
            "src.smart.tokens.compare_trajectory_token_reconstruction \\"
        ),
        f"  --input-path {Path(args.input_tfrecord).expanduser().resolve()} \\",
        *(
            [
                "  --reconstruction-root "
                f"{Path(args.reconstruction_root).expanduser().resolve()} \\"
            ]
            if args.reconstruction_root
            else []
        ),
        f"  --output-dir {output_dir.resolve()} \\",
        f"  --vocab-output-dir {_vocab_export_path(args).parent} \\",
        f"  --vocab-output-name {_vocab_export_path(args).name} \\",
        f"  --method {args.method} --filter-strength {args.filter_strength} \\",
        f"  --num-workers {args.num_workers} --worker-backend {args.worker_backend} \\",
        num_cluster_line,
        *(
            ["  --write-reconstructed-tfrecord"]
            if args.write_reconstructed_tfrecord
            else []
        ),
        "```",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run_config.json", vars(args))
    if args.stage in ("all", "preprocess"):
        preprocessing_summary = preprocess_dataset(args, output_dir)
        print(json.dumps(preprocessing_summary, ensure_ascii=False, indent=2))
    if args.stage in ("all", "cluster"):
        if args.stage == "cluster":
            preprocessing_summary = _refresh_preprocessing_summary(args, output_dir)
            print(json.dumps(preprocessing_summary, ensure_ascii=False, indent=2))
        comparison_summary = cluster_and_visualize(args, output_dir)
        print(json.dumps(comparison_summary, ensure_ascii=False, indent=2))
    _write_output_readme(output_dir, args)


if __name__ == "__main__":
    main()
