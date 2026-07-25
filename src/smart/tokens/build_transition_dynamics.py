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

"""Build fixed token-transition dynamics from the CatK training cache."""

from __future__ import annotations

import copy
import json
import pickle
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch

from src.smart.tokens.future_token_dynamics import (
    build_future_token_dynamics_lookup,
)
from src.smart.tokens.transition_dynamics import (
    TransitionDynamicsAccumulator,
    extract_full_trajectory_dynamics,
)
from src.smart.tokens.transition_dynamics_artifact import (
    make_transition_dynamics_artifact,
    save_transition_dynamics_artifact,
    vocabulary_sha256,
)


VALID_SOURCES = ("raw", "reconstructed")
TOKEN_SHIFT = 5


def build_parser() -> ArgumentParser:
    """Return the training-only transition-table command parser."""

    parser = ArgumentParser(
        description=(
            "Build a fixed CatK token-transition dynamics table from one "
            "training cache."
        )
    )
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--agent-token-file", required=True)
    parser.add_argument(
        "--source",
        choices=VALID_SOURCES,
        default="raw",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--map-token-file",
        default="map_traj_token5.pkl",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--shrinkage-count", type=float, default=8.0)
    parser.add_argument("--max-scenarios", type=int)
    return parser


def build_transition_dynamics(
    training_dir: str | Path,
    agent_token_file: str | Path,
    output: str | Path,
    *,
    source: str = "raw",
    map_token_file: str | Path = "map_traj_token5.pkl",
    batch_size: int = 8,
    num_workers: int = 8,
    max_scenarios: int | None = None,
    shrinkage_count: float = 8.0,
) -> Path:
    """Build one transition table from an explicit training cache."""

    training_dir = Path(training_dir)
    agent_token_file = Path(agent_token_file)
    output = Path(output)
    if not training_dir.is_dir():
        raise FileNotFoundError(
            f"training cache directory does not exist: {training_dir}"
        )
    cache_files = sorted(path for path in training_dir.iterdir() if path.is_file())
    if not cache_files:
        raise ValueError(f"training directory contains no cache files: {training_dir}")
    if not agent_token_file.is_file():
        raise FileNotFoundError(
            f"agent vocabulary does not exist: {agent_token_file}"
        )
    agent_token_file = agent_token_file.resolve()
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if max_scenarios is not None and max_scenarios < 1:
        raise ValueError("max_scenarios must be positive when provided")
    if not np.isfinite(shrinkage_count) or shrinkage_count <= 0.0:
        raise ValueError("shrinkage_count must be finite and positive")

    isolated_fallback = _load_isolated_fallback(agent_token_file)
    n_token = int(isolated_fallback.shape[1])
    accumulator = TransitionDynamicsAccumulator(
        n_agent_types=3,
        n_token=n_token,
    )
    runtime = _load_runtime_components()
    dataset = runtime.MultiDataset(
        raw_dir=str(training_dir),
        transform=lambda value: runtime.HeteroData(value),
    )
    scenario_count = len(dataset)
    if scenario_count < 1:
        raise ValueError(f"training directory contains no cache files: {training_dir}")
    if max_scenarios is not None and max_scenarios < scenario_count:
        scenario_count = max_scenarios
        dataset = runtime.Subset(dataset, range(scenario_count))
    loader = runtime.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    sampling = SimpleNamespace(num_k=1, temp=1.0)
    processor = runtime.TokenProcessor(
        map_token_file=str(map_token_file),
        agent_token_file=str(agent_token_file),
        map_token_sampling=sampling,
        agent_token_sampling=sampling,
        history_dynamics={"is_active": False},
        future_token_dynamics={"is_active": False},
    )
    processor.eval()

    scan_statistics = {
        "candidate_occurrences": 0,
        "accepted_occurrences": 0,
        "skipped_occurrences": 0,
    }
    for batch in _progress(loader, description="training transition dynamics"):
        source_snapshot = {"agent": _snapshot_agent_store(batch["agent"])}
        tokenized_agent = processor.tokenize_agent(batch)
        batch_statistics = accumulate_tokenized_batch(
            accumulator,
            source_snapshot,
            tokenized_agent,
            source=source,
        )
        for key in scan_statistics:
            scan_statistics[key] += int(batch_statistics[key])

    values, coverage_statistics = accumulator.finalize(
        isolated_fallback,
        shrinkage_count=shrinkage_count,
    )
    summary = {
        "source": source,
        "scenarios": int(scenario_count),
        "vocabulary_sha256": vocabulary_sha256(agent_token_file),
        "vocabulary_size": n_token,
        **scan_statistics,
        **coverage_statistics,
    }
    artifact = make_transition_dynamics_artifact(
        values,
        vocabulary_path=agent_token_file,
        source=source,
        dt=0.1,
        clipping_limits=(15.0, 3.0, 15.0),
        shrinkage_count=shrinkage_count,
        statistics=summary,
    )
    result = save_transition_dynamics_artifact(
        output,
        artifact,
        vocabulary_path=agent_token_file,
    )
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _load_runtime_components():
    """Import training dependencies only when a real cache scan starts."""

    from torch.utils.data import Subset
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader

    from src.smart.datasets import MultiDataset
    from src.smart.tokens.token_processor import TokenProcessor

    return SimpleNamespace(
        MultiDataset=MultiDataset,
        DataLoader=DataLoader,
        HeteroData=HeteroData,
        TokenProcessor=TokenProcessor,
        Subset=Subset,
    )


def _progress(iterable, *, description: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=description)


def _snapshot_agent_store(agent_store: Mapping) -> dict:
    keys = ("position", "heading", "valid_mask", "trajectory_reconstructed")
    snapshot = {}
    for key in keys:
        if key not in agent_store:
            continue
        value = agent_store[key]
        if isinstance(value, torch.Tensor):
            snapshot[key] = value.detach().clone()
        else:
            snapshot[key] = copy.deepcopy(value)
    return snapshot


def _load_isolated_fallback(agent_token_file: Path) -> np.ndarray:
    with agent_token_file.open("rb") as handle:
        vocabulary = pickle.load(handle)
    if not isinstance(vocabulary, dict) or not isinstance(
        vocabulary.get("token_all"),
        dict,
    ):
        raise ValueError(
            f"{agent_token_file}: expected a token_all dictionary"
        )
    class_values = []
    token_count = None
    for class_name in ("veh", "ped", "cyc"):
        trajectory = vocabulary["token_all"].get(class_name)
        if trajectory is None:
            raise ValueError(
                f"{agent_token_file}: token_all is missing class {class_name}"
            )
        tensor = torch.as_tensor(trajectory, dtype=torch.float32)
        lookup = build_future_token_dynamics_lookup(
            tensor,
            context=f"{agent_token_file} class {class_name}",
        )
        if token_count is None:
            token_count = int(lookup.shape[0])
        elif int(lookup.shape[0]) != token_count:
            raise ValueError(
                f"{agent_token_file}: agent classes must share one token count"
            )
        class_values.append(lookup.cpu().numpy())
    return np.stack(class_values, axis=0).astype(np.float64)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = build_transition_dynamics(
        training_dir=args.training_dir,
        agent_token_file=args.agent_token_file,
        output=args.output,
        source=args.source,
        map_token_file=args.map_token_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_scenarios=args.max_scenarios,
        shrinkage_count=args.shrinkage_count,
    )
    print(f"Transition dynamics artifact: {output}")
    print(f"Summary: {output.with_suffix('.summary.json')}")


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_source_provenance(
    agent_store: Mapping,
    source: str,
    *,
    context: str,
) -> None:
    """Require a cache whose provenance matches the requested table family."""

    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}")
    marker = agent_store.get("trajectory_reconstructed")
    if marker is None:
        if source == "reconstructed":
            raise ValueError(
                f"{context}: reconstructed source requires "
                "trajectory_reconstructed provenance"
            )
        return

    marker = _as_numpy(marker).astype(bool, copy=False).reshape(-1)
    position = agent_store.get("position")
    if position is not None and marker.size != len(position):
        raise ValueError(
            f"{context}: trajectory_reconstructed must contain one value "
            "per agent"
        )
    if source == "raw" and marker.any():
        raise ValueError(
            f"{context}: raw source cannot use trajectory_reconstructed agents"
        )
    if source == "reconstructed" and (marker.size == 0 or not marker.all()):
        raise ValueError(
            f"{context}: reconstructed source requires every "
            "trajectory_reconstructed marker to be true"
        )


def _clean_raw_heading(valid: np.ndarray, heading: np.ndarray) -> np.ndarray:
    cleaned = np.array(heading, dtype=np.float64, copy=True)
    valid = np.asarray(valid, dtype=bool)
    for index in range(cleaned.shape[1] - 1):
        difference = (
            cleaned[:, index] - cleaned[:, index + 1] + np.pi
        ) % (2.0 * np.pi) - np.pi
        change = (
            (np.abs(difference) > 1.5)
            & valid[:, index]
            & valid[:, index + 1]
        )
        cleaned[change, index + 1] = cleaned[change, index]
    return cleaned


def accumulate_tokenized_batch(
    accumulator: TransitionDynamicsAccumulator,
    data,
    tokenized_agent: Mapping,
    *,
    source: str,
) -> dict:
    """Accumulate endpoint dynamics for adjacent matched token IDs."""

    agent_store = data["agent"]
    validate_source_provenance(
        agent_store,
        source,
        context="training batch",
    )
    position = _as_numpy(agent_store["position"])
    heading = _as_numpy(agent_store["heading"])
    valid = _as_numpy(agent_store["valid_mask"]).astype(bool, copy=False)
    agent_type = _as_numpy(tokenized_agent["type"]).astype(
        np.int64,
        copy=False,
    )
    token_index = _as_numpy(tokenized_agent["gt_idx"]).astype(
        np.int64,
        copy=False,
    )
    token_valid = _as_numpy(tokenized_agent["valid_mask"]).astype(
        bool,
        copy=False,
    )
    if (
        position.ndim != 3
        or position.shape[0] != len(agent_type)
        or position.shape[1] != heading.shape[1]
        or heading.shape != valid.shape
        or heading.shape[0] != len(agent_type)
    ):
        raise ValueError(
            "training position, heading, validity, and type have "
            "incompatible shapes"
        )
    if position.shape[1] != 91:
        raise ValueError(
            "transition dynamics require complete 91-frame training trajectories"
        )
    if token_index.ndim != 2 or token_valid.shape != token_index.shape:
        raise ValueError("gt_idx and token validity must have shape [agent, token]")
    if token_index.shape[0] != len(agent_type):
        raise ValueError("tokenized and source agent counts must match")
    endpoint_index = (
        np.arange(1, token_index.shape[1] + 1, dtype=np.int64)
        * TOKEN_SHIFT
    )
    if endpoint_index.size < 2 or endpoint_index[-1] >= position.shape[1]:
        raise ValueError(
            "source trajectory does not cover every token endpoint"
        )

    if source == "raw":
        heading = _clean_raw_heading(valid, heading)

    endpoint_values = np.zeros(
        (len(agent_type), len(endpoint_index), 3),
        dtype=np.float64,
    )
    endpoint_valid = np.zeros(
        (len(agent_type), len(endpoint_index)),
        dtype=bool,
    )
    for agent_index in range(len(agent_type)):
        dynamics = extract_full_trajectory_dynamics(
            position=position[agent_index],
            heading=heading[agent_index],
            valid_mask=valid[agent_index],
        )
        endpoint_values[agent_index] = dynamics.values[endpoint_index]
        endpoint_valid[agent_index] = dynamics.valid[endpoint_index]

    pair_valid = (
        token_valid[:, :-1]
        & token_valid[:, 1:]
        & endpoint_valid[:, 1:]
    )
    accumulator.add(
        agent_type=np.broadcast_to(
            agent_type[:, None],
            token_index[:, 1:].shape,
        ),
        previous_token=token_index[:, :-1],
        current_token=token_index[:, 1:],
        values=endpoint_values[:, 1:],
        valid=pair_valid,
    )
    candidate_count = int(pair_valid.size)
    accepted_count = int(pair_valid.sum())
    return {
        "candidate_occurrences": candidate_count,
        "accepted_occurrences": accepted_count,
        "skipped_occurrences": candidate_count - accepted_count,
    }


if __name__ == "__main__":
    main()
