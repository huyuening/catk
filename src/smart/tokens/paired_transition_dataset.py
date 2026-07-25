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

"""Pair original and reconstructed CatK caches for transition lookup builds."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


ASSIGNMENT_AGENT_FIELDS = (
    "id",
    "type",
    "position",
    "heading",
    "valid_mask",
)
DYNAMICS_AGENT_FIELDS = (
    *ASSIGNMENT_AGENT_FIELDS,
    "trajectory_reconstructed",
)


def _cache_paths(
    directory: str | Path,
    *,
    label: str,
) -> dict[str, Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{label} directory does not exist: {directory}"
        )
    paths = {
        path.name: path
        for path in sorted(directory.glob("*.pkl"))
        if path.is_file()
    }
    return paths


def _as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _require_agent_fields(
    agent: Mapping,
    fields: tuple[str, ...],
    *,
    context: str,
) -> None:
    for field in fields:
        if field not in agent:
            raise ValueError(f"{context}: agent.{field} is required")


def _unique_agent_ids(agent: Mapping, *, context: str) -> np.ndarray:
    ids = _as_numpy(agent["id"])
    if ids.ndim != 1:
        raise ValueError(f"{context}: agent.id must be one-dimensional")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{context}: agent.id values must be unique")
    return ids.astype(np.int64, copy=False)


def _take_agents(
    value,
    order: np.ndarray,
    *,
    expected_agents: int,
    context: str,
):
    if isinstance(value, torch.Tensor):
        if value.ndim < 1 or value.shape[0] != expected_agents:
            raise ValueError(
                f"{context}: agent field must have {expected_agents} entries"
            )
        index = torch.as_tensor(
            order,
            dtype=torch.long,
            device=value.device,
        )
        return value.index_select(0, index)

    array = np.asarray(value)
    if array.ndim < 1 or array.shape[0] != expected_agents:
        raise ValueError(
            f"{context}: agent field must have {expected_agents} entries"
        )
    return array[order]


def align_reconstructed_cache(
    assignment: Mapping,
    reconstructed: Mapping,
    *,
    context: str,
) -> dict:
    """Return reconstructed dynamics fields in original-cache agent order."""

    if "scenario_id" not in assignment or "scenario_id" not in reconstructed:
        raise ValueError(f"{context}: both caches require scenario_id")
    if assignment["scenario_id"] != reconstructed["scenario_id"]:
        raise ValueError(f"{context}: scenario_id mismatch")

    assignment_has_current = "current_time_index" in assignment
    reconstructed_has_current = "current_time_index" in reconstructed
    if assignment_has_current != reconstructed_has_current:
        raise ValueError(f"{context}: current_time_index presence mismatch")
    if (
        assignment_has_current
        and assignment["current_time_index"]
        != reconstructed["current_time_index"]
    ):
        raise ValueError(f"{context}: current_time_index mismatch")

    assignment_agent = assignment.get("agent")
    reconstructed_agent = reconstructed.get("agent")
    if not isinstance(assignment_agent, Mapping) or not isinstance(
        reconstructed_agent,
        Mapping,
    ):
        raise ValueError(f"{context}: both caches require an agent store")
    _require_agent_fields(
        assignment_agent,
        ASSIGNMENT_AGENT_FIELDS,
        context=f"{context} assignment",
    )
    _require_agent_fields(
        reconstructed_agent,
        DYNAMICS_AGENT_FIELDS,
        context=f"{context} reconstructed",
    )

    assignment_ids = _unique_agent_ids(
        assignment_agent,
        context=f"{context} assignment",
    )
    reconstructed_ids = _unique_agent_ids(
        reconstructed_agent,
        context=f"{context} reconstructed",
    )
    if set(assignment_ids.tolist()) != set(reconstructed_ids.tolist()):
        raise ValueError(f"{context}: agent.id sets differ")

    reconstructed_index = {
        int(agent_id): index
        for index, agent_id in enumerate(reconstructed_ids.tolist())
    }
    order = np.asarray(
        [
            reconstructed_index[int(agent_id)]
            for agent_id in assignment_ids.tolist()
        ],
        dtype=np.int64,
    )
    aligned_agent = {
        field: _take_agents(
            reconstructed_agent[field],
            order,
            expected_agents=len(reconstructed_ids),
            context=f"{context} reconstructed agent.{field}",
        )
        for field in DYNAMICS_AGENT_FIELDS
    }
    aligned_agent["num_nodes"] = len(assignment_ids)

    assignment_type = _as_numpy(assignment_agent["type"]).astype(
        np.int64,
        copy=False,
    )
    aligned_type = _as_numpy(aligned_agent["type"]).astype(
        np.int64,
        copy=False,
    )
    if not np.array_equal(assignment_type, aligned_type):
        raise ValueError(f"{context}: aligned agent.type values differ")

    for field in ("position", "heading", "valid_mask"):
        assignment_shape = _as_numpy(assignment_agent[field]).shape
        reconstructed_shape = _as_numpy(aligned_agent[field]).shape
        if assignment_shape != reconstructed_shape:
            raise ValueError(f"{context}: agent.{field} shapes differ")

    result = {
        "scenario_id": assignment["scenario_id"],
        "agent": aligned_agent,
    }
    if assignment_has_current:
        result["current_time_index"] = assignment["current_time_index"]
    return result


class PairedTransitionDataset(Dataset):
    """Load matched original/reconstructed scenario caches deterministically."""

    def __init__(
        self,
        assignment_dir: str | Path,
        dynamics_dir: str | Path,
        transform: Callable[[Mapping], Any],
    ) -> None:
        assignment_paths = _cache_paths(
            assignment_dir,
            label="assignment training",
        )
        dynamics_paths = _cache_paths(
            dynamics_dir,
            label="dynamics training",
        )
        if set(assignment_paths) != set(dynamics_paths):
            raise ValueError(
                "assignment and dynamics cache file sets differ"
            )
        if not assignment_paths:
            raise ValueError(
                "paired training directories contain no .pkl cache files"
            )
        self._pairs = [
            (assignment_paths[name], dynamics_paths[name])
            for name in sorted(assignment_paths)
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int):
        assignment_path, dynamics_path = self._pairs[index]
        with assignment_path.open("rb") as stream:
            assignment = pickle.load(stream)
        with dynamics_path.open("rb") as stream:
            reconstructed = pickle.load(stream)
        dynamics = align_reconstructed_cache(
            assignment,
            reconstructed,
            context=assignment_path.name,
        )
        return self.transform(assignment), self.transform(dynamics)
