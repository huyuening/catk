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

"""CatK agent feature extraction with causal object dimensions."""

from typing import Any, Dict

import numpy as np
import torch
from scipy.interpolate import interp1d


def get_causal_object_shape(
    states: np.ndarray, valid: np.ndarray, history_index: int
) -> np.ndarray:
    """Return last-history dimensions with causal non-zero mean fallback.

    A finite positive component at the last history frame is kept unchanged.
    A missing, zero, negative, or non-finite component is replaced by the mean
    of that component over raw-valid history states with finite positive
    values.  Future dimensions are never inspected.
    """

    states = np.asarray(states)
    valid = np.asarray(valid, dtype=bool)
    if states.ndim != 2 or states.shape[-1] < 6:
        raise ValueError("states must have shape [steps, >=6]")
    if valid.shape != states.shape[:1]:
        raise ValueError(
            "states and valid must have the same step dimension: "
            f"{states.shape[:1]} != {valid.shape}"
        )
    if history_index < 0 or history_index >= len(states):
        raise IndexError(history_index)

    shape = states[history_index, 3:6].astype(np.float32, copy=True)
    history_dimensions = states[: history_index + 1, 3:6]
    history_valid = valid[: history_index + 1]
    for dimension in range(3):
        if np.isfinite(shape[dimension]) and shape[dimension] > 0:
            continue
        values = history_dimensions[:, dimension]
        usable = history_valid & np.isfinite(values) & (values > 0)
        if usable.any():
            shape[dimension] = np.mean(values[usable], dtype=np.float64)
    return shape


def get_agent_features(
    track_infos: Dict[str, np.ndarray],
    split,
    num_historical_steps: int,
    num_steps: int,
) -> Dict[str, Any]:
    """Convert decoded Scenario tracks to the unchanged CatK model tensors.

    Position, heading, and velocity retain CatK's legacy interpolation.  Only
    object dimensions change: they come from the last observable history frame,
    with a non-zero history-only mean for missing components, instead of an
    average that can include future frames.
    """

    del split  # Kept in the public signature for existing preprocessing callers.

    idx_agents_to_add = []
    for i in range(len(track_infos["object_id"])):
        add_agent = track_infos["valid"][i, num_historical_steps - 1]
        if add_agent:
            idx_agents_to_add.append(i)

    num_agents = len(idx_agents_to_add)
    out_dict = {
        "num_nodes": num_agents,
        "valid_mask": torch.zeros([num_agents, num_steps], dtype=torch.bool),
        "role": torch.zeros([num_agents, 3], dtype=torch.bool),
        "id": torch.zeros(num_agents, dtype=torch.int64) - 1,
        "type": torch.zeros(num_agents, dtype=torch.uint8),
        "position": torch.zeros([num_agents, num_steps, 3], dtype=torch.float32),
        "heading": torch.zeros([num_agents, num_steps], dtype=torch.float32),
        "velocity": torch.zeros([num_agents, num_steps, 2], dtype=torch.float32),
        "shape": torch.zeros([num_agents, 3], dtype=torch.float32),
    }

    for i, idx in enumerate(idx_agents_to_add):
        out_dict["role"][i] = torch.from_numpy(track_infos["role"][idx])
        out_dict["id"][i] = track_infos["object_id"][idx]
        out_dict["type"][i] = track_infos["object_type"][idx]

        valid = track_infos["valid"][idx]
        states = track_infos["states"][idx]

        # WOSAC fixes box dimensions at the last observable history frame.  The
        # same causal value is used for every split; malformed components fall
        # back only to positive observations from the available history.
        object_shape = get_causal_object_shape(
            states, valid, num_historical_steps - 1
        )
        out_dict["shape"][i] = torch.from_numpy(object_shape)

        valid_steps = np.where(valid)[0]
        position = states[:, :3]
        velocity = states[:, 7:9]
        heading = states[:, 6]
        if valid.sum() > 1:
            t_start, t_end = valid_steps[0], valid_steps[-1]
            f_pos = interp1d(valid_steps, position[valid], axis=0)
            f_vel = interp1d(valid_steps, velocity[valid], axis=0)
            f_yaw = interp1d(valid_steps, np.unwrap(heading[valid], axis=0), axis=0)
            t_in = np.arange(t_start, t_end + 1)
            out_dict["valid_mask"][i, t_start : t_end + 1] = True
            out_dict["position"][i, t_start : t_end + 1] = torch.from_numpy(f_pos(t_in))
            out_dict["velocity"][i, t_start : t_end + 1] = torch.from_numpy(f_vel(t_in))
            out_dict["heading"][i, t_start : t_end + 1] = torch.from_numpy(f_yaw(t_in))
        else:
            t = valid_steps[0]
            out_dict["valid_mask"][i, t] = True
            out_dict["position"][i, t] = torch.from_numpy(position[t])
            out_dict["velocity"][i, t] = torch.from_numpy(velocity[t])
            out_dict["heading"][i, t] = torch.tensor(heading[t])

    return out_dict
