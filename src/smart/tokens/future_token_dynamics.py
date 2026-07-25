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

"""Body-frame endpoint dynamics derived directly from CatK token trajectories."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor


def _error_prefix(context: Optional[str]) -> str:
    return f"{context}: " if context else ""


def _second_order_gradient(values: Tensor, dt: float) -> Tensor:
    """Differentiate along the six-frame axis like NumPy edge_order=2."""

    gradient = torch.empty_like(values)
    gradient[:, 0] = (
        -3.0 * values[:, 0] + 4.0 * values[:, 1] - values[:, 2]
    ) / (2.0 * dt)
    gradient[:, -1] = (
        3.0 * values[:, -1] - 4.0 * values[:, -2] + values[:, -3]
    ) / (2.0 * dt)
    gradient[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * dt)
    return gradient


def _unwrap_heading(heading: Tensor) -> Tensor:
    delta = heading[:, 1:] - heading[:, :-1]
    delta = torch.remainder(delta + math.pi, 2.0 * math.pi) - math.pi
    return torch.cat(
        (
            heading[:, :1],
            heading[:, :1] + torch.cumsum(delta, dim=1),
        ),
        dim=1,
    )


@torch.no_grad()
def build_future_token_dynamics_lookup(
    token_trajectory: Tensor,
    *,
    dt: float = 0.1,
    clipping_limits: Sequence[float] = (15.0, 3.0, 15.0),
    context: Optional[str] = None,
) -> Tensor:
    """Return `[a_lon, angular_speed, a_lat]` at frame 5 for every token."""

    prefix = _error_prefix(context)
    if (
        not isinstance(token_trajectory, Tensor)
        or token_trajectory.ndim != 4
        or tuple(token_trajectory.shape[1:]) != (6, 4, 2)
    ):
        shape = getattr(token_trajectory, "shape", None)
        raise ValueError(
            f"{prefix}token trajectory must have shape [n_token, 6, 4, 2], "
            f"got {shape}"
        )
    if not token_trajectory.is_floating_point():
        raise ValueError(f"{prefix}token trajectory must use a floating dtype")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"{prefix}dt must be finite and positive")
    if len(clipping_limits) != 3:
        raise ValueError(f"{prefix}clipping_limits must contain three values")

    limits = token_trajectory.new_tensor(tuple(float(v) for v in clipping_limits))
    if not torch.isfinite(limits).all() or bool((limits <= 0).any()):
        raise ValueError(
            f"{prefix}clipping_limits must contain three finite positive values"
        )
    if not torch.isfinite(token_trajectory).all():
        raise ValueError(f"{prefix}token trajectory contains non-finite values")

    center = token_trajectory.mean(dim=2)
    forward = token_trajectory[:, :, 0] - token_trajectory[:, :, 3]
    heading = torch.atan2(forward[..., 1], forward[..., 0])
    unwrapped_heading = _unwrap_heading(heading)

    velocity = _second_order_gradient(center, dt)
    acceleration = _second_order_gradient(velocity, dt)
    angular_speed = _second_order_gradient(unwrapped_heading, dt)

    endpoint_heading = unwrapped_heading[:, -1]
    cosine = endpoint_heading.cos()
    sine = endpoint_heading.sin()
    endpoint_acceleration = acceleration[:, -1]
    longitudinal_acceleration = (
        endpoint_acceleration[:, 0] * cosine
        + endpoint_acceleration[:, 1] * sine
    )
    lateral_acceleration = (
        -endpoint_acceleration[:, 0] * sine
        + endpoint_acceleration[:, 1] * cosine
    )
    dynamics = torch.stack(
        (
            longitudinal_acceleration,
            angular_speed[:, -1],
            lateral_acceleration,
        ),
        dim=-1,
    )
    if not torch.isfinite(dynamics).all():
        raise ValueError(f"{prefix}derived dynamics contain non-finite values")

    return torch.maximum(torch.minimum(dynamics, limits), -limits)


@torch.no_grad()
def gather_future_token_dynamics(
    token_index: Tensor,
    agent_type: Tensor,
    dynamics_veh: Tensor,
    dynamics_ped: Tensor,
    dynamics_cyc: Tensor,
) -> Tensor:
    """Gather class-specific lookup rows for each agent and token position."""

    if token_index.ndim < 1:
        raise ValueError("token_index must have at least one dimension")
    if agent_type.shape != token_index.shape[:1]:
        raise ValueError("agent_type must have shape [n_agent]")
    if token_index.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("token_index must use an integer dtype")
    if bool(((agent_type < 0) | (agent_type > 2)).any()):
        raise ValueError("agent_type values must be 0, 1, or 2")

    lookup_tables = (dynamics_veh, dynamics_ped, dynamics_cyc)
    reference = dynamics_veh
    for name, table in zip(("veh", "ped", "cyc"), lookup_tables):
        if table.ndim != 2 or table.shape[1] != 3:
            raise ValueError(f"dynamics_{name} must have shape [n_token, 3]")
        if table.device != reference.device or table.dtype != reference.dtype:
            raise ValueError("all dynamics lookup tables must share dtype and device")
    if token_index.device != reference.device or agent_type.device != reference.device:
        raise ValueError(
            "token_index, agent_type, and dynamics tables must share a device"
        )

    gathered = reference.new_empty((*token_index.shape, 3))
    for class_index, table in enumerate(lookup_tables):
        mask = agent_type == class_index
        class_token_index = token_index[mask].long()
        if bool(
            ((class_token_index < 0) | (class_token_index >= table.shape[0])).any()
        ):
            raise IndexError(
                f"token_index is outside the dynamics lookup for agent type "
                f"{class_index}"
            )
        gathered[mask] = table[class_token_index]
    return gathered
