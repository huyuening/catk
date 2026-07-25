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

"""Optional causal conditioning from selected future vocabulary tokens."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.smart.layers.fourier_embedding import MLPEmbedding
from src.smart.tokens.future_token_dynamics import (
    gather_transition_dynamics,
)


class FutureTokenDynamicsConditioner(nn.Module):
    """Embed selected-token dynamics without exposing a token to its own logits."""

    def __init__(self, hidden_dim: int, config: Optional[Any] = None) -> None:
        super().__init__()
        self.is_active = bool(
            config is not None and config.get("is_active", False)
        )
        if not self.is_active:
            return

        normalization_scale = tuple(
            float(value)
            for value in config.get(
                "normalization_scale",
                [5.0, 1.0, 5.0],
            )
        )
        if len(normalization_scale) != 3 or any(
            not math.isfinite(value) or value <= 0.0
            for value in normalization_scale
        ):
            raise ValueError(
                "future_token_dynamics.normalization_scale must contain "
                "three finite positive values"
            )
        initial_gate = float(config.get("initial_gate", 1.0))
        if not math.isfinite(initial_gate):
            raise ValueError(
                "future_token_dynamics.initial_gate must be finite"
            )

        self.embedding = MLPEmbedding(input_dim=3, hidden_dim=hidden_dim)
        self.register_buffer(
            "normalization_scale",
            torch.tensor(normalization_scale, dtype=torch.float32),
            persistent=False,
        )
        self.gate = nn.Parameter(torch.tensor(initial_gate, dtype=torch.float32))

    @staticmethod
    def _require_lookups(
        dynamics_veh: Optional[Tensor],
        dynamics_ped: Optional[Tensor],
        dynamics_cyc: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        lookups = (dynamics_veh, dynamics_ped, dynamics_cyc)
        if any(lookup is None for lookup in lookups):
            raise KeyError(
                "future_token_dynamics is active but tokenized agent data "
                "does not contain every agent_token_dynamics lookup"
            )
        return lookups  # type: ignore[return-value]

    def _embedded_dynamics(
        self,
        previous_token_index: Tensor,
        current_token_index: Tensor,
        agent_type: Tensor,
        dynamics_veh: Optional[Tensor],
        dynamics_ped: Optional[Tensor],
        dynamics_cyc: Optional[Tensor],
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        dynamics_veh, dynamics_ped, dynamics_cyc = self._require_lookups(
            dynamics_veh,
            dynamics_ped,
            dynamics_cyc,
        )
        dynamics = gather_transition_dynamics(
            previous_token_index=previous_token_index,
            current_token_index=current_token_index,
            agent_type=agent_type,
            dynamics_veh=dynamics_veh,
            dynamics_ped=dynamics_ped,
            dynamics_cyc=dynamics_cyc,
        ).to(dtype=dtype)
        normalized = dynamics / self.normalization_scale.to(
            device=dynamics.device,
            dtype=dtype,
        )
        return self.embedding(normalized)

    def add_open_loop(
        self,
        feature: Tensor,
        token_index: Tensor,
        agent_type: Tensor,
        dynamics_veh: Optional[Tensor] = None,
        dynamics_ped: Optional[Tensor] = None,
        dynamics_cyc: Optional[Tensor] = None,
        num_historical_tokens: int = 2,
    ) -> Tensor:
        """Condition position `t` with `D(k[t-1], k[t])` after history."""

        if not self.is_active:
            return feature
        if feature.ndim != 3 or token_index.shape != feature.shape[:2]:
            raise ValueError(
                "open-loop feature and token_index must have shapes "
                "[n_agent, n_step, hidden_dim] and [n_agent, n_step]"
            )
        if num_historical_tokens < 0:
            raise ValueError("num_historical_tokens must be non-negative")

        previous_token_index = torch.cat(
            (token_index[:, :1], token_index[:, :-1]),
            dim=1,
        )
        embedded = self._embedded_dynamics(
            previous_token_index,
            token_index,
            agent_type,
            dynamics_veh,
            dynamics_ped,
            dynamics_cyc,
            dtype=feature.dtype,
        )
        causal_mask = (
            torch.arange(feature.shape[1], device=feature.device)
            >= num_historical_tokens
        )
        embedded = embedded * causal_mask.view(1, -1, 1).to(embedded.dtype)
        return feature + self.gate.to(feature.dtype) * embedded.to(feature.dtype)

    def add_selected(
        self,
        feature: Tensor,
        previous_token_index: Tensor,
        current_token_index: Tensor,
        agent_type: Tensor,
        dynamics_veh: Optional[Tensor] = None,
        dynamics_ped: Optional[Tensor] = None,
        dynamics_cyc: Optional[Tensor] = None,
    ) -> Tensor:
        """Condition a selected token with its incoming token transition."""

        if not self.is_active:
            return feature
        if (
            feature.ndim != 2
            or previous_token_index.shape != feature.shape[:1]
            or current_token_index.shape != feature.shape[:1]
        ):
            raise ValueError(
                "selected feature and token indices must have shapes "
                "[n_agent, hidden_dim] and [n_agent]"
            )

        embedded = self._embedded_dynamics(
            previous_token_index,
            current_token_index,
            agent_type,
            dynamics_veh,
            dynamics_ped,
            dynamics_cyc,
            dtype=feature.dtype,
        )
        return feature + self.gate.to(feature.dtype) * embedded.to(feature.dtype)
