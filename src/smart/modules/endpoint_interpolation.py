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

"""Inference-only reconstruction of 10 Hz trajectories from token endpoints.

This module mirrors TrajTok's endpoint interpolation post-processing while
keeping it independent from CatK's learned decoder and checkpoint state.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F


def wrap_angle(
    angle: torch.Tensor,
    min_value: float = -math.pi,
    max_value: float = math.pi,
) -> torch.Tensor:
    """Wrap angles without importing the decoder's heavier utility package."""
    return min_value + (angle + max_value) % (max_value - min_value)


class EndpointInterpolator:
    """Reconstruct token-internal points without changing token generation."""

    def __init__(
        self,
        config: Optional[Any] = None,
        shift: int = 5,
        step_seconds: float = 0.1,
    ) -> None:
        self.config = config
        self.shift = int(shift)
        self.step_seconds = float(step_seconds)

    @property
    def is_active(self) -> bool:
        return bool(self._get("is_active", False))

    def _get(self, key: str, default: Any) -> Any:
        if self.config is None:
            return default
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _uses_tangent_heading(self) -> bool:
        heading_method = str(self._get("heading_method", "endpoint_cubic")).lower()
        return heading_method in {"tangent", "xy_tangent"}

    @staticmethod
    def _unwrapped_endpoint_heading(
        start_head: torch.Tensor, endpoint_head: torch.Tensor
    ) -> torch.Tensor:
        control_head = torch.cat([start_head.unsqueeze(1), endpoint_head], dim=1)
        delta = wrap_angle(control_head[:, 1:] - control_head[:, :-1])
        return torch.cat(
            [control_head[:, :1], control_head[:, :1] + torch.cumsum(delta, dim=1)],
            dim=1,
        )

    def _interpolate_endpoint_heading(
        self, start_head: torch.Tensor, endpoint_head: torch.Tensor
    ) -> torch.Tensor:
        heading_method = str(self._get("heading_method", "endpoint_cubic")).lower()
        control_head = self._unwrapped_endpoint_heading(start_head, endpoint_head)
        tau = (
            torch.arange(
                1,
                self.shift + 1,
                device=start_head.device,
                dtype=start_head.dtype,
            )
            / self.shift
        )

        if heading_method in {"endpoint_cubic", "cubic", "natural_cubic"}:
            second_derivatives = self._natural_cubic_second_derivatives(control_head)
            tau_view = tau.view(1, 1, self.shift)
            one_minus_tau = 1.0 - tau_view
            h0 = control_head[:, :-1].unsqueeze(-1)
            h1 = control_head[:, 1:].unsqueeze(-1)
            m0 = second_derivatives[:, :-1].unsqueeze(-1)
            m1 = second_derivatives[:, 1:].unsqueeze(-1)
            segment_head = (
                m0 * one_minus_tau.pow(3) / 6.0
                + m1 * tau_view.pow(3) / 6.0
                + (h0 - m0 / 6.0) * one_minus_tau
                + (h1 - m1 / 6.0) * tau_view
            )
            return wrap_angle(segment_head.flatten(1, 2))

        h0 = control_head[:, :-1].unsqueeze(-1)
        h1 = control_head[:, 1:].unsqueeze(-1)
        delta = wrap_angle(h1 - h0)
        if heading_method in {"endpoint_smoothstep", "smoothstep"}:
            tau = tau * tau * (3.0 - 2.0 * tau)
        segment_head = wrap_angle(h0 + delta * tau.view(1, 1, self.shift))
        return segment_head.flatten(1, 2)

    @staticmethod
    def _smooth_temporal_sequence(
        values: torch.Tensor, window_size: int, iterations: int
    ) -> torch.Tensor:
        window_size = int(window_size)
        iterations = int(iterations)
        if window_size <= 1 or iterations <= 0:
            return values
        if window_size % 2 == 0:
            window_size += 1

        squeeze_last = values.dim() == 2
        if squeeze_last:
            values = values.unsqueeze(-1)

        smoothed = values.transpose(1, 2)
        padding = window_size // 2
        for _ in range(iterations):
            smoothed = F.pad(smoothed, (padding, padding), mode="replicate")
            smoothed = F.avg_pool1d(smoothed, kernel_size=window_size, stride=1)
        smoothed = smoothed.transpose(1, 2)
        return smoothed.squeeze(-1) if squeeze_last else smoothed

    def _preserve_smoothing_start(
        self, original: torch.Tensor, smoothed: torch.Tensor
    ) -> torch.Tensor:
        preserve_steps = int(self._get("smooth_preserve_start_steps", self.shift))
        transition_steps = int(
            self._get("smooth_start_transition_steps", self.shift)
        )
        if preserve_steps <= 0 and transition_steps <= 0:
            return smoothed

        n_step = original.shape[1]
        preserve_steps = max(0, min(preserve_steps, n_step))
        transition_steps = max(0, min(transition_steps, n_step - preserve_steps))
        if preserve_steps > 0:
            smoothed = smoothed.clone()
            smoothed[:, :preserve_steps] = original[:, :preserve_steps]
        if transition_steps > 0:
            start = preserve_steps
            end = preserve_steps + transition_steps
            blend = torch.linspace(
                0.0,
                1.0,
                steps=transition_steps + 2,
                dtype=original.dtype,
                device=original.device,
            )[1:-1]
            view_shape = [1, transition_steps] + [1] * (original.dim() - 2)
            blend = blend.view(*view_shape)
            smoothed[:, start:end] = (
                original[:, start:end] * (1.0 - blend)
                + smoothed[:, start:end] * blend
            )
        return smoothed

    def _smooth_output(
        self,
        pred_traj: torch.Tensor,
        pred_head: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not bool(self._get("smooth_output", True)):
            return pred_traj, pred_head

        if not bool(self._get("smooth_moving_agents_only", True)):
            agent_mask = torch.ones_like(agent_mask)

        smooth_traj = self._smooth_temporal_sequence(
            pred_traj,
            self._get("xy_smoothing_window", 11),
            self._get("xy_smoothing_iterations", 2),
        )
        unwrapped_head = self._unwrapped_endpoint_heading(
            pred_head[:, 0], pred_head[:, 1:]
        )
        smooth_head = self._smooth_temporal_sequence(
            unwrapped_head,
            self._get("heading_smoothing_window", 15),
            self._get("heading_smoothing_iterations", 2),
        )
        smooth_head = wrap_angle(smooth_head)
        smooth_traj = self._preserve_smoothing_start(pred_traj, smooth_traj)
        smooth_head = wrap_angle(
            self._preserve_smoothing_start(
                unwrapped_head,
                self._unwrapped_endpoint_heading(
                    smooth_head[:, 0], smooth_head[:, 1:]
                ),
            )
        )

        pred_traj = torch.where(
            agent_mask.view(-1, 1, 1), smooth_traj, pred_traj
        )
        pred_head = torch.where(agent_mask.view(-1, 1), smooth_head, pred_head)
        return pred_traj, pred_head

    def _endpoint_segment_speed(
        self, start_pos: torch.Tensor, endpoint_pos: torch.Tensor
    ) -> torch.Tensor:
        control_pos = torch.cat([start_pos.unsqueeze(1), endpoint_pos], dim=1)
        step_dist = torch.norm(
            control_pos[:, 1:] - control_pos[:, :-1], p=2, dim=-1
        )
        return step_dist / (self.shift * self.step_seconds)

    def _moving_agent_mask(self, segment_speed: torch.Tensor) -> torch.Tensor:
        if not bool(self._get("moving_only", True)):
            return torch.ones(
                segment_speed.shape[0],
                dtype=torch.bool,
                device=segment_speed.device,
            )
        threshold_mps = float(self._get("moving_speed_threshold_mps", 0.5))
        return segment_speed.mean(dim=-1) >= threshold_mps

    def _moving_segment_mask(self, segment_speed: torch.Tensor) -> torch.Tensor:
        if not bool(self._get("moving_segment_only", True)):
            return torch.ones_like(segment_speed, dtype=torch.bool)
        threshold_mps = float(
            self._get(
                "moving_segment_speed_threshold_mps",
                self._get("moving_speed_threshold_mps", 0.5),
            )
        )
        return segment_speed >= threshold_mps

    def _low_speed_agent_mask(self, segment_speed: torch.Tensor) -> torch.Tensor:
        if not bool(self._get("low_speed_reconstruction", False)):
            return torch.zeros(
                segment_speed.shape[0],
                dtype=torch.bool,
                device=segment_speed.device,
            )
        threshold_mps = float(self._get("low_speed_min_segment_threshold_mps", 0.1))
        return segment_speed.min(dim=-1).values <= threshold_mps

    def _static_agent_mask(
        self,
        start_pos: torch.Tensor,
        endpoint_pos: torch.Tensor,
        segment_speed: torch.Tensor,
        agent_type: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not bool(self._get("static_reconstruction", False)):
            return torch.zeros(
                segment_speed.shape[0],
                dtype=torch.bool,
                device=segment_speed.device,
            )

        vehicle_speed_threshold = float(
            self._get("static_max_segment_speed_threshold_mps", 0.5)
        )
        vehicle_span_threshold = float(
            self._get("static_endpoint_span_threshold_m", 1.0)
        )
        pedestrian_speed_threshold = float(
            self._get("static_pedestrian_max_segment_speed_threshold_mps", 0.2)
        )
        pedestrian_span_threshold = float(
            self._get("static_pedestrian_endpoint_span_threshold_m", 0.5)
        )
        cyclist_speed_threshold = float(
            self._get("static_cyclist_max_segment_speed_threshold_mps", 0.3)
        )
        cyclist_span_threshold = float(
            self._get("static_cyclist_endpoint_span_threshold_m", 0.7)
        )

        max_speed = segment_speed.max(dim=-1).values
        control_pos = torch.cat([start_pos.unsqueeze(1), endpoint_pos], dim=1)
        endpoint_span = torch.norm(
            control_pos.max(dim=1).values - control_pos.min(dim=1).values,
            p=2,
            dim=-1,
        )

        speed_threshold = torch.full_like(max_speed, vehicle_speed_threshold)
        span_threshold = torch.full_like(endpoint_span, vehicle_span_threshold)
        pedestrian_mask = torch.zeros_like(max_speed, dtype=torch.bool)
        cyclist_mask = torch.zeros_like(max_speed, dtype=torch.bool)
        if agent_type is not None:
            agent_type = agent_type.to(device=segment_speed.device)
            pedestrian_mask = agent_type == 1
            cyclist_mask = agent_type == 2
            speed_threshold = torch.where(
                pedestrian_mask,
                torch.full_like(speed_threshold, pedestrian_speed_threshold),
                speed_threshold,
            )
            speed_threshold = torch.where(
                cyclist_mask,
                torch.full_like(speed_threshold, cyclist_speed_threshold),
                speed_threshold,
            )
            span_threshold = torch.where(
                pedestrian_mask,
                torch.full_like(span_threshold, pedestrian_span_threshold),
                span_threshold,
            )
            span_threshold = torch.where(
                cyclist_mask,
                torch.full_like(span_threshold, cyclist_span_threshold),
                span_threshold,
            )

        static_by_speed = max_speed <= speed_threshold
        static_by_span = endpoint_span <= span_threshold
        non_vehicle_mask = pedestrian_mask | cyclist_mask
        return torch.where(
            non_vehicle_mask,
            static_by_speed & static_by_span,
            static_by_speed | static_by_span,
        )

    @staticmethod
    def _static_reconstruction(
        start_pos: torch.Tensor, start_head: torch.Tensor, n_step: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            start_pos.unsqueeze(1).expand(-1, n_step, -1),
            start_head.unsqueeze(1).expand(-1, n_step),
        )

    def _interpolate_low_speed(
        self,
        start_pos: torch.Tensor,
        start_head: torch.Tensor,
        endpoint_pos: torch.Tensor,
        endpoint_head: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        control_pos = torch.cat([start_pos.unsqueeze(1), endpoint_pos], dim=1)
        control_head = self._unwrapped_endpoint_heading(start_head, endpoint_head)
        tau = (
            torch.arange(
                1,
                self.shift + 1,
                device=start_pos.device,
                dtype=start_pos.dtype,
            )
            / self.shift
        )

        tau_pos = tau.view(1, 1, self.shift, 1)
        segment_pos = (
            control_pos[:, :-1].unsqueeze(2) * (1.0 - tau_pos)
            + control_pos[:, 1:].unsqueeze(2) * tau_pos
        )
        segment_pos[:, :, -1] = control_pos[:, 1:]

        h0 = control_head[:, :-1].unsqueeze(-1)
        h1 = control_head[:, 1:].unsqueeze(-1)
        heading_method = str(
            self._get("low_speed_heading_method", "endpoint_smoothstep")
        ).lower()
        if heading_method in {"constant", "hold", "start"}:
            segment_head = h0.expand(-1, -1, self.shift).clone()
            segment_head[:, :, -1] = h1.squeeze(-1)
        else:
            tau_head = tau
            if heading_method in {"endpoint_smoothstep", "smoothstep"}:
                tau_head = tau_head * tau_head * (3.0 - 2.0 * tau_head)
            segment_head = h0 + (h1 - h0) * tau_head.view(1, 1, self.shift)

        return (
            segment_pos.flatten(1, 2),
            wrap_angle(segment_head.flatten(1, 2)),
        )

    @staticmethod
    def _natural_cubic_second_derivatives(
        control_pos: torch.Tensor,
    ) -> torch.Tensor:
        n_control = control_pos.shape[1]
        n_inner = n_control - 2
        second_derivatives = control_pos.new_zeros(control_pos.shape)
        if n_inner <= 0:
            return second_derivatives

        rhs = 6.0 * (
            control_pos[:, 2:]
            - 2.0 * control_pos[:, 1:-1]
            + control_pos[:, :-2]
        )
        matrix = control_pos.new_zeros(n_inner, n_inner)
        diagonal = torch.arange(n_inner, device=control_pos.device)
        matrix[diagonal, diagonal] = 4.0
        if n_inner > 1:
            matrix[diagonal[:-1], diagonal[1:]] = 1.0
            matrix[diagonal[1:], diagonal[:-1]] = 1.0

        solution = torch.linalg.solve(
            matrix.unsqueeze(0).expand(control_pos.shape[0], -1, -1), rhs
        )
        second_derivatives[:, 1:-1] = solution
        return second_derivatives

    def _interpolate_global_cubic(
        self,
        start_pos: torch.Tensor,
        start_head: torch.Tensor,
        endpoint_pos: torch.Tensor,
        endpoint_head: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        control_pos = torch.cat([start_pos.unsqueeze(1), endpoint_pos], dim=1)
        n_segment = endpoint_pos.shape[1]
        tau = (
            torch.arange(
                1,
                self.shift + 1,
                device=start_pos.device,
                dtype=start_pos.dtype,
            )
            / self.shift
        )
        tau_view = tau.view(1, 1, self.shift, 1)

        if n_segment == 1:
            segment_pos = (
                control_pos[:, :-1].unsqueeze(2) * (1.0 - tau_view)
                + control_pos[:, 1:].unsqueeze(2) * tau_view
            )
        else:
            second_derivatives = self._natural_cubic_second_derivatives(control_pos)
            p0 = control_pos[:, :-1].unsqueeze(2)
            p1 = control_pos[:, 1:].unsqueeze(2)
            m0 = second_derivatives[:, :-1].unsqueeze(2)
            m1 = second_derivatives[:, 1:].unsqueeze(2)
            one_minus_tau = 1.0 - tau_view
            segment_pos = (
                m0 * one_minus_tau.pow(3) / 6.0
                + m1 * tau_view.pow(3) / 6.0
                + (p0 - m0 / 6.0) * one_minus_tau
                + (p1 - m1 / 6.0) * tau_view
            )

        segment_pos[:, :, -1] = control_pos[:, 1:]
        pred_pos = segment_pos.flatten(1, 2)
        if not self._uses_tangent_heading():
            return pred_pos, self._interpolate_endpoint_heading(
                start_head, endpoint_head
            )

        full_points = torch.cat([start_pos.unsqueeze(1), pred_pos], dim=1)
        step_vector = full_points[:, 1:] - full_points[:, :-1]
        step_distance = torch.norm(step_vector, p=2, dim=-1)
        step_head = torch.atan2(step_vector[..., 1], step_vector[..., 0])
        carried_head = torch.cat([start_head.unsqueeze(1), step_head[:, :-1]], dim=1)
        step_head = torch.where(step_distance > 1e-3, step_head, carried_head)
        return pred_pos, wrap_angle(step_head)

    def _interpolate_endpoint_trajectory(
        self,
        start_pos: torch.Tensor,
        start_head: torch.Tensor,
        endpoint_pos: torch.Tensor,
        endpoint_head: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        method = str(self._get("method", "hermite")).lower()
        if method in {"global_cubic", "natural_cubic", "cubic_spline"}:
            return self._interpolate_global_cubic(
                start_pos, start_head, endpoint_pos, endpoint_head
            )

        min_segment_distance = float(self._get("min_segment_distance_m", 0.25))
        heading_alignment_threshold = math.radians(
            float(self._get("heading_alignment_threshold_deg", 75.0))
        )
        control_pos = torch.cat([start_pos.unsqueeze(1), endpoint_pos], dim=1)
        control_head = torch.cat([start_head.unsqueeze(1), endpoint_head], dim=1)
        tau = (
            torch.arange(
                1,
                self.shift + 1,
                device=start_pos.device,
                dtype=start_pos.dtype,
            )
            / self.shift
        )
        tau_squared = tau * tau
        tau_cubed = tau_squared * tau

        pred_pos_segments = []
        pred_head_segments = []
        for segment_index in range(endpoint_pos.shape[1]):
            p0 = control_pos[:, segment_index]
            p1 = control_pos[:, segment_index + 1]
            h0 = control_head[:, segment_index]
            h1 = control_head[:, segment_index + 1]

            displacement = p1 - p0
            distance = torch.norm(displacement, p=2, dim=-1)
            tau_view = tau.view(1, self.shift, 1)
            linear_pos = (
                p0.unsqueeze(1) * (1.0 - tau_view) + p1.unsqueeze(1) * tau_view
            )

            if method == "linear":
                segment_pos = linear_pos
            else:
                direction0 = torch.stack([h0.cos(), h0.sin()], dim=-1)
                direction1 = torch.stack([h1.cos(), h1.sin()], dim=-1)
                tangent_scale = distance.unsqueeze(-1)
                tangent0 = direction0 * tangent_scale
                tangent1 = direction1 * tangent_scale

                h00 = (
                    2.0 * tau_cubed - 3.0 * tau_squared + 1.0
                ).view(1, self.shift, 1)
                h10 = (tau_cubed - 2.0 * tau_squared + tau).view(
                    1, self.shift, 1
                )
                h01 = (-2.0 * tau_cubed + 3.0 * tau_squared).view(
                    1, self.shift, 1
                )
                h11 = (tau_cubed - tau_squared).view(1, self.shift, 1)
                hermite_pos = (
                    h00 * p0.unsqueeze(1)
                    + h10 * tangent0.unsqueeze(1)
                    + h01 * p1.unsqueeze(1)
                    + h11 * tangent1.unsqueeze(1)
                )

                displacement_head = torch.atan2(
                    displacement[:, 1], displacement[:, 0]
                )
                bad_alignment = (
                    torch.abs(wrap_angle(h0 - displacement_head))
                    > heading_alignment_threshold
                ) | (
                    torch.abs(wrap_angle(h1 - displacement_head))
                    > heading_alignment_threshold
                )
                use_linear = (distance < min_segment_distance) | bad_alignment
                segment_pos = torch.where(
                    use_linear.view(-1, 1, 1), linear_pos, hermite_pos
                )

            segment_pos[:, -1] = p1
            segment_points = torch.cat([p0.unsqueeze(1), segment_pos], dim=1)
            step_vector = segment_points[:, 1:] - segment_points[:, :-1]
            step_distance = torch.norm(step_vector, p=2, dim=-1)
            step_head = torch.atan2(step_vector[..., 1], step_vector[..., 0])
            carried_head = torch.cat([h0.unsqueeze(1), step_head[:, :-1]], dim=1)
            step_head = torch.where(step_distance > 1e-3, step_head, carried_head)
            pred_pos_segments.append(segment_pos)
            pred_head_segments.append(wrap_angle(step_head))

        pred_pos = torch.cat(pred_pos_segments, dim=1)
        if not self._uses_tangent_heading():
            return pred_pos, self._interpolate_endpoint_heading(
                start_head, endpoint_head
            )
        return pred_pos, torch.cat(pred_head_segments, dim=1)

    def reconstruct(
        self,
        raw_traj: torch.Tensor,
        raw_head: torch.Tensor,
        start_pos: torch.Tensor,
        start_head: torch.Tensor,
        endpoint_pos: torch.Tensor,
        endpoint_head: torch.Tensor,
        agent_type: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply TrajTok endpoint reconstruction to one CatK rollout."""
        if not self.is_active:
            return raw_traj, raw_head

        expected_steps = endpoint_pos.shape[1] * self.shift
        if raw_traj.shape[1] != expected_steps or raw_head.shape[1] != expected_steps:
            raise ValueError(
                "Endpoint interpolation shape mismatch: expected "
                f"{expected_steps} future steps, got trajectory "
                f"{raw_traj.shape[1]} and heading {raw_head.shape[1]}."
            )

        interp_traj, interp_head = self._interpolate_endpoint_trajectory(
            start_pos, start_head, endpoint_pos, endpoint_head
        )
        segment_speed = self._endpoint_segment_speed(start_pos, endpoint_pos)

        if bool(self._get("low_speed_reconstruction", False)):
            static_agent_mask = self._static_agent_mask(
                start_pos, endpoint_pos, segment_speed, agent_type
            )
            low_speed_agent_mask = (
                self._low_speed_agent_mask(segment_speed) & ~static_agent_mask
            )
            low_speed_traj, low_speed_head = self._interpolate_low_speed(
                start_pos, start_head, endpoint_pos, endpoint_head
            )
            pred_traj = torch.where(
                low_speed_agent_mask.view(-1, 1, 1), low_speed_traj, raw_traj
            )
            pred_head = torch.where(
                low_speed_agent_mask.view(-1, 1), low_speed_head, raw_head
            )

            interpolation_agent_mask = ~(
                static_agent_mask | low_speed_agent_mask
            )
            pred_traj = torch.where(
                interpolation_agent_mask.view(-1, 1, 1), interp_traj, pred_traj
            )
            pred_head = torch.where(
                interpolation_agent_mask.view(-1, 1), interp_head, pred_head
            )

            static_traj, static_head = self._static_reconstruction(
                start_pos, start_head, raw_traj.shape[1]
            )
            pred_traj = torch.where(
                static_agent_mask.view(-1, 1, 1), static_traj, pred_traj
            )
            pred_head = torch.where(
                static_agent_mask.view(-1, 1), static_head, pred_head
            )
            smoothing_agent_mask = torch.ones_like(low_speed_agent_mask)
        else:
            interpolation_agent_mask = self._moving_agent_mask(segment_speed)
            interpolation_segment_mask = self._moving_segment_mask(segment_speed)
            interpolation_step_mask = (
                interpolation_agent_mask.unsqueeze(1)
                & interpolation_segment_mask
            ).repeat_interleave(self.shift, dim=1)
            pred_traj = torch.where(
                interpolation_step_mask.unsqueeze(-1), interp_traj, raw_traj
            )
            pred_head = torch.where(
                interpolation_step_mask, interp_head, raw_head
            )
            smoothing_agent_mask = interpolation_agent_mask

        return self._smooth_output(pred_traj, pred_head, smoothing_agent_mask)
