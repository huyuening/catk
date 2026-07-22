# Adapted from WOMD-Traffic-Signal-Data-Improvement for CatK vocabulary-only
# reconstruction. Distributed under the PolyForm Noncommercial License 1.0.0;
# see LICENSE.WOMD_TRAJECTORY_RECONSTRUCTION.txt in this directory.

"""Reverse-aware geometric trajectory filtering for WOMD Scenario protos.

Positions are filtered first. For vehicles moving at a reliable speed, the
filtered xy direction then provides a robust heading observation while a
two-candidate continuity model preserves sustained reverse motion. All
velocity fields are derived again from the reconstructed positions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np
from scipy.signal import savgol_filter


class MotionMode(IntEnum):
    UNKNOWN = 0
    STATIONARY = 1
    FORWARD = 2
    REVERSE = 3
    LATERAL = 4


FILTER_STRENGTHS = ("light", "balanced", "strong")


@dataclass(frozen=True)
class TrajectoryFilterConfig:
    processed_object_types: tuple[int, ...] = (1, 2, 3)
    max_gap_frames: int | None = None
    min_observed_frames: int = 2
    position_window: int = 11
    z_window: int = 11
    heading_window: int = 11
    pedestrian_heading_window: int = 21
    pedestrian_heading_polynomial_order: int = 1
    pedestrian_detailed_heading_window: int = 21
    pedestrian_detailed_heading_polynomial_order: int = 2
    pedestrian_detailed_heading_min_observed_ratio: float = 0.80
    pedestrian_detailed_heading_max_gap_frames: int = 2
    pedestrian_detailed_heading_max_second_difference_deg: float = 5.0
    polynomial_order: int = 2
    outlier_sigma: float = 6.0
    outlier_support_ratio: float = 1.5
    position_outlier_floor_m: float = 0.75
    vehicle_position_jump_speed_mps: float = 70.0
    pedestrian_position_jump_speed_mps: float = 15.0
    cyclist_position_jump_speed_mps: float = 35.0
    vehicle_adaptive_jump_speed_floor_mps: float = 15.0
    pedestrian_adaptive_jump_speed_floor_mps: float = 5.0
    cyclist_adaptive_jump_speed_floor_mps: float = 10.0
    adaptive_jump_speed_factor: float = 4.0
    position_jump_distance_margin_m: float = 1.0
    position_excursion_bridge_speed_floor_mps: float = 2.0
    position_excursion_bridge_speed_factor: float = 2.0
    position_excursion_return_distance_ratio: float = 1.0
    position_endpoint_outlier_max_frames: int = 8
    position_endpoint_outlier_max_ratio: float = 0.35
    position_consistent_path_max_skipped_observations: int = 8
    vehicle_closed_excursion_max_frames: int = 8
    vehicle_closed_excursion_path_ratio: float = 2.5
    vehicle_closed_excursion_min_path_excess_m: float = 1.0
    vehicle_closed_excursion_max_rejected_ratio: float = 0.70
    rough_position_jerk_rms_mps3: float = 50.0
    rough_position_window: int = 21
    rough_position_max_correction_m: float = 3.0
    z_outlier_floor_m: float = 0.40
    heading_outlier_floor_deg: float = 45.0
    heading_flip_deg: float = 120.0
    heading_flip_support_deg: float = 45.0
    heading_flip_max_run: int = 3
    max_trusted_xy_correction_m: float = 1.50
    max_trusted_z_correction_m: float = 0.80
    max_trusted_heading_correction_deg: float = 45.0
    stationary_speed_mps: float = 0.20
    direction_speed_mps: float = 0.30
    direction_min_run: int = 3
    motion_heading_object_types: tuple[int, ...] = (1,)
    motion_heading_min_speed_mps: float = 0.50
    motion_heading_min_run: int = 3
    motion_heading_velocity_window: int = 5
    motion_heading_outlier_deg: float = 20.0
    motion_heading_branch_evidence_deg: float = 15.0
    motion_heading_forward_lock_ratio: float = 0.60
    motion_heading_reverse_lock_ratio: float = 0.75
    motion_heading_reverse_min_run: int = 5
    motion_heading_observation_cap_deg: float = 60.0
    motion_heading_transition_weight: float = 25.0
    motion_heading_seed_max_gap_s: float = 1.0
    motion_heading_seed_margin_deg: float = 30.0
    motion_heading_support_max_gap_s: float = 3.0
    endpoint_heading_jump_deg: float = 45.0
    endpoint_heading_max_yaw_rate_radps: float = 1.5
    endpoint_heading_max_curvature_radpm: float = 0.5
    vehicle_endpoint_heading_residual_deg: float = 30.0
    vehicle_endpoint_heading_support_deg: float = 15.0
    vehicle_endpoint_heading_support_frames: int = 5
    vehicle_pi_flip_min_jump_deg: float = 120.0


def config_for_filter_strength(
    strength: str,
    max_gap_frames: int | None = None,
) -> TrajectoryFilterConfig:
    windows = {
        "light": 5,
        "balanced": 7,
        "strong": 11,
    }
    try:
        window = windows[strength]
    except KeyError as exc:
        valid = ", ".join(FILTER_STRENGTHS)
        raise ValueError(f"Unknown filter strength '{strength}'. Choose one of: {valid}") from exc
    if max_gap_frames is not None and max_gap_frames < 0:
        max_gap_frames = None
    return TrajectoryFilterConfig(
        max_gap_frames=max_gap_frames,
        position_window=window,
        z_window=window,
        heading_window=window,
    )


@dataclass
class KinematicFeatures:
    velocity_x: np.ndarray
    velocity_y: np.ndarray
    velocity_z: np.ndarray
    planar_speed: np.ndarray
    linear_speed: np.ndarray
    planar_acceleration: np.ndarray
    linear_acceleration: np.ndarray
    linear_jerk: np.ndarray
    angular_speed: np.ndarray
    angular_acceleration: np.ndarray
    angular_jerk: np.ndarray
    longitudinal_speed: np.ndarray
    lateral_speed: np.ndarray
    speed_validity: np.ndarray
    acceleration_validity: np.ndarray
    jerk_validity: np.ndarray
    motion_mode: np.ndarray


@dataclass
class TrackFilterResult:
    reconstructed: bool = False
    insufficient_support: bool = False
    filtered_frames: int = 0
    filled_frames: int = 0
    position_outliers: int = 0
    kinematic_position_outliers: int = 0
    relaxed_position_outlier_segments: int = 0
    z_outliers: int = 0
    heading_outliers: int = 0
    reverse_frames: int = 0
    rejected_segments: int = 0
    best_effort_segments: int = 0
    fallback_filled_frames: int = 0


@dataclass
class ArrayTrajectoryReconstruction:
    """Reconstructed array trajectory and the support accepted by the filter."""

    positions: np.ndarray
    heading: np.ndarray
    valid: np.ndarray
    result: TrackFilterResult


@dataclass
class ReconstructionStats:
    total_tracks: int = 0
    reconstructed_tracks: int = 0
    skipped_tracks: int = 0
    insufficient_support_tracks: int = 0
    filtered_frames: int = 0
    filled_frames: int = 0
    position_outliers: int = 0
    kinematic_position_outliers: int = 0
    relaxed_position_outlier_segments: int = 0
    z_outliers: int = 0
    heading_outliers: int = 0
    reverse_frames: int = 0
    rejected_segments: int = 0
    best_effort_segments: int = 0
    fallback_filled_frames: int = 0


@dataclass
class _SegmentCandidate:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    heading: np.ndarray
    position_outliers: np.ndarray
    kinematic_position_outliers: np.ndarray
    z_outliers: np.ndarray
    heading_outliers: np.ndarray
    position_outliers_relaxed: bool = False


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def angle_diff(angle, reference):
    return wrap_angle(np.asarray(angle) - np.asarray(reference))


def _timestamps_for_count(timestamps: Iterable[float], count: int) -> np.ndarray:
    time = np.asarray(list(timestamps)[:count], dtype=float)
    if len(time) != count:
        return np.arange(count, dtype=float) * 0.1

    if count > 1:
        finite_diff = np.diff(time)
        positive_diff = finite_diff[np.isfinite(finite_diff) & (finite_diff > 0)]
        if positive_diff.size and float(np.median(positive_diff)) > 10.0:
            time = time / 1e6

    if not np.all(np.isfinite(time)) or (count > 1 and np.any(np.diff(time) <= 0)):
        return np.arange(count, dtype=float) * 0.1
    return time - time[0]


def _true_runs(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(mask)


def _fill_short_internal_gaps(
    valid: np.ndarray,
    max_gap_frames: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(valid, dtype=bool)
    eligible = valid.copy()
    filled = np.zeros_like(valid)
    if max_gap_frames == 0:
        return eligible, filled

    index = 0
    while index < len(valid):
        if valid[index]:
            index += 1
            continue
        start = index
        while index < len(valid) and not valid[index]:
            index += 1
        end = index
        gap_is_allowed = max_gap_frames is None or end - start <= max_gap_frames
        if start > 0 and end < len(valid) and gap_is_allowed:
            eligible[start:end] = True
            filled[start:end] = True
    return eligible, filled


def _effective_window(length: int, requested: int, polynomial_order: int) -> int:
    if length < 3 or requested < 3:
        return 1
    window = min(requested, length if length % 2 else length - 1)
    if window % 2 == 0:
        window -= 1
    if window <= polynomial_order:
        return 1
    return window


def _smooth_scalar(
    values: np.ndarray,
    window: int,
    polynomial_order: int,
    coordinate: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if coordinate is not None and len(values) >= 3:
        coordinate = np.asarray(coordinate, dtype=float)
        if len(coordinate) == len(values) and np.all(np.isfinite(coordinate)):
            step = np.diff(coordinate)
            positive = step[step > 0.0]
            uniform = len(positive) == len(step) and np.allclose(
                step,
                np.median(positive),
                rtol=1e-5,
                atol=1e-8,
            )
            if len(positive) and not uniform:
                nominal_step = float(np.median(positive))
                sample_count = int(
                    np.round((coordinate[-1] - coordinate[0]) / nominal_step)
                ) + 1
                sample_count = min(max(sample_count, len(values)), 4 * len(values))
                regular_coordinate = np.linspace(
                    coordinate[0],
                    coordinate[-1],
                    sample_count,
                )
                regular_values = np.interp(
                    regular_coordinate,
                    coordinate,
                    values,
                )
                regular_smoothed = _smooth_scalar(
                    regular_values,
                    window,
                    polynomial_order,
                )
                return np.interp(
                    coordinate,
                    regular_coordinate,
                    regular_smoothed,
                )
    effective_window = _effective_window(len(values), window, polynomial_order)
    if effective_window == 1:
        return values.copy()
    return savgol_filter(
        values,
        window_length=effective_window,
        polyorder=min(polynomial_order, effective_window - 1),
        mode="interp",
    )


def _interpolate_scalar(
    values: np.ndarray,
    anchor_mask: np.ndarray,
    *,
    coordinate: np.ndarray | None = None,
    extrapolate_endpoints: bool = False,
) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    anchor_mask = np.asarray(anchor_mask, dtype=bool) & np.isfinite(values)
    anchor_index = np.flatnonzero(anchor_mask)
    if len(anchor_index) == 0:
        return None
    if len(anchor_index) == 1:
        return np.full_like(values, values[anchor_index[0]], dtype=float)
    if coordinate is None:
        coordinate = np.arange(len(values), dtype=float)
    else:
        coordinate = np.asarray(coordinate, dtype=float)
        if len(coordinate) != len(values) or not np.all(np.isfinite(coordinate)):
            coordinate = np.arange(len(values), dtype=float)

    anchor_coordinate = coordinate[anchor_index]
    output = np.interp(coordinate, anchor_coordinate, values[anchor_index])
    if not extrapolate_endpoints:
        return output

    def robust_endpoint_slope(indices: np.ndarray) -> float:
        delta_coordinate = np.diff(coordinate[indices])
        delta_value = np.diff(values[indices])
        usable = (
            np.isfinite(delta_coordinate)
            & (delta_coordinate > 0.0)
            & np.isfinite(delta_value)
        )
        if not np.any(usable):
            return 0.0
        return float(np.median(delta_value[usable] / delta_coordinate[usable]))

    support = min(6, len(anchor_index))
    first = int(anchor_index[0])
    if first > 0:
        slope = robust_endpoint_slope(anchor_index[:support])
        output[:first] = values[first] + slope * (
            coordinate[:first] - coordinate[first]
        )
    last = int(anchor_index[-1])
    if last + 1 < len(values):
        slope = robust_endpoint_slope(anchor_index[-support:])
        output[last + 1 :] = values[last] + slope * (
            coordinate[last + 1 :] - coordinate[last]
        )
    return output


def _smoothing_windows(length: int, requested: int, polynomial_order: int):
    window = _effective_window(length, requested, polynomial_order)
    while window > 1:
        yield window
        window -= 2
        if window <= polynomial_order:
            break
    yield 1


def _smooth_xy_with_limit(
    x_filled: np.ndarray,
    y_filled: np.ndarray,
    x_observed: np.ndarray,
    y_observed: np.ndarray,
    trusted: np.ndarray,
    coordinate: np.ndarray,
    config: TrajectoryFilterConfig,
    requested_window: int | None = None,
    maximum_correction: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    origin_x = float(x_filled[0])
    origin_y = float(y_filled[0])
    for window in _smoothing_windows(
        len(x_filled),
        requested_window or config.position_window,
        config.polynomial_order,
    ):
        x_candidate = origin_x + _smooth_scalar(
            x_filled - origin_x,
            window,
            config.polynomial_order,
            coordinate,
        )
        y_candidate = origin_y + _smooth_scalar(
            y_filled - origin_y,
            window,
            config.polynomial_order,
            coordinate,
        )
        correction = np.hypot(
            x_candidate - x_observed,
            y_candidate - y_observed,
        )[trusted]
        correction_limit = (
            config.max_trusted_xy_correction_m
            if maximum_correction is None
            else maximum_correction
        )
        if _percentile(correction) <= correction_limit:
            return x_candidate, y_candidate
    return x_filled.copy(), y_filled.copy()


def _dense_linear_jerk_rms(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    time: np.ndarray,
) -> float:
    if len(x) < 7:
        return 0.0
    position = np.column_stack((x, y, z))
    velocity_duration = time[2:] - time[:-2]
    velocity = (position[2:] - position[:-2]) / velocity_duration[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    acceleration_duration = time[3:-1] - time[1:-3]
    acceleration = (speed[2:] - speed[:-2]) / acceleration_duration
    jerk_duration = time[4:-2] - time[2:-4]
    jerk = (acceleration[2:] - acceleration[:-2]) / jerk_duration
    finite = jerk[np.isfinite(jerk)]
    return float(np.sqrt(np.mean(finite**2))) if len(finite) else 0.0


def _smooth_scalar_with_limit(
    filled: np.ndarray,
    observed_values: np.ndarray,
    trusted: np.ndarray,
    requested_window: int,
    polynomial_order: int,
    maximum_correction: float,
    coordinate: np.ndarray,
) -> np.ndarray:
    for window in _smoothing_windows(
        len(filled),
        requested_window,
        polynomial_order,
    ):
        candidate = _smooth_scalar(
            filled,
            window,
            polynomial_order,
            coordinate,
        )
        correction = np.abs(candidate - observed_values)[trusted]
        if _percentile(correction) <= maximum_correction:
            return candidate
    return filled.copy()


def _interpolate_heading(values: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    anchor_mask = np.asarray(anchor_mask, dtype=bool) & np.isfinite(values)
    anchor_index = np.flatnonzero(anchor_mask)
    if len(anchor_index) == 0:
        return None
    unwrapped = np.unwrap(values[anchor_index])
    if len(anchor_index) == 1:
        interpolated = np.full_like(values, unwrapped[0], dtype=float)
    else:
        interpolated = np.interp(np.arange(len(values)), anchor_index, unwrapped)
    return interpolated


def _interpolate_pi_ambiguous_heading(
    values: np.ndarray,
    anchor_mask: np.ndarray,
) -> np.ndarray | None:
    """Interpolate pedestrian heading after resolving framewise pi ambiguity."""
    values = np.asarray(values, dtype=float)
    anchor_mask = np.asarray(anchor_mask, dtype=bool) & np.isfinite(values)
    anchor_index = np.flatnonzero(anchor_mask)
    if len(anchor_index) == 0:
        return None

    aligned = np.empty(len(anchor_index), dtype=float)
    aligned[0] = values[anchor_index[0]]
    for position in range(1, len(anchor_index)):
        value = values[anchor_index[position]]
        pi_shift = np.round((aligned[position - 1] - value) / np.pi)
        aligned[position] = value + pi_shift * np.pi
    if len(anchor_index) == 1:
        return np.full_like(values, aligned[0], dtype=float)
    return np.interp(np.arange(len(values)), anchor_index, aligned)


def _supports_detailed_pedestrian_heading(
    heading: np.ndarray,
    anchor_mask: np.ndarray,
    config: TrajectoryFilterConfig,
) -> bool:
    anchor_index = np.flatnonzero(anchor_mask & np.isfinite(heading))
    if len(anchor_index) < 5:
        return False
    start = int(anchor_index[0])
    end = int(anchor_index[-1]) + 1
    span_anchor = np.asarray(anchor_mask[start:end], dtype=bool)
    if (
        float(np.mean(span_anchor))
        < config.pedestrian_detailed_heading_min_observed_ratio
    ):
        return False
    gaps = np.diff(np.flatnonzero(span_anchor)) - 1
    if (
        len(gaps)
        and int(np.max(gaps))
        > config.pedestrian_detailed_heading_max_gap_frames
    ):
        return False
    unwrapped = _interpolate_pi_ambiguous_heading(
        np.asarray(heading[start:end], dtype=float),
        span_anchor,
    )
    if unwrapped is None or len(unwrapped) < 3:
        return False
    second_difference_rms = np.sqrt(np.mean(np.diff(unwrapped, n=2) ** 2))
    return second_difference_rms <= np.deg2rad(
        config.pedestrian_detailed_heading_max_second_difference_deg
    )


def _interpolate_vehicle_heading(
    values: np.ndarray,
    anchor_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    preferred_anchor: int,
    config: TrajectoryFilterConfig,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Resolve only kinematically infeasible vehicle pi-branch transitions."""
    values = np.asarray(values, dtype=float)
    anchor_mask = np.asarray(anchor_mask, dtype=bool) & np.isfinite(values)
    anchor_index = np.flatnonzero(anchor_mask)
    corrected = np.zeros(len(values), dtype=bool)
    if len(anchor_index) == 0:
        return None, corrected

    anchor_position = int(np.argmin(np.abs(anchor_index - preferred_anchor)))
    aligned = np.empty(len(anchor_index), dtype=float)
    aligned[anchor_position] = values[anchor_index[anchor_position]]
    minimum_jump = np.deg2rad(config.vehicle_pi_flip_min_jump_deg)

    def align_to_reference(index: int, reference_index: int, reference: float) -> float:
        value = values[index]
        direct = value + np.round((reference - value) / (2.0 * np.pi)) * 2.0 * np.pi
        flipped_value = value + np.pi
        flipped = (
            flipped_value
            + np.round((reference - flipped_value) / (2.0 * np.pi)) * 2.0 * np.pi
        )
        direct_change = abs(float(direct - reference))
        flipped_change = abs(float(flipped - reference))
        elapsed = abs(float(time[index] - time[reference_index]))
        distance = float(np.hypot(x[index] - x[reference_index], y[index] - y[reference_index]))
        infeasible = (
            direct_change >= minimum_jump
            and elapsed > 0.0
            and direct_change / elapsed > config.endpoint_heading_max_yaw_rate_radps
            and direct_change / max(distance, 1e-3)
            > config.endpoint_heading_max_curvature_radpm
        )
        if infeasible and flipped_change < direct_change:
            corrected[index] = True
            return float(flipped)
        return float(direct)

    for position in range(anchor_position + 1, len(anchor_index)):
        index = int(anchor_index[position])
        reference_index = int(anchor_index[position - 1])
        aligned[position] = align_to_reference(
            index,
            reference_index,
            aligned[position - 1],
        )
    for position in range(anchor_position - 1, -1, -1):
        index = int(anchor_index[position])
        reference_index = int(anchor_index[position + 1])
        aligned[position] = align_to_reference(
            index,
            reference_index,
            aligned[position + 1],
        )

    if len(anchor_index) == 1:
        return np.full_like(values, aligned[0], dtype=float), corrected
    interpolated = np.interp(np.arange(len(values)), anchor_index, aligned)
    return interpolated, corrected


def _preferred_vehicle_heading_anchor(
    heading_anchor: np.ndarray,
    position_observed: np.ndarray,
    speed: np.ndarray,
    time: np.ndarray,
    config: TrajectoryFilterConfig,
) -> int:
    """Prefer a pre-motion orientation seed, otherwise the strongest motion frame."""
    heading_index = np.flatnonzero(heading_anchor & np.isfinite(speed))
    if len(heading_index) == 0:
        return 0

    position_supported = _keep_sustained_runs(
        np.asarray(position_observed, dtype=bool),
        config.motion_heading_min_run,
    )
    moving = _keep_sustained_runs(
        position_supported
        & np.isfinite(speed)
        & (speed >= config.motion_heading_min_speed_mps),
        config.motion_heading_min_run,
    )
    moving_index = np.flatnonzero(moving)
    if len(moving_index):
        first_moving = int(moving_index[0])
        observed_run_start = first_moving
        while observed_run_start > 0 and position_observed[observed_run_start - 1]:
            observed_run_start -= 1
        before_observed_run = heading_index[heading_index < observed_run_start]
        if len(before_observed_run):
            seed_index = int(before_observed_run[-1])
            if time[first_moving] - time[seed_index] <= config.motion_heading_seed_max_gap_s:
                return seed_index
    return int(heading_index[np.argmax(speed[heading_index])])


def smooth_heading_series(
    heading: np.ndarray,
    valid: np.ndarray,
    window: int = 5,
    polynomial_order: int = 2,
) -> np.ndarray:
    """Interpolate and smooth a circular heading series without using velocity direction."""
    unwrapped = _interpolate_heading(heading, valid)
    if unwrapped is None:
        return np.full_like(np.asarray(heading, dtype=float), np.nan)
    return wrap_angle(_smooth_scalar(unwrapped, window, polynomial_order))


def _remove_short_heading_flips(
    heading: np.ndarray,
    valid: np.ndarray,
    spike_deg: float,
    support_deg: float,
    max_run: int,
) -> np.ndarray:
    rejected = np.zeros_like(valid, dtype=bool)
    finite_index = np.flatnonzero(valid & np.isfinite(heading))
    if max_run <= 0 or len(finite_index) < 3:
        return rejected

    finite_heading = heading[finite_index]
    jumps = np.rad2deg(np.abs(angle_diff(finite_heading[1:], finite_heading[:-1])))
    large_jumps = np.flatnonzero(jumps > spike_deg)
    for left_jump, right_jump in zip(large_jumps[:-1], large_jumps[1:]):
        run_length = right_jump - left_jump
        if run_length < 1 or run_length > max_run:
            continue
        before = finite_heading[left_jump]
        after = finite_heading[right_jump + 1]
        if np.rad2deg(abs(float(angle_diff(before, after)))) > support_deg:
            continue
        rejected[finite_index[left_jump + 1 : right_jump + 1]] = True
    return rejected


def _interpolation_residual_outliers(
    values: np.ndarray,
    valid: np.ndarray,
    time: np.ndarray,
    minimum_residual: float,
    sigma: float,
    support_ratio: float,
    angular: bool = False,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim == 1:
        finite = np.isfinite(values)
    else:
        finite = np.all(np.isfinite(values), axis=1)
    valid_index = np.flatnonzero(valid & finite)
    rejected = np.zeros(len(valid), dtype=bool)
    if len(valid_index) < 3:
        return rejected

    residual = np.full(len(valid), np.nan, dtype=float)
    support = np.full(len(valid), np.nan, dtype=float)
    for position in range(1, len(valid_index) - 1):
        left = valid_index[position - 1]
        center = valid_index[position]
        right = valid_index[position + 1]
        duration = time[right] - time[left]
        if duration <= 0:
            continue
        ratio = (time[center] - time[left]) / duration
        if angular:
            span = float(angle_diff(values[right], values[left]))
            prediction = float(wrap_angle(values[left] + ratio * span))
            residual[center] = abs(float(angle_diff(values[center], prediction)))
            support[center] = abs(span)
        else:
            prediction = values[left] + ratio * (values[right] - values[left])
            residual[center] = float(np.linalg.norm(values[center] - prediction))
            support[center] = float(np.linalg.norm(values[right] - values[left]))

    finite_residual = residual[np.isfinite(residual)]
    if len(finite_residual) == 0:
        return rejected
    median = float(np.median(finite_residual))
    mad = float(np.median(np.abs(finite_residual - median)))
    threshold = max(minimum_residual, median + sigma * 1.4826 * mad)
    candidate = np.isfinite(residual) & (residual > threshold)
    isolated = residual > support_ratio * np.nan_to_num(support, nan=np.inf)
    rejected[candidate & isolated] = True
    return rejected


def _maximum_plausible_position_speed_mps(
    object_type: int,
    config: TrajectoryFilterConfig,
) -> float:
    if object_type == 2:
        return config.pedestrian_position_jump_speed_mps
    if object_type == 3:
        return config.cyclist_position_jump_speed_mps
    return config.vehicle_position_jump_speed_mps


def _adaptive_jump_speed_floor_mps(
    object_type: int,
    config: TrajectoryFilterConfig,
) -> float:
    if object_type == 2:
        return config.pedestrian_adaptive_jump_speed_floor_mps
    if object_type == 3:
        return config.cyclist_adaptive_jump_speed_floor_mps
    return config.vehicle_adaptive_jump_speed_floor_mps


def _dominant_consistent_position_path_outliers(
    xy: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    object_type: int,
    config: TrajectoryFilterConfig,
) -> np.ndarray:
    """Keep the largest time-ordered path that does not contain a teleport.

    A single object id can occasionally alternate between two coordinate
    branches or switch to another branch for the rest of the scenario. Pairing
    jump edges cannot resolve those cases reliably. This bounded longest-path
    pass selects the best-supported physically connected observation sequence.
    It runs only after a consecutive jump has been detected.
    """
    xy = np.asarray(xy, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    time = np.asarray(time, dtype=float)
    finite = np.all(np.isfinite(xy), axis=1) & np.isfinite(time)
    observed_index = np.flatnonzero(observed & finite)
    rejected = np.zeros(len(observed), dtype=bool)
    if len(observed_index) < 3:
        return rejected

    observed_xy = xy[observed_index]
    observed_time = time[observed_index]
    consecutive_duration = np.diff(observed_time)
    if np.any(consecutive_duration <= 0.0):
        return rejected
    consecutive_distance = np.linalg.norm(np.diff(observed_xy, axis=0), axis=1)
    consecutive_speed = consecutive_distance / consecutive_duration

    maximum_speed = _maximum_plausible_position_speed_mps(object_type, config)
    typical_speed = float(np.median(consecutive_speed))
    adaptive_speed_limit = max(
        _adaptive_jump_speed_floor_mps(object_type, config),
        typical_speed * config.adaptive_jump_speed_factor,
    )

    def plausible_edges(
        distance: np.ndarray,
        duration: np.ndarray,
    ) -> np.ndarray:
        speed = distance / duration
        absolute_limit = (
            config.position_jump_distance_margin_m + maximum_speed * duration
        )
        return (distance <= absolute_limit) & (
            (distance <= config.position_jump_distance_margin_m)
            | (speed <= adaptive_speed_limit)
        )

    if np.all(
        plausible_edges(consecutive_distance, consecutive_duration)
    ):
        return rejected

    count = len(observed_index)
    support_count = np.ones(count, dtype=np.int32)
    covered_duration = np.zeros(count, dtype=float)
    predecessor = np.full(count, -1, dtype=np.int32)
    maximum_skipped = max(
        0,
        int(config.position_consistent_path_max_skipped_observations),
    )

    for right in range(count):
        if right == 0:
            continue
        left = np.arange(right, dtype=np.int32)
        duration = observed_time[right] - observed_time[:right]
        distance = np.linalg.norm(
            observed_xy[right] - observed_xy[:right],
            axis=1,
        )
        skipped = right - left - 1
        usable = (
            (duration > 0.0)
            & (skipped <= maximum_skipped)
            & plausible_edges(distance, duration)
        )
        for candidate in left[usable]:
            candidate = int(candidate)
            candidate_support = int(support_count[candidate]) + 1
            candidate_duration = (
                covered_duration[candidate]
                + observed_time[right]
                - observed_time[candidate]
            )
            if (
                candidate_support > support_count[right]
                or (
                    candidate_support == support_count[right]
                    and candidate_duration > covered_duration[right]
                )
            ):
                support_count[right] = candidate_support
                covered_duration[right] = candidate_duration
                predecessor[right] = candidate

    endpoint = max(
        range(count),
        key=lambda index: (
            int(support_count[index]),
            float(covered_duration[index]),
            index,
        ),
    )
    kept_positions = []
    while endpoint >= 0:
        kept_positions.append(endpoint)
        endpoint = int(predecessor[endpoint])

    rejected[observed_index] = True
    rejected[observed_index[np.asarray(kept_positions, dtype=int)]] = False
    return rejected


def _kinematic_excursion_position_outliers(
    xy: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    object_type: int,
    config: TrajectoryFilterConfig,
) -> np.ndarray:
    """Reject bracketed excursions and short endpoints around impossible jumps."""
    xy = np.asarray(xy, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    time = np.asarray(time, dtype=float)
    finite = np.all(np.isfinite(xy), axis=1) & np.isfinite(time)
    observed_index = np.flatnonzero(observed & finite)
    rejected = np.zeros(len(observed), dtype=bool)
    if len(observed_index) < 3:
        return rejected

    left_index = observed_index[:-1]
    right_index = observed_index[1:]
    duration = time[right_index] - time[left_index]
    displacement = np.linalg.norm(xy[right_index] - xy[left_index], axis=1)
    maximum_speed = _maximum_plausible_position_speed_mps(object_type, config)
    usable = np.isfinite(duration) & (duration > 0) & np.isfinite(displacement)
    speed = np.full_like(displacement, np.nan, dtype=float)
    speed[usable] = displacement[usable] / duration[usable]
    typical_observed_speed = (
        float(np.median(speed[usable])) if np.any(usable) else 0.0
    )
    adaptive_speed_limit = max(
        _adaptive_jump_speed_floor_mps(object_type, config),
        typical_observed_speed * config.adaptive_jump_speed_factor,
    )
    jump_limit = config.position_jump_distance_margin_m + maximum_speed * duration
    absolute_jump = usable & (displacement > jump_limit)
    adaptive_jump = (
        usable
        & (displacement > config.position_jump_distance_margin_m)
        & (speed > adaptive_speed_limit)
    )
    jump_mask = absolute_jump | adaptive_jump
    jump_edges = np.flatnonzero(jump_mask)
    if len(jump_edges) == 0:
        return rejected

    ordinary_speed = speed[usable & ~jump_mask]
    typical_speed = float(np.median(ordinary_speed)) if len(ordinary_speed) else 0.0
    bridge_speed = min(
        maximum_speed,
        max(
            config.position_excursion_bridge_speed_floor_mps,
            _adaptive_jump_speed_floor_mps(object_type, config),
            typical_speed * config.position_excursion_bridge_speed_factor,
        ),
    )

    bracketed_jump_edges = np.zeros(len(displacement), dtype=bool)
    for left_edge, right_edge in zip(jump_edges[:-1], jump_edges[1:]):
        outside_left = observed_index[left_edge]
        outside_right = observed_index[right_edge + 1]
        outside_duration = time[outside_right] - time[outside_left]
        if not np.isfinite(outside_duration) or outside_duration <= 0:
            continue
        outside_displacement = float(
            np.linalg.norm(xy[outside_right] - xy[outside_left])
        )
        bridge_limit = (
            config.position_jump_distance_margin_m
            + bridge_speed * outside_duration
        )
        return_limit = config.position_jump_distance_margin_m + (
            config.position_excursion_return_distance_ratio
            * (displacement[left_edge] + displacement[right_edge])
        )
        if outside_displacement > min(bridge_limit, return_limit):
            continue

        excursion_index = observed_index[left_edge + 1 : right_edge + 1]
        rejected[excursion_index] = True
        bracketed_jump_edges[left_edge] = True
        bracketed_jump_edges[right_edge] = True

    def is_supported_endpoint_branch(branch_count: int) -> bool:
        return (
            branch_count <= config.position_endpoint_outlier_max_frames
            and branch_count / len(observed_index)
            <= config.position_endpoint_outlier_max_ratio
        )

    first_jump = int(jump_edges[0])
    prefix_count = first_jump + 1
    first_jump_isolated = len(jump_edges) == 1 or jump_edges[1] > first_jump + 1
    if (
        not bracketed_jump_edges[first_jump]
        and is_supported_endpoint_branch(prefix_count)
        and first_jump_isolated
    ):
        rejected[observed_index[:prefix_count]] = True

    last_jump = int(jump_edges[-1])
    suffix_start = last_jump + 1
    suffix_count = len(observed_index) - suffix_start
    last_jump_isolated = len(jump_edges) == 1 or jump_edges[-2] < last_jump - 1
    if (
        not bracketed_jump_edges[last_jump]
        and is_supported_endpoint_branch(suffix_count)
        and last_jump_isolated
    ):
        rejected[observed_index[suffix_start:]] = True
    return rejected


def _short_closed_position_excursion_outliers(
    xy: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    object_type: int,
    config: TrajectoryFilterConfig,
) -> np.ndarray:
    """Reject short vehicle detours that return to the surrounding path."""
    xy = np.asarray(xy, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    time = np.asarray(time, dtype=float)
    rejected = np.zeros(len(observed), dtype=bool)
    if object_type != 1 or config.vehicle_closed_excursion_max_frames <= 0:
        return rejected

    finite = np.all(np.isfinite(xy), axis=1) & np.isfinite(time)
    observed_index = np.flatnonzero(observed & finite)
    if len(observed_index) < 3:
        return rejected

    observed_xy = xy[observed_index]
    observed_time = time[observed_index]
    if np.any(np.diff(observed_time) <= 0.0):
        return rejected
    edge_length = np.linalg.norm(np.diff(observed_xy, axis=0), axis=1)
    cumulative_path_length = np.r_[0.0, np.cumsum(edge_length)]

    for left_position in range(len(observed_index) - 2):
        maximum_right = min(
            len(observed_index),
            left_position + config.vehicle_closed_excursion_max_frames + 2,
        )
        for right_position in range(left_position + 2, maximum_right):
            path_length = float(
                cumulative_path_length[right_position]
                - cumulative_path_length[left_position]
            )
            direct_distance = float(
                np.linalg.norm(
                    observed_xy[right_position] - observed_xy[left_position]
                )
            )
            if (
                path_length - direct_distance
                <= config.vehicle_closed_excursion_min_path_excess_m
                or path_length
                <= config.vehicle_closed_excursion_path_ratio
                * max(direct_distance, 1e-6)
            ):
                continue

            elapsed = float(
                observed_time[right_position] - observed_time[left_position]
            )
            interior_slice = slice(left_position + 1, right_position)
            ratio = (
                observed_time[interior_slice] - observed_time[left_position]
            ) / elapsed
            chord = observed_xy[left_position] + ratio[:, None] * (
                observed_xy[right_position] - observed_xy[left_position]
            )
            chord_residual = np.linalg.norm(
                observed_xy[interior_slice] - chord,
                axis=1,
            )
            if (
                len(chord_residual) == 0
                or float(np.max(chord_residual))
                <= config.position_outlier_floor_m
            ):
                continue
            rejected[observed_index[interior_slice]] = True
    return rejected


def _isolated_endpoint_heading_outliers(
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    config: TrajectoryFilterConfig,
) -> np.ndarray:
    """Reject endpoint headings that imply an infeasible turn across a gap."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    heading = np.asarray(heading, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    finite = (
        observed
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(heading)
        & np.isfinite(time)
    )
    valid_index = np.flatnonzero(finite)
    rejected = np.zeros(len(observed), dtype=bool)
    if len(valid_index) < 2:
        return rejected

    endpoint_pairs = (
        (int(valid_index[0]), int(valid_index[1]), int(valid_index[0])),
        (int(valid_index[-2]), int(valid_index[-1]), int(valid_index[-1])),
    )
    minimum_jump = np.deg2rad(config.endpoint_heading_jump_deg)
    for left, right, endpoint in endpoint_pairs:
        if right - left <= 1:
            continue
        elapsed = float(time[right] - time[left])
        if elapsed <= 0.0:
            continue
        heading_change = abs(float(angle_diff(heading[right], heading[left])))
        if heading_change <= minimum_jump:
            continue
        distance = float(np.hypot(x[right] - x[left], y[right] - y[left]))
        yaw_rate = heading_change / elapsed
        curvature = heading_change / max(distance, 1e-3)
        if (
            yaw_rate > config.endpoint_heading_max_yaw_rate_radps
            and curvature > config.endpoint_heading_max_curvature_radpm
        ):
            rejected[endpoint] = True
    return rejected


def _vehicle_endpoint_heading_outliers(
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    config: TrajectoryFilterConfig,
) -> np.ndarray:
    """Reject an infeasible endpoint heading using stable inward support."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    heading = np.asarray(heading, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    finite = (
        observed
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(heading)
        & np.isfinite(time)
    )
    valid_index = np.flatnonzero(finite)
    rejected = np.zeros(len(observed), dtype=bool)
    support_count = config.vehicle_endpoint_heading_support_frames
    if len(valid_index) < support_count + 2:
        return rejected

    residual_threshold = np.deg2rad(
        config.vehicle_endpoint_heading_residual_deg
    )
    support_threshold = np.deg2rad(config.vehicle_endpoint_heading_support_deg)
    for ordered_index in (valid_index, valid_index[::-1]):
        endpoint = int(ordered_index[0])
        neighbor = int(ordered_index[1])
        support_index = ordered_index[2 : support_count + 2]
        support_heading = heading[support_index]
        support_unwrapped = support_heading[0] + angle_diff(
            support_heading,
            support_heading[0],
        )
        support_reference = float(np.median(support_unwrapped))
        support_spread = float(
            np.max(np.abs(angle_diff(support_heading, support_reference)))
        )
        endpoint_residual = abs(
            float(angle_diff(heading[endpoint], support_reference))
        )
        if (
            support_spread > support_threshold
            or endpoint_residual <= residual_threshold
        ):
            continue

        elapsed = abs(float(time[neighbor] - time[endpoint]))
        if elapsed <= 0.0:
            continue
        neighbor_change = abs(
            float(angle_diff(heading[neighbor], heading[endpoint]))
        )
        distance = float(
            np.hypot(x[neighbor] - x[endpoint], y[neighbor] - y[endpoint])
        )
        yaw_rate = neighbor_change / elapsed
        curvature = neighbor_change / max(distance, 1e-3)
        if (
            yaw_rate > config.endpoint_heading_max_yaw_rate_radps
            and curvature > config.endpoint_heading_max_curvature_radpm
        ):
            rejected[endpoint] = True
    return rejected


def _percentile(values: np.ndarray, q: float = 95.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, q)) if len(finite) else 0.0


def _motion_body_heading_observation(
    x: np.ndarray,
    y: np.ndarray,
    raw_heading: np.ndarray,
    raw_heading_valid: np.ndarray,
    position_observed: np.ndarray,
    time: np.ndarray,
    object_type: int,
    config: TrajectoryFilterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Select motion heading or motion heading + pi with temporal continuity."""
    count = len(x)
    selected = np.full(count, np.nan, dtype=float)
    reliable = np.zeros(count, dtype=bool)
    position_observed = np.asarray(position_observed, dtype=bool)
    if object_type not in config.motion_heading_object_types or count < 2:
        return selected, reliable

    velocity_x = _smooth_scalar(
        _gradient(x, time),
        config.motion_heading_velocity_window,
        config.polynomial_order,
        time,
    )
    velocity_y = _smooth_scalar(
        _gradient(y, time),
        config.motion_heading_velocity_window,
        config.polynomial_order,
        time,
    )
    speed = np.hypot(velocity_x, velocity_y)
    motion_heading = np.arctan2(velocity_y, velocity_x)
    position_supported = _keep_sustained_runs(
        np.asarray(position_observed, dtype=bool),
        config.motion_heading_min_run,
    )
    reliable = (
        np.isfinite(speed)
        & np.isfinite(motion_heading)
        & (speed >= config.motion_heading_min_speed_mps)
        & position_supported
    )
    reliable = _keep_sustained_runs(reliable, config.motion_heading_min_run)
    moving_index = np.flatnonzero(reliable)
    if len(moving_index) == 0:
        return selected, reliable

    candidates = np.column_stack(
        (
            motion_heading[moving_index],
            wrap_angle(motion_heading[moving_index] + np.pi),
        )
    )
    observation_cap = np.deg2rad(config.motion_heading_observation_cap_deg)
    observation_cost = np.zeros_like(candidates)
    candidate_residual = np.full_like(candidates, np.inf)
    for row, frame_index in enumerate(moving_index):
        if not raw_heading_valid[frame_index]:
            continue
        residual = np.abs(angle_diff(candidates[row], raw_heading[frame_index]))
        candidate_residual[row] = residual
        observation_cost[row] = np.minimum(residual, observation_cap) ** 2

    positive_step = np.diff(time)
    positive_step = positive_step[np.isfinite(positive_step) & (positive_step > 0)]
    nominal_step = float(np.median(positive_step)) if len(positive_step) else 0.1
    first_moving = int(moving_index[0])
    heading_anchor_index = np.flatnonzero(raw_heading_valid)
    seed_index = None
    observed_run_start = first_moving
    while observed_run_start > 0 and position_observed[observed_run_start - 1]:
        observed_run_start -= 1
    before_observed_run = heading_anchor_index[
        heading_anchor_index < observed_run_start
    ]
    if observed_run_start > 0 and len(before_observed_run):
        candidate_index = int(before_observed_run[-1])
        if time[first_moving] - time[candidate_index] <= config.motion_heading_seed_max_gap_s:
            seed_index = candidate_index
    if seed_index is None:
        preceding_anchor = heading_anchor_index[heading_anchor_index <= first_moving]
        following_anchor = heading_anchor_index[heading_anchor_index > first_moving]
        if len(preceding_anchor):
            candidate_index = int(preceding_anchor[-1])
            if time[first_moving] - time[candidate_index] <= config.motion_heading_seed_max_gap_s:
                seed_index = candidate_index
        elif len(following_anchor):
            candidate_index = int(following_anchor[0])
            if time[candidate_index] - time[first_moving] <= config.motion_heading_seed_max_gap_s:
                seed_index = candidate_index
    seed_state = None
    seed_margin = 0.0
    if seed_index is not None:
        seed_residual = np.abs(angle_diff(candidates[0], raw_heading[seed_index]))
        seed_state = int(np.argmin(seed_residual))
        seed_margin = abs(float(seed_residual[0] - seed_residual[1]))

    evidence_threshold = np.deg2rad(config.motion_heading_branch_evidence_deg)
    forward_evidence = np.zeros(count, dtype=bool)
    reverse_evidence = np.zeros(count, dtype=bool)
    moving_heading_valid = raw_heading_valid[moving_index]
    forward_evidence[moving_index] = (
        moving_heading_valid & (candidate_residual[:, 0] <= evidence_threshold)
    )
    reverse_evidence[moving_index] = (
        moving_heading_valid & (candidate_residual[:, 1] <= evidence_threshold)
    )
    sustained_forward = _keep_sustained_runs(
        forward_evidence,
        config.motion_heading_min_run,
    )
    sustained_reverse = _keep_sustained_runs(
        reverse_evidence,
        config.motion_heading_reverse_min_run,
    )
    valid_motion_observations = int(np.sum(moving_heading_valid))
    forward_ratio = (
        float(np.sum(forward_evidence)) / valid_motion_observations
        if valid_motion_observations
        else 0.0
    )
    reverse_ratio = (
        float(np.sum(reverse_evidence)) / valid_motion_observations
        if valid_motion_observations
        else 0.0
    )
    seed_is_strong = seed_state is not None and seed_margin >= np.deg2rad(
        config.motion_heading_seed_margin_deg
    )
    clearly_forward = (
        np.any(sustained_forward)
        and not np.any(sustained_reverse)
        and forward_ratio >= config.motion_heading_forward_lock_ratio
        and not (seed_is_strong and seed_state == 1)
    )
    clearly_reverse = (
        np.any(sustained_reverse)
        and not np.any(sustained_forward)
        and reverse_ratio >= config.motion_heading_reverse_lock_ratio
        and not (seed_is_strong and seed_state == 0)
    )
    if clearly_forward or clearly_reverse:
        selected[moving_index] = candidates[:, 1 if clearly_reverse else 0]
        return selected, reliable

    accumulated_cost = np.full_like(candidates, np.inf)
    predecessor = np.zeros_like(candidates, dtype=np.int8)
    accumulated_cost[0] = observation_cost[0]
    if seed_index is not None:
        if seed_margin >= np.deg2rad(config.motion_heading_seed_margin_deg):
            accumulated_cost[0] = np.inf
            accumulated_cost[0, seed_state] = observation_cost[0, seed_state]
            if seed_index != first_moving:
                accumulated_cost[0, seed_state] += min(
                    float(seed_residual[seed_state]), observation_cap
                ) ** 2
        elif seed_index != first_moving:
            accumulated_cost[0] += np.minimum(seed_residual, observation_cap) ** 2

    for row in range(1, len(moving_index)):
        elapsed = max(float(time[moving_index[row]] - time[moving_index[row - 1]]), nominal_step)
        gap_scale = max(1.0, elapsed / nominal_step)
        for state in range(2):
            transition_residual = np.abs(
                angle_diff(candidates[row, state], candidates[row - 1])
            )
            transition_cost = (
                config.motion_heading_transition_weight
                * transition_residual**2
                / gap_scale
            )
            choices = accumulated_cost[row - 1] + transition_cost
            predecessor[row, state] = int(np.argmin(choices))
            accumulated_cost[row, state] = observation_cost[row, state] + float(
                np.min(choices)
            )

    state = int(np.argmin(accumulated_cost[-1]))
    selected[moving_index[-1]] = candidates[-1, state]
    for row in range(len(moving_index) - 1, 0, -1):
        state = int(predecessor[row, state])
        selected[moving_index[row - 1]] = candidates[row - 1, state]
    return selected, reliable


def _filter_segment(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    heading: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    object_type: int,
    config: TrajectoryFilterConfig,
) -> _SegmentCandidate | None:
    finite_xy = np.isfinite(x) & np.isfinite(y)
    finite_z = np.isfinite(z)
    finite_heading = np.isfinite(heading)

    interpolation_position_outliers = _interpolation_residual_outliers(
        np.column_stack((x, y)),
        observed & finite_xy,
        time,
        config.position_outlier_floor_m,
        config.outlier_sigma,
        config.outlier_support_ratio,
    )
    kinematic_position_outliers = _kinematic_excursion_position_outliers(
        np.column_stack((x, y)),
        observed & finite_xy,
        time,
        object_type,
        config,
    )
    consistent_path_outliers = _dominant_consistent_position_path_outliers(
        np.column_stack((x, y)),
        observed & finite_xy & ~kinematic_position_outliers,
        time,
        object_type,
        config,
    )
    if np.any(consistent_path_outliers):
        kinematic_position_outliers |= consistent_path_outliers
    else:
        closed_excursion_outliers = _short_closed_position_excursion_outliers(
            np.column_stack((x, y)),
            observed & finite_xy,
            time,
            object_type,
            config,
        )
        observed_position_count = int(np.sum(observed & finite_xy))
        maximum_closed_rejections = (
            config.vehicle_closed_excursion_max_rejected_ratio
            * observed_position_count
        )
        if int(np.sum(closed_excursion_outliers)) <= maximum_closed_rejections:
            kinematic_position_outliers |= closed_excursion_outliers
    position_outliers = (
        interpolation_position_outliers | kinematic_position_outliers
    )
    z_outliers = _interpolation_residual_outliers(
        z,
        observed & finite_z,
        time,
        config.z_outlier_floor_m,
        config.outlier_sigma,
        config.outlier_support_ratio,
    )
    xy_anchor = observed & finite_xy & ~position_outliers
    z_anchor = observed & finite_z & ~z_outliers & ~position_outliers
    position_outliers_relaxed = False
    if int(np.sum(xy_anchor)) < config.min_observed_frames:
        interpolation_position_outliers = np.zeros_like(observed, dtype=bool)
        kinematic_position_outliers = np.zeros_like(observed, dtype=bool)
        position_outliers = np.zeros_like(observed, dtype=bool)
        xy_anchor = observed & finite_xy
        z_anchor = observed & finite_z & ~z_outliers
        position_outliers_relaxed = True
    if int(np.sum(xy_anchor)) < config.min_observed_frames:
        return None

    x_filled = _interpolate_scalar(
        x,
        xy_anchor,
        coordinate=time,
        extrapolate_endpoints=True,
    )
    y_filled = _interpolate_scalar(
        y,
        xy_anchor,
        coordinate=time,
        extrapolate_endpoints=True,
    )
    z_filled = _interpolate_scalar(
        z,
        z_anchor,
        coordinate=time,
        extrapolate_endpoints=True,
    )
    if x_filled is None or y_filled is None:
        return None
    if z_filled is None:
        z_filled = np.zeros_like(x_filled)

    position_jerk_rms = _dense_linear_jerk_rms(
        x_filled,
        y_filled,
        z_filled,
        time,
    )
    rough_position = position_jerk_rms > config.rough_position_jerk_rms_mps3
    position_window = (
        max(config.position_window, config.rough_position_window)
        if rough_position
        else config.position_window
    )
    position_correction_limit = (
        max(
            config.max_trusted_xy_correction_m,
            config.rough_position_max_correction_m,
        )
        if rough_position
        else config.max_trusted_xy_correction_m
    )
    x_filtered, y_filtered = _smooth_xy_with_limit(
        x_filled,
        y_filled,
        x,
        y,
        xy_anchor,
        time,
        config,
        requested_window=position_window,
        maximum_correction=position_correction_limit,
    )
    z_filtered = _smooth_scalar_with_limit(
        z_filled,
        z,
        z_anchor,
        config.z_window,
        config.polynomial_order,
        config.max_trusted_z_correction_m,
        time,
    )

    heading_outliers = _remove_short_heading_flips(
        heading,
        observed & finite_heading,
        config.heading_flip_deg,
        config.heading_flip_support_deg,
        config.heading_flip_max_run,
    )
    heading_outliers |= _interpolation_residual_outliers(
        heading,
        observed & finite_heading & ~heading_outliers,
        time,
        np.deg2rad(config.heading_outlier_floor_deg),
        config.outlier_sigma,
        config.outlier_support_ratio,
        angular=True,
    )
    heading_outliers |= position_outliers
    heading_outliers |= _isolated_endpoint_heading_outliers(
        x,
        y,
        heading,
        observed,
        time,
        config,
    )
    if object_type in (1, 3):
        heading_outliers |= _vehicle_endpoint_heading_outliers(
            x,
            y,
            heading,
            observed & finite_heading & ~heading_outliers,
            time,
            config,
        )
    heading_for_motion = heading.copy()
    heading_branch_corrections = np.zeros_like(observed, dtype=bool)
    if object_type in (1, 3):
        velocity_x = _gradient(x_filtered, time)
        velocity_y = _gradient(y_filtered, time)
        speed = np.hypot(velocity_x, velocity_y)
        preliminary_heading_anchor = observed & finite_heading & ~heading_outliers
        preferred_anchor = _preferred_vehicle_heading_anchor(
            preliminary_heading_anchor,
            xy_anchor,
            speed,
            time,
            config,
        )
        branch_heading, heading_branch_corrections = _interpolate_vehicle_heading(
            heading,
            preliminary_heading_anchor,
            x_filtered,
            y_filtered,
            time,
            preferred_anchor,
            config,
        )
        if branch_heading is not None:
            heading_for_motion[preliminary_heading_anchor] = wrap_angle(
                branch_heading[preliminary_heading_anchor]
            )
    motion_heading, motion_reliable = _motion_body_heading_observation(
        x_filtered,
        y_filtered,
        heading_for_motion,
        observed & finite_heading & ~heading_outliers,
        xy_anchor,
        time,
        object_type,
        config,
    )
    motion_reference = np.full_like(heading, np.nan, dtype=float)
    motion_supported = np.zeros_like(observed, dtype=bool)
    if np.any(motion_reliable):
        moving_index = np.flatnonzero(motion_reliable)
        motion_gap = np.abs(time[:, None] - time[moving_index][None, :])
        nearest_motion_position = np.argmin(motion_gap, axis=1)
        nearest_motion_gap = motion_gap[
            np.arange(len(time)),
            nearest_motion_position,
        ]
        motion_supported = (
            nearest_motion_gap <= config.motion_heading_support_max_gap_s
        )
        nearest_motion_index = moving_index[nearest_motion_position]
        motion_reference[motion_supported] = motion_heading[
            nearest_motion_index[motion_supported]
        ]
    motion_reference_unwrapped = _interpolate_heading(
        motion_reference,
        motion_supported,
    )
    motion_residual = np.abs(angle_diff(heading_for_motion, motion_reference))
    motion_outliers = (
        observed
        & finite_heading
        & motion_supported
        & (motion_residual > np.deg2rad(config.motion_heading_outlier_deg))
    )
    heading_outliers |= motion_outliers

    heading_values = heading_for_motion.copy()
    heading_values[heading_outliers | ~observed | ~finite_heading] = np.nan
    trusted_heading_anchor = observed & finite_heading & ~heading_outliers
    if object_type in (2, 3):
        heading_unwrapped = _interpolate_pi_ambiguous_heading(
            heading_values,
            trusted_heading_anchor,
        )
        if object_type == 2:
            if _supports_detailed_pedestrian_heading(
                heading_for_motion,
                trusted_heading_anchor,
                config,
            ):
                heading_window = config.pedestrian_detailed_heading_window
                heading_polynomial_order = (
                    config.pedestrian_detailed_heading_polynomial_order
                )
            else:
                heading_window = config.pedestrian_heading_window
                heading_polynomial_order = (
                    config.pedestrian_heading_polynomial_order
                )
        else:
            heading_window = config.heading_window
            heading_polynomial_order = config.polynomial_order
    elif object_type == 1:
        heading_unwrapped = _interpolate_heading(
            heading_values,
            trusted_heading_anchor,
        )
        heading_window = config.heading_window
        heading_polynomial_order = config.polynomial_order
    else:
        heading_unwrapped = _interpolate_heading(
            heading_values,
            trusted_heading_anchor,
        )
        heading_window = config.heading_window
        heading_polynomial_order = config.polynomial_order
    if heading_unwrapped is None:
        heading_unwrapped = motion_reference_unwrapped
    if heading_unwrapped is None:
        heading_unwrapped = np.zeros_like(x_filled)
    heading_outliers |= heading_branch_corrections
    quality_heading_anchor = trusted_heading_anchor & ~heading_branch_corrections
    heading_filtered = None
    for window in _smoothing_windows(
        len(heading_unwrapped),
        heading_window,
        heading_polynomial_order,
    ):
        candidate = wrap_angle(
            _smooth_scalar(
                heading_unwrapped,
                window,
                heading_polynomial_order,
                time,
            )
        )
        heading_correction = np.abs(angle_diff(candidate, heading))
        if object_type in (2, 3):
            opposite_correction = np.abs(
                angle_diff(candidate, wrap_angle(heading + np.pi))
            )
            heading_correction = np.minimum(
                heading_correction,
                opposite_correction,
            )
        trusted_correction = np.rad2deg(heading_correction)[quality_heading_anchor]
        if (
            _percentile(trusted_correction)
            <= (
                max(90.0, config.max_trusted_heading_correction_deg)
                if object_type in (2, 3)
                else config.max_trusted_heading_correction_deg
            )
        ):
            heading_filtered = candidate
            break
    if heading_filtered is None:
        heading_filtered = wrap_angle(heading_unwrapped)

    return _SegmentCandidate(
        x=x_filtered,
        y=y_filtered,
        z=z_filtered,
        heading=heading_filtered,
        position_outliers=position_outliers,
        kinematic_position_outliers=kinematic_position_outliers,
        z_outliers=z_outliers,
        heading_outliers=heading_outliers,
        position_outliers_relaxed=position_outliers_relaxed,
    )


def _best_effort_candidate(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    heading: np.ndarray,
    observed: np.ndarray,
    time: np.ndarray,
    object_type: int,
    config: TrajectoryFilterConfig,
) -> _SegmentCandidate | None:
    """Always return a smoothed finite trajectory when two xy anchors exist."""
    finite_xy = observed & np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite_xy)) < 2:
        return None

    x_filled = _interpolate_scalar(
        x,
        finite_xy,
        coordinate=time,
        extrapolate_endpoints=True,
    )
    y_filled = _interpolate_scalar(
        y,
        finite_xy,
        coordinate=time,
        extrapolate_endpoints=True,
    )
    z_filled = _interpolate_scalar(
        z,
        observed & np.isfinite(z),
        coordinate=time,
        extrapolate_endpoints=True,
    )
    heading_anchor = observed & np.isfinite(heading)
    if object_type in (2, 3):
        heading_filled = _interpolate_pi_ambiguous_heading(
            heading,
            heading_anchor,
        )
        if object_type == 2:
            heading_window = config.pedestrian_heading_window
            heading_order = config.pedestrian_heading_polynomial_order
        else:
            heading_window = config.heading_window
            heading_order = config.polynomial_order
    else:
        heading_filled = _interpolate_heading(heading, heading_anchor)
        heading_window = config.heading_window
        heading_order = config.polynomial_order
    if x_filled is None or y_filled is None:
        return None
    if z_filled is None:
        z_filled = np.zeros_like(x_filled)
    if heading_filled is None:
        heading_filled = np.zeros_like(x_filled)

    origin_x = float(x_filled[0])
    origin_y = float(y_filled[0])
    x_output = origin_x + _smooth_scalar(
        x_filled - origin_x,
        config.position_window,
        config.polynomial_order,
        time,
    )
    y_output = origin_y + _smooth_scalar(
        y_filled - origin_y,
        config.position_window,
        config.polynomial_order,
        time,
    )
    z_output = _smooth_scalar(
        z_filled,
        config.z_window,
        config.polynomial_order,
        time,
    )
    heading_output = wrap_angle(
        _smooth_scalar(
            heading_filled,
            heading_window,
            heading_order,
            time,
        )
    )
    empty_mask = np.zeros_like(observed, dtype=bool)
    return _SegmentCandidate(
        x=x_output,
        y=y_output,
        z=z_output,
        heading=heading_output,
        position_outliers=empty_mask.copy(),
        kinematic_position_outliers=empty_mask.copy(),
        z_outliers=empty_mask.copy(),
        heading_outliers=empty_mask.copy(),
    )


def _gradient(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=float)
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(np.asarray(values, dtype=float), time, edge_order=edge_order)


def _keep_sustained_runs(mask: np.ndarray, minimum_run: int) -> np.ndarray:
    kept = np.zeros_like(mask, dtype=bool)
    for start, end in _true_runs(mask):
        if end - start >= minimum_run:
            kept[start:end] = True
    return kept


def classify_motion_modes(
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray | None = None,
    config: TrajectoryFilterConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify motion without changing body heading during reverse motion."""
    config = config or TrajectoryFilterConfig()
    velocity_x = np.asarray(velocity_x, dtype=float)
    velocity_y = np.asarray(velocity_y, dtype=float)
    heading = np.asarray(heading, dtype=float)
    if valid is None:
        valid = np.ones(len(heading), dtype=bool)
    else:
        valid = np.asarray(valid, dtype=bool)

    longitudinal = velocity_x * np.cos(heading) + velocity_y * np.sin(heading)
    lateral = -velocity_x * np.sin(heading) + velocity_y * np.cos(heading)
    speed = np.hypot(velocity_x, velocity_y)
    finite = valid & np.isfinite(speed) & np.isfinite(heading)

    modes = np.full(len(heading), int(MotionMode.UNKNOWN), dtype=np.int8)
    stationary = finite & (speed < config.stationary_speed_mps)
    moving = finite & ~stationary
    reverse_candidate = moving & (longitudinal <= -config.direction_speed_mps)
    forward_candidate = moving & (longitudinal >= config.direction_speed_mps)
    reverse = _keep_sustained_runs(reverse_candidate, config.direction_min_run)
    forward = _keep_sustained_runs(forward_candidate, config.direction_min_run)

    modes[stationary] = int(MotionMode.STATIONARY)
    modes[moving] = int(MotionMode.LATERAL)
    modes[forward] = int(MotionMode.FORWARD)
    modes[reverse] = int(MotionMode.REVERSE)
    return modes, longitudinal, lateral


def _central_logical_and(valid: np.ndarray) -> np.ndarray:
    output = np.zeros_like(valid, dtype=bool)
    if len(valid) >= 3:
        output[1:-1] = valid[:-2] & valid[2:]
    return output


def compute_kinematic_features(
    positions: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    timestamps: Iterable[float],
    config: TrajectoryFilterConfig | None = None,
) -> KinematicFeatures:
    """Compute TrajTok-style centered kinematics from xyz and body heading."""
    config = config or TrajectoryFilterConfig()
    positions = np.asarray(positions, dtype=float)
    heading = np.asarray(heading, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    count = len(valid)
    time = _timestamps_for_count(timestamps, count)

    velocity = np.full((count, 3), np.nan, dtype=float)
    angular_speed = np.full(count, np.nan, dtype=float)
    angular_step = np.full(count, np.nan, dtype=float)
    speed_validity = _central_logical_and(valid)
    if count >= 3:
        duration = time[2:] - time[:-2]
        usable = speed_validity[1:-1] & (duration > 0)
        delta = positions[2:] - positions[:-2]
        interior_velocity = velocity[1:-1]
        interior_velocity[usable] = delta[usable] / duration[usable, None]
        heading_usable = usable & np.isfinite(heading[2:]) & np.isfinite(heading[:-2])
        angular_delta = angle_diff(heading[2:], heading[:-2])
        interior_angular_speed = angular_speed[1:-1]
        interior_angular_step = angular_step[1:-1]
        interior_angular_speed[heading_usable] = (
            angular_delta[heading_usable] / duration[heading_usable]
        )
        interior_angular_step[heading_usable] = angular_delta[heading_usable] / 2.0

    planar_speed = np.hypot(velocity[:, 0], velocity[:, 1])
    linear_speed = np.linalg.norm(velocity, axis=1)
    acceleration_validity = _central_logical_and(speed_validity)
    planar_acceleration = np.full(count, np.nan, dtype=float)
    linear_acceleration = np.full(count, np.nan, dtype=float)
    angular_acceleration = np.full(count, np.nan, dtype=float)
    if count >= 3:
        duration = time[2:] - time[:-2]
        usable = acceleration_validity[1:-1] & (duration > 0)
        planar_delta = planar_speed[2:] - planar_speed[:-2]
        linear_delta = linear_speed[2:] - linear_speed[:-2]
        interior_planar_acceleration = planar_acceleration[1:-1]
        interior_linear_acceleration = linear_acceleration[1:-1]
        interior_planar_acceleration[usable] = planar_delta[usable] / duration[usable]
        interior_linear_acceleration[usable] = linear_delta[usable] / duration[usable]

        time_step = np.diff(time)
        uniform_time = len(time_step) > 0 and np.allclose(
            time_step, np.median(time_step), rtol=1e-5, atol=1e-8
        )
        if uniform_time:
            dt = float(np.median(time_step))
            angular_delta = angle_diff(angular_step[2:], angular_step[:-2])
            angular_values = angular_delta / (2.0 * dt * dt)
        else:
            angular_delta = angular_speed[2:] - angular_speed[:-2]
            angular_values = angular_delta / duration
        angular_usable = usable & np.isfinite(angular_values)
        interior_angular_acceleration = angular_acceleration[1:-1]
        interior_angular_acceleration[angular_usable] = angular_values[angular_usable]

    jerk_validity = _central_logical_and(acceleration_validity)
    linear_jerk = np.full(count, np.nan, dtype=float)
    angular_jerk = np.full(count, np.nan, dtype=float)
    if count >= 3:
        duration = time[2:] - time[:-2]
        usable = jerk_validity[1:-1] & (duration > 0)
        linear_values = (linear_acceleration[2:] - linear_acceleration[:-2]) / duration
        angular_values = (angular_acceleration[2:] - angular_acceleration[:-2]) / duration
        linear_usable = usable & np.isfinite(linear_values)
        angular_usable = usable & np.isfinite(angular_values)
        interior_linear_jerk = linear_jerk[1:-1]
        interior_angular_jerk = angular_jerk[1:-1]
        interior_linear_jerk[linear_usable] = linear_values[linear_usable]
        interior_angular_jerk[angular_usable] = angular_values[angular_usable]

    frame_usable = speed_validity & valid & np.isfinite(heading)
    modes, longitudinal, lateral = classify_motion_modes(
        velocity[:, 0], velocity[:, 1], heading, frame_usable, config
    )
    return KinematicFeatures(
        velocity_x=velocity[:, 0],
        velocity_y=velocity[:, 1],
        velocity_z=velocity[:, 2],
        planar_speed=planar_speed,
        linear_speed=linear_speed,
        planar_acceleration=planar_acceleration,
        linear_acceleration=linear_acceleration,
        linear_jerk=linear_jerk,
        angular_speed=angular_speed,
        angular_acceleration=angular_acceleration,
        angular_jerk=angular_jerk,
        longitudinal_speed=longitudinal,
        lateral_speed=lateral,
        speed_validity=speed_validity,
        acceleration_validity=acceleration_validity,
        jerk_validity=jerk_validity,
        motion_mode=modes,
    )


def compute_track_kinematics(
    track,
    timestamps: Iterable[float],
    config: TrajectoryFilterConfig | None = None,
) -> KinematicFeatures:
    time = list(timestamps)
    count = min(len(track.states), len(time))
    time = time[:count]
    states = track.states[:count]
    positions = np.array(
        [[state.center_x, state.center_y, state.center_z] for state in states], dtype=float
    )
    heading = np.array([state.heading for state in states], dtype=float)
    valid = np.array([state.valid for state in states], dtype=bool)
    return compute_kinematic_features(positions, heading, valid, time, config)


def _interpolated_sizes(track, start: int, end: int, observed: np.ndarray) -> np.ndarray:
    sizes = np.array(
        [[state.length, state.width, state.height] for state in track.states[start:end]],
        dtype=float,
    )
    for dimension in range(3):
        anchors = observed & np.isfinite(sizes[:, dimension]) & (sizes[:, dimension] > 0)
        interpolated = _interpolate_scalar(sizes[:, dimension], anchors)
        if interpolated is not None:
            sizes[:, dimension] = interpolated
    return sizes


def reconstruct_trajectory_arrays(
    positions: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    timestamps: Iterable[float],
    object_type: int,
    config: TrajectoryFilterConfig | None = None,
) -> ArrayTrajectoryReconstruction:
    """Apply the proto reconstructor to one array-backed trajectory.

    ``object_type`` uses WOMD's one-based convention: vehicle=1,
    pedestrian=2, cyclist=3. Invalid samples are never used as observations.
    With ``max_gap_frames=None`` all internal gaps between the first and last
    observations belong to one continuous reconstruction block.
    """

    config = config or TrajectoryFilterConfig()
    positions = np.asarray(positions, dtype=float)
    heading = np.asarray(heading, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("positions must have shape [steps, >=2]")
    if heading.shape != positions.shape[:1] or valid.shape != positions.shape[:1]:
        raise ValueError("positions, heading, and valid must share the step dimension")

    count = len(valid)
    time = _timestamps_for_count(timestamps, count)
    if positions.shape[1] == 2:
        positions_xyz = np.column_stack(
            (positions, np.zeros(count, dtype=positions.dtype))
        )
    else:
        positions_xyz = positions[:, :3].copy()

    output_positions = positions_xyz.copy()
    output_heading = heading.copy()
    output_valid = np.zeros(count, dtype=bool)
    result = TrackFilterResult()
    if object_type not in config.processed_object_types:
        return ArrayTrajectoryReconstruction(
            output_positions, output_heading, output_valid, result
        )
    if (
        count < config.min_observed_frames
        or int(np.sum(valid)) < config.min_observed_frames
    ):
        result.insufficient_support = True
        return ArrayTrajectoryReconstruction(
            output_positions, output_heading, output_valid, result
        )

    eligible, gap_candidates = _fill_short_internal_gaps(
        valid, config.max_gap_frames
    )
    accepted_runs: list[tuple[int, int]] = []
    for start, end in _true_runs(eligible):
        observed = valid[start:end]
        if int(np.sum(observed)) < config.min_observed_frames:
            continue
        candidate = _filter_segment(
            output_positions[start:end, 0],
            output_positions[start:end, 1],
            output_positions[start:end, 2],
            output_heading[start:end],
            observed,
            time[start:end],
            object_type,
            config,
        )
        used_fallback = False
        if candidate is None:
            result.rejected_segments += 1
            candidate = _best_effort_candidate(
                output_positions[start:end, 0],
                output_positions[start:end, 1],
                output_positions[start:end, 2],
                output_heading[start:end],
                observed,
                time[start:end],
                object_type,
                config,
            )
            if candidate is None:
                continue
            used_fallback = True
            result.best_effort_segments += 1

        output_positions[start:end] = np.column_stack(
            (candidate.x, candidate.y, candidate.z)
        )
        output_heading[start:end] = candidate.heading
        output_valid[start:end] = True
        accepted_runs.append((start, end))

        velocity_x = _gradient(candidate.x, time[start:end])
        velocity_y = _gradient(candidate.y, time[start:end])
        modes, _, _ = classify_motion_modes(
            velocity_x,
            velocity_y,
            candidate.heading,
            np.ones(end - start, dtype=bool),
            config,
        )
        filled_count = int(np.sum(gap_candidates[start:end]))
        result.filtered_frames += end - start
        result.filled_frames += filled_count
        if used_fallback:
            result.fallback_filled_frames += filled_count
        result.position_outliers += int(np.sum(candidate.position_outliers))
        result.kinematic_position_outliers += int(
            np.sum(candidate.kinematic_position_outliers)
        )
        result.relaxed_position_outlier_segments += int(
            candidate.position_outliers_relaxed
        )
        result.z_outliers += int(np.sum(candidate.z_outliers))
        result.heading_outliers += int(np.sum(candidate.heading_outliers))
        result.reverse_frames += int(np.sum(modes == int(MotionMode.REVERSE)))

    result.reconstructed = bool(accepted_runs)
    return ArrayTrajectoryReconstruction(
        output_positions, output_heading, output_valid, result
    )


def reconstruct_track(
    track,
    timestamps: Iterable[float],
    config: TrajectoryFilterConfig | None = None,
) -> TrackFilterResult:
    config = config or TrajectoryFilterConfig()
    result = TrackFilterResult()
    if int(track.object_type) not in config.processed_object_types:
        return result

    timestamps = list(timestamps)
    count = min(len(track.states), len(timestamps))
    if count < config.min_observed_frames:
        result.insufficient_support = True
        return result
    original_valid = np.array([track.states[i].valid for i in range(count)], dtype=bool)
    positions = np.array(
        [
            [state.center_x, state.center_y, state.center_z]
            for state in track.states[:count]
        ],
        dtype=float,
    )
    heading = np.array(
        [state.heading for state in track.states[:count]], dtype=float
    )
    reconstruction = reconstruct_trajectory_arrays(
        positions,
        heading,
        original_valid,
        timestamps,
        int(track.object_type),
        config,
    )
    time = _timestamps_for_count(timestamps, count)

    for start, end in _true_runs(reconstruction.valid):
        observed = original_valid[start:end]
        sizes = _interpolated_sizes(track, start, end, observed)
        for local_index, global_index in enumerate(range(start, end)):
            state = track.states[global_index]
            state.center_x = float(reconstruction.positions[global_index, 0])
            state.center_y = float(reconstruction.positions[global_index, 1])
            state.center_z = float(reconstruction.positions[global_index, 2])
            state.heading = float(reconstruction.heading[global_index])
            if not original_valid[global_index]:
                state.length = float(sizes[local_index, 0])
                state.width = float(sizes[local_index, 1])
                state.height = float(sizes[local_index, 2])
            state.valid = True

        velocity_x = _gradient(reconstruction.positions[start:end, 0], time[start:end])
        velocity_y = _gradient(reconstruction.positions[start:end, 1], time[start:end])
        for local_index, global_index in enumerate(range(start, end)):
            track.states[global_index].velocity_x = float(velocity_x[local_index])
            track.states[global_index].velocity_y = float(velocity_y[local_index])
    return reconstruction.result


def reconstruct_scenario_agents(
    scenario,
    config: TrajectoryFilterConfig | None = None,
):
    reconstructed = copy.deepcopy(scenario)
    timestamps = list(reconstructed.timestamps_seconds)
    stats = ReconstructionStats(total_tracks=len(reconstructed.tracks))

    for track in reconstructed.tracks:
        result = reconstruct_track(track, timestamps, config=config)
        if result.reconstructed:
            stats.reconstructed_tracks += 1
        else:
            stats.skipped_tracks += 1
        if result.insufficient_support:
            stats.insufficient_support_tracks += 1
        stats.filtered_frames += result.filtered_frames
        stats.filled_frames += result.filled_frames
        stats.position_outliers += result.position_outliers
        stats.kinematic_position_outliers += result.kinematic_position_outliers
        stats.relaxed_position_outlier_segments += (
            result.relaxed_position_outlier_segments
        )
        stats.z_outliers += result.z_outliers
        stats.heading_outliers += result.heading_outliers
        stats.reverse_frames += result.reverse_frames
        stats.rejected_segments += result.rejected_segments
        stats.best_effort_segments += result.best_effort_segments
        stats.fallback_filled_frames += result.fallback_filled_frames

    return reconstructed, stats
