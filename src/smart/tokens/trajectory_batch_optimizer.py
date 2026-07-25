# Adapted from WOMD-Traffic-Signal-Data-Improvement for CatK vocabulary-only
# reconstruction. Distributed under the PolyForm Noncommercial License 1.0.0;
# see LICENSE.WOMD_TRAJECTORY_RECONSTRUCTION.txt in this directory.

"""Batch trajectory smoothing with decoupled position and heading stages.

The batch method follows a geometry-first order:

1. use the geometric filter output as the filled/smoothed position observation,
2. optimize positions using only position fidelity and linear jerk terms,
3. build motion-aware heading observations from the optimized positions and
   the original raw headings,
4. optimize heading with heading fidelity and angular jerk terms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .trajectory_filter_reconstructor import (
    ReconstructionStats,
    _interpolate_heading,
    _interpolate_pi_ambiguous_heading,
    _interpolate_vehicle_heading,
    _interpolation_residual_outliers,
    _isolated_endpoint_heading_outliers,
    _keep_sustained_runs,
    _motion_body_heading_observation,
    _percentile,
    _preferred_vehicle_heading_anchor,
    _remove_short_heading_flips,
    _smooth_scalar,
    _smoothing_windows,
    _supports_detailed_pedestrian_heading,
    _vehicle_endpoint_heading_outliers,
    angle_diff,
    config_for_filter_strength,
    reconstruct_scenario_agents as filter_scenario_agents,
    wrap_angle,
)


@dataclass(frozen=True)
class BatchTrajectoryConfig:
    processed_object_types: tuple[int, ...] = (1, 2, 3)
    min_optimization_frames: int = 7
    position_observation_scale_m: float = 0.20
    heading_observation_scale_rad: float = np.deg2rad(10.0)
    heading_rate_observation_scale_radps: float = np.deg2rad(5.0)
    heading_rate_observation_weight: float = 1.0
    vehicle_heading_rate_weight_scale: float = 0.0
    pedestrian_heading_rate_weight_scale: float = 0.0
    cyclist_heading_rate_weight_scale: float = 0.005
    filled_observation_weight: float = 0.10
    endpoint_observation_weight: float = 4.0
    robust_observation_delta: float = 2.0
    linear_jerk_scale_mps3: float = 10.0
    planar_vector_jerk_scale_mps3: float = 10.0
    angular_jerk_scale_radps3: float = 5.0
    linear_jerk_weight: float = 1.0
    planar_vector_jerk_weight: float = 2.0
    angular_jerk_weight: float = 1.0
    adjacent_planar_jerk_weight: float = 1.0
    adjacent_angular_jerk_weight: float = 1.0
    pedestrian_regularization_scale: float = 0.50
    cyclist_regularization_scale: float = 0.75
    short_segment_linear_regularization_frames: int = 25
    short_segment_linear_regularization_max_scale: float = 4.0
    pedestrian_heading_jerk_weight_scale: float = 2.0
    prefilter_trusted_position_correction_m: float = 0.25
    vehicle_velocity_heading_outlier_deg: float = 75.0
    vehicle_short_heading_excursion_deg: float = 60.0
    vehicle_prefilter_disagreement_p95_deg: float = 45.0
    vehicle_prefilter_excess_turn_deg: float = 60.0
    vehicle_heading_target_window: int = 5
    vehicle_min_forward_heading_anchors: int = 2
    vehicle_reverse_speed_tolerance_mps: float = 0.0
    vehicle_motion_heading_weight: float = 0.20
    motion_heading_scale_rad: float = np.deg2rad(10.0)
    max_observed_position_shift_m: float = 0.50
    max_filled_position_shift_m: float = 2.00
    max_observed_heading_shift_rad: float = np.deg2rad(30.0)
    max_filled_heading_shift_rad: float = np.deg2rad(90.0)
    max_trusted_position_correction_p95_m: float = 0.75
    max_trusted_heading_correction_p95_rad: float = np.deg2rad(20.0)
    linear_jerk_safety_rms_mps3: float = 8.0
    vehicle_linear_jerk_safety_correction_p95_m: float = 8.0
    pedestrian_linear_jerk_safety_correction_p95_m: float = 1.5
    cyclist_linear_jerk_safety_correction_p95_m: float = 2.5
    angular_jerk_safety_rms_radps3: float = 2.0
    angular_jerk_safety_correction_p95_rad: float = np.deg2rad(90.0)
    max_nfev: int = 120
    ftol: float = 1e-7
    xtol: float = 1e-7
    gtol: float = 1e-7


@dataclass
class TrackBatchResult:
    processed: bool = False
    optimized: bool = False
    processed_segments: int = 0
    optimized_segments: int = 0
    optimized_frames: int = 0
    short_segments: int = 0
    failed_segments: int = 0
    position_optimized_segments: int = 0
    heading_optimized_segments: int = 0
    position_limited_segments: int = 0
    heading_limited_segments: int = 0
    position_solver_failures: int = 0
    heading_solver_failures: int = 0
    linear_acceleration_sumsq_before: float = 0.0
    linear_acceleration_sumsq_after: float = 0.0
    linear_acceleration_count: int = 0
    angular_acceleration_sumsq_before: float = 0.0
    angular_acceleration_sumsq_after: float = 0.0
    angular_acceleration_count: int = 0
    linear_jerk_sumsq_before: float = 0.0
    linear_jerk_sumsq_after: float = 0.0
    linear_jerk_count: int = 0
    angular_jerk_sumsq_before: float = 0.0
    angular_jerk_sumsq_after: float = 0.0
    angular_jerk_count: int = 0


@dataclass
class BatchReconstructionStats(ReconstructionStats):
    processed_tracks: int = 0
    processed_segments: int = 0
    optimized_tracks: int = 0
    optimized_segments: int = 0
    optimized_frames: int = 0
    short_segments: int = 0
    optimization_failures: int = 0
    position_optimized_segments: int = 0
    heading_optimized_segments: int = 0
    position_limited_segments: int = 0
    heading_limited_segments: int = 0
    position_solver_failures: int = 0
    heading_solver_failures: int = 0
    linear_acceleration_rms_before: float = 0.0
    linear_acceleration_rms_after: float = 0.0
    angular_acceleration_rms_before: float = 0.0
    angular_acceleration_rms_after: float = 0.0
    linear_jerk_rms_before: float = 0.0
    linear_jerk_rms_after: float = 0.0
    angular_jerk_rms_before: float = 0.0
    angular_jerk_rms_after: float = 0.0


@dataclass
class _SegmentSolution:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    heading: np.ndarray
    linear_acceleration_before: np.ndarray
    linear_acceleration_after: np.ndarray
    angular_acceleration_before: np.ndarray
    angular_acceleration_after: np.ndarray
    linear_jerk_before: np.ndarray
    linear_jerk_after: np.ndarray
    angular_jerk_before: np.ndarray
    angular_jerk_after: np.ndarray
    position_optimized: bool
    heading_optimized: bool
    position_limited: bool
    heading_limited: bool
    position_solver_failed: bool
    heading_solver_failed: bool


@dataclass
class _PositionSolution:
    x: np.ndarray
    y: np.ndarray
    optimized: bool
    limited: bool
    solver_failed: bool


@dataclass
class _HeadingSolution:
    heading: np.ndarray
    optimized: bool
    limited: bool
    solver_failed: bool


@dataclass
class _HeadingObservation:
    heading: np.ndarray
    trusted: np.ndarray
    motion_reference: np.ndarray
    motion_reliable: np.ndarray
    outliers: np.ndarray


def _true_runs(mask: np.ndarray):
    start = None
    for index, value in enumerate(np.r_[np.asarray(mask, dtype=bool), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield start, index
            start = None


def _timestamps_for_count(timestamps: Iterable[float], count: int) -> np.ndarray:
    time = np.asarray(list(timestamps)[:count], dtype=float)
    if len(time) != count:
        return np.arange(count, dtype=float) * 0.1
    if count > 1:
        step = np.diff(time)
        positive = step[np.isfinite(step) & (step > 0)]
        if len(positive) and float(np.median(positive)) > 10.0:
            time = time / 1e6
    if not np.all(np.isfinite(time)) or (count > 1 and np.any(np.diff(time) <= 0)):
        return np.arange(count, dtype=float) * 0.1
    return time - time[0]


def _seconds_per_step(time: np.ndarray) -> float:
    step = np.diff(time)
    positive = step[np.isfinite(step) & (step > 0)]
    return float(np.median(positive)) if len(positive) else 0.1


def _gradient(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=float)
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(np.asarray(values, dtype=float), time, edge_order=edge_order)


def _kinematic_time_axis(
    count: int,
    time_or_step: float | np.ndarray,
) -> np.ndarray:
    value = np.asarray(time_or_step, dtype=float)
    if value.ndim == 0:
        step = float(value)
        if not np.isfinite(step) or step <= 0.0:
            step = 0.1
        return np.arange(count, dtype=float) * step
    if value.ndim != 1 or len(value) != count:
        raise ValueError("Kinematic timestamps must match the trajectory length")
    if not np.all(np.isfinite(value)) or (count > 1 and np.any(np.diff(value) <= 0)):
        raise ValueError("Kinematic timestamps must be finite and strictly increasing")
    return value - value[0]


def wosac_acceleration_features(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    heading_unwrapped: np.ndarray,
    time_or_step: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return WOSAC linear and angular acceleration on their valid centers."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    heading_unwrapped = np.asarray(heading_unwrapped, dtype=float)
    if len(x) < 5:
        empty = np.empty(0, dtype=float)
        return empty, empty.copy()

    time = _kinematic_time_axis(len(x), time_or_step)
    velocity_duration = time[2:] - time[:-2]
    dpos_x = (x[2:] - x[:-2]) / velocity_duration
    dpos_y = (y[2:] - y[:-2]) / velocity_duration
    dpos_z = (z[2:] - z[:-2]) / velocity_duration
    linear_speed = np.sqrt(dpos_x**2 + dpos_y**2 + dpos_z**2)
    acceleration_duration = time[3:-1] - time[1:-3]
    linear_acceleration = (
        linear_speed[2:] - linear_speed[:-2]
    ) / acceleration_duration

    time_step = np.diff(time)
    uniform_time = len(time_step) > 0 and np.allclose(
        time_step,
        np.median(time_step),
        rtol=1e-5,
        atol=1e-8,
    )
    heading_delta = wrap_angle(heading_unwrapped[2:] - heading_unwrapped[:-2])
    if uniform_time:
        step = float(np.median(time_step))
        angular_step = heading_delta / 2.0
        angular_acceleration = wrap_angle(
            angular_step[2:] - angular_step[:-2]
        ) / (2.0 * step**2)
    else:
        angular_speed = heading_delta / velocity_duration
        angular_acceleration = (
            angular_speed[2:] - angular_speed[:-2]
        ) / acceleration_duration
    return linear_acceleration, angular_acceleration


def wosac_jerk_features(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    heading_unwrapped: np.ndarray,
    time_or_step: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend the WOSAC centered-difference chain by one derivative."""
    linear_acceleration, angular_acceleration = wosac_acceleration_features(
        x,
        y,
        z,
        heading_unwrapped,
        time_or_step,
    )
    if len(linear_acceleration) < 3:
        empty = np.empty(0, dtype=float)
        return empty, empty.copy()
    time = _kinematic_time_axis(len(x), time_or_step)
    jerk_duration = time[4:-2] - time[2:-4]
    linear_jerk = (
        linear_acceleration[2:] - linear_acceleration[:-2]
    ) / jerk_duration
    angular_jerk = (
        angular_acceleration[2:] - angular_acceleration[:-2]
    ) / jerk_duration
    return linear_jerk, angular_jerk


def _centered_planar_vector_jerk(
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return centered x/y jerk to keep the optimized path direction smooth."""
    if len(x) < 7:
        empty = np.empty(0, dtype=float)
        return empty, empty.copy()
    time = _kinematic_time_axis(len(x), time)
    velocity_duration = time[2:] - time[:-2]
    velocity_x = (x[2:] - x[:-2]) / velocity_duration
    velocity_y = (y[2:] - y[:-2]) / velocity_duration
    acceleration_duration = time[3:-1] - time[1:-3]
    acceleration_x = (velocity_x[2:] - velocity_x[:-2]) / acceleration_duration
    acceleration_y = (velocity_y[2:] - velocity_y[:-2]) / acceleration_duration
    jerk_duration = time[4:-2] - time[2:-4]
    jerk_x = (acceleration_x[2:] - acceleration_x[:-2]) / jerk_duration
    jerk_y = (acceleration_y[2:] - acceleration_y[:-2]) / jerk_duration
    return jerk_x, jerk_y


def _adjacent_scalar_jerk(
    values: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    """Return third divided differences that couple adjacent frame parity."""
    values = np.asarray(values, dtype=float)
    if len(values) < 4:
        return np.empty(0, dtype=float)
    time = _kinematic_time_axis(len(values), time)

    velocity_time = 0.5 * (time[1:] + time[:-1])
    velocity = np.diff(values) / np.diff(time)
    acceleration_time = 0.5 * (velocity_time[1:] + velocity_time[:-1])
    acceleration = np.diff(velocity) / np.diff(velocity_time)
    return np.diff(acceleration) / np.diff(acceleration_time)


def _regularization_scale(object_type: int, config: BatchTrajectoryConfig) -> float:
    if object_type == 2:
        return config.pedestrian_regularization_scale
    if object_type == 3:
        return config.cyclist_regularization_scale
    return 1.0


def _short_segment_linear_regularization_scale(
    count: int,
    config: BatchTrajectoryConfig,
) -> float:
    full_frames = max(
        config.min_optimization_frames,
        config.short_segment_linear_regularization_frames,
    )
    if count >= full_frames or full_frames == config.min_optimization_frames:
        return 1.0
    fraction = (full_frames - count) / (
        full_frames - config.min_optimization_frames
    )
    return 1.0 + fraction * (
        config.short_segment_linear_regularization_max_scale - 1.0
    )


def _finite_array_rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0
    return float(np.sqrt(np.mean(finite**2)))


def _jerk_support_from_frame_validity(
    frame_validity: np.ndarray | None,
    count: int,
) -> np.ndarray | None:
    if frame_validity is None:
        return None
    valid = np.asarray(frame_validity, dtype=bool)
    if len(valid) != count:
        raise ValueError("frame validity must match the trajectory length")
    if count < 7:
        return np.zeros(0, dtype=bool)
    return valid[:-6] & valid[2:-4] & valid[4:-2] & valid[6:]


def _safety_jerk_rms(
    jerk: np.ndarray,
    matched_support: np.ndarray | None,
) -> float:
    rms = _finite_array_rms(jerk)
    if matched_support is None or not np.any(matched_support):
        return rms
    if len(matched_support) != len(jerk):
        raise ValueError("jerk support must match the jerk feature length")
    return max(rms, _finite_array_rms(np.asarray(jerk)[matched_support]))


def _safety_smoothing_windows(count: int):
    maximum = count if count % 2 else count - 1
    if maximum < 7:
        return range(0)
    return range(7, maximum + 1, 2)


def _global_time_polynomial_fit(
    values: np.ndarray,
    time: np.ndarray,
    polynomial_order: int,
) -> np.ndarray:
    """Fit a low-order trend against the actual, possibly irregular time axis."""
    values = np.asarray(values, dtype=float)
    time = _kinematic_time_axis(len(values), np.asarray(time, dtype=float))
    if len(values) <= polynomial_order or time[-1] <= time[0]:
        return values.copy()
    normalized_time = 2.0 * (time - time[0]) / (time[-1] - time[0]) - 1.0
    design = np.polynomial.polynomial.polyvander(
        normalized_time,
        polynomial_order,
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return design @ coefficients


def _linear_jerk_safety_correction_limit(
    object_type: int,
    config: BatchTrajectoryConfig,
) -> float:
    if object_type == 2:
        return config.pedestrian_linear_jerk_safety_correction_p95_m
    if object_type == 3:
        return config.cyclist_linear_jerk_safety_correction_p95_m
    return config.vehicle_linear_jerk_safety_correction_p95_m


def _apply_linear_jerk_safety(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    time: np.ndarray,
    object_type: int,
    config: BatchTrajectoryConfig,
    matched_frame_validity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Apply a bounded fallback when full or matched-support jerk is extreme."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    matched_support = _jerk_support_from_frame_validity(
        matched_frame_validity,
        len(x),
    )
    linear_jerk, _ = wosac_jerk_features(
        x,
        y,
        z,
        np.zeros(len(x), dtype=float),
        time,
    )
    initial_rms = _safety_jerk_rms(linear_jerk, matched_support)
    target_rms = config.linear_jerk_safety_rms_mps3
    if initial_rms <= target_rms or len(x) < 7:
        return x.copy(), y.copy(), z.copy(), False

    correction_limit = _linear_jerk_safety_correction_limit(
        object_type,
        config,
    )
    reference_x = x.copy()
    reference_y = y.copy()
    reference_z = z.copy()
    base_x = reference_x
    base_y = reference_y
    base_z = reference_z
    current_rms = initial_rms
    target_with_margin = 0.98 * target_rms
    best = None

    for _ in range(3):
        candidate_coordinates = []
        for polynomial_order in (2, 1):
            for window in _safety_smoothing_windows(len(x)):
                candidate_coordinates.append(
                    (
                        _smooth_scalar(
                            base_x,
                            window,
                            polynomial_order,
                            time,
                        ),
                        _smooth_scalar(
                            base_y,
                            window,
                            polynomial_order,
                            time,
                        ),
                        _smooth_scalar(
                            base_z,
                            window,
                            polynomial_order,
                            time,
                        ),
                    )
                )
            candidate_coordinates.append(
                (
                    _global_time_polynomial_fit(
                        base_x,
                        time,
                        polynomial_order,
                    ),
                    _global_time_polynomial_fit(
                        base_y,
                        time,
                        polynomial_order,
                    ),
                    _global_time_polynomial_fit(
                        base_z,
                        time,
                        polynomial_order,
                    ),
                )
            )

        candidates = []
        for candidate_x, candidate_y, candidate_z in candidate_coordinates:
            correction = np.sqrt(
                (candidate_x - reference_x) ** 2
                + (candidate_y - reference_y) ** 2
                + (candidate_z - reference_z) ** 2
            )
            correction_p95 = float(np.percentile(correction, 95))
            if (
                correction_p95 > correction_limit
                or float(np.max(correction)) > 2.0 * correction_limit
            ):
                continue
            candidate_jerk, _ = wosac_jerk_features(
                candidate_x,
                candidate_y,
                candidate_z,
                np.zeros(len(x), dtype=float),
                time,
            )
            candidate_rms = _safety_jerk_rms(
                candidate_jerk,
                matched_support,
            )
            if candidate_rms >= current_rms:
                continue
            candidates.append(
                (
                    candidate_rms,
                    correction_p95,
                    candidate_x,
                    candidate_y,
                    candidate_z,
                )
            )

        if not candidates:
            break
        within_target = [
            row for row in candidates if row[0] <= target_with_margin
        ]
        if within_target:
            selected = min(within_target, key=lambda row: (row[1], row[0]))
        else:
            selected = min(candidates, key=lambda row: (row[0], row[1]))
        if best is None or (selected[0], selected[1]) < (best[0], best[1]):
            best = selected
        if selected[0] <= target_with_margin:
            break
        current_rms = selected[0]
        base_x, base_y, base_z = selected[2:5]

    if best is None:
        return x.copy(), y.copy(), z.copy(), False
    return best[2], best[3], best[4], True


def _apply_angular_jerk_safety(
    heading: np.ndarray,
    time: np.ndarray,
    config: BatchTrajectoryConfig,
    matched_frame_validity: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Smooth a heading when full or matched-support jerk is extreme."""
    heading = wrap_angle(np.asarray(heading, dtype=float))
    matched_support = _jerk_support_from_frame_validity(
        matched_frame_validity,
        len(heading),
    )
    _, angular_jerk = wosac_jerk_features(
        np.zeros(len(heading), dtype=float),
        np.zeros(len(heading), dtype=float),
        np.zeros(len(heading), dtype=float),
        np.unwrap(heading),
        time,
    )
    initial_rms = _safety_jerk_rms(angular_jerk, matched_support)
    target_rms = config.angular_jerk_safety_rms_radps3
    if initial_rms <= target_rms or len(heading) < 7:
        return heading.copy(), False

    reference_heading = heading.copy()
    base_unwrapped = np.unwrap(reference_heading)
    current_rms = initial_rms
    target_with_margin = 0.98 * target_rms
    best = None

    for _ in range(3):
        candidate_headings = []
        for polynomial_order in (2, 1):
            for window in _safety_smoothing_windows(len(heading)):
                candidate_headings.append(
                    wrap_angle(
                        _smooth_scalar(
                            base_unwrapped,
                            window,
                            polynomial_order,
                            time,
                        )
                    )
                )
            candidate_headings.append(
                wrap_angle(
                    _global_time_polynomial_fit(
                        base_unwrapped,
                        time,
                        polynomial_order,
                    )
                )
            )

        candidates = []
        for candidate in candidate_headings:
            correction = np.abs(angle_diff(candidate, reference_heading))
            correction_p95 = float(np.percentile(correction, 95))
            if correction_p95 > config.angular_jerk_safety_correction_p95_rad:
                continue
            _, candidate_jerk = wosac_jerk_features(
                np.zeros(len(candidate), dtype=float),
                np.zeros(len(candidate), dtype=float),
                np.zeros(len(candidate), dtype=float),
                np.unwrap(candidate),
                time,
            )
            candidate_rms = _safety_jerk_rms(
                candidate_jerk,
                matched_support,
            )
            if candidate_rms >= current_rms:
                continue
            candidates.append((candidate_rms, correction_p95, candidate))

        if not candidates:
            break
        within_target = [
            row for row in candidates if row[0] <= target_with_margin
        ]
        if within_target:
            selected = min(within_target, key=lambda row: (row[1], row[0]))
        else:
            selected = min(candidates, key=lambda row: (row[0], row[1]))
        if best is None or (selected[0], selected[1]) < (best[0], best[1]):
            best = selected
        if selected[0] <= target_with_margin:
            break
        current_rms = selected[0]
        base_unwrapped = np.unwrap(selected[2])

    if best is None:
        return heading.copy(), False
    return best[2], True


def _position_jacobian_sparsity(count: int):
    centered_jerk_count = max(0, count - 6)
    adjacent_jerk_count = max(0, count - 3)
    residual_count = 2 * count + 3 * centered_jerk_count + 2 * adjacent_jerk_count
    sparsity = lil_matrix((residual_count, 2 * count), dtype=np.int8)
    for index in range(count):
        sparsity[index, index] = 1
        sparsity[count + index, count + index] = 1

    linear_offset = 2 * count
    vector_x_offset = linear_offset + centered_jerk_count
    vector_y_offset = vector_x_offset + centered_jerk_count
    for index in range(centered_jerk_count):
        for frame in (index, index + 2, index + 4, index + 6):
            sparsity[linear_offset + index, frame] = 1
            sparsity[linear_offset + index, count + frame] = 1
            sparsity[vector_x_offset + index, frame] = 1
            sparsity[vector_y_offset + index, count + frame] = 1

    adjacent_x_offset = vector_y_offset + centered_jerk_count
    adjacent_y_offset = adjacent_x_offset + adjacent_jerk_count
    for index in range(adjacent_jerk_count):
        for frame in (index, index + 1, index + 2, index + 3):
            sparsity[adjacent_x_offset + index, frame] = 1
            sparsity[adjacent_y_offset + index, count + frame] = 1
    return sparsity.tocsr()


def _heading_jacobian_sparsity(count: int):
    heading_rate_count = max(0, count - 2)
    centered_jerk_count = max(0, count - 6)
    adjacent_jerk_count = max(0, count - 3)
    residual_count = (
        2 * count
        + heading_rate_count
        + centered_jerk_count
        + adjacent_jerk_count
    )
    sparsity = lil_matrix((residual_count, count), dtype=np.int8)
    for index in range(count):
        sparsity[index, index] = 1
        sparsity[count + index, index] = 1
    heading_rate_offset = 2 * count
    for index in range(heading_rate_count):
        sparsity[heading_rate_offset + index, index] = 1
        sparsity[heading_rate_offset + index, index + 2] = 1
    angular_offset = heading_rate_offset + heading_rate_count
    for index in range(centered_jerk_count):
        for frame in (index, index + 2, index + 4, index + 6):
            sparsity[angular_offset + index, frame] = 1

    adjacent_angular_offset = angular_offset + centered_jerk_count
    for index in range(adjacent_jerk_count):
        for frame in (index, index + 1, index + 2, index + 3):
            sparsity[adjacent_angular_offset + index, frame] = 1
    return sparsity.tocsr()


def _robust_observation_residual(
    normalized_residual: np.ndarray,
    delta: float,
) -> np.ndarray:
    """Encode pseudo-Huber observation cost while leaving jerk quadratic."""
    residual = np.asarray(normalized_residual, dtype=float)
    if delta <= 0.0:
        return residual
    ratio = residual / delta
    return residual * np.sqrt(2.0 / (np.sqrt(1.0 + ratio**2) + 1.0))


def _observation_scale(
    trusted: np.ndarray,
    config: BatchTrajectoryConfig,
) -> np.ndarray:
    trusted = np.asarray(trusted, dtype=bool)
    weight = np.where(trusted, 1.0, config.filled_observation_weight).astype(float)
    if len(weight):
        if trusted[0]:
            weight[0] *= config.endpoint_observation_weight
        if trusted[-1]:
            weight[-1] *= config.endpoint_observation_weight
    return np.sqrt(weight)


def _solve_position_stage(
    x_observation: np.ndarray,
    y_observation: np.ndarray,
    z_fixed: np.ndarray,
    x_trusted: np.ndarray,
    y_trusted: np.ndarray,
    time: np.ndarray,
    object_type: int,
    config: BatchTrajectoryConfig,
) -> _PositionSolution:
    count = len(x_observation)
    x_observation_scale = _observation_scale(x_trusted, config)
    y_observation_scale = _observation_scale(y_trusted, config)
    regularization_scale = _regularization_scale(object_type, config)
    regularization_scale *= _short_segment_linear_regularization_scale(
        count,
        config,
    )
    linear_scale = (
        np.sqrt(config.linear_jerk_weight * regularization_scale)
        / config.linear_jerk_scale_mps3
    )
    vector_scale = (
        np.sqrt(
            config.linear_jerk_weight
            * config.planar_vector_jerk_weight
            * regularization_scale
        )
        / config.planar_vector_jerk_scale_mps3
    )
    adjacent_vector_scale = (
        np.sqrt(
            config.linear_jerk_weight
            * config.adjacent_planar_jerk_weight
            * regularization_scale
        )
        / config.planar_vector_jerk_scale_mps3
    )
    initial = np.concatenate((x_observation, y_observation))

    def residual(decision: np.ndarray) -> np.ndarray:
        x = decision[:count]
        y = decision[count:]
        linear_jerk, _ = wosac_jerk_features(
            x,
            y,
            z_fixed,
            np.zeros(count, dtype=float),
            time,
        )
        vector_jerk_x, vector_jerk_y = _centered_planar_vector_jerk(
            x,
            y,
            time,
        )
        adjacent_jerk_x = _adjacent_scalar_jerk(x, time)
        adjacent_jerk_y = _adjacent_scalar_jerk(y, time)
        x_residual = x_observation_scale * (
            (x - x_observation) / config.position_observation_scale_m
        )
        y_residual = y_observation_scale * (
            (y - y_observation) / config.position_observation_scale_m
        )
        return np.concatenate(
            (
                _robust_observation_residual(
                    x_residual,
                    config.robust_observation_delta,
                ),
                _robust_observation_residual(
                    y_residual,
                    config.robust_observation_delta,
                ),
                linear_scale * linear_jerk,
                vector_scale * vector_jerk_x,
                vector_scale * vector_jerk_y,
                adjacent_vector_scale * adjacent_jerk_x,
                adjacent_vector_scale * adjacent_jerk_y,
            )
        )

    x_bound = np.where(
        x_trusted,
        config.max_observed_position_shift_m,
        config.max_filled_position_shift_m,
    )
    y_bound = np.where(
        y_trusted,
        config.max_observed_position_shift_m,
        config.max_filled_position_shift_m,
    )
    lower = np.concatenate(
        (x_observation - x_bound, y_observation - y_bound)
    )
    upper = np.concatenate(
        (x_observation + x_bound, y_observation + y_bound)
    )
    initial_cost = 0.5 * float(np.sum(residual(initial) ** 2))
    try:
        solution = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            jac_sparsity=_position_jacobian_sparsity(count),
            loss="linear",
            x_scale="jac",
            max_nfev=config.max_nfev,
            ftol=config.ftol,
            xtol=config.xtol,
            gtol=config.gtol,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return _PositionSolution(
            x=x_observation.copy(),
            y=y_observation.copy(),
            optimized=False,
            limited=False,
            solver_failed=True,
        )

    usable = (
        np.all(np.isfinite(solution.x))
        and np.isfinite(solution.cost)
        and float(solution.cost) <= initial_cost * (1.0 + 1e-6)
    )
    if not usable:
        return _PositionSolution(
            x=x_observation.copy(),
            y=y_observation.copy(),
            optimized=False,
            limited=False,
            solver_failed=True,
        )

    x = solution.x[:count]
    y = solution.x[count:]
    limited = False
    position_trusted = np.asarray(x_trusted) & np.asarray(y_trusted)
    if np.any(position_trusted):
        correction = np.hypot(
            x[position_trusted] - x_observation[position_trusted],
            y[position_trusted] - y_observation[position_trusted],
        )
        correction_p95 = float(np.percentile(correction, 95))
        if correction_p95 > config.max_trusted_position_correction_p95_m:
            fraction = config.max_trusted_position_correction_p95_m / correction_p95
            x = x_observation + fraction * (x - x_observation)
            y = y_observation + fraction * (y - y_observation)
            limited = True
    return _PositionSolution(
        x=x,
        y=y,
        optimized=True,
        limited=limited,
        solver_failed=False,
    )


def _nearest_motion_reference(
    motion_heading: np.ndarray,
    motion_reliable: np.ndarray,
    time: np.ndarray,
    max_gap_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.full(len(time), np.nan, dtype=float)
    supported = np.zeros(len(time), dtype=bool)
    moving_index = np.flatnonzero(motion_reliable)
    if len(moving_index) == 0:
        return reference, supported

    motion_gap = np.abs(time[:, None] - time[moving_index][None, :])
    nearest_position = np.argmin(motion_gap, axis=1)
    nearest_gap = motion_gap[np.arange(len(time)), nearest_position]
    supported = nearest_gap <= max_gap_s
    nearest_index = moving_index[nearest_position]
    reference[supported] = motion_heading[nearest_index[supported]]
    return reference, supported


def _fallback_motion_heading(
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    velocity_x = _gradient(x, time)
    velocity_y = _gradient(y, time)
    motion_heading = np.arctan2(velocity_y, velocity_x)
    if np.isfinite(motion_heading).any():
        return np.unwrap(motion_heading)
    return np.zeros(len(x), dtype=float)


def _raw_vehicle_heading_is_smooth(
    heading: np.ndarray,
    valid: np.ndarray,
    time: np.ndarray,
    max_yaw_rate_radps: float,
) -> bool:
    """Return whether adjacent raw anchors form a physically continuous turn."""
    index = np.flatnonzero(np.asarray(valid, dtype=bool) & np.isfinite(heading))
    if len(index) < 2:
        return True
    elapsed = np.diff(time[index])
    usable = np.isfinite(elapsed) & (elapsed > 0.0)
    if not np.any(usable):
        return True
    heading_change = np.abs(angle_diff(heading[index[1:]], heading[index[:-1]]))
    return bool(
        np.all(heading_change[usable] / elapsed[usable] <= max_yaw_rate_radps)
    )


def _build_vehicle_velocity_heading_observation(
    x_optimized: np.ndarray,
    y_optimized: np.ndarray,
    raw_heading: np.ndarray,
    raw_heading_valid: np.ndarray,
    prefilter_heading: np.ndarray | None,
    time: np.ndarray,
    batch_config: BatchTrajectoryConfig,
    motion_config,
) -> _HeadingObservation:
    """Build a reverse-safe vehicle target from sustained optimized motion."""
    raw_clean = np.asarray(raw_heading, dtype=float).copy()
    raw_heading_valid = (
        np.asarray(raw_heading_valid, dtype=bool) & np.isfinite(raw_clean)
    )
    if prefilter_heading is None:
        prefilter_clean = np.full(len(raw_clean), np.nan, dtype=float)
    else:
        prefilter_clean = np.asarray(prefilter_heading, dtype=float)
        if prefilter_clean.shape != raw_clean.shape:
            prefilter_clean = np.full(len(raw_clean), np.nan, dtype=float)
    prefilter_valid = np.isfinite(prefilter_clean)

    short_flip = _remove_short_heading_flips(
        raw_clean,
        raw_heading_valid,
        motion_config.heading_flip_deg,
        motion_config.heading_flip_support_deg,
        motion_config.heading_flip_max_run,
    )
    raw_clean[short_flip] = np.nan
    clean_valid = raw_heading_valid & np.isfinite(raw_clean)

    short_excursion = _remove_short_heading_flips(
        raw_clean,
        clean_valid,
        batch_config.vehicle_short_heading_excursion_deg,
        motion_config.heading_flip_support_deg,
        motion_config.heading_flip_max_run,
    )
    raw_clean[short_excursion] = np.nan
    clean_valid = raw_heading_valid & np.isfinite(raw_clean)
    rejected = short_flip | short_excursion

    if _raw_vehicle_heading_is_smooth(
        raw_clean,
        clean_valid,
        time,
        motion_config.endpoint_heading_max_yaw_rate_radps,
    ):
        heading = _interpolate_heading(raw_clean, clean_valid)
        if heading is None:
            heading = _fallback_motion_heading(x_optimized, y_optimized, time)
        return _HeadingObservation(
            heading=wrap_angle(heading),
            trusted=clean_valid,
            motion_reference=np.full(len(time), np.nan, dtype=float),
            motion_reliable=np.zeros(len(time), dtype=bool),
            outliers=rejected,
        )

    velocity_x = _smooth_scalar(
        _gradient(x_optimized, time),
        motion_config.motion_heading_velocity_window,
        motion_config.polynomial_order,
        time,
    )
    velocity_y = _smooth_scalar(
        _gradient(y_optimized, time),
        motion_config.motion_heading_velocity_window,
        motion_config.polynomial_order,
        time,
    )
    speed = np.hypot(velocity_x, velocity_y)
    motion_heading = np.arctan2(velocity_y, velocity_x)
    motion_supported = _keep_sustained_runs(
        np.isfinite(speed)
        & np.isfinite(motion_heading)
        & (speed >= motion_config.motion_heading_min_speed_mps),
        motion_config.motion_heading_min_run,
    )

    base_reliable = clean_valid & motion_supported
    longitudinal_speed = (
        velocity_x * np.cos(raw_clean) + velocity_y * np.sin(raw_clean)
    )
    reverse_protected = (
        base_reliable
        & np.isfinite(longitudinal_speed)
        & (
            longitudinal_speed
            < -batch_config.vehicle_reverse_speed_tolerance_mps
        )
    )
    reliable_forward = base_reliable & ~reverse_protected
    residual = np.abs(angle_diff(raw_clean, motion_heading))
    velocity_outlier = reliable_forward & (
        np.rad2deg(residual)
        > batch_config.vehicle_velocity_heading_outlier_deg
    )

    corrected_anchor = np.full(len(time), np.nan, dtype=float)
    corrected_anchor[reliable_forward & ~velocity_outlier] = raw_clean[
        reliable_forward & ~velocity_outlier
    ]
    corrected_anchor[velocity_outlier] = motion_heading[velocity_outlier]
    corrected_mask = np.isfinite(corrected_anchor)
    prefilter_target = _interpolate_heading(prefilter_clean, prefilter_valid)

    if (
        int(np.sum(corrected_mask))
        >= batch_config.vehicle_min_forward_heading_anchors
    ):
        candidate_heading = _interpolate_heading(corrected_anchor, corrected_mask)
        candidate_heading = _smooth_scalar(
            candidate_heading,
            batch_config.vehicle_heading_target_window,
            1,
            time,
        )
        use_prefilter = False
        if prefilter_target is not None:
            disagreement_deg = np.rad2deg(
                np.abs(angle_diff(candidate_heading, prefilter_target))
            )
            candidate_turn_deg = np.rad2deg(
                np.sum(
                    np.abs(
                        angle_diff(candidate_heading[1:], candidate_heading[:-1])
                    )
                )
            )
            prefilter_turn_deg = np.rad2deg(
                np.sum(
                    np.abs(
                        angle_diff(prefilter_target[1:], prefilter_target[:-1])
                    )
                )
            )
            use_prefilter = bool(
                _percentile(disagreement_deg[prefilter_valid])
                > batch_config.vehicle_prefilter_disagreement_p95_deg
                and candidate_turn_deg - prefilter_turn_deg
                > batch_config.vehicle_prefilter_excess_turn_deg
            )
        if use_prefilter:
            heading = prefilter_target
            trusted = prefilter_valid
            motion_reliable = np.zeros(len(time), dtype=bool)
        else:
            heading = candidate_heading
            trusted = corrected_mask
            motion_reliable = velocity_outlier
        rejected |= velocity_outlier
    else:
        heading = prefilter_target
        trusted = prefilter_valid
        if heading is None:
            heading = _interpolate_heading(raw_clean, clean_valid)
            trusted = clean_valid
        if heading is None:
            heading = _fallback_motion_heading(x_optimized, y_optimized, time)
            trusted = np.zeros(len(time), dtype=bool)
        motion_reliable = np.zeros(len(time), dtype=bool)

    return _HeadingObservation(
        heading=wrap_angle(heading),
        trusted=trusted,
        motion_reference=motion_heading,
        motion_reliable=motion_reliable,
        outliers=rejected,
    )


def _build_heading_observation_after_position(
    x_optimized: np.ndarray,
    y_optimized: np.ndarray,
    raw_heading: np.ndarray,
    raw_heading_valid: np.ndarray,
    prefilter_heading: np.ndarray | None,
    time: np.ndarray,
    object_type: int,
    batch_config: BatchTrajectoryConfig,
    motion_config,
) -> _HeadingObservation:
    """Create theta_bar after position optimization using raw heading anchors.

    All motion-heading evidence is derived from ``x_optimized`` and
    ``y_optimized``. Vehicles use sustained forward evidence with explicit
    reverse protection; other object types retain the generic anchor pipeline.
    """
    count = len(raw_heading)
    raw_heading = np.asarray(raw_heading, dtype=float)
    raw_heading_valid = np.asarray(raw_heading_valid, dtype=bool) & np.isfinite(
        raw_heading
    )
    position_supported = np.ones(count, dtype=bool)

    if object_type == 1:
        return _build_vehicle_velocity_heading_observation(
            x_optimized,
            y_optimized,
            raw_heading,
            raw_heading_valid,
            prefilter_heading,
            time,
            batch_config,
            motion_config,
        )

    heading_outliers = _remove_short_heading_flips(
        raw_heading,
        raw_heading_valid,
        motion_config.heading_flip_deg,
        motion_config.heading_flip_support_deg,
        motion_config.heading_flip_max_run,
    )
    heading_outliers |= _interpolation_residual_outliers(
        raw_heading,
        raw_heading_valid & ~heading_outliers,
        time,
        np.deg2rad(motion_config.heading_outlier_floor_deg),
        motion_config.outlier_sigma,
        motion_config.outlier_support_ratio,
        angular=True,
    )
    heading_outliers |= _isolated_endpoint_heading_outliers(
        x_optimized,
        y_optimized,
        raw_heading,
        raw_heading_valid,
        time,
        motion_config,
    )
    if object_type in (1, 3):
        heading_outliers |= _vehicle_endpoint_heading_outliers(
            x_optimized,
            y_optimized,
            raw_heading,
            raw_heading_valid & ~heading_outliers,
            time,
            motion_config,
        )

    heading_for_motion = raw_heading.copy()
    heading_branch_corrections = np.zeros(count, dtype=bool)
    if object_type in (1, 3):
        velocity_x = _gradient(x_optimized, time)
        velocity_y = _gradient(y_optimized, time)
        speed = np.hypot(velocity_x, velocity_y)
        preliminary_anchor = raw_heading_valid & ~heading_outliers
        preferred_anchor = _preferred_vehicle_heading_anchor(
            preliminary_anchor,
            position_supported,
            speed,
            time,
            motion_config,
        )
        branch_heading, heading_branch_corrections = _interpolate_vehicle_heading(
            raw_heading,
            preliminary_anchor,
            x_optimized,
            y_optimized,
            time,
            preferred_anchor,
            motion_config,
        )
        if branch_heading is not None:
            heading_for_motion[preliminary_anchor] = wrap_angle(
                branch_heading[preliminary_anchor]
            )

    motion_heading, motion_reliable = _motion_body_heading_observation(
        x_optimized,
        y_optimized,
        heading_for_motion,
        raw_heading_valid & ~heading_outliers,
        position_supported,
        time,
        object_type,
        motion_config,
    )
    motion_reference, motion_supported = _nearest_motion_reference(
        motion_heading,
        motion_reliable,
        time,
        motion_config.motion_heading_support_max_gap_s,
    )
    motion_reference_unwrapped = _interpolate_heading(
        motion_reference,
        motion_supported,
    )

    heading_values = heading_for_motion.copy()
    heading_values[heading_outliers | ~raw_heading_valid] = np.nan
    trusted_anchor = raw_heading_valid & ~heading_outliers

    if object_type in (2, 3):
        heading_unwrapped = _interpolate_pi_ambiguous_heading(
            heading_values,
            trusted_anchor,
        )
        if object_type == 2:
            if _supports_detailed_pedestrian_heading(
                heading_for_motion,
                trusted_anchor,
                motion_config,
            ):
                heading_window = motion_config.pedestrian_detailed_heading_window
                heading_order = motion_config.pedestrian_detailed_heading_polynomial_order
            else:
                heading_window = motion_config.pedestrian_heading_window
                heading_order = motion_config.pedestrian_heading_polynomial_order
        else:
            heading_window = motion_config.heading_window
            heading_order = motion_config.polynomial_order
    else:
        heading_unwrapped = _interpolate_heading(
            heading_values,
            trusted_anchor,
        )
        heading_window = motion_config.heading_window
        heading_order = motion_config.polynomial_order

    if heading_unwrapped is None:
        heading_unwrapped = motion_reference_unwrapped
    if heading_unwrapped is None:
        heading_unwrapped = _fallback_motion_heading(x_optimized, y_optimized, time)

    heading_outliers |= heading_branch_corrections
    quality_anchor = trusted_anchor & ~heading_branch_corrections
    heading_filtered = None
    for window in _smoothing_windows(len(heading_unwrapped), heading_window, heading_order):
        candidate = wrap_angle(
            _smooth_scalar(
                heading_unwrapped,
                window,
                heading_order,
                time,
            )
        )
        heading_correction = np.abs(angle_diff(candidate, raw_heading))
        if object_type in (2, 3):
            opposite_correction = np.abs(
                angle_diff(candidate, wrap_angle(raw_heading + np.pi))
            )
            heading_correction = np.minimum(heading_correction, opposite_correction)
        trusted_correction = np.rad2deg(heading_correction)[quality_anchor]
        correction_limit = (
            max(90.0, motion_config.max_trusted_heading_correction_deg)
            if object_type in (2, 3)
            else motion_config.max_trusted_heading_correction_deg
        )
        if _percentile(trusted_correction) <= correction_limit:
            heading_filtered = candidate
            break
    if heading_filtered is None:
        heading_filtered = wrap_angle(heading_unwrapped)

    heading_motion_reliable = (
        motion_reliable
        & np.isfinite(motion_heading)
        & ~quality_anchor
    )
    return _HeadingObservation(
        heading=wrap_angle(heading_filtered),
        trusted=quality_anchor,
        motion_reference=motion_heading,
        motion_reliable=heading_motion_reliable,
        outliers=heading_outliers,
    )


def _solve_heading_stage(
    heading_observation: np.ndarray,
    heading_trusted: np.ndarray,
    motion_reference: np.ndarray,
    motion_reliable: np.ndarray,
    time: np.ndarray,
    object_type: int,
    config: BatchTrajectoryConfig,
) -> _HeadingSolution:
    count = len(heading_observation)
    heading_unwrapped = np.unwrap(np.asarray(heading_observation, dtype=float))
    observation_scale = _observation_scale(heading_trusted, config)
    regularization_scale = _regularization_scale(object_type, config)
    heading_jerk_object_scale = (
        config.pedestrian_heading_jerk_weight_scale
        if object_type == 2
        else 1.0
    )
    angular_scale = (
        np.sqrt(
            config.angular_jerk_weight
            * regularization_scale
            * heading_jerk_object_scale
        )
        / config.angular_jerk_scale_radps3
    )
    adjacent_angular_scale = (
        np.sqrt(
            config.angular_jerk_weight
            * config.adjacent_angular_jerk_weight
            * regularization_scale
            * heading_jerk_object_scale
        )
        / config.angular_jerk_scale_radps3
    )
    heading_rate_object_scale = 0.0
    if object_type == 1:
        heading_rate_object_scale = config.vehicle_heading_rate_weight_scale
    elif object_type == 2:
        heading_rate_object_scale = config.pedestrian_heading_rate_weight_scale
    elif object_type == 3:
        heading_rate_object_scale = config.cyclist_heading_rate_weight_scale
    heading_rate_scale = (
        np.sqrt(config.heading_rate_observation_weight * heading_rate_object_scale)
        / config.heading_rate_observation_scale_radps
    )
    heading_rate_duration = time[2:] - time[:-2]
    observed_heading_rate = (
        heading_unwrapped[2:] - heading_unwrapped[:-2]
    ) / heading_rate_duration
    motion_reliable = np.asarray(motion_reliable, dtype=bool)
    motion_aligned = heading_unwrapped.copy()
    if np.any(motion_reliable):
        motion_aligned[motion_reliable] = (
            heading_unwrapped[motion_reliable]
            + wrap_angle(
                motion_reference[motion_reliable]
                - heading_observation[motion_reliable]
            )
        )

    def residual(candidate: np.ndarray) -> np.ndarray:
        _, angular_jerk = wosac_jerk_features(
            np.zeros(count, dtype=float),
            np.zeros(count, dtype=float),
            np.zeros(count, dtype=float),
            candidate,
            time,
        )
        adjacent_angular_jerk = _adjacent_scalar_jerk(candidate, time)
        observation_residual = observation_scale * (
            (candidate - heading_unwrapped) / config.heading_observation_scale_rad
        )
        motion_residual = np.zeros(count, dtype=float)
        if np.any(motion_reliable):
            motion_residual[motion_reliable] = (
                np.sqrt(config.vehicle_motion_heading_weight)
                * (candidate[motion_reliable] - motion_aligned[motion_reliable])
                / config.motion_heading_scale_rad
            )
        candidate_heading_rate = (
            candidate[2:] - candidate[:-2]
        ) / heading_rate_duration
        heading_rate_residual = heading_rate_scale * (
            candidate_heading_rate - observed_heading_rate
        )
        return np.concatenate(
            (
                _robust_observation_residual(
                    observation_residual,
                    config.robust_observation_delta,
                ),
                _robust_observation_residual(
                    motion_residual,
                    config.robust_observation_delta,
                ),
                heading_rate_residual,
                angular_scale * angular_jerk,
                adjacent_angular_scale * adjacent_angular_jerk,
            )
        )

    heading_bound = np.where(
        heading_trusted,
        config.max_observed_heading_shift_rad,
        config.max_filled_heading_shift_rad,
    )
    lower = heading_unwrapped - heading_bound
    upper = heading_unwrapped + heading_bound
    initial_cost = 0.5 * float(np.sum(residual(heading_unwrapped) ** 2))
    try:
        solution = least_squares(
            residual,
            heading_unwrapped,
            bounds=(lower, upper),
            jac_sparsity=_heading_jacobian_sparsity(count),
            loss="linear",
            x_scale="jac",
            max_nfev=config.max_nfev,
            ftol=config.ftol,
            xtol=config.xtol,
            gtol=config.gtol,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return _HeadingSolution(
            heading=wrap_angle(heading_unwrapped),
            optimized=False,
            limited=False,
            solver_failed=True,
        )

    usable = (
        np.all(np.isfinite(solution.x))
        and np.isfinite(solution.cost)
        and float(solution.cost) <= initial_cost * (1.0 + 1e-6)
    )
    if not usable:
        return _HeadingSolution(
            heading=wrap_angle(heading_unwrapped),
            optimized=False,
            limited=False,
            solver_failed=True,
        )

    heading = solution.x
    limited = False
    if np.any(heading_trusted):
        correction = np.abs(
            wrap_angle(heading[heading_trusted] - heading_unwrapped[heading_trusted])
        )
        correction_p95 = float(np.percentile(correction, 95))
        if correction_p95 > config.max_trusted_heading_correction_p95_rad:
            fraction = config.max_trusted_heading_correction_p95_rad / correction_p95
            heading = heading_unwrapped + fraction * (heading - heading_unwrapped)
            limited = True
    return _HeadingSolution(
        heading=wrap_angle(heading),
        optimized=True,
        limited=limited,
        solver_failed=False,
    )


def _optimize_segment(
    x_observation: np.ndarray,
    y_observation: np.ndarray,
    z_fixed: np.ndarray,
    raw_heading: np.ndarray,
    raw_heading_valid: np.ndarray,
    original_observed: np.ndarray,
    seconds_per_step: float,
    object_type: int,
    config: BatchTrajectoryConfig,
    *,
    position_trusted: np.ndarray | None = None,
    x_trusted: np.ndarray | None = None,
    y_trusted: np.ndarray | None = None,
    heading_trusted: np.ndarray | None = None,
    prefilter_heading: np.ndarray | None = None,
    time: np.ndarray | None = None,
    motion_config=None,
) -> _SegmentSolution | None:
    count = len(x_observation)
    if count < config.min_optimization_frames:
        return None
    original_observed = np.asarray(original_observed, dtype=bool)
    if position_trusted is None:
        position_trusted = original_observed
    else:
        position_trusted = np.asarray(position_trusted, dtype=bool)
    if x_trusted is None:
        x_trusted = position_trusted
    else:
        x_trusted = np.asarray(x_trusted, dtype=bool)
    if y_trusted is None:
        y_trusted = position_trusted
    else:
        y_trusted = np.asarray(y_trusted, dtype=bool)
    raw_heading = np.asarray(raw_heading, dtype=float)
    raw_heading_valid = (
        np.asarray(raw_heading_valid, dtype=bool)
        & original_observed
        & np.isfinite(raw_heading)
    )
    if heading_trusted is not None:
        raw_heading_valid &= np.asarray(heading_trusted, dtype=bool)

    if time is None:
        time = np.arange(count, dtype=float) * seconds_per_step
    else:
        time = _kinematic_time_axis(count, np.asarray(time, dtype=float))

    position_solution = _solve_position_stage(
        x_observation,
        y_observation,
        z_fixed,
        x_trusted,
        y_trusted,
        time,
        object_type,
        config,
    )
    safe_x, safe_y, safe_z, _ = _apply_linear_jerk_safety(
        position_solution.x,
        position_solution.y,
        z_fixed,
        time,
        object_type,
        config,
        matched_frame_validity=original_observed,
    )
    position_solution.x = safe_x
    position_solution.y = safe_y

    if motion_config is None:
        motion_config = config_for_filter_strength("strong")
    heading_observation = _build_heading_observation_after_position(
        position_solution.x,
        position_solution.y,
        raw_heading,
        raw_heading_valid,
        prefilter_heading,
        np.asarray(time, dtype=float),
        object_type,
        config,
        motion_config,
    )
    heading_solution = _solve_heading_stage(
        heading_observation.heading,
        heading_observation.trusted,
        heading_observation.motion_reference,
        heading_observation.motion_reliable,
        time,
        object_type,
        config,
    )
    safe_heading, _ = _apply_angular_jerk_safety(
        heading_solution.heading,
        time,
        config,
        matched_frame_validity=raw_heading_valid,
    )
    heading_solution.heading = safe_heading

    x = position_solution.x
    y = position_solution.y
    z = safe_z
    heading = heading_solution.heading
    heading_before = np.unwrap(np.asarray(heading_observation.heading, dtype=float))
    heading_after = np.unwrap(np.asarray(heading, dtype=float))

    linear_before, angular_before = wosac_acceleration_features(
        x_observation,
        y_observation,
        z_fixed,
        heading_before,
        time,
    )
    linear_after, angular_after = wosac_acceleration_features(
        x,
        y,
        z,
        heading_after,
        time,
    )
    linear_jerk_before, angular_jerk_before = wosac_jerk_features(
        x_observation,
        y_observation,
        z_fixed,
        heading_before,
        time,
    )
    linear_jerk_after, angular_jerk_after = wosac_jerk_features(
        x,
        y,
        z,
        heading_after,
        time,
    )
    return _SegmentSolution(
        x=x,
        y=y,
        z=z,
        heading=wrap_angle(heading),
        linear_acceleration_before=linear_before,
        linear_acceleration_after=linear_after,
        angular_acceleration_before=angular_before,
        angular_acceleration_after=angular_after,
        linear_jerk_before=linear_jerk_before,
        linear_jerk_after=linear_jerk_after,
        angular_jerk_before=angular_jerk_before,
        angular_jerk_after=angular_jerk_after,
        position_optimized=position_solution.optimized,
        heading_optimized=heading_solution.optimized,
        position_limited=position_solution.limited,
        heading_limited=heading_solution.limited,
        position_solver_failed=position_solution.solver_failed,
        heading_solver_failed=heading_solution.solver_failed,
    )


def optimize_track(
    original_track,
    reconstructed_track,
    timestamps: Iterable[float],
    config: BatchTrajectoryConfig | None = None,
    motion_config=None,
) -> TrackBatchResult:
    config = config or BatchTrajectoryConfig()
    result = TrackBatchResult()
    object_type = int(reconstructed_track.object_type)
    if object_type not in config.processed_object_types:
        return result

    timestamps = list(timestamps)
    count = min(
        len(original_track.states),
        len(reconstructed_track.states),
        len(timestamps),
    )
    if count == 0:
        return result
    time = _timestamps_for_count(timestamps, count)
    valid = np.array(
        [reconstructed_track.states[index].valid for index in range(count)],
        dtype=bool,
    )
    original_valid = np.array(
        [original_track.states[index].valid for index in range(count)],
        dtype=bool,
    )

    for start, end in _true_runs(valid):
        result.processed = True
        result.processed_segments += 1
        if end - start < config.min_optimization_frames:
            result.short_segments += 1
            continue
        states = reconstructed_track.states[start:end]
        original_states = original_track.states[start:end]
        x = np.array([state.center_x for state in states], dtype=float)
        y = np.array([state.center_y for state in states], dtype=float)
        z = np.array([state.center_z for state in states], dtype=float)
        prefilter_heading = np.array(
            [state.heading for state in states],
            dtype=float,
        )
        raw_heading = np.array([state.heading for state in original_states], dtype=float)
        original_x = np.array(
            [state.center_x for state in original_states],
            dtype=float,
        )
        original_y = np.array(
            [state.center_y for state in original_states],
            dtype=float,
        )
        segment_original_valid = original_valid[start:end]
        x_trusted = (
            segment_original_valid
            & np.isfinite(original_x)
            & (
                np.abs(x - original_x)
                <= config.prefilter_trusted_position_correction_m
            )
        )
        y_trusted = (
            segment_original_valid
            & np.isfinite(original_y)
            & (
                np.abs(y - original_y)
                <= config.prefilter_trusted_position_correction_m
            )
        )
        position_trusted = x_trusted & y_trusted
        raw_heading_valid = segment_original_valid & np.isfinite(raw_heading)
        solution = _optimize_segment(
            x,
            y,
            z,
            raw_heading,
            raw_heading_valid,
            segment_original_valid,
            _seconds_per_step(time[start:end]),
            object_type,
            config,
            position_trusted=position_trusted,
            x_trusted=x_trusted,
            y_trusted=y_trusted,
            heading_trusted=raw_heading_valid,
            prefilter_heading=prefilter_heading,
            time=time[start:end],
            motion_config=motion_config,
        )
        if solution is None:
            result.failed_segments += 1
            continue

        for local_index, global_index in enumerate(range(start, end)):
            state = reconstructed_track.states[global_index]
            state.center_x = float(solution.x[local_index])
            state.center_y = float(solution.y[local_index])
            state.center_z = float(solution.z[local_index])
            state.heading = float(solution.heading[local_index])

        if solution.position_optimized:
            result.position_optimized_segments += 1
        if solution.heading_optimized:
            result.heading_optimized_segments += 1
        if solution.position_limited:
            result.position_limited_segments += 1
        if solution.heading_limited:
            result.heading_limited_segments += 1
        if solution.position_solver_failed:
            result.position_solver_failures += 1
        if solution.heading_solver_failed:
            result.heading_solver_failures += 1
        if solution.position_solver_failed and solution.heading_solver_failed:
            result.failed_segments += 1
        if solution.position_optimized or solution.heading_optimized:
            result.optimized = True
            result.optimized_segments += 1
            result.optimized_frames += end - start
        result.linear_acceleration_sumsq_before += float(
            np.sum(solution.linear_acceleration_before**2)
        )
        result.linear_acceleration_sumsq_after += float(
            np.sum(solution.linear_acceleration_after**2)
        )
        result.linear_acceleration_count += len(solution.linear_acceleration_after)
        result.angular_acceleration_sumsq_before += float(
            np.sum(solution.angular_acceleration_before**2)
        )
        result.angular_acceleration_sumsq_after += float(
            np.sum(solution.angular_acceleration_after**2)
        )
        result.angular_acceleration_count += len(solution.angular_acceleration_after)
        result.linear_jerk_sumsq_before += float(
            np.sum(solution.linear_jerk_before**2)
        )
        result.linear_jerk_sumsq_after += float(
            np.sum(solution.linear_jerk_after**2)
        )
        result.linear_jerk_count += len(solution.linear_jerk_after)
        result.angular_jerk_sumsq_before += float(
            np.sum(solution.angular_jerk_before**2)
        )
        result.angular_jerk_sumsq_after += float(
            np.sum(solution.angular_jerk_after**2)
        )
        result.angular_jerk_count += len(solution.angular_jerk_after)

    if result.processed:
        for start, end in _true_runs(valid):
            segment_time = time[start:end]
            x = np.array(
                [reconstructed_track.states[index].center_x for index in range(start, end)]
            )
            y = np.array(
                [reconstructed_track.states[index].center_y for index in range(start, end)]
            )
            velocity_x = _gradient(x, segment_time)
            velocity_y = _gradient(y, segment_time)
            for local_index, global_index in enumerate(range(start, end)):
                state = reconstructed_track.states[global_index]
                state.velocity_x = float(velocity_x[local_index])
                state.velocity_y = float(velocity_y[local_index])
    return result


def _rms(sum_of_squares: float, count: int) -> float:
    return float(np.sqrt(sum_of_squares / count)) if count else 0.0


def reconstruct_scenario_agents(
    scenario,
    config: BatchTrajectoryConfig | None = None,
    filter_strength: str = "strong",
    max_gap_frames: int | None = None,
):
    config = config or BatchTrajectoryConfig()
    filter_config = config_for_filter_strength(
        filter_strength,
        max_gap_frames=max_gap_frames,
    )
    reconstructed, filter_stats = filter_scenario_agents(scenario, filter_config)
    stats = BatchReconstructionStats(**asdict(filter_stats))

    linear_before = 0.0
    linear_after = 0.0
    linear_count = 0
    angular_before = 0.0
    angular_after = 0.0
    angular_count = 0
    linear_jerk_before = 0.0
    linear_jerk_after = 0.0
    linear_jerk_count = 0
    angular_jerk_before = 0.0
    angular_jerk_after = 0.0
    angular_jerk_count = 0
    for original_track, reconstructed_track in zip(scenario.tracks, reconstructed.tracks):
        track_result = optimize_track(
            original_track,
            reconstructed_track,
            reconstructed.timestamps_seconds,
            config,
            motion_config=filter_config,
        )
        if track_result.processed:
            stats.processed_tracks += 1
        if track_result.optimized:
            stats.optimized_tracks += 1
        stats.processed_segments += track_result.processed_segments
        stats.optimized_segments += track_result.optimized_segments
        stats.optimized_frames += track_result.optimized_frames
        stats.short_segments += track_result.short_segments
        stats.optimization_failures += track_result.failed_segments
        stats.position_optimized_segments += track_result.position_optimized_segments
        stats.heading_optimized_segments += track_result.heading_optimized_segments
        stats.position_limited_segments += track_result.position_limited_segments
        stats.heading_limited_segments += track_result.heading_limited_segments
        stats.position_solver_failures += track_result.position_solver_failures
        stats.heading_solver_failures += track_result.heading_solver_failures
        linear_before += track_result.linear_acceleration_sumsq_before
        linear_after += track_result.linear_acceleration_sumsq_after
        linear_count += track_result.linear_acceleration_count
        angular_before += track_result.angular_acceleration_sumsq_before
        angular_after += track_result.angular_acceleration_sumsq_after
        angular_count += track_result.angular_acceleration_count
        linear_jerk_before += track_result.linear_jerk_sumsq_before
        linear_jerk_after += track_result.linear_jerk_sumsq_after
        linear_jerk_count += track_result.linear_jerk_count
        angular_jerk_before += track_result.angular_jerk_sumsq_before
        angular_jerk_after += track_result.angular_jerk_sumsq_after
        angular_jerk_count += track_result.angular_jerk_count

    stats.linear_acceleration_rms_before = _rms(linear_before, linear_count)
    stats.linear_acceleration_rms_after = _rms(linear_after, linear_count)
    stats.angular_acceleration_rms_before = _rms(angular_before, angular_count)
    stats.angular_acceleration_rms_after = _rms(angular_after, angular_count)
    stats.linear_jerk_rms_before = _rms(linear_jerk_before, linear_jerk_count)
    stats.linear_jerk_rms_after = _rms(linear_jerk_after, linear_jerk_count)
    stats.angular_jerk_rms_before = _rms(angular_jerk_before, angular_jerk_count)
    stats.angular_jerk_rms_after = _rms(angular_jerk_after, angular_jerk_count)
    return reconstructed, stats
