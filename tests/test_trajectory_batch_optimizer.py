import copy
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np


warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"(vector|keypoint|label|map|dataset|scenario)_pb2",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PB2_ROOT = PROJECT_ROOT / "src" / "smart" / "tokens" / "womd_proto" / "pb2"
if str(PB2_ROOT) not in sys.path:
    sys.path.insert(0, str(PB2_ROOT))

from src.smart.tokens import trajectory_batch_optimizer as batch_optimizer  # noqa: E402
from src.smart.tokens.trajectory_batch_optimizer import (  # noqa: E402
    BatchTrajectoryConfig,
    optimize_track,
    reconstruct_scenario_agents as batch_reconstruct,
    wosac_acceleration_features,
    wosac_jerk_features,
)
from src.smart.tokens.trajectory_filter_reconstructor import (  # noqa: E402
    angle_diff,
    compute_kinematic_features,
    reconstruct_scenario_agents as filter_reconstruct,
)
import scenario_pb2  # noqa: E402


def build_scenario(x, y, heading, valid=None, z=None, object_type=1):
    count = len(x)
    if valid is None:
        valid = np.ones(count, dtype=bool)
    if z is None:
        z = np.zeros(count, dtype=float)
    scenario = scenario_pb2.Scenario()
    scenario.timestamps_seconds.extend(np.arange(count, dtype=float) * 0.1)
    track = scenario.tracks.add()
    track.id = 1
    track.object_type = object_type
    for index in range(count):
        state = track.states.add()
        state.center_x = float(x[index])
        state.center_y = float(y[index])
        state.center_z = float(z[index])
        state.heading = float(heading[index])
        state.length = 4.5
        state.width = 1.8
        state.height = 1.5
        state.valid = bool(valid[index])
    return scenario


class TrajectoryBatchOptimizerTest(unittest.TestCase):
    def test_acceleration_features_match_wosac_centered_differences(self):
        count = 13
        time = np.arange(count, dtype=float) * 0.1
        x = 0.03 * np.arange(count, dtype=float) ** 2
        y = 0.02 * np.sin(np.arange(count, dtype=float))
        z = 0.01 * np.arange(count, dtype=float)
        heading = 0.04 * np.arange(count, dtype=float) ** 2
        positions = np.column_stack((x, y, z))
        valid = np.ones(count, dtype=bool)

        expected = compute_kinematic_features(positions, heading, valid, time)
        linear, angular = wosac_acceleration_features(x, y, z, heading, 0.1)
        linear_jerk, angular_jerk = wosac_jerk_features(x, y, z, heading, 0.1)

        np.testing.assert_allclose(linear, expected.linear_acceleration[2:-2])
        np.testing.assert_allclose(angular, expected.angular_acceleration[2:-2])
        np.testing.assert_allclose(linear_jerk, expected.linear_jerk[3:-3])
        np.testing.assert_allclose(
            angular_jerk,
            expected.angular_jerk[3:-3],
            atol=1e-12,
        )

    def test_features_match_nonuniform_centered_differences(self):
        count = 13
        time = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4]
        )
        x = 2.0 * time + 0.2 * time**2
        y = 0.1 * np.sin(2.0 * time)
        z = 0.03 * time
        heading = 0.2 * time + 0.04 * time**2
        positions = np.column_stack((x, y, z))
        valid = np.ones(count, dtype=bool)

        expected = compute_kinematic_features(positions, heading, valid, time)
        linear, angular = wosac_acceleration_features(x, y, z, heading, time)
        linear_jerk, angular_jerk = wosac_jerk_features(
            x,
            y,
            z,
            heading,
            time,
        )

        np.testing.assert_allclose(linear, expected.linear_acceleration[2:-2])
        np.testing.assert_allclose(angular, expected.angular_acceleration[2:-2])
        np.testing.assert_allclose(linear_jerk, expected.linear_jerk[3:-3])
        np.testing.assert_allclose(angular_jerk, expected.angular_jerk[3:-3])

    def test_batch_uses_actual_nonuniform_timestamps(self):
        time = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4]
        )
        scenario = build_scenario(
            x=15.0 * time,
            y=np.zeros(len(time)),
            heading=0.1 * time,
        )
        del scenario.timestamps_seconds[:]
        scenario.timestamps_seconds.extend(time)

        reconstructed, stats = batch_reconstruct(scenario)
        features = compute_kinematic_features(
            np.array(
                [
                    [state.center_x, state.center_y, state.center_z]
                    for state in reconstructed.tracks[0].states
                ]
            ),
            np.array([state.heading for state in reconstructed.tracks[0].states]),
            np.ones(len(time), dtype=bool),
            time,
        )

        self.assertEqual(stats.optimization_failures, 0)
        self.assertLess(np.nanmax(np.abs(features.linear_jerk)), 1e-4)
        self.assertLess(np.nanmax(np.abs(features.angular_jerk)), 1e-4)

    def test_batch_optimization_suppresses_adjacent_frame_sawtooth(self):
        count = 41
        index = np.arange(count, dtype=float)
        scenario = build_scenario(
            x=0.2 * index + 0.04 * (-1.0) ** index,
            y=0.025 * np.sin(1.7 * index),
            heading=0.015 * index + 0.035 * (-1.0) ** index,
        )

        parity_decoupled, _ = batch_reconstruct(
            scenario,
            BatchTrajectoryConfig(
                adjacent_planar_jerk_weight=0.0,
                adjacent_angular_jerk_weight=0.0,
            ),
        )
        reconstructed, stats = batch_reconstruct(scenario)
        time = np.asarray(scenario.timestamps_seconds, dtype=float)
        parity_decoupled_x = np.array(
            [state.center_x for state in parity_decoupled.tracks[0].states]
        )
        reconstructed_x = np.array(
            [state.center_x for state in reconstructed.tracks[0].states]
        )
        parity_decoupled_heading = np.unwrap(
            np.array([state.heading for state in parity_decoupled.tracks[0].states])
        )
        reconstructed_heading = np.unwrap(
            np.array([state.heading for state in reconstructed.tracks[0].states])
        )

        parity_decoupled_linear_jerk = batch_optimizer._adjacent_scalar_jerk(
            parity_decoupled_x,
            time,
        )
        reconstructed_linear_jerk = batch_optimizer._adjacent_scalar_jerk(
            reconstructed_x,
            time,
        )
        parity_decoupled_angular_jerk = batch_optimizer._adjacent_scalar_jerk(
            parity_decoupled_heading,
            time,
        )
        reconstructed_angular_jerk = batch_optimizer._adjacent_scalar_jerk(
            reconstructed_heading,
            time,
        )

        self.assertEqual(stats.optimized_tracks, 1)
        self.assertEqual(stats.optimization_failures, 0)
        self.assertLess(
            np.sqrt(np.mean(reconstructed_linear_jerk**2)),
            0.1 * np.sqrt(np.mean(parity_decoupled_linear_jerk**2)),
        )
        self.assertLess(
            np.sqrt(np.mean(reconstructed_angular_jerk**2)),
            0.1 * np.sqrt(np.mean(parity_decoupled_angular_jerk**2)),
        )
        self.assertTrue(all(state.valid for state in reconstructed.tracks[0].states))

    def test_constant_linear_and_angular_acceleration_are_preserved(self):
        count = 31
        time = np.arange(count, dtype=float) * 0.1
        linear_acceleration = 2.0
        angular_acceleration = 0.3
        scenario = build_scenario(
            x=0.5 * linear_acceleration * time**2,
            y=np.zeros(count),
            heading=0.5 * angular_acceleration * time**2,
        )

        reconstructed_track = copy.deepcopy(scenario.tracks[0])
        result = optimize_track(
            scenario.tracks[0],
            reconstructed_track,
            scenario.timestamps_seconds,
            BatchTrajectoryConfig(),
        )
        states = reconstructed_track.states
        x = np.array([state.center_x for state in states])
        heading = np.unwrap(np.array([state.heading for state in states]))

        np.testing.assert_allclose(
            x,
            0.5 * linear_acceleration * time**2,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            heading,
            0.5 * angular_acceleration * time**2,
            atol=1e-6,
        )
        linear_jerk, angular_jerk = wosac_jerk_features(
            x,
            np.zeros(count),
            np.zeros(count),
            heading,
            0.1,
        )
        self.assertTrue(result.optimized)
        self.assertLess(np.max(np.abs(linear_jerk)), 1e-6)
        self.assertLess(np.max(np.abs(angular_jerk)), 1e-4)

    def test_reverse_heading_is_preserved(self):
        count = 21
        scenario = build_scenario(
            x=-0.2 * np.arange(count),
            y=np.zeros(count),
            heading=np.zeros(count),
        )

        reconstructed, stats = batch_reconstruct(scenario)
        heading = np.array([state.heading for state in reconstructed.tracks[0].states])

        self.assertEqual(stats.optimized_tracks, 1)
        self.assertTrue(np.all(np.abs(angle_diff(heading, 0.0)) < 1e-6))

    def test_low_speed_heading_branch_uses_prefilter_fallback(self):
        count = 8
        heading = np.array(
            [-0.80, -0.79, -3.94, -3.95, -3.93, -3.91, -3.90, -3.89]
        )
        scenario = build_scenario(
            x=0.01 * np.arange(count, dtype=float),
            y=np.zeros(count),
            heading=heading,
        )

        filtered, _ = filter_reconstruct(scenario)
        reconstructed, stats = batch_reconstruct(scenario)
        filtered_heading = np.array(
            [state.heading for state in filtered.tracks[0].states]
        )
        reconstructed_heading = np.array(
            [state.heading for state in reconstructed.tracks[0].states]
        )
        features = compute_kinematic_features(
            np.array(
                [
                    [state.center_x, state.center_y, state.center_z]
                    for state in reconstructed.tracks[0].states
                ]
            ),
            reconstructed_heading,
            np.ones(count, dtype=bool),
            reconstructed.timestamps_seconds,
        )

        self.assertEqual(stats.optimization_failures, 0)
        self.assertLess(
            np.max(np.abs(angle_diff(reconstructed_heading, filtered_heading))),
            np.deg2rad(1.0),
        )
        self.assertLess(np.nanmax(np.abs(features.angular_jerk)), 0.1)

    def test_noisy_motion_direction_cannot_override_smooth_prefilter_heading(self):
        x = np.array(
            [
                -2980.75,
                -2980.83,
                -2981.17,
                -2981.27,
                -2981.25,
                -2981.30,
                -2981.35,
                -2981.40,
                -2981.36,
                -2981.39,
                -2981.42,
                -2981.43,
                -2981.51,
                -2981.59,
                -2981.78,
                -2981.87,
            ]
        )
        y = np.array(
            [
                1159.69,
                1159.69,
                1160.99,
                1161.07,
                1161.15,
                1161.18,
                1161.20,
                1161.23,
                1161.23,
                1161.23,
                1161.26,
                1161.26,
                1161.28,
                1161.31,
                1161.39,
                1161.40,
            ]
        )
        heading = np.array(
            [
                1.183,
                1.210,
                -3.441,
                -3.474,
                -3.530,
                -3.542,
                -3.536,
                -3.541,
                -3.567,
                -3.565,
                -3.572,
                -3.585,
                -3.554,
                -3.540,
                -3.521,
                -3.511,
            ]
        )
        scenario = build_scenario(x=x, y=y, heading=heading)

        filtered, _ = filter_reconstruct(scenario)
        reconstructed, stats = batch_reconstruct(scenario)
        filtered_heading = np.array(
            [state.heading for state in filtered.tracks[0].states]
        )
        reconstructed_heading = np.array(
            [state.heading for state in reconstructed.tracks[0].states]
        )
        features = compute_kinematic_features(
            np.array(
                [
                    [state.center_x, state.center_y, state.center_z]
                    for state in reconstructed.tracks[0].states
                ]
            ),
            reconstructed_heading,
            np.ones(len(x), dtype=bool),
            reconstructed.timestamps_seconds,
        )

        self.assertEqual(stats.optimization_failures, 0)
        self.assertLess(
            np.max(np.abs(angle_diff(reconstructed_heading, filtered_heading))),
            np.deg2rad(2.0),
        )
        self.assertLess(np.nanmax(np.abs(features.linear_jerk)), 8.0)

    def test_sparse_short_pedestrian_uses_extended_linear_regularization(self):
        x = np.array(
            [
                0.00,
                0.33,
                0.40,
                0.00,
                0.00,
                -0.59,
                0.00,
                -0.04,
                0.00,
                -0.50,
                -0.19,
                0.00,
                -0.15,
                0.54,
                0.56,
            ]
        )
        y = np.array(
            [
                0.00,
                0.64,
                0.54,
                0.03,
                0.00,
                -0.14,
                0.00,
                -0.12,
                0.00,
                -0.22,
                -0.17,
                0.00,
                -0.19,
                0.69,
                0.67,
            ]
        )
        valid = np.array(
            [
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
            ]
        )
        scenario = build_scenario(
            x=x,
            y=y,
            heading=np.zeros(len(x)),
            valid=valid,
            object_type=2,
        )

        reconstructed, stats = batch_reconstruct(scenario)
        features = compute_kinematic_features(
            np.array(
                [
                    [state.center_x, state.center_y, state.center_z]
                    for state in reconstructed.tracks[0].states
                ]
            ),
            np.array(
                [state.heading for state in reconstructed.tracks[0].states]
            ),
            np.array(
                [state.valid for state in reconstructed.tracks[0].states],
                dtype=bool,
            ),
            reconstructed.timestamps_seconds,
        )

        self.assertEqual(stats.optimization_failures, 0)
        self.assertLess(np.nanmax(np.abs(features.linear_jerk)), 12.0)
        self.assertLess(np.nanmax(np.abs(features.angular_jerk)), 1.0)

    def test_noisy_vehicle_heading_uses_sustained_forward_evidence(self):
        count = 42
        index = np.arange(count, dtype=float)
        x = np.where(index < 15, 0.02 * np.sin(index), 0.10 * (index - 15.0))
        heading = np.full(count, np.pi)
        heading[19:22] = 0.0
        heading[22] = np.deg2rad(-117.0)
        heading[23:25] = np.deg2rad(90.0)
        heading[25:33] = np.pi
        heading[33:38] = np.deg2rad(4.0)
        heading[38:41] = np.deg2rad(90.0)
        heading[41] = np.deg2rad(5.0)
        valid = np.ones(count, dtype=bool)
        valid[16:19] = False
        valid[30] = False
        scenario = build_scenario(
            x=x,
            y=np.zeros(count),
            heading=heading,
            valid=valid,
        )

        reconstructed, stats = batch_reconstruct(scenario)
        reconstructed_heading = np.array(
            [state.heading for state in reconstructed.tracks[0].states]
        )

        self.assertEqual(stats.optimization_failures, 0)
        self.assertLess(
            np.max(np.abs(angle_diff(reconstructed_heading, 0.0))),
            np.deg2rad(15.0),
        )

    def test_lateral_cyclist_heading_is_not_forced_to_motion(self):
        count = 21
        scenario = build_scenario(
            x=np.zeros(count),
            y=0.1 * np.arange(count),
            heading=np.zeros(count),
            object_type=3,
        )

        reconstructed, stats = batch_reconstruct(scenario)
        heading = np.array([state.heading for state in reconstructed.tracks[0].states])

        self.assertEqual(stats.optimized_tracks, 1)
        self.assertTrue(np.all(np.abs(angle_diff(heading, 0.0)) < 1e-6))

    def test_sparse_pedestrian_heading_does_not_create_quadratic_turn(self):
        count = 12
        heading = np.array(
            [-0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -0.17, -0.23, -0.8, 0.24, 0.21]
        )
        valid = np.ones(count, dtype=bool)
        valid[2:7] = False
        scenario = build_scenario(
            x=0.05 * np.arange(count),
            y=np.zeros(count),
            heading=heading,
            valid=valid,
            object_type=2,
        )

        reconstructed, stats = batch_reconstruct(scenario)
        features = compute_kinematic_features(
            np.array(
                [
                    [state.center_x, state.center_y, state.center_z]
                    for state in reconstructed.tracks[0].states
                ]
            ),
            np.array([state.heading for state in reconstructed.tracks[0].states]),
            np.array([state.valid for state in reconstructed.tracks[0].states]),
            reconstructed.timestamps_seconds,
        )

        self.assertEqual(stats.optimized_tracks, 1)
        self.assertLess(
            np.nanmax(np.abs(features.angular_speed[features.speed_validity])),
            0.4,
        )
        self.assertLess(
            np.nanmax(
                np.abs(features.angular_acceleration[features.acceleration_validity])
            ),
            0.5,
        )

    def test_batch_preprocessing_still_fills_internal_gaps(self):
        count = 21
        valid = np.ones(count, dtype=bool)
        valid[8:12] = False
        scenario = build_scenario(
            x=0.1 * np.arange(count),
            y=np.zeros(count),
            heading=np.zeros(count),
            valid=valid,
        )

        filtered, _ = filter_reconstruct(scenario)
        reconstructed, stats = batch_reconstruct(scenario)

        self.assertTrue(all(state.valid for state in reconstructed.tracks[0].states))
        self.assertEqual(stats.filled_frames, 4)
        self.assertEqual(stats.optimized_tracks, 1)
        self.assertLess(
            max(
                abs(
                    reconstructed.tracks[0].states[index].center_x
                    - filtered.tracks[0].states[index].center_x
                )
                for index in range(count)
            ),
            0.2,
        )

    def test_batch_trusts_the_complete_preprocessed_heading_curve(self):
        count = 21
        scenario = build_scenario(
            x=0.1 * np.arange(count),
            y=np.zeros(count),
            heading=np.zeros(count),
        )
        reconstructed_track = copy.deepcopy(scenario.tracks[0])
        for index, state in enumerate(reconstructed_track.states):
            state.heading = 0.02 * index

        with patch.object(
            batch_optimizer,
            "_optimize_segment",
            wraps=batch_optimizer._optimize_segment,
        ) as optimize_segment:
            optimize_track(
                scenario.tracks[0],
                reconstructed_track,
                scenario.timestamps_seconds,
                BatchTrajectoryConfig(),
            )

        heading_trusted = optimize_segment.call_args.kwargs["heading_trusted"]
        self.assertTrue(np.all(heading_trusted))

    def test_cyclist_heading_rate_constraint_preserves_turn_shape(self):
        count = 41
        time = 0.1 * np.arange(count, dtype=float)
        heading = 0.25 * np.sin(np.linspace(-1.2, 1.2, count))
        scenario = build_scenario(
            x=2.0 * time,
            y=np.zeros(count),
            heading=heading,
            object_type=3,
        )
        without_rate = copy.deepcopy(scenario.tracks[0])
        with_rate = copy.deepcopy(scenario.tracks[0])

        optimize_track(
            scenario.tracks[0],
            without_rate,
            scenario.timestamps_seconds,
            BatchTrajectoryConfig(cyclist_heading_rate_weight_scale=0.0),
        )
        optimize_track(
            scenario.tracks[0],
            with_rate,
            scenario.timestamps_seconds,
            BatchTrajectoryConfig(),
        )
        without_heading = np.array([state.heading for state in without_rate.states])
        with_heading = np.array([state.heading for state in with_rate.states])
        without_rmse = np.sqrt(np.mean(angle_diff(without_heading, heading) ** 2))
        with_rmse = np.sqrt(np.mean(angle_diff(with_heading, heading) ** 2))

        self.assertLess(with_rmse, without_rmse)

    def test_position_and_heading_are_optimized_independently(self):
        count = 21
        index = np.arange(count, dtype=float)
        x = 0.15 * index + 0.05 * np.sin(1.2 * index)
        y = 0.02 * np.sin(0.7 * index)
        heading = 0.02 * index + 0.10 * np.sin(1.3 * index)
        scenario = build_scenario(
            x=x,
            y=y,
            heading=heading,
            object_type=2,
        )
        config = BatchTrajectoryConfig(
            max_trusted_heading_correction_p95_rad=np.deg2rad(0.1),
        )

        reconstructed_track = copy.deepcopy(scenario.tracks[0])
        result = optimize_track(
            scenario.tracks[0],
            reconstructed_track,
            scenario.timestamps_seconds,
            config,
        )
        features = compute_kinematic_features(
            np.array(
                [
                    [state.center_x, state.center_y, state.center_z]
                    for state in reconstructed_track.states
                ]
            ),
            np.array([state.heading for state in reconstructed_track.states]),
            np.ones(count, dtype=bool),
            scenario.timestamps_seconds,
        )

        self.assertEqual(result.failed_segments, 0)
        self.assertEqual(result.position_optimized_segments, 1)
        self.assertEqual(result.heading_optimized_segments, 1)
        self.assertEqual(result.heading_limited_segments, 0)
        self.assertLess(
            np.nanstd(features.linear_jerk),
            np.nanstd(
                compute_kinematic_features(
                    np.column_stack(
                        (
                            x,
                            y,
                            np.zeros(count),
                        )
                    ),
                    heading,
                    np.ones(count, dtype=bool),
                    scenario.timestamps_seconds,
                ).linear_jerk
            ),
        )

    def test_persistent_endpoint_branch_does_not_force_batch_failure(self):
        count = 31
        expected_x = 0.2 * np.arange(count, dtype=float)
        x = expected_x.copy()
        x[-5:] += 6.0
        scenario = build_scenario(
            x=x,
            y=np.zeros(count),
            heading=np.zeros(count),
        )

        reconstructed, stats = batch_reconstruct(scenario)
        reconstructed_x = np.array(
            [state.center_x for state in reconstructed.tracks[0].states]
        )

        self.assertEqual(stats.optimization_failures, 0)
        self.assertEqual(stats.position_optimized_segments, 1)
        np.testing.assert_allclose(reconstructed_x[-5:], expected_x[-5:], atol=0.2)

    def test_linear_jerk_safety_smooths_only_extreme_output(self):
        count = 31
        time = np.arange(count, dtype=float) * 0.1
        x = (
            0.2 * np.arange(count, dtype=float)
            + 0.2 * np.sin(2.1 * np.arange(count, dtype=float))
        )
        y = np.zeros(count, dtype=float)
        z = np.zeros(count, dtype=float)
        before, _ = wosac_jerk_features(
            x,
            y,
            z,
            np.zeros(count),
            time,
        )

        safe_x, safe_y, safe_z, applied = batch_optimizer._apply_linear_jerk_safety(
            x,
            y,
            z,
            time,
            1,
            BatchTrajectoryConfig(),
        )
        after, _ = wosac_jerk_features(
            safe_x,
            safe_y,
            safe_z,
            np.zeros(count),
            time,
        )

        self.assertTrue(applied)
        self.assertGreater(np.sqrt(np.mean(before**2)), 50.0)
        self.assertLessEqual(np.sqrt(np.mean(after**2)), 8.0)

    def test_linear_jerk_safety_also_bounds_sparse_matched_support(self):
        count = 31
        time = np.arange(count, dtype=float) * 0.1
        x = (
            0.2 * np.arange(count, dtype=float)
            + 0.015 * np.sin(2.1 * np.arange(count, dtype=float))
        )
        y = np.zeros(count, dtype=float)
        z = np.zeros(count, dtype=float)
        matched_frames = np.zeros(count, dtype=bool)
        matched_frames[[0, 2, 4, 6]] = True
        support = batch_optimizer._jerk_support_from_frame_validity(
            matched_frames,
            count,
        )
        before, _ = wosac_jerk_features(
            x,
            y,
            z,
            np.zeros(count),
            time,
        )

        safe_x, safe_y, safe_z, applied = batch_optimizer._apply_linear_jerk_safety(
            x,
            y,
            z,
            time,
            1,
            BatchTrajectoryConfig(),
            matched_frame_validity=matched_frames,
        )
        after, _ = wosac_jerk_features(
            safe_x,
            safe_y,
            safe_z,
            np.zeros(count),
            time,
        )

        self.assertLess(np.sqrt(np.mean(before**2)), 8.0)
        self.assertGreater(np.sqrt(np.mean(before[support] ** 2)), 8.0)
        self.assertTrue(applied)
        self.assertLessEqual(np.sqrt(np.mean(after**2)), 8.0)
        self.assertLessEqual(np.sqrt(np.mean(after[support] ** 2)), 8.0)

    def test_angular_jerk_safety_keeps_the_resolved_angle_branch(self):
        count = 31
        time = np.arange(count, dtype=float) * 0.1
        heading = np.sin(1.7 * np.arange(count, dtype=float))
        config = BatchTrajectoryConfig()

        safe_heading, applied = batch_optimizer._apply_angular_jerk_safety(
            heading,
            time,
            config,
        )
        _, angular_jerk = wosac_jerk_features(
            np.zeros(count),
            np.zeros(count),
            np.zeros(count),
            np.unwrap(safe_heading),
            time,
        )

        self.assertTrue(applied)
        self.assertLessEqual(
            np.sqrt(np.mean(angular_jerk**2)),
            config.angular_jerk_safety_rms_radps3,
        )
        self.assertLessEqual(
            np.percentile(np.abs(angle_diff(safe_heading, heading)), 95),
            config.angular_jerk_safety_correction_p95_rad,
        )

    def test_angular_jerk_safety_handles_short_irregular_time_axis(self):
        time = np.array(
            [0.0, 0.09998, 0.2, 0.3, 0.39997, 0.59998, 0.69998]
        )
        heading = np.array(
            [
                -2.96672320,
                -3.03249073,
                -3.06303883,
                -3.05805898,
                -3.01676416,
                -2.82113481,
                -2.66613698,
            ]
        )
        config = BatchTrajectoryConfig()

        safe_heading, applied = batch_optimizer._apply_angular_jerk_safety(
            heading,
            time,
            config,
        )
        _, angular_jerk = wosac_jerk_features(
            np.zeros(len(heading)),
            np.zeros(len(heading)),
            np.zeros(len(heading)),
            np.unwrap(safe_heading),
            time,
        )

        self.assertTrue(applied)
        self.assertLessEqual(
            np.sqrt(np.mean(angular_jerk**2)),
            config.angular_jerk_safety_rms_radps3,
        )
        self.assertLessEqual(
            np.percentile(np.abs(angle_diff(safe_heading, heading)), 95),
            config.angular_jerk_safety_correction_p95_rad,
        )

    def test_angular_jerk_safety_also_bounds_sparse_matched_support(self):
        count = 31
        time = np.arange(count, dtype=float) * 0.1
        heading = 0.0025 * np.sin(1.7 * np.arange(count, dtype=float))
        matched_frames = np.zeros(count, dtype=bool)
        matched_frames[[21, 23, 25, 27]] = True
        support = batch_optimizer._jerk_support_from_frame_validity(
            matched_frames,
            count,
        )
        config = BatchTrajectoryConfig()
        zeros = np.zeros(count)
        _, before = wosac_jerk_features(
            zeros,
            zeros,
            zeros,
            np.unwrap(heading),
            time,
        )

        safe_heading, applied = batch_optimizer._apply_angular_jerk_safety(
            heading,
            time,
            config,
            matched_frame_validity=matched_frames,
        )
        _, after = wosac_jerk_features(
            zeros,
            zeros,
            zeros,
            np.unwrap(safe_heading),
            time,
        )

        self.assertLess(np.sqrt(np.mean(before**2)), 2.0)
        self.assertGreater(np.sqrt(np.mean(before[support] ** 2)), 2.0)
        self.assertTrue(applied)
        self.assertLessEqual(np.sqrt(np.mean(after**2)), 2.0)
        self.assertLessEqual(np.sqrt(np.mean(after[support] ** 2)), 2.0)

    def test_severe_along_track_timing_jitter_uses_rough_profile(self):
        count = 31
        interval = np.arange(count - 1, dtype=float)
        speed = 25.0 + 20.0 * np.sin(2.1 * interval)
        y = np.r_[0.0, np.cumsum(0.1 * speed)]
        scenario = build_scenario(
            x=np.zeros(count),
            y=y,
            heading=np.full(count, np.pi / 2.0),
        )

        reconstructed, stats = batch_reconstruct(scenario)
        raw_features = compute_kinematic_features(
            np.column_stack((np.zeros(count), y, np.zeros(count))),
            np.full(count, np.pi / 2.0),
            np.ones(count, dtype=bool),
            scenario.timestamps_seconds,
        )
        reconstructed_features = compute_kinematic_features(
            np.array(
                [
                    [state.center_x, state.center_y, state.center_z]
                    for state in reconstructed.tracks[0].states
                ]
            ),
            np.array([state.heading for state in reconstructed.tracks[0].states]),
            np.ones(count, dtype=bool),
            scenario.timestamps_seconds,
        )

        self.assertGreater(np.nanstd(raw_features.linear_jerk), 100.0)
        self.assertLess(np.nanstd(reconstructed_features.linear_jerk), 0.1)
        self.assertLess(
            max(abs(state.center_x) for state in reconstructed.tracks[0].states),
            1e-3,
        )
        self.assertEqual(stats.optimization_failures, 0)

    def test_short_track_is_reported_as_processed_not_failed(self):
        count = 5
        scenario = build_scenario(
            x=0.1 * np.arange(count),
            y=np.zeros(count),
            heading=np.zeros(count),
        )

        _, stats = batch_reconstruct(scenario)

        self.assertEqual(stats.processed_tracks, 1)
        self.assertEqual(stats.short_segments, 1)
        self.assertEqual(stats.optimization_failures, 0)


if __name__ == "__main__":
    unittest.main()
