import copy
import unittest
import warnings

import numpy as np

from src.smart.tokens.compare_trajectory_token_reconstruction import (
    _load_scenario_class,
)
from src.smart.tokens.reconstruction_evaluation import (
    AGENT_METRICS,
    FRAME_METRICS,
    EvaluationAccumulator,
    RunningMoments,
    evaluate_scenario_pair,
    evaluate_track,
)


def build_scenario(count: int = 11, *, object_type: int = 1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        scenario_class = _load_scenario_class()
    scenario = scenario_class()
    scenario.scenario_id = "metric-test"
    scenario.current_time_index = min(10, count - 1)
    scenario.timestamps_seconds.extend(
        (np.arange(count, dtype=float) * 0.1).tolist()
    )
    track = scenario.tracks.add()
    track.id = 42
    track.object_type = object_type
    for value in np.arange(count, dtype=float) * 0.1:
        state = track.states.add()
        state.center_x = float(value**3)
        state.center_y = float(0.5 * value**2)
        state.center_z = 0.0
        state.heading = float(0.2 * value**3)
        state.length = 4.5
        state.width = 1.8
        state.height = 1.5
        state.valid = True
    return scenario


class ReconstructionEvaluationTest(unittest.TestCase):
    def test_running_moments_filters_nonfinite_and_round_trips(self):
        moments = RunningMoments()
        moments.update_many(
            np.asarray([1.0, 2.0, 3.0, np.nan, np.inf])
        )

        restored = RunningMoments.from_state(moments.to_state())

        self.assertEqual(restored.count, 3)
        self.assertEqual(restored.mean, 2.0)
        self.assertAlmostEqual(
            restored.std,
            float(np.std([1.0, 2.0, 3.0])),
        )
        self.assertEqual(restored.minimum, 1.0)
        self.assertEqual(restored.maximum, 3.0)

    def test_metric_schema_covers_agent_and_frame_outputs(self):
        self.assertEqual(len(AGENT_METRICS), 9)
        self.assertEqual(len(FRAME_METRICS), 6)
        self.assertIn(
            "xy_rmse_m",
            {definition.key for definition in AGENT_METRICS},
        )
        self.assertIn(
            "reconstructed_full_angular_jerk_radps3",
            {definition.key for definition in FRAME_METRICS},
        )

    def test_track_metrics_use_raw_support_and_xy_rmse(self):
        raw_scenario = build_scenario(count=25)
        reconstructed_scenario = copy.deepcopy(raw_scenario)
        raw_track = raw_scenario.tracks[0]
        reconstructed_track = reconstructed_scenario.tracks[0]
        for state in reconstructed_track.states:
            state.center_x += 3.0
            state.center_y += 4.0
        for index in range(11, 14):
            raw_track.states[index].valid = False

        evaluation = evaluate_track(
            raw_track,
            reconstructed_track,
            raw_scenario.timestamps_seconds,
        )

        self.assertAlmostEqual(evaluation.agent_values["xy_rmse_m"], 5.0)
        self.assertEqual(
            len(evaluation.frame_values["raw_linear_jerk_mps3"]),
            len(
                evaluation.frame_values[
                    "reconstructed_linear_jerk_mps3"
                ]
            ),
        )
        self.assertGreater(
            len(
                evaluation.frame_values[
                    "reconstructed_full_linear_jerk_mps3"
                ]
            ),
            len(evaluation.frame_values["raw_linear_jerk_mps3"]),
        )

    def test_scenario_pair_groups_values_by_object_type(self):
        raw_scenario = build_scenario(count=25)
        reconstructed_scenario = copy.deepcopy(raw_scenario)

        batch = evaluate_scenario_pair(
            raw_scenario,
            reconstructed_scenario,
        )

        self.assertEqual(batch.scenario_id, "metric-test")
        self.assertEqual(batch.agent_count, 1)
        self.assertEqual(set(batch.agent_values), {"vehicle"})
        self.assertEqual(
            batch.agent_values["vehicle"]["xy_rmse_m"].shape,
            (1,),
        )

    def test_incomplete_reconstructed_support_invalidates_paired_jerk(self):
        raw_scenario = build_scenario(count=15)
        reconstructed_scenario = copy.deepcopy(raw_scenario)
        reconstructed_scenario.tracks[0].states[7].valid = False

        evaluation = evaluate_track(
            raw_scenario.tracks[0],
            reconstructed_scenario.tracks[0],
            raw_scenario.timestamps_seconds,
        )

        self.assertLess(
            evaluation.agent_values["linear_jerk_matched_coverage"],
            1.0,
        )
        self.assertTrue(
            np.isnan(
                evaluation.agent_values[
                    "raw_linear_jerk_rms_mps3"
                ]
            )
        )
        self.assertEqual(
            evaluation.frame_values["raw_linear_jerk_mps3"].size,
            0,
        )

    def test_accumulator_tracks_all_and_type_scopes_and_round_trips(self):
        raw_scenario = build_scenario(count=25)
        batch = evaluate_scenario_pair(
            raw_scenario,
            copy.deepcopy(raw_scenario),
        )
        accumulator = EvaluationAccumulator()

        accumulator.add_batch(batch)
        restored = EvaluationAccumulator.from_state(
            accumulator.to_state()
        )

        self.assertEqual(restored.scenarios, 1)
        self.assertEqual(restored.agents, 1)
        self.assertEqual(
            restored.agent_moments["all"]["xy_rmse_m"].count,
            1,
        )
        self.assertEqual(
            restored.agent_moments["vehicle"]["xy_rmse_m"].mean,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
