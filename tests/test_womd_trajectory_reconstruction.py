import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.smart.tokens.womd_trajectory_reconstruction import (
    TrajectoryReconstructionConfig,
    reconstruct_scenario_agents,
    reconstruct_scenario_for_vocabulary,
)


class WomdTrajectoryReconstructionTest(unittest.TestCase):
    def test_disabled_reconstruction_is_a_noop(self):
        scenario = {"scenario_id": "example"}

        reconstructed, stats = reconstruct_scenario_agents(
            scenario, TrajectoryReconstructionConfig()
        )

        self.assertIs(reconstructed, scenario)
        self.assertIsNone(stats)

    def test_only_optimizer_requires_project_root(self):
        config = TrajectoryReconstructionConfig(method="batch")

        self.assertTrue(config.is_active)
        self.assertIsNone(config.project_root)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            TrajectoryReconstructionConfig(method="optimizer")

    def test_bundled_filter_does_not_require_project_root(self):
        config = TrajectoryReconstructionConfig(method="filter")

        self.assertTrue(config.is_active)
        self.assertIsNone(config.project_root)

    def test_bundled_filter_reconstructs_without_external_checkout(self):
        scenario = SimpleNamespace(timestamps_seconds=[], tracks=[])

        reconstructed, stats = reconstruct_scenario_agents(
            scenario, TrajectoryReconstructionConfig(method="filter")
        )

        self.assertIsNot(reconstructed, scenario)
        self.assertEqual(stats.total_tracks, 0)

    def test_bundled_batch_reconstructs_without_external_checkout(self):
        scenario = SimpleNamespace(timestamps_seconds=[], tracks=[])

        reconstructed, stats = reconstruct_scenario_agents(
            scenario, TrajectoryReconstructionConfig(method="batch")
        )

        self.assertIsNot(reconstructed, scenario)
        self.assertEqual(type(stats).__name__, "BatchReconstructionStats")
        self.assertEqual(stats.total_tracks, 0)

    @patch(
        "src.smart.tokens.trajectory_batch_optimizer."
        "reconstruct_scenario_agents"
    )
    def test_bundled_batch_receives_exposed_weights(self, reconstruct):
        reconstruct.return_value = ("scenario", "stats")
        config = TrajectoryReconstructionConfig(
            method="batch",
            filter_strength="balanced",
            max_gap_frames=4,
            batch_linear_jerk_weight=1.5,
            batch_angular_jerk_weight=2.5,
        )

        result = reconstruct_scenario_agents(object(), config)

        self.assertEqual(result, ("scenario", "stats"))
        call = reconstruct.call_args
        self.assertEqual(call.kwargs["filter_strength"], "balanced")
        self.assertEqual(call.kwargs["max_gap_frames"], 4)
        self.assertEqual(call.kwargs["config"].linear_jerk_weight, 1.5)
        self.assertEqual(call.kwargs["config"].angular_jerk_weight, 2.5)

    def test_external_project_is_loaded_in_an_isolated_namespace(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            source_dir = project_root / "src"
            source_dir.mkdir()
            (source_dir / "trajectory_reconstruction.py").write_text(
                textwrap.dedent(
                    """
                    def reconstruct_scenario_agents(scenario, **kwargs):
                        output = dict(scenario)
                        output["method"] = kwargs["method"]
                        output["filter_strength"] = kwargs["filter_strength"]
                        return output, {"max_gap_frames": kwargs["max_gap_frames"]}
                    """
                ),
                encoding="utf-8",
            )
            config = TrajectoryReconstructionConfig(
                method="batch",
                project_root=str(project_root),
                filter_strength="balanced",
                max_gap_frames=4,
            )

            reconstructed, stats = reconstruct_scenario_agents(
                {"scenario_id": "example"}, config
            )

            self.assertEqual(reconstructed["method"], "batch")
            self.assertEqual(reconstructed["filter_strength"], "balanced")
            self.assertEqual(stats["max_gap_frames"], 4)

    def test_vocabulary_reconstruction_uses_the_full_scenario(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            source_dir = project_root / "src"
            source_dir.mkdir()
            (source_dir / "trajectory_reconstruction.py").write_text(
                textwrap.dedent(
                    """
                    def reconstruct_scenario_agents(scenario, **kwargs):
                        output = dict(scenario)
                        output["frame_count_seen"] = len(scenario["frames"])
                        return output, {"frame_count_seen": len(scenario["frames"])}
                    """
                ),
                encoding="utf-8",
            )
            scenario = {"frames": list(range(91))}
            config = TrajectoryReconstructionConfig(
                method="filter",
                project_root=str(project_root),
            )

            reconstructed, _ = reconstruct_scenario_for_vocabulary(scenario, config)

            self.assertEqual(reconstructed["frame_count_seen"], 91)


if __name__ == "__main__":
    unittest.main()
