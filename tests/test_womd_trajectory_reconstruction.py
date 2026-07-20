import tempfile
import textwrap
import unittest
from pathlib import Path

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

    def test_active_reconstruction_requires_project_root(self):
        with self.assertRaisesRegex(ValueError, "reconstruction_root"):
            TrajectoryReconstructionConfig(method="filter")

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
