import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.smart.tokens.paired_transition_dataset import PairedTransitionDataset


def make_cache(
    scenario_id,
    agent_ids,
    agent_types,
    x_accelerations,
    *,
    reconstructed,
):
    time = torch.arange(91, dtype=torch.float64) * 0.1
    position = torch.zeros(len(agent_ids), 91, 3, dtype=torch.float64)
    for index, acceleration in enumerate(x_accelerations):
        position[index, :, 0] = 0.5 * acceleration * time.square()
    return {
        "scenario_id": scenario_id,
        "current_time_index": 10,
        "agent": {
            "num_nodes": len(agent_ids),
            "id": torch.tensor(agent_ids, dtype=torch.long),
            "type": torch.tensor(agent_types, dtype=torch.long),
            "position": position,
            "heading": torch.zeros(
                len(agent_ids),
                91,
                dtype=torch.float64,
            ),
            "valid_mask": torch.ones(
                len(agent_ids),
                91,
                dtype=torch.bool,
            ),
            "trajectory_reconstructed": torch.full(
                (len(agent_ids),),
                reconstructed,
                dtype=torch.bool,
            ),
        },
    }


class PairedTransitionDatasetTest(unittest.TestCase):
    @staticmethod
    def write_cache(path, value):
        with path.open("wb") as stream:
            pickle.dump(value, stream)

    def make_directories(self, root):
        assignment_dir = root / "original"
        dynamics_dir = root / "reconstructed"
        assignment_dir.mkdir()
        dynamics_dir.mkdir()
        return assignment_dir, dynamics_dir

    def test_reconstructed_agents_are_aligned_by_id(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment_dir, dynamics_dir = self.make_directories(root)
            self.write_cache(
                assignment_dir / "scene.pkl",
                make_cache(
                    "scene",
                    [11, 22],
                    [0, 1],
                    [1.0, 2.0],
                    reconstructed=False,
                ),
            )
            self.write_cache(
                dynamics_dir / "scene.pkl",
                make_cache(
                    "scene",
                    [22, 11],
                    [1, 0],
                    [8.0, 4.0],
                    reconstructed=True,
                ),
            )

            dataset = PairedTransitionDataset(
                assignment_dir=assignment_dir,
                dynamics_dir=dynamics_dir,
                transform=lambda value: value,
            )
            assignment, dynamics = dataset[0]

            self.assertEqual(assignment["agent"]["id"].tolist(), [11, 22])
            self.assertEqual(dynamics["agent"]["id"].tolist(), [11, 22])
            self.assertAlmostEqual(
                float(dynamics["agent"]["position"][0, 10, 0]),
                2.0,
            )
            self.assertAlmostEqual(
                float(dynamics["agent"]["position"][1, 10, 0]),
                4.0,
            )

    def test_file_sets_must_match(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment_dir, dynamics_dir = self.make_directories(root)
            self.write_cache(
                assignment_dir / "only-original.pkl",
                make_cache(
                    "scene",
                    [1],
                    [0],
                    [1.0],
                    reconstructed=False,
                ),
            )

            with self.assertRaisesRegex(ValueError, "file sets"):
                PairedTransitionDataset(
                    assignment_dir=assignment_dir,
                    dynamics_dir=dynamics_dir,
                    transform=lambda value: value,
                )

    def test_duplicate_agent_ids_are_rejected(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment_dir, dynamics_dir = self.make_directories(root)
            self.write_cache(
                assignment_dir / "scene.pkl",
                make_cache(
                    "scene",
                    [1, 1],
                    [0, 0],
                    [1.0, 1.0],
                    reconstructed=False,
                ),
            )
            self.write_cache(
                dynamics_dir / "scene.pkl",
                make_cache(
                    "scene",
                    [1, 2],
                    [0, 0],
                    [2.0, 2.0],
                    reconstructed=True,
                ),
            )
            dataset = PairedTransitionDataset(
                assignment_dir=assignment_dir,
                dynamics_dir=dynamics_dir,
                transform=lambda value: value,
            )

            with self.assertRaisesRegex(ValueError, "unique"):
                dataset[0]

    def test_missing_agent_id_is_rejected(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment_dir, dynamics_dir = self.make_directories(root)
            self.write_cache(
                assignment_dir / "scene.pkl",
                make_cache(
                    "scene",
                    [1, 2],
                    [0, 0],
                    [1.0, 1.0],
                    reconstructed=False,
                ),
            )
            self.write_cache(
                dynamics_dir / "scene.pkl",
                make_cache(
                    "scene",
                    [1, 3],
                    [0, 0],
                    [2.0, 2.0],
                    reconstructed=True,
                ),
            )
            dataset = PairedTransitionDataset(
                assignment_dir=assignment_dir,
                dynamics_dir=dynamics_dir,
                transform=lambda value: value,
            )

            with self.assertRaisesRegex(ValueError, "agent.id sets"):
                dataset[0]

    def test_agent_type_mismatch_is_rejected_after_id_alignment(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment_dir, dynamics_dir = self.make_directories(root)
            self.write_cache(
                assignment_dir / "scene.pkl",
                make_cache(
                    "scene",
                    [1],
                    [0],
                    [1.0],
                    reconstructed=False,
                ),
            )
            self.write_cache(
                dynamics_dir / "scene.pkl",
                make_cache(
                    "scene",
                    [1],
                    [2],
                    [2.0],
                    reconstructed=True,
                ),
            )
            dataset = PairedTransitionDataset(
                assignment_dir=assignment_dir,
                dynamics_dir=dynamics_dir,
                transform=lambda value: value,
            )

            with self.assertRaisesRegex(ValueError, "agent.type"):
                dataset[0]

    def test_scenario_and_current_time_must_match(self):
        for field, changed_value, message in (
            ("scenario_id", "other-scene", "scenario_id"),
            ("current_time_index", 9, "current_time_index"),
        ):
            with self.subTest(field=field):
                with TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    assignment_dir, dynamics_dir = self.make_directories(root)
                    assignment = make_cache(
                        "scene",
                        [1],
                        [0],
                        [1.0],
                        reconstructed=False,
                    )
                    reconstructed = make_cache(
                        "scene",
                        [1],
                        [0],
                        [2.0],
                        reconstructed=True,
                    )
                    reconstructed[field] = changed_value
                    self.write_cache(
                        assignment_dir / "scene.pkl",
                        assignment,
                    )
                    self.write_cache(
                        dynamics_dir / "scene.pkl",
                        reconstructed,
                    )
                    dataset = PairedTransitionDataset(
                        assignment_dir=assignment_dir,
                        dynamics_dir=dynamics_dir,
                        transform=lambda value: value,
                    )

                    with self.assertRaisesRegex(ValueError, message):
                        dataset[0]

    def test_required_agent_fields_are_validated(self):
        for cache_name, field in (
            ("assignment", "heading"),
            ("reconstructed", "valid_mask"),
        ):
            with self.subTest(cache=cache_name, field=field):
                with TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    assignment_dir, dynamics_dir = self.make_directories(root)
                    assignment = make_cache(
                        "scene",
                        [1],
                        [0],
                        [1.0],
                        reconstructed=False,
                    )
                    reconstructed = make_cache(
                        "scene",
                        [1],
                        [0],
                        [2.0],
                        reconstructed=True,
                    )
                    target = (
                        assignment
                        if cache_name == "assignment"
                        else reconstructed
                    )
                    del target["agent"][field]
                    self.write_cache(
                        assignment_dir / "scene.pkl",
                        assignment,
                    )
                    self.write_cache(
                        dynamics_dir / "scene.pkl",
                        reconstructed,
                    )
                    dataset = PairedTransitionDataset(
                        assignment_dir=assignment_dir,
                        dynamics_dir=dynamics_dir,
                        transform=lambda value: value,
                    )

                    with self.assertRaisesRegex(ValueError, field):
                        dataset[0]

    def test_trajectory_shapes_must_match(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment_dir, dynamics_dir = self.make_directories(root)
            assignment = make_cache(
                "scene",
                [1],
                [0],
                [1.0],
                reconstructed=False,
            )
            reconstructed = make_cache(
                "scene",
                [1],
                [0],
                [2.0],
                reconstructed=True,
            )
            for field in ("position", "heading", "valid_mask"):
                reconstructed["agent"][field] = reconstructed["agent"][field][
                    :, :90
                ]
            self.write_cache(assignment_dir / "scene.pkl", assignment)
            self.write_cache(dynamics_dir / "scene.pkl", reconstructed)
            dataset = PairedTransitionDataset(
                assignment_dir=assignment_dir,
                dynamics_dir=dynamics_dir,
                transform=lambda value: value,
            )

            with self.assertRaisesRegex(ValueError, "shapes differ"):
                dataset[0]

    def test_paired_samples_collate_as_two_batches(self):
        try:
            from torch_geometric.data import HeteroData
            from torch_geometric.loader import DataLoader
        except ModuleNotFoundError:
            self.skipTest("PyTorch Geometric is not installed")

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment_dir, dynamics_dir = self.make_directories(root)
            self.write_cache(
                assignment_dir / "scene.pkl",
                make_cache(
                    "scene",
                    [1],
                    [0],
                    [1.0],
                    reconstructed=False,
                ),
            )
            self.write_cache(
                dynamics_dir / "scene.pkl",
                make_cache(
                    "scene",
                    [1],
                    [0],
                    [2.0],
                    reconstructed=True,
                ),
            )
            dataset = PairedTransitionDataset(
                assignment_dir=assignment_dir,
                dynamics_dir=dynamics_dir,
                transform=HeteroData,
            )
            assignment_batch, dynamics_batch = next(
                iter(DataLoader(dataset, batch_size=1, shuffle=False))
            )

            self.assertEqual(
                assignment_batch["agent"]["id"].tolist(),
                [1],
            )
            self.assertEqual(
                dynamics_batch["agent"]["id"].tolist(),
                [1],
            )


if __name__ == "__main__":
    unittest.main()
