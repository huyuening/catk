import contextlib
import io
import json
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.smart.tokens.transition_dynamics import TransitionDynamicsAccumulator
from src.smart.tokens.transition_dynamics_artifact import (
    load_transition_dynamics_artifact,
    vocabulary_sha256,
)

from src.smart.tokens.build_transition_dynamics import (
    accumulate_tokenized_batch,
    validate_source_provenance,
)

try:
    from src.smart.tokens.build_transition_dynamics import build_parser
except ImportError:
    build_parser = None

try:
    from src.smart.tokens.build_transition_dynamics import (
        build_transition_dynamics,
    )
except ImportError:
    build_transition_dynamics = None

try:
    from src.smart.tokens.build_transition_dynamics import main
except ImportError:
    main = None


class TransitionDynamicsSourceTest(unittest.TestCase):
    def test_reconstructed_source_requires_provenance_marker(self):
        self.assertIsNotNone(
            validate_source_provenance,
            "training-only transition builder is not implemented",
        )
        agent_store = {
            "position": torch.zeros(1, 91, 3),
        }

        with self.assertRaisesRegex(ValueError, "trajectory_reconstructed"):
            validate_source_provenance(
                agent_store,
                "reconstructed",
                context="scene.pkl",
            )

    def test_raw_and_reconstructed_markers_cannot_be_mixed(self):
        validate_source_provenance(
            {
                "position": torch.zeros(2, 91, 3),
                "trajectory_reconstructed": torch.tensor([False, False]),
            },
            "raw",
            context="raw.pkl",
        )
        validate_source_provenance(
            {
                "position": torch.zeros(2, 91, 3),
                "trajectory_reconstructed": torch.tensor([True, True]),
            },
            "reconstructed",
            context="reconstructed.pkl",
        )
        with self.assertRaisesRegex(ValueError, "raw source"):
            validate_source_provenance(
                {
                    "position": torch.zeros(2, 91, 3),
                    "trajectory_reconstructed": torch.tensor([False, True]),
                },
                "raw",
                context="mixed.pkl",
            )
        with self.assertRaisesRegex(ValueError, "every"):
            validate_source_provenance(
                {
                    "position": torch.zeros(2, 91, 3),
                    "trajectory_reconstructed": torch.tensor([False, True]),
                },
                "reconstructed",
                context="mixed.pkl",
            )
        with self.assertRaisesRegex(ValueError, "one value per agent"):
            validate_source_provenance(
                {
                    "position": torch.zeros(2, 91, 3),
                    "trajectory_reconstructed": torch.tensor([True]),
                },
                "reconstructed",
                context="short-marker.pkl",
            )
        with self.assertRaisesRegex(ValueError, "source"):
            validate_source_provenance(
                {"position": torch.zeros(1, 91, 3)},
                "validation",
                context="validation.pkl",
            )


class TransitionDynamicsBatchTest(unittest.TestCase):
    @staticmethod
    def _constant_acceleration_batch():
        time = torch.arange(91, dtype=torch.float64) * 0.1
        position = torch.zeros(1, 91, 3, dtype=torch.float64)
        position[0, :, 0] = 2.0 * time + time.square()
        return {
            "agent": {
                "position": position,
                "heading": torch.zeros(1, 91, dtype=torch.float64),
                "valid_mask": torch.ones(1, 91, dtype=torch.bool),
                "type": torch.tensor([0], dtype=torch.long),
            }
        }

    def test_batch_accumulates_current_endpoint_for_adjacent_pair(self):
        self.assertIsNotNone(
            accumulate_tokenized_batch,
            "training-only transition builder is not implemented",
        )
        accumulator = TransitionDynamicsAccumulator(3, 2)
        tokenized_agent = {
            "type": torch.tensor([0]),
            "gt_idx": torch.tensor([[0, 1, 1]]),
            "valid_mask": torch.tensor([[True, True, True]]),
        }

        statistics = accumulate_tokenized_batch(
            accumulator,
            self._constant_acceleration_batch(),
            tokenized_agent,
            source="raw",
        )

        self.assertEqual(accumulator.pair_count[0, 0, 1], 1)
        self.assertEqual(accumulator.pair_count[0, 1, 1], 1)
        np.testing.assert_allclose(
            accumulator.pair_sum[0, 0, 1],
            [2.0, 0.0, 0.0],
            atol=1e-8,
            rtol=0.0,
        )
        self.assertEqual(statistics["accepted_occurrences"], 2)
        self.assertEqual(statistics["skipped_occurrences"], 0)

    def test_batch_uses_current_endpoint_from_complete_trajectory(self):
        time = torch.arange(91, dtype=torch.float64) * 0.1
        data = self._constant_acceleration_batch()
        data["agent"]["position"][0, :, 0] = time**4
        accumulator = TransitionDynamicsAccumulator(3, 2)

        accumulate_tokenized_batch(
            accumulator,
            data,
            {
                "type": torch.tensor([0]),
                "gt_idx": torch.tensor([[0, 1]]),
                "valid_mask": torch.tensor([[True, True]]),
            },
            source="raw",
        )

        # Current token 1 ends at frame 10 (t=1.0). Two centered finite
        # differences over the complete trajectory give 12*t^2+8*dt^2.
        self.assertAlmostEqual(
            float(accumulator.pair_sum[0, 0, 1, 0]),
            12.08,
            delta=1e-5,
        )

    def test_invalid_current_endpoint_is_skipped(self):
        data = self._constant_acceleration_batch()
        data["agent"]["valid_mask"][0, 10] = False
        accumulator = TransitionDynamicsAccumulator(3, 2)

        statistics = accumulate_tokenized_batch(
            accumulator,
            data,
            {
                "type": torch.tensor([0]),
                "gt_idx": torch.tensor([[0, 1]]),
                "valid_mask": torch.tensor([[True, True]]),
            },
            source="raw",
        )

        self.assertEqual(statistics["accepted_occurrences"], 0)
        self.assertEqual(statistics["skipped_occurrences"], 1)
        self.assertEqual(int(accumulator.pair_count.sum()), 0)

    def test_batch_requires_complete_ninety_one_frame_trajectory(self):
        data = self._constant_acceleration_batch()
        for key in ("position", "heading", "valid_mask"):
            data["agent"][key] = data["agent"][key][:, :90]
        accumulator = TransitionDynamicsAccumulator(3, 2)

        with self.assertRaisesRegex(ValueError, "91"):
            accumulate_tokenized_batch(
                accumulator,
                data,
                {
                    "type": torch.tensor([0]),
                    "gt_idx": torch.tensor([[0, 1]]),
                    "valid_mask": torch.tensor([[True, True]]),
                },
                source="raw",
            )


class TransitionDynamicsCliTest(unittest.TestCase):
    def test_module_entrypoint_runs_after_batch_helpers_are_defined(self):
        module_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "smart"
            / "tokens"
            / "build_transition_dynamics.py"
        )
        source = module_path.read_text(encoding="utf-8")

        self.assertGreater(
            source.index('if __name__ == "__main__":'),
            source.index("def accumulate_tokenized_batch("),
        )

    def test_parser_accepts_training_only_inputs(self):
        self.assertIsNotNone(
            build_parser,
            "transition dynamics CLI is not implemented",
        )
        parser = build_parser()

        args = parser.parse_args(
            [
                "--training-dir",
                "/cache/training",
                "--agent-token-file",
                "/tokens/agent.pkl",
                "--source",
                "raw",
                "--output",
                "/cache/lookup.pt",
            ]
        )

        self.assertEqual(args.training_dir, "/cache/training")
        self.assertEqual(args.source, "raw")
        self.assertFalse(hasattr(args, "validation_dir"))
        self.assertFalse(hasattr(args, "test_dir"))

    def test_builder_rejects_missing_or_empty_training_directory_first(self):
        self.assertIsNotNone(
            build_transition_dynamics,
            "transition dynamics build entry point is not implemented",
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vocabulary = root / "agent.pkl"
            vocabulary.write_bytes(b"vocabulary")

            with self.assertRaisesRegex(FileNotFoundError, "training"):
                build_transition_dynamics(
                    training_dir=root / "missing-training",
                    agent_token_file=vocabulary,
                    output=root / "lookup.pt",
                    source="raw",
                )
            empty_training = root / "training"
            empty_training.mkdir()
            with self.assertRaisesRegex(ValueError, "no cache"):
                build_transition_dynamics(
                    training_dir=empty_training,
                    agent_token_file=vocabulary,
                    output=root / "lookup.pt",
                    source="raw",
                )

    def test_builder_writes_real_artifact_and_summary_from_training_batch(self):
        self.assertIsNotNone(build_transition_dynamics)
        with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            training_dir = root / "training"
            training_dir.mkdir()
            (training_dir / "scene.pkl").write_bytes(b"dataset-owned")
            vocabulary = root / "agent.pkl"
            token_all = {
                name: np.zeros((2, 6, 4, 2), dtype=np.float32)
                for name in ("veh", "ped", "cyc")
            }
            with vocabulary.open("wb") as handle:
                pickle.dump({"token_all": token_all}, handle)

            batch = TransitionDynamicsBatchTest._constant_acceleration_batch()
            tokenized = {
                "type": torch.tensor([0]),
                "gt_idx": torch.tensor([[0, 1, 1]]),
                "valid_mask": torch.tensor([[True, True, True]]),
            }
            processor_arguments = {}

            class FakeDataset:
                def __init__(self, raw_dir, transform):
                    self.raw_dir = raw_dir
                    self.transform = transform
                    self.items = [batch]

                def __len__(self):
                    return len(self.items)

            class FakeLoader:
                def __init__(self, dataset, **kwargs):
                    self.dataset = dataset
                    self.kwargs = kwargs

                def __iter__(self):
                    return iter(self.dataset.items)

                def __len__(self):
                    return len(self.dataset)

            class FakeProcessor:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    processor_arguments.update(kwargs)

                def eval(self):
                    return self

                def tokenize_agent(self, data):
                    return tokenized

            runtime = SimpleNamespace(
                MultiDataset=FakeDataset,
                DataLoader=FakeLoader,
                HeteroData=dict,
                TokenProcessor=FakeProcessor,
                Subset=lambda dataset, indices: dataset,
            )
            output = root / "lookup.pt"
            with patch(
                "src.smart.tokens.build_transition_dynamics."
                "_load_runtime_components",
                return_value=runtime,
                create=True,
            ):
                result = build_transition_dynamics(
                    training_dir=training_dir,
                    agent_token_file=vocabulary.relative_to(Path.cwd()),
                    output=output,
                    source="raw",
                    batch_size=1,
                    num_workers=0,
                )

            self.assertEqual(result, output)
            self.assertTrue(
                Path(processor_arguments["agent_token_file"]).is_absolute()
            )
            table = load_transition_dynamics_artifact(
                output,
                vocabulary_path=vocabulary,
                expected_source="raw",
                expected_n_token=2,
            )
            self.assertEqual(tuple(table.shape), (3, 2, 2, 3))
            self.assertAlmostEqual(float(table[0, 0, 1, 0]), 2.0, delta=2e-3)
            summary_path = root / "lookup.summary.json"
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["scenarios"], 1)
            self.assertEqual(summary["accepted_occurrences"], 2)
            self.assertEqual(summary["source"], "raw")
            self.assertEqual(summary["vocabulary_size"], 2)
            self.assertEqual(
                summary["vocabulary_sha256"],
                vocabulary_sha256(vocabulary),
            )
            self.assertEqual(
                summary["occurrences_by_class"],
                {"veh": 2, "ped": 0, "cyc": 0},
            )

    def test_module_help_exposes_training_only_command(self):
        self.assertIsNotNone(main, "transition dynamics CLI entry point is missing")
        stdout = io.StringIO()
        with self.assertRaises(SystemExit) as raised:
            with contextlib.redirect_stdout(stdout):
                main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("--training-dir", help_text)
        self.assertIn("--agent-token-file", help_text)
        self.assertIn("--source", help_text)
        self.assertIn("--output", help_text)
        self.assertNotIn("--validation-dir", help_text)
        self.assertNotIn("--test-dir", help_text)


if __name__ == "__main__":
    unittest.main()
