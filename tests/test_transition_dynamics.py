import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

try:
    from src.smart.tokens.transition_dynamics import (
        TransitionDynamicsAccumulator,
        extract_full_trajectory_dynamics,
    )
except (ImportError, ModuleNotFoundError):
    TransitionDynamicsAccumulator = None
    extract_full_trajectory_dynamics = None

try:
    from src.smart.tokens.transition_dynamics_artifact import (
        load_transition_dynamics_artifact,
        make_transition_dynamics_artifact,
        save_transition_dynamics_artifact,
    )
except (ImportError, ModuleNotFoundError):
    load_transition_dynamics_artifact = None
    make_transition_dynamics_artifact = None
    save_transition_dynamics_artifact = None

try:
    from src.smart.tokens.transition_dynamics_artifact import HYBRID_SOURCE
except (ImportError, ModuleNotFoundError):
    HYBRID_SOURCE = None


class FullTrajectoryDynamicsTest(unittest.TestCase):
    def test_complete_constant_acceleration_trajectory(self):
        self.assertIsNotNone(
            extract_full_trajectory_dynamics,
            "full-trajectory dynamics extractor is not implemented",
        )
        time = np.arange(91, dtype=np.float64) * 0.1
        position = np.column_stack(
            (2.0 * time + time**2, np.zeros_like(time))
        )

        result = extract_full_trajectory_dynamics(
            position=position,
            heading=np.zeros(91, dtype=np.float64),
            valid_mask=np.ones(91, dtype=bool),
        )

        np.testing.assert_allclose(
            result.values[[5, 10, 90], 0],
            2.0,
            atol=1e-8,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            result.values[:, 1:],
            0.0,
            atol=1e-8,
            rtol=0.0,
        )
        np.testing.assert_array_equal(result.valid, np.ones(91, dtype=bool))

    def test_constant_radius_trajectory_has_body_frame_lateral_acceleration(self):
        self.assertIsNotNone(extract_full_trajectory_dynamics)
        time = np.arange(91, dtype=np.float64) * 0.1
        radius = 10.0
        angular_speed = 0.4
        heading = angular_speed * time
        position = np.column_stack(
            (
                radius * np.sin(heading),
                radius * (1.0 - np.cos(heading)),
            )
        )

        result = extract_full_trajectory_dynamics(
            position,
            heading,
            np.ones(91, dtype=bool),
        )

        self.assertAlmostEqual(float(result.values[45, 0]), 0.0, delta=2e-3)
        self.assertAlmostEqual(
            float(result.values[45, 1]),
            angular_speed,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            float(result.values[45, 2]),
            radius * angular_speed**2,
            delta=2e-3,
        )

    def test_token_endpoint_uses_full_trajectory_neighbors(self):
        self.assertIsNotNone(extract_full_trajectory_dynamics)
        time = np.arange(91, dtype=np.float64) * 0.1
        position = np.column_stack((time**4, np.zeros_like(time)))

        result = extract_full_trajectory_dynamics(
            position,
            np.zeros(91, dtype=np.float64),
            np.ones(91, dtype=bool),
        )

        # Two centered 0.1 s finite differences of x=t^4 at t=0.5 give
        # 12*t^2 + 8*dt^2 = 3.08. Frame 6 is across the first token boundary.
        self.assertAlmostEqual(float(result.values[5, 0]), 3.08, delta=1e-6)

    def test_heading_is_unwrapped_over_the_complete_valid_run(self):
        self.assertIsNotNone(extract_full_trajectory_dynamics)
        time = np.arange(91, dtype=np.float64) * 0.1
        angular_speed = 0.4
        heading_unwrapped = 3.0 + angular_speed * time
        heading_wrapped = (
            (heading_unwrapped + np.pi) % (2.0 * np.pi)
        ) - np.pi

        result = extract_full_trajectory_dynamics(
            np.zeros((91, 2), dtype=np.float64),
            heading_wrapped,
            np.ones(91, dtype=bool),
        )

        np.testing.assert_allclose(
            result.values[:, 1],
            angular_speed,
            atol=1e-6,
            rtol=0.0,
        )

    def test_short_and_nonfinite_runs_are_invalid(self):
        self.assertIsNotNone(extract_full_trajectory_dynamics)
        position = np.zeros((10, 2), dtype=np.float64)
        heading = np.zeros(10, dtype=np.float64)
        valid = np.zeros(10, dtype=bool)
        valid[0:2] = True
        valid[4:10] = True
        position[7] = np.nan

        result = extract_full_trajectory_dynamics(position, heading, valid)

        np.testing.assert_array_equal(
            result.valid,
            np.array(
                [False, False, False, False, True, True, True, False, False, False]
            ),
        )

    def test_rejects_invalid_shapes_and_numeric_configuration(self):
        self.assertIsNotNone(extract_full_trajectory_dynamics)
        position = np.zeros((5, 2), dtype=np.float64)
        heading = np.zeros(5, dtype=np.float64)
        valid = np.ones(5, dtype=bool)

        with self.assertRaisesRegex(ValueError, "position"):
            extract_full_trajectory_dynamics(
                np.zeros(5),
                heading,
                valid,
            )
        with self.assertRaisesRegex(ValueError, "share"):
            extract_full_trajectory_dynamics(
                position,
                np.zeros(4),
                valid,
            )
        with self.assertRaisesRegex(ValueError, "dt"):
            extract_full_trajectory_dynamics(
                position,
                heading,
                valid,
                dt=0.0,
            )
        with self.assertRaisesRegex(ValueError, "clipping_limits"):
            extract_full_trajectory_dynamics(
                position,
                heading,
                valid,
                clipping_limits=(15.0, 3.0),
            )


class TransitionDynamicsAccumulatorTest(unittest.TestCase):
    def test_unseen_pair_falls_back_to_current_token_marginal(self):
        self.assertIsNotNone(
            TransitionDynamicsAccumulator,
            "transition accumulator is not implemented",
        )
        accumulator = TransitionDynamicsAccumulator(
            n_agent_types=3,
            n_token=2,
        )
        accumulator.add(
            agent_type=np.array([0, 0]),
            previous_token=np.array([0, 0]),
            current_token=np.array([1, 1]),
            values=np.array(
                [
                    [4.0, 0.4, 2.0],
                    [6.0, 0.6, 4.0],
                ]
            ),
            valid=np.array([True, True]),
        )

        table, statistics = accumulator.finalize(
            isolated_fallback=np.zeros((3, 2, 3), dtype=np.float64),
            shrinkage_count=2.0,
        )

        np.testing.assert_allclose(
            table[0, 0, 1],
            np.array([5.0, 0.5, 3.0]),
            atol=2e-3,
            rtol=0.0,
        )
        np.testing.assert_array_equal(table[0, 1, 1], table[0, 0, 1])
        self.assertEqual(table.shape, (3, 2, 2, 3))
        self.assertEqual(table.dtype, np.float16)
        self.assertEqual(statistics["observed_pairs"]["veh"], 1)

    def test_rare_pair_is_shrunk_toward_current_token_marginal(self):
        self.assertIsNotNone(TransitionDynamicsAccumulator)
        accumulator = TransitionDynamicsAccumulator(3, 2)
        accumulator.add(
            agent_type=np.zeros(4, dtype=np.int64),
            previous_token=np.array([0, 1, 1, 1]),
            current_token=np.ones(4, dtype=np.int64),
            values=np.array(
                [
                    [10.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            valid=np.ones(4, dtype=bool),
        )
        isolated = np.zeros((3, 2, 3), dtype=np.float64)
        isolated[0, 0] = [7.0, 0.7, -1.0]

        table, _ = accumulator.finalize(
            isolated,
            shrinkage_count=1.0,
        )

        # The current-token marginal is 2.5. The one-observation (0 -> 1)
        # pair is therefore (10 + 2.5) / 2 = 6.25.
        self.assertAlmostEqual(float(table[0, 0, 1, 0]), 6.25, delta=2e-3)
        # Token 0 was never observed, so every transition into it uses the
        # isolated-token fallback.
        np.testing.assert_allclose(
            table[0, :, 0],
            np.array([[7.0, 0.7, -1.0], [7.0, 0.7, -1.0]]),
            atol=2e-3,
            rtol=0.0,
        )

    def test_agent_classes_are_separate_and_invalid_occurrences_are_skipped(self):
        self.assertIsNotNone(TransitionDynamicsAccumulator)
        accumulator = TransitionDynamicsAccumulator(3, 2)
        accumulator.add(
            agent_type=np.array([0, 1, 2, 9]),
            previous_token=np.array([0, 0, 0, 9]),
            current_token=np.array([1, 1, 1, 9]),
            values=np.array(
                [
                    [1.0, 0.1, 2.0],
                    [3.0, 0.3, 4.0],
                    [5.0, 0.5, 6.0],
                    [np.nan, np.nan, np.nan],
                ]
            ),
            valid=np.array([True, True, True, False]),
        )

        table, statistics = accumulator.finalize(
            np.zeros((3, 2, 3)),
            shrinkage_count=1.0,
        )

        np.testing.assert_allclose(table[0, 0, 1], [1.0, 0.1, 2.0], atol=2e-3)
        np.testing.assert_allclose(table[1, 0, 1], [3.0, 0.3, 4.0], atol=2e-3)
        np.testing.assert_allclose(table[2, 0, 1], [5.0, 0.5, 6.0], atol=2e-3)
        self.assertEqual(statistics["occurrences"], 3)
        self.assertEqual(
            statistics["occurrences_by_class"],
            {"veh": 1, "ped": 1, "cyc": 1},
        )

    def test_accumulator_rejects_invalid_shapes_indices_and_fallback(self):
        self.assertIsNotNone(TransitionDynamicsAccumulator)
        with self.assertRaisesRegex(ValueError, "n_agent_types"):
            TransitionDynamicsAccumulator(2, 2)
        with self.assertRaisesRegex(ValueError, "n_token"):
            TransitionDynamicsAccumulator(3, 0)

        accumulator = TransitionDynamicsAccumulator(3, 2)
        with self.assertRaisesRegex(ValueError, "share"):
            accumulator.add(
                agent_type=np.array([0]),
                previous_token=np.array([0, 1]),
                current_token=np.array([1]),
                values=np.zeros((1, 3)),
                valid=np.array([True]),
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            accumulator.add(
                agent_type=np.array([0]),
                previous_token=np.array([2]),
                current_token=np.array([1]),
                values=np.zeros((1, 3)),
                valid=np.array([True]),
            )
        with self.assertRaisesRegex(ValueError, "isolated_fallback"):
            accumulator.finalize(np.zeros((2, 2, 3)))
        with self.assertRaisesRegex(ValueError, "shrinkage_count"):
            accumulator.finalize(
                np.zeros((3, 2, 3)),
                shrinkage_count=0.0,
            )


class TransitionDynamicsArtifactTest(unittest.TestCase):
    @staticmethod
    def _artifact(vocabulary):
        return make_transition_dynamics_artifact(
            np.zeros((3, 2, 2, 3), dtype=np.float16),
            vocabulary_path=vocabulary,
            source="raw",
            dt=0.1,
            clipping_limits=(15.0, 3.0, 15.0),
            shrinkage_count=8.0,
            statistics={"occurrences": 0},
        )

    def test_artifact_round_trip_is_bound_to_vocabulary(self):
        self.assertIsNotNone(
            make_transition_dynamics_artifact,
            "transition artifact support is not implemented",
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vocabulary = root / "agent_vocab.pkl"
            vocabulary.write_bytes(b"vocabulary-a")
            values = np.arange(36, dtype=np.float16).reshape(3, 2, 2, 3)
            artifact = make_transition_dynamics_artifact(
                values,
                vocabulary_path=vocabulary,
                source="raw",
                dt=0.1,
                clipping_limits=(15.0, 3.0, 15.0),
                shrinkage_count=8.0,
                statistics={"occurrences": 12},
            )

            output = save_transition_dynamics_artifact(
                root / "lookup.pt",
                artifact,
                vocabulary_path=vocabulary,
            )
            loaded = load_transition_dynamics_artifact(
                output,
                vocabulary_path=vocabulary,
                expected_source="raw",
                expected_n_token=2,
            )

            torch.testing.assert_close(
                loaded,
                torch.from_numpy(values),
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(loaded.dtype, torch.float16)
            self.assertFalse((root / "lookup.pt.tmp").exists())

    def test_hybrid_source_round_trip_is_explicit(self):
        self.assertIsNotNone(
            HYBRID_SOURCE,
            "hybrid transition-dynamics provenance is not implemented",
        )
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vocabulary = root / "agent_vocab.pkl"
            vocabulary.write_bytes(b"original-vocabulary")
            artifact = make_transition_dynamics_artifact(
                np.zeros((3, 2, 2, 3), dtype=np.float16),
                vocabulary_path=vocabulary,
                source=HYBRID_SOURCE,
                dt=0.1,
                clipping_limits=(15.0, 3.0, 15.0),
                shrinkage_count=8.0,
                statistics={
                    "assignment_source": "raw",
                    "dynamics_source": "reconstructed",
                },
            )
            output = save_transition_dynamics_artifact(
                root / "hybrid.pt",
                artifact,
                vocabulary_path=vocabulary,
            )

            loaded = load_transition_dynamics_artifact(
                output,
                vocabulary_path=vocabulary,
                expected_source=HYBRID_SOURCE,
                expected_n_token=2,
            )

            self.assertEqual(tuple(loaded.shape), (3, 2, 2, 3))
            with self.assertRaisesRegex(ValueError, "source"):
                load_transition_dynamics_artifact(
                    output,
                    vocabulary_path=vocabulary,
                    expected_source="raw",
                    expected_n_token=2,
                )

    def test_load_rejects_missing_numeric_metadata(self):
        self.assertIsNotNone(make_transition_dynamics_artifact)
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vocabulary = root / "agent_vocab.pkl"
            vocabulary.write_bytes(b"vocabulary-a")
            artifact = make_transition_dynamics_artifact(
                np.zeros((3, 2, 2, 3), dtype=np.float16),
                vocabulary_path=vocabulary,
                source="raw",
                dt=0.1,
                clipping_limits=(15.0, 3.0, 15.0),
                shrinkage_count=8.0,
                statistics={},
            )
            del artifact["dt"]
            output = root / "missing-dt.pt"
            torch.save(artifact, output)

            with self.assertRaisesRegex(ValueError, "dt"):
                load_transition_dynamics_artifact(
                    output,
                    vocabulary_path=vocabulary,
                    expected_source="raw",
                    expected_n_token=2,
                )

    def test_load_rejects_vocabulary_and_source_mismatch(self):
        self.assertIsNotNone(load_transition_dynamics_artifact)
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vocabulary = root / "agent_vocab.pkl"
            vocabulary.write_bytes(b"vocabulary-a")
            output = root / "lookup.pt"
            torch.save(self._artifact(vocabulary), output)

            vocabulary.write_bytes(b"vocabulary-b")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_transition_dynamics_artifact(
                    output,
                    vocabulary_path=vocabulary,
                    expected_source="raw",
                    expected_n_token=2,
                )

            vocabulary.write_bytes(b"vocabulary-a")
            with self.assertRaisesRegex(ValueError, "source"):
                load_transition_dynamics_artifact(
                    output,
                    vocabulary_path=vocabulary,
                    expected_source="reconstructed",
                    expected_n_token=2,
                )

    def test_load_rejects_corrupt_structure_and_values(self):
        self.assertIsNotNone(load_transition_dynamics_artifact)
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vocabulary = root / "agent_vocab.pkl"
            vocabulary.write_bytes(b"vocabulary-a")
            corruptions = {
                "version": ("format_version", 99, "format_version"),
                "feature": ("feature_order", ("wrong",), "feature_order"),
                "size": ("vocabulary_size", 3, "vocabulary_size"),
                "shape": (
                    "values",
                    torch.zeros(3, 2, 3, dtype=torch.float16),
                    "shape",
                ),
                "dtype": (
                    "values",
                    torch.zeros(3, 2, 2, 3, dtype=torch.float32),
                    "float16",
                ),
                "finite": (
                    "values",
                    torch.full(
                        (3, 2, 2, 3),
                        float("nan"),
                        dtype=torch.float16,
                    ),
                    "non-finite",
                ),
                "limits": ("clipping_limits", (15.0, -1.0, 15.0), "clipping"),
                "shrinkage": ("shrinkage_count", 0.0, "shrinkage"),
                "statistics": ("statistics", None, "statistics"),
            }
            for name, (key, value, message) in corruptions.items():
                with self.subTest(name=name):
                    artifact = self._artifact(vocabulary)
                    artifact[key] = value
                    output = root / f"{name}.pt"
                    torch.save(artifact, output)
                    with self.assertRaisesRegex(ValueError, message):
                        load_transition_dynamics_artifact(
                            output,
                            vocabulary_path=vocabulary,
                            expected_source="raw",
                            expected_n_token=2,
                        )

    def test_failed_atomic_save_preserves_existing_output(self):
        self.assertIsNotNone(save_transition_dynamics_artifact)
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            vocabulary = root / "agent_vocab.pkl"
            vocabulary.write_bytes(b"vocabulary-a")
            output = root / "lookup.pt"
            output.write_bytes(b"existing-output")
            artifact = self._artifact(vocabulary)
            artifact["values"] = torch.full(
                (3, 2, 2, 3),
                float("nan"),
                dtype=torch.float16,
            )

            with self.assertRaisesRegex(ValueError, "non-finite"):
                save_transition_dynamics_artifact(
                    output,
                    artifact,
                    vocabulary_path=vocabulary,
                )

            self.assertEqual(output.read_bytes(), b"existing-output")
            self.assertFalse((root / "lookup.pt.tmp").exists())

    def test_make_rejects_invalid_table_and_parameters(self):
        self.assertIsNotNone(make_transition_dynamics_artifact)
        with TemporaryDirectory() as temp_dir:
            vocabulary = Path(temp_dir) / "agent_vocab.pkl"
            vocabulary.write_bytes(b"vocabulary-a")
            values = np.zeros((3, 2, 2, 3), dtype=np.float16)
            common = {
                "vocabulary_path": vocabulary,
                "source": "raw",
                "dt": 0.1,
                "clipping_limits": (15.0, 3.0, 15.0),
                "shrinkage_count": 8.0,
                "statistics": {},
            }

            with self.assertRaisesRegex(ValueError, "shape"):
                make_transition_dynamics_artifact(
                    np.zeros((3, 2, 3), dtype=np.float16),
                    **common,
                )
            with self.assertRaisesRegex(ValueError, "source"):
                make_transition_dynamics_artifact(
                    values,
                    **{**common, "source": "validation"},
                )
            with self.assertRaisesRegex(ValueError, "dt"):
                make_transition_dynamics_artifact(
                    values,
                    **{**common, "dt": 0.0},
                )
            with self.assertRaisesRegex(ValueError, "clipping_limits"):
                make_transition_dynamics_artifact(
                    values,
                    **{**common, "clipping_limits": (15.0, 3.0)},
                )
            with self.assertRaisesRegex(ValueError, "shrinkage_count"):
                make_transition_dynamics_artifact(
                    values,
                    **{**common, "shrinkage_count": 0.0},
                )


if __name__ == "__main__":
    unittest.main()
