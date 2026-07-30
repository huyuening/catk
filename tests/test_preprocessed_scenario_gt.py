import importlib.util
import pickle
import tempfile
import unittest
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/smart/metrics/preprocessed_scenario_gt.py"
SPEC = importlib.util.spec_from_file_location(
    "preprocessed_scenario_gt",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
PreprocessedScenarioGT = MODULE.PreprocessedScenarioGT


class PreprocessedScenarioGTTest(unittest.TestCase):
    def test_required_store_rejects_missing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "validation_gt"
            with self.assertRaisesRegex(
                FileNotFoundError,
                "strict Fast WOSAC.*validation_gt",
            ):
                PreprocessedScenarioGT(missing, required=True)

    def test_required_store_rejects_missing_scenario(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = PreprocessedScenarioGT(temporary, required=True)
            with self.assertRaisesRegex(
                FileNotFoundError,
                "scenario-123.*pkl",
            ):
                store.load("scenario-123")

    def test_store_loads_valid_dictionary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scenario-123.pkl"
            expected = {"scenario_id": "scenario-123", "tracks": []}
            with path.open("wb") as file:
                pickle.dump(expected, file)
            store = PreprocessedScenarioGT(temporary, required=True)
            self.assertEqual(store.load("scenario-123"), expected)

    def test_optional_store_keeps_fallback_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "validation_gt"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store = PreprocessedScenarioGT(missing, required=False)
            self.assertIsNone(store.directory)
            self.assertIsNone(store.load("scenario-123"))
            self.assertTrue(
                any("falling back" in str(item.message) for item in caught)
            )


if __name__ == "__main__":
    unittest.main()
