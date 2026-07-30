"""Load TrajTok-preprocessed ground truth for Fast WOSAC evaluation."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path


class PreprocessedScenarioGT:
    """Resolve and load scenario dictionaries with optional strict semantics."""

    def __init__(
        self,
        directory: str | Path | None,
        *,
        required: bool = False,
    ) -> None:
        self.required = bool(required)
        self.directory = (
            Path(directory).expanduser().resolve() if directory else None
        )
        if self.directory is not None and self.directory.is_dir():
            return
        if self.required:
            target = (
                "<not configured>"
                if self.directory is None
                else str(self.directory)
            )
            raise FileNotFoundError(
                "strict Fast WOSAC requires a validation_gt directory: "
                f"{target}"
            )
        if self.directory is not None:
            warnings.warn(
                "Fast WOSAC GT directory does not exist; falling back to "
                f"per-scenario TFRecords: {self.directory}",
                stacklevel=2,
            )
        self.directory = None

    def load(self, scenario_id: str) -> dict | None:
        """Load one preprocessed scenario or return ``None`` for fallback."""
        if self.directory is None:
            return None
        path = self.directory / f"{scenario_id}.pkl"
        if not path.is_file():
            if self.required:
                raise FileNotFoundError(
                    "strict Fast WOSAC requires preprocessed GT for "
                    f"scenario {scenario_id}: {path}"
                )
            return None
        with path.open("rb") as file:
            scenario = pickle.load(file)
        if hasattr(scenario, "value"):
            scenario = scenario.value
        if not isinstance(scenario, dict):
            raise TypeError(
                "Fast WOSAC GT must be a dict, got "
                f"{type(scenario).__name__}: {path}"
            )
        return scenario
