# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Vocabulary-only WOMD trajectory reconstruction for CatK.

The geometric ``filter`` implementation is bundled with CatK.  The heavier
``batch`` and ``optimizer`` implementations can still be loaded from an
external WOMD-Traffic-Signal-Data-Improvement checkout when explicitly used.

The reconstructed Scenario produced here is a source for trajectory-vocabulary
clustering only.  CatK model inputs and training labels remain untouched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional


TRAJECTORY_RECONSTRUCTION_METHODS = ("none", "filter", "batch", "optimizer")


@dataclass(frozen=True)
class TrajectoryReconstructionConfig:
    """Configuration for bundled filtering or an optional external solver."""

    method: str = "none"
    project_root: Optional[str] = None
    filter_strength: str = "strong"
    max_gap_frames: int = -1
    batch_linear_jerk_weight: float = 1.0
    batch_angular_jerk_weight: float = 1.0
    show_solver_warnings: bool = False

    def __post_init__(self) -> None:
        if self.method not in TRAJECTORY_RECONSTRUCTION_METHODS:
            valid = ", ".join(TRAJECTORY_RECONSTRUCTION_METHODS)
            raise ValueError(
                f"Unknown trajectory reconstruction method '{self.method}'. "
                f"Choose one of: {valid}"
            )
        if self.method in ("batch", "optimizer") and not self.project_root:
            raise ValueError(
                "--reconstruction-root is required for batch and optimizer "
                "trajectory reconstruction"
            )
        if self.method != "none" and self.project_root:
            entrypoint = (
                Path(self.project_root).expanduser()
                / "src"
                / "trajectory_reconstruction.py"
            )
            if not entrypoint.is_file():
                raise FileNotFoundError(
                    "Could not find WOMD trajectory reconstruction entrypoint at "
                    f"{entrypoint}"
                )

    @property
    def is_active(self) -> bool:
        return self.method != "none"


@lru_cache(maxsize=None)
def _load_reconstruction_entrypoint(project_root: str) -> Callable[..., Any]:
    """Load the external wrapper without colliding with CatK's ``src`` package."""

    root = Path(project_root).expanduser().resolve()
    source_dir = root / "src"
    entrypoint_path = source_dir / "trajectory_reconstruction.py"
    if not entrypoint_path.is_file():
        raise FileNotFoundError(
            "Could not find WOMD trajectory reconstruction entrypoint at "
            f"{entrypoint_path}"
        )

    path_digest = hashlib.sha1(str(source_dir).encode("utf-8")).hexdigest()[:12]
    package_name = f"_catk_womd_reconstruction_{path_digest}"
    module_name = f"{package_name}.trajectory_reconstruction"

    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        package = sys.modules.get(package_name)
        if package is None:
            package = types.ModuleType(package_name)
            package.__path__ = [str(source_dir)]
            package.__package__ = package_name
            package.__spec__ = importlib.util.spec_from_loader(
                package_name, loader=None, is_package=True
            )
            sys.modules[package_name] = package

        spec = importlib.util.spec_from_file_location(module_name, entrypoint_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Unable to load trajectory reconstruction module from {entrypoint_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    entrypoint = getattr(module, "reconstruct_scenario_agents", None)
    if entrypoint is None:
        raise AttributeError(
            f"{entrypoint_path} does not define reconstruct_scenario_agents"
        )
    return entrypoint


def reconstruct_scenario_agents(
    scenario: Any,
    config: TrajectoryReconstructionConfig,
):
    """Reconstruct one Scenario, or return it unchanged when disabled."""

    if not config.is_active:
        return scenario, None

    if config.method == "filter" and not config.project_root:
        from .trajectory_filter_reconstructor import (
            config_for_filter_strength,
            reconstruct_scenario_agents as reconstruct_with_filter,
        )

        return reconstruct_with_filter(
            scenario,
            config=config_for_filter_strength(
                config.filter_strength,
                max_gap_frames=config.max_gap_frames,
            ),
        )

    entrypoint = _load_reconstruction_entrypoint(config.project_root or "")
    return entrypoint(
        scenario,
        method=config.method,
        show_solver_warnings=config.show_solver_warnings,
        filter_strength=config.filter_strength,
        max_gap_frames=config.max_gap_frames,
        batch_linear_jerk_weight=config.batch_linear_jerk_weight,
        batch_angular_jerk_weight=config.batch_angular_jerk_weight,
    )


def reconstruct_scenario_for_vocabulary(
    scenario: Any,
    config: TrajectoryReconstructionConfig,
):
    """Reconstruct the complete observed trajectory for vocabulary clustering.

    No history/future split is applied: every available training frame may
    contribute to the offline vocabulary.  The returned Scenario must not be
    substituted for CatK's model-input or label cache.
    """

    return reconstruct_scenario_agents(scenario, config)
