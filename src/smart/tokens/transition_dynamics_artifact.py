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

"""Portable artifacts for training-derived token-transition dynamics."""

from __future__ import annotations

import hashlib
import inspect
import math
import os
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


FORMAT_VERSION = 1
FEATURE_ORDER = ("a_lon", "angular_speed", "a_lat")
VALID_SOURCES = ("raw", "reconstructed")


def vocabulary_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of the exact vocabulary file bytes."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"agent vocabulary does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_transition_dynamics_artifact(
    values: np.ndarray | Tensor,
    *,
    vocabulary_path: str | Path,
    source: str,
    dt: float,
    clipping_limits: Sequence[float],
    shrinkage_count: float,
    statistics: Mapping,
) -> dict:
    """Construct a tensor-only, vocabulary-bound transition artifact."""

    values = torch.as_tensor(values, dtype=torch.float16).cpu().contiguous()
    if (
        values.ndim != 4
        or values.shape[0] != 3
        or values.shape[-1] != 3
        or values.shape[1] != values.shape[2]
    ):
        raise ValueError(
            "values must have shape [3, n_token, n_token, 3]"
        )
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    if source not in VALID_SOURCES:
        raise ValueError(f"source must be one of {VALID_SOURCES}")
    limits = tuple(float(value) for value in clipping_limits)
    if (
        len(limits) != 3
        or any(not math.isfinite(value) or value <= 0.0 for value in limits)
    ):
        raise ValueError(
            "clipping_limits must contain three finite positive values"
        )
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(shrinkage_count) or shrinkage_count <= 0.0:
        raise ValueError("shrinkage_count must be finite and positive")

    return {
        "format_version": FORMAT_VERSION,
        "feature_order": FEATURE_ORDER,
        "values": values,
        "vocabulary_sha256": vocabulary_sha256(vocabulary_path),
        "vocabulary_size": int(values.shape[1]),
        "source": source,
        "dt": float(dt),
        "clipping_limits": limits,
        "shrinkage_count": float(shrinkage_count),
        "statistics": dict(statistics),
    }


def _torch_load(path: Path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def load_transition_dynamics_artifact(
    path: str | Path,
    *,
    vocabulary_path: str | Path,
    expected_source: str,
    expected_n_token: int,
) -> Tensor:
    """Load and validate a transition table before model initialization."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"transition dynamics artifact does not exist: {path}")
    if expected_source not in VALID_SOURCES:
        raise ValueError(f"expected_source must be one of {VALID_SOURCES}")
    if expected_n_token < 1:
        raise ValueError("expected_n_token must be positive")

    artifact = _torch_load(path)
    if not isinstance(artifact, dict):
        raise ValueError(f"{path}: expected a dictionary artifact")
    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path}: unsupported format_version "
            f"{artifact.get('format_version')!r}"
        )
    if tuple(artifact.get("feature_order", ())) != FEATURE_ORDER:
        raise ValueError(f"{path}: unexpected feature_order")
    if artifact.get("source") != expected_source:
        raise ValueError(
            f"{path}: source {artifact.get('source')!r} does not match "
            f"{expected_source!r}"
        )
    expected_digest = vocabulary_sha256(vocabulary_path)
    if artifact.get("vocabulary_sha256") != expected_digest:
        raise ValueError(f"{path}: vocabulary SHA-256 mismatch")
    if artifact.get("vocabulary_size") != expected_n_token:
        raise ValueError(f"{path}: vocabulary_size mismatch")
    try:
        dt = float(artifact.get("dt"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: dt must be finite and positive") from error
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"{path}: dt must be finite and positive")
    try:
        limits = tuple(float(value) for value in artifact.get("clipping_limits", ()))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: clipping_limits must contain three finite positive values"
        ) from error
    if (
        len(limits) != 3
        or any(not math.isfinite(value) or value <= 0.0 for value in limits)
    ):
        raise ValueError(
            f"{path}: clipping_limits must contain three finite positive values"
        )
    try:
        shrinkage_count = float(artifact.get("shrinkage_count"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: shrinkage_count must be finite and positive"
        ) from error
    if not math.isfinite(shrinkage_count) or shrinkage_count <= 0.0:
        raise ValueError(
            f"{path}: shrinkage_count must be finite and positive"
        )
    if not isinstance(artifact.get("statistics"), MappingABC):
        raise ValueError(f"{path}: statistics must be a dictionary")

    values = artifact.get("values")
    expected_shape = (3, expected_n_token, expected_n_token, 3)
    if not isinstance(values, Tensor) or tuple(values.shape) != expected_shape:
        raise ValueError(f"{path}: values must have shape {expected_shape}")
    if values.dtype != torch.float16:
        raise ValueError(f"{path}: values must use torch.float16")
    if not torch.isfinite(values).all():
        raise ValueError(f"{path}: values contain non-finite entries")
    return values.cpu().contiguous()


def save_transition_dynamics_artifact(
    path: str | Path,
    artifact: Mapping,
    *,
    vocabulary_path: str | Path,
) -> Path:
    """Atomically save an artifact after validating its temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    try:
        torch.save(dict(artifact), temporary)
        load_transition_dynamics_artifact(
            temporary,
            vocabulary_path=vocabulary_path,
            expected_source=str(artifact.get("source")),
            expected_n_token=int(artifact.get("vocabulary_size", 0)),
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path
