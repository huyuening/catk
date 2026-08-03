from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class WarmStartReport:
    checkpoint_path: str
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]
    loaded_epoch: Optional[int]
    loaded_global_step: Optional[int]
    restored_trainer_state: bool = False


def _trusted_torch_load(path: Path) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError as exc:
        # PyTorch before the weights_only argument existed. The checkpoint is
        # still treated as a trusted, project-produced artifact.
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def _extract_state_dict(payload: Any) -> tuple[Dict[str, torch.Tensor], Mapping]:
    metadata: Mapping = {}
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state = payload["state_dict"]
        metadata = payload
    elif isinstance(payload, Mapping):
        state = payload
    else:
        raise RuntimeError(
            "warm-start checkpoint must be a raw state dict or contain "
            "a 'state_dict' mapping"
        )
    if not isinstance(state, Mapping):
        raise RuntimeError("checkpoint 'state_dict' must be a mapping")

    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise RuntimeError("checkpoint state-dict keys must be strings")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"checkpoint value for {key!r} is not a tensor")
        normalized[key] = value

    if normalized and all(key.startswith("module.") for key in normalized):
        normalized = {
            key[len("module.") :]: value for key, value in normalized.items()
        }
    return normalized, metadata


def _dtype_class(value: torch.Tensor) -> str:
    if value.dtype == torch.bool:
        return "bool"
    if value.is_floating_point():
        return "floating"
    if value.is_complex():
        return "complex"
    if value.is_quantized:
        return "quantized"
    return "integer"


def _optional_int(value: Any) -> Optional[int]:
    if isinstance(value, Integral):
        return int(value)
    return None


def load_warm_start_state_dict(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    allowed_missing_prefixes: Sequence[str] = (),
) -> WarmStartReport:
    """Load model tensors only and return the audited compatibility report."""

    prefixes = tuple(allowed_missing_prefixes)
    if any(not isinstance(prefix, str) for prefix in prefixes):
        raise TypeError("allowed missing prefixes must be strings")
    if any(not prefix for prefix in prefixes):
        raise ValueError("allowed missing prefixes must not contain an empty prefix")

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"warm-start checkpoint does not exist: {path}")
    checkpoint_state, metadata = _extract_state_dict(_trusted_torch_load(path))
    model_state = model.state_dict()

    checkpoint_keys = set(checkpoint_state)
    model_keys = set(model_state)
    missing = tuple(sorted(model_keys - checkpoint_keys))
    unexpected = tuple(sorted(checkpoint_keys - model_keys))
    disallowed_missing = tuple(
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in prefixes)
    )
    shared = tuple(sorted(checkpoint_keys & model_keys))
    loaded_base = tuple(
        key for key in shared if ".text_control_adapter." not in key
    )

    non_finite = []
    shape_mismatches = []
    dtype_class_mismatches = []
    for key, checkpoint_value in checkpoint_state.items():
        if (
            checkpoint_value.is_floating_point()
            or checkpoint_value.is_complex()
        ) and not bool(torch.isfinite(checkpoint_value).all()):
            non_finite.append(key)
        if key not in model_state:
            continue
        model_value = model_state[key]
        if tuple(checkpoint_value.shape) != tuple(model_value.shape):
            shape_mismatches.append(
                f"{key}: checkpoint={tuple(checkpoint_value.shape)}, "
                f"model={tuple(model_value.shape)}"
            )
        checkpoint_class = _dtype_class(checkpoint_value)
        model_class = _dtype_class(model_value)
        if checkpoint_class != model_class:
            dtype_class_mismatches.append(
                f"{key}: checkpoint={checkpoint_class}, model={model_class}"
            )

    problems = []
    if disallowed_missing:
        problems.append("disallowed missing keys: " + ", ".join(disallowed_missing))
    if unexpected:
        problems.append("unexpected keys: " + ", ".join(unexpected))
    if not loaded_base:
        problems.append("checkpoint loads no non-text base tensor")
    if non_finite:
        problems.append("non-finite tensors: " + ", ".join(sorted(non_finite)))
    if shape_mismatches:
        problems.append("shape mismatches: " + "; ".join(shape_mismatches))
    if dtype_class_mismatches:
        problems.append(
            "dtype class mismatches: " + "; ".join(dtype_class_mismatches)
        )
    if problems:
        raise RuntimeError("warm-start checkpoint audit failed; " + " | ".join(problems))

    incompatible = model.load_state_dict(checkpoint_state, strict=False)
    actual_missing = tuple(sorted(incompatible.missing_keys))
    actual_unexpected = tuple(sorted(incompatible.unexpected_keys))
    if actual_missing != missing or actual_unexpected != unexpected:
        raise RuntimeError(
            "load_state_dict compatibility result differed from the pre-load audit"
        )

    return WarmStartReport(
        checkpoint_path=str(path),
        missing_keys=actual_missing,
        unexpected_keys=actual_unexpected,
        loaded_epoch=_optional_int(metadata.get("epoch")),
        loaded_global_step=_optional_int(metadata.get("global_step")),
        restored_trainer_state=False,
    )


__all__ = ["WarmStartReport", "load_warm_start_state_dict"]
