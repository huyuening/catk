from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import yaml


def _trusted_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _set(config: Any, key: str, value: Any) -> None:
    if isinstance(config, Mapping):
        config[key] = value
    else:
        setattr(config, key, value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hard_ce_contract(model_config: Any) -> tuple[Any, Any]:
    history = _get(model_config, "history_dynamics", {})
    loss = _get(model_config, "training_loss", {})
    if not bool(_get(history, "is_active", False)):
        raise RuntimeError("selected PRE_BC checkpoint has history dynamics disabled")

    missing = object()
    history_mode = _get(history, "mode", missing)
    if history_mode is missing:
        # Checkpoints created before history modes were introduced only had
        # the cached reconstructed path, so preserve that legacy default.
        history_mode = "cached_reconstructed"
        _set(history, "mode", history_mode)
    if history_mode != "cached_reconstructed":
        raise RuntimeError(
            "selected PRE_BC checkpoint must use history mode "
            f"cached_reconstructed, got {history_mode!r}"
        )

    spatial_smoothing = _get(loss, "spatial_aware_smoothing", missing)
    if spatial_smoothing is missing:
        raise RuntimeError(
            "selected PRE_BC checkpoint lacks spatial_aware_smoothing metadata"
        )
    if bool(spatial_smoothing):
        raise RuntimeError(
            "selected PRE_BC checkpoint must disable spatial smoothing"
        )

    label_smoothing = _get(loss, "label_smoothing", missing)
    if label_smoothing is missing:
        raise RuntimeError(
            "selected PRE_BC checkpoint must use label_smoothing=0.0"
        )
    try:
        label_smoothing = float(label_smoothing)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "selected PRE_BC checkpoint must use label_smoothing=0.0"
        ) from exc
    if label_smoothing != 0.0:
        raise RuntimeError(
            "selected PRE_BC checkpoint must use label_smoothing=0.0, "
            f"got {label_smoothing}"
        )
    return history, loss


def audit_pre_bc_for_text_control(
    checkpoint_path: Path,
    *,
    text_model_path: str,
    local_files_only: bool,
) -> int:
    from src.smart.inference.text_control import _TextControlRuntime
    from src.smart.utils.finetune import set_model_for_text_control
    from src.utils.checkpoint import load_warm_start_state_dict

    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PRE_BC checkpoint not found: {checkpoint_path}")
    checkpoint = _trusted_torch_load(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("PRE_BC checkpoint must be a mapping")
    hyperparameters = checkpoint.get("hyper_parameters", {})
    if not isinstance(hyperparameters, Mapping) or "model_config" not in hyperparameters:
        raise RuntimeError("PRE_BC checkpoint lacks hyper_parameters.model_config")
    model_config = copy.deepcopy(hyperparameters["model_config"])

    default_path = Path(__file__).resolve().parents[3] / "configs/model/smart.yaml"
    text_config = yaml.safe_load(default_path.read_text(encoding="utf-8"))[
        "model_config"
    ]["text_control"]
    text_config["is_active"] = True
    text_config["freeze_base"] = True
    text_config["model_name_or_path"] = text_model_path
    text_config["local_files_only"] = bool(local_files_only)
    _set(model_config, "text_control", text_config)
    _set(model_config, "finetune", False)

    history, loss = validate_hard_ce_contract(model_config)

    runtime = _TextControlRuntime(model_config)
    report = load_warm_start_state_dict(
        runtime,
        checkpoint_path,
        allowed_missing_prefixes=(
            "encoder.agent_encoder.text_control_adapter.",
        ),
    )
    trainable_names = set_model_for_text_control(runtime.encoder)

    parameters = list(runtime.named_parameters())
    total = sum(parameter.numel() for _, parameter in parameters)
    trainable = sum(
        parameter.numel()
        for _, parameter in parameters
        if parameter.requires_grad
    )
    frozen = total - trainable
    token_config = _get(model_config, "token_processor")
    vocabulary_name = str(_get(token_config, "agent_token_file"))
    vocabulary_path = (
        Path(__file__).resolve().parents[1] / "tokens" / vocabulary_name
    ).resolve()
    if not vocabulary_path.is_file():
        raise FileNotFoundError(f"agent vocabulary not found: {vocabulary_path}")

    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint_epoch: {report.loaded_epoch}")
    print(f"checkpoint_global_step: {report.loaded_global_step}")
    print("allowed_missing_keys:")
    for key in report.missing_keys:
        print(f"  {key}")
    print(f"unexpected_keys: {len(report.unexpected_keys)}")
    print(f"parameters_total: {total}")
    print(f"parameters_frozen: {frozen}")
    print(f"parameters_trainable: {trainable}")
    print("trainable_parameter_tensors:")
    for name in trainable_names:
        print(f"  {name}")
    print(f"agent_vocabulary: {vocabulary_path}")
    print(f"agent_vocabulary_sha256: {_sha256(vocabulary_path)}")
    print(f"history_dynamics_mode: {_get(history, 'mode')}")
    print(
        "loss_mode: "
        f"spatial={_get(loss, 'spatial_aware_smoothing')}, "
        f"label_smoothing={float(_get(loss, 'label_smoothing'))}"
    )
    print("CFG disabled")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a PRE_BC checkpoint for frozen text-control warm start."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--text-model-path",
        default=os.environ.get("TEXT_MODEL_PATH", "distilbert-base-uncased"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    return audit_pre_bc_for_text_control(
        args.checkpoint,
        text_model_path=args.text_model_path,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
