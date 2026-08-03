from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import shutil
from typing import Any, Mapping, Optional, Sequence
import warnings

import torch
from torch import nn


HISTORY_FRAMES = 11
FUTURE_STEPS = 80
PROBABILITY_ONLY_CRITERIUM = "topk_prob"


@dataclass(frozen=True)
class TextControlInferenceRequest:
    checkpoint: Path
    scenario_pickle: Path
    target_agent_id: int
    prompt: str
    output_dir: Path
    n_rollouts: int = 32
    seed: int = 0


@dataclass(frozen=True)
class TextControlInferenceResult:
    trajectories: torch.Tensor
    z: torch.Tensor
    headings: torch.Tensor
    pred_idx: torch.Tensor
    agent_ids: torch.Tensor
    target_agent_id: int
    prompt: str
    scenario_id: str
    seed: int
    output_dir: Path
    rendered_video: Optional[Path] = None


def build_single_agent_override(
    agent_ids: torch.Tensor | Sequence[int],
    target_agent_id: int,
    prompt: str,
) -> tuple[list[str], torch.Tensor]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("text-control prompt must be a non-empty string")
    ids = torch.as_tensor(agent_ids).reshape(-1).detach().cpu()
    matches = ids == int(target_agent_id)
    if int(matches.sum()) != 1:
        raise ValueError(
            f"target agent ID {target_agent_id} must occur exactly once; "
            f"found {int(matches.sum())}"
        )
    cleaned = prompt.strip()
    prompts = [""] * int(ids.numel())
    prompts[int(torch.where(matches)[0][0])] = cleaned
    return prompts, matches.to(dtype=torch.bool)


def _mapping_keys(value: Any) -> list[str]:
    if hasattr(value, "keys"):
        return list(value.keys())
    return []


def make_history_only_inference_view(data: Any) -> Any:
    """Deep-copy a scenario and remove every agent future signal."""

    view = copy.deepcopy(data)
    if "agent" not in view:
        raise KeyError("preprocessed scenario has no 'agent' store")
    agent = view["agent"]
    required = ("position", "heading", "velocity", "valid_mask", "id")
    missing = [key for key in required if key not in agent]
    if missing:
        raise KeyError(f"preprocessed scenario agent store is missing {missing}")

    position = agent["position"]
    heading = agent["heading"]
    velocity = agent["velocity"]
    valid = agent["valid_mask"]
    for name, value in (
        ("position", position),
        ("heading", heading),
        ("velocity", velocity),
        ("valid_mask", valid),
    ):
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            raise ValueError(f"agent.{name} must be a time-indexed tensor")
        if value.shape[1] < HISTORY_FRAMES:
            raise ValueError(
                f"agent.{name} must contain at least {HISTORY_FRAMES} frames"
            )
    n_agent, n_frame = position.shape[:2]
    if any(value.shape[0] != n_agent or value.shape[1] != n_frame for value in (heading, velocity, valid)):
        raise ValueError("agent time-indexed tensors have inconsistent shapes")

    for name in ("position", "heading", "velocity"):
        value = agent[name].clone()
        value[:, HISTORY_FRAMES:] = 0
        agent[name] = value

    sanitized_valid = valid.clone().bool()
    if n_frame > HISTORY_FRAMES:
        sanitized_valid[:, HISTORY_FRAMES:] = sanitized_valid[
            :, HISTORY_FRAMES - 1 : HISTORY_FRAMES
        ].expand(-1, n_frame - HISTORY_FRAMES)
    agent["valid_mask"] = sanitized_valid

    temporal_markers = (
        "future",
        "dynamics",
        "acceleration",
        "angular_speed",
        "yaw_rate",
    )
    protected = {"position", "heading", "velocity", "valid_mask"}
    for key in _mapping_keys(agent):
        if key in protected:
            continue
        value = agent[key]
        lowered = str(key).lower()
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and value.shape[0] == n_agent
            and value.shape[1] == n_frame
            and any(marker in lowered for marker in temporal_markers)
        ):
            value = value.clone()
            value[:, HISTORY_FRAMES:] = 0
            agent[key] = value

    for key in (
        "text_prompt",
        "text_prompt_mask",
        "action_tags",
        "future_action",
        "future_actions",
        "future_gt",
        "validation_gt",
    ):
        if key in agent:
            del agent[key]
    for key in ("future_gt", "validation_gt", "action_tags"):
        if key in view:
            del view[key]
    return view


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class _TextControlRuntime(nn.Module):
    """Only the inference modules; deliberately excludes metrics and GT stores."""

    def __init__(self, model_config: Any) -> None:
        super().__init__()
        from src.smart.modules.smart_decoder import SMARTDecoder
        from src.smart.tokens.token_processor import TokenProcessor

        history_dynamics = _config_get(model_config, "history_dynamics")
        future_token_dynamics = _config_get(model_config, "future_token_dynamics")
        text_control = _config_get(model_config, "text_control")
        if text_control is None or not bool(_config_get(text_control, "is_active", False)):
            raise ValueError("checkpoint model_config does not enable text control")
        token_processor_config = _config_get(model_config, "token_processor")
        decoder_config = _config_get(model_config, "decoder")
        if token_processor_config is None or decoder_config is None:
            raise KeyError("checkpoint model_config lacks token_processor or decoder")

        self.token_processor = TokenProcessor(
            **token_processor_config,
            history_dynamics=history_dynamics,
            future_token_dynamics=future_token_dynamics,
        )
        self.encoder = SMARTDecoder(
            **decoder_config,
            n_token_agent=self.token_processor.n_token_agent,
            history_dynamics=history_dynamics,
            future_token_dynamics=future_token_dynamics,
            text_control=text_control,
        )
        self.validation_rollout_sampling = _config_get(
            model_config,
            "validation_rollout_sampling",
        )


def _trusted_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def _load_runtime(checkpoint_path: Path) -> _TextControlRuntime:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"text-control checkpoint not found: {checkpoint_path}")
    checkpoint = _trusted_torch_load(checkpoint_path)
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise RuntimeError("text-control inference requires a Lightning checkpoint")
    hyperparameters = checkpoint.get("hyper_parameters", {})
    if not isinstance(hyperparameters, Mapping) or "model_config" not in hyperparameters:
        raise RuntimeError("checkpoint lacks hyper_parameters.model_config")
    runtime = _TextControlRuntime(hyperparameters["model_config"])

    state = checkpoint["state_dict"]
    if not isinstance(state, Mapping):
        raise RuntimeError("checkpoint state_dict is not a mapping")
    runtime_state = {
        key: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("encoder.")
    }
    if not runtime_state:
        raise RuntimeError("checkpoint contains no encoder tensors")
    runtime.load_state_dict(runtime_state, strict=True)
    return runtime


def _model_device(model: Any, *, loaded_here: bool) -> torch.device:
    if loaded_here and torch.cuda.is_available():
        return torch.device("cuda")
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _move_nested(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_nested(child, device) for key, child in value.items()}
    return value


def _prepare_single_graph(data: Any, device: torch.device) -> Any:
    n_agent = int(data["agent"]["id"].shape[0])
    n_polyline = int(data["pt_token"]["type"].shape[0])
    data["agent"]["batch"] = torch.zeros(n_agent, dtype=torch.long)
    data["pt_token"]["batch"] = torch.zeros(n_polyline, dtype=torch.long)
    try:
        from torch_geometric.data import HeteroData
    except ImportError:
        return _move_nested(data, device)
    graph = data if isinstance(data, HeteroData) else HeteroData(data)
    graph.num_graphs = 1
    return graph.to(device)


def _validate_sampling_scheme(sampling_scheme: Any) -> None:
    criterium = _config_get(sampling_scheme, "criterium")
    if criterium != PROBABILITY_ONLY_CRITERIUM:
        raise ValueError(
            "custom text inference supports only criterium='topk_prob'; "
            f"{criterium!r} may consult hidden future GT"
        )
    if int(_config_get(sampling_scheme, "num_k", 0)) <= 0:
        raise ValueError("topk_prob sampling requires num_k > 0")
    if float(_config_get(sampling_scheme, "temp", 0.0)) <= 0:
        raise ValueError("topk_prob sampling requires temp > 0")


def _scenario_id(data: Any, fallback: Path) -> str:
    value = data.get("scenario_id") if hasattr(data, "get") else None
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    return str(value) if value is not None else fallback.stem


def _save_outputs(
    request: TextControlInferenceRequest,
    result: TextControlInferenceResult,
) -> None:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pred_traj_10hz": result.trajectories,
            "pred_z_10hz": result.z,
            "pred_head_10hz": result.headings,
            "pred_idx": result.pred_idx,
            "agent_ids": result.agent_ids,
            "seed": result.seed,
        },
        request.output_dir / "rollouts.pt",
    )
    metadata = {
        "checkpoint": str(request.checkpoint),
        "scenario_pickle": str(request.scenario_pickle),
        "scenario_id": result.scenario_id,
        "target_agent_id": result.target_agent_id,
        "prompt": result.prompt,
        "n_rollouts": int(request.n_rollouts),
        "seed": int(request.seed),
    }
    (request.output_dir / "request.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _render_rollout_if_available(
    data: Any,
    result: TextControlInferenceResult,
) -> Optional[Path]:
    tfrecord_path = data.get("tfrecord_path") if hasattr(data, "get") else None
    if isinstance(tfrecord_path, (list, tuple)) and tfrecord_path:
        tfrecord_path = tfrecord_path[0]
    if not tfrecord_path or not Path(str(tfrecord_path)).is_file():
        warnings.warn(
            "No valid TFRecord path is present; tensor rollouts were saved "
            "without rollout.mp4.",
            stacklevel=2,
        )
        return None
    try:
        from src.utils.vis_waymo import VisWaymo
        from src.utils.wosac_utils import (
            get_scenario_id_int_tensor,
            get_scenario_rollouts,
        )

        device = result.trajectories.device
        scenario_rollouts = get_scenario_rollouts(
            scenario_id=get_scenario_id_int_tensor([result.scenario_id], device),
            agent_id=result.agent_ids.to(device),
            agent_batch=torch.zeros(
                result.agent_ids.numel(), dtype=torch.long, device=device
            ),
            pred_traj=result.trajectories.to(device),
            pred_z=result.z.to(device),
            pred_head=result.headings.to(device),
        )
        render_dir = result.output_dir / "waymo_render"
        renderer = VisWaymo(str(tfrecord_path), render_dir)
        renderer.save_video_scenario_rollout(scenario_rollouts[0], 1)
        source = Path(renderer.video_paths[-1])
        destination = result.output_dir / "rollout.mp4"
        shutil.copyfile(source, destination)
        return destination
    except Exception as exc:  # rendering is optional; tensors remain authoritative
        warnings.warn(f"Waymo rollout rendering failed: {exc}", stacklevel=2)
        return None


def run_text_control_inference(
    request: TextControlInferenceRequest,
    model: Optional[Any] = None,
) -> TextControlInferenceResult:
    if int(request.n_rollouts) <= 0:
        raise ValueError("n_rollouts must be positive")
    scenario_path = Path(request.scenario_pickle)
    if not scenario_path.is_file():
        raise FileNotFoundError(f"scenario pickle not found: {scenario_path}")

    with scenario_path.open("rb") as handle:
        raw_data = pickle.load(handle)
    history_data = make_history_only_inference_view(raw_data)
    agent_ids = torch.as_tensor(history_data["agent"]["id"]).reshape(-1).clone()
    prompts, control_mask = build_single_agent_override(
        agent_ids,
        request.target_agent_id,
        request.prompt,
    )
    history_data["agent"]["text_prompt"] = prompts
    history_data["agent"]["text_prompt_mask"] = control_mask

    loaded_here = model is None
    if model is None:
        model = _load_runtime(Path(request.checkpoint))
    device = _model_device(model, loaded_here=loaded_here)
    model = model.to(device)
    model.eval()
    data = _prepare_single_graph(history_data, device)
    tokenized_map, tokenized_agent = model.token_processor(data)

    sampling_scheme = model.validation_rollout_sampling
    _validate_sampling_scheme(sampling_scheme)
    encoded_text_control = model.encoder.encode_text_control(
        prompts,
        control_mask.to(device),
        device,
        training=False,
    )
    if encoded_text_control is None:
        raise RuntimeError("selected text prompt produced no encoded control")

    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    trajectories = []
    z_values = []
    headings = []
    pred_indices = []
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(request.seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(request.seed))
        for _ in range(int(request.n_rollouts)):
            prediction = model.encoder.inference(
                tokenized_map,
                tokenized_agent,
                sampling_scheme,
                encoded_text_control=encoded_text_control,
            )
            trajectories.append(prediction["pred_traj_10hz"].detach().cpu())
            z_values.append(prediction["pred_z_10hz"].detach().cpu())
            headings.append(prediction["pred_head_10hz"].detach().cpu())
            pred_indices.append(prediction["pred_idx"].detach().cpu())

    trajectory_tensor = torch.stack(trajectories, dim=1)
    z_tensor = torch.stack(z_values, dim=1)
    heading_tensor = torch.stack(headings, dim=1)
    pred_idx_tensor = torch.stack(pred_indices, dim=1)
    if trajectory_tensor.shape[0] != agent_ids.numel() or tuple(
        trajectory_tensor.shape[-2:]
    ) != (FUTURE_STEPS, 2):
        raise RuntimeError(
            "decoder returned an invalid trajectory shape: "
            f"{tuple(trajectory_tensor.shape)}"
        )

    result = TextControlInferenceResult(
        trajectories=trajectory_tensor,
        z=z_tensor,
        headings=heading_tensor,
        pred_idx=pred_idx_tensor,
        agent_ids=agent_ids.detach().cpu(),
        target_agent_id=int(request.target_agent_id),
        prompt=request.prompt.strip(),
        scenario_id=_scenario_id(history_data, scenario_path),
        seed=int(request.seed),
        output_dir=Path(request.output_dir),
    )
    _save_outputs(request, result)
    rendered_video = _render_rollout_if_available(history_data, result)
    if rendered_video is not None:
        result = TextControlInferenceResult(
            **{
                **result.__dict__,
                "rendered_video": rendered_video,
            }
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one-agent ECoSim-style text-controlled CatK rollouts."
    )
    parser.add_argument("checkpoint", type=Path, help="trained text-control checkpoint")
    parser.add_argument("scenario_pickle", type=Path, help="one preprocessed scenario")
    parser.add_argument("target_agent_id", type=int, help="agent ID to control")
    parser.add_argument("prompt", help="one non-empty text instruction")
    parser.add_argument("output_dir", type=Path, help="directory for tensor/video outputs")
    parser.add_argument("n_rollouts", type=int, nargs="?", default=32)
    parser.add_argument("seed", type=int, nargs="?", default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    request = TextControlInferenceRequest(
        checkpoint=args.checkpoint,
        scenario_pickle=args.scenario_pickle,
        target_agent_id=args.target_agent_id,
        prompt=args.prompt,
        output_dir=args.output_dir,
        n_rollouts=args.n_rollouts,
        seed=args.seed,
    )
    result = run_text_control_inference(request)
    print(f"Saved rollouts to {result.output_dir / 'rollouts.pt'}")
    if result.rendered_video is not None:
        print(f"Saved visualization to {result.rendered_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TextControlInferenceRequest",
    "TextControlInferenceResult",
    "build_single_agent_override",
    "make_history_only_inference_view",
    "run_text_control_inference",
]
