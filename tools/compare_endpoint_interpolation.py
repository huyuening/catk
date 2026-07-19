#!/usr/bin/env python3
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

"""Compare CatK raw token expansion with TrajTok endpoint interpolation.

The selected validation agent is shown in the same six-panel diagnostic used
by TrajTok: XY, heading, linear speed/acceleration, and angular
speed/acceleration. Both passes use the same checkpoint, seed, and token path.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJTOK_ROOT = os.environ.get("TRAJTOK_ROOT", "/root/workspace/TrajTok")
TYPE_NAMES = {0: "vehicle", 1: "pedestrian", 2: "cyclist"}

# Running ``python tools/compare_endpoint_interpolation.py`` sets sys.path[0]
# to ``tools/`` rather than the repository root. Hydra resolves targets such as
# ``src.smart.datamodules.MultiDataModule`` through normal Python imports.
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)


def parse_args():
    checkpoint_default = os.environ.get("CATK_CKPT")
    parser = argparse.ArgumentParser(
        description=(
            "Generate one CatK scene twice with an identical token path and "
            "compare raw 10 Hz token expansion against endpoint interpolation."
        )
    )
    parser.add_argument(
        "--ckpt-path",
        default=checkpoint_default,
        required=checkpoint_default is None,
        help="CatK checkpoint; defaults to CATK_CKPT.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--rollout-index", type=int, default=0)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--agent-index", type=int, default=None)
    parser.add_argument("--agent-id", type=int, default=None)
    parser.add_argument(
        "--select-motion-mode",
        choices=[
            "any",
            "endpoint_interpolation",
            "low_speed_reconstruction",
            "static_reconstruction",
            "raw_token_expansion",
            "mixed",
        ],
        default="any",
        help=(
            "When no agent id/index is given, restrict automatic selection to "
            "one endpoint postprocessing mode."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=817)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--sampling-num-k",
        type=int,
        default=1,
        help="Top-k sampling size; K=1 is the clearest deterministic comparison.",
    )
    parser.add_argument("--sampling-temp", type=float, default=1.0)
    parser.add_argument("--trajtok-root", default=DEFAULT_TRAJTOK_ROOT)
    parser.add_argument(
        "--output-dir",
        default="outputs/catk_endpoint_interpolation_check",
    )
    parser.add_argument(
        "--config-overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides; place this option last.",
    )
    return parser.parse_args()


def load_trajtok_plotter(trajtok_root):
    plotter_path = Path(trajtok_root).expanduser().resolve() / "tools" / "visualize_reconstruction.py"
    if not plotter_path.is_file():
        raise FileNotFoundError(
            f"TrajTok visualization tool was not found: {plotter_path}. "
            "Set --trajtok-root or TRAJTOK_ROOT."
        )
    spec = importlib.util.spec_from_file_location(
        "catk_trajtok_visualize_reconstruction", plotter_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load TrajTok visualization tool: {plotter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_device(device_arg):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_cfg(args):
    import hydra
    from omegaconf import open_dict

    overrides = ["experiment=inference", *args.config_overrides]
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"), version_base=None
    ):
        cfg = hydra.compose(config_name="run.yaml", overrides=overrides)

    with open_dict(cfg):
        cfg.data.val_batch_size = 1
        cfg.data.test_batch_size = 1
        cfg.data.num_workers = 0
        cfg.data.shuffle = False
        cfg.data.pin_memory = False
        cfg.data.persistent_workers = False
        cfg.model.model_config.n_rollout_closed_val = args.num_rollouts
        cfg.model.model_config.n_batch_wosac_metric = 0
        cfg.model.model_config.n_vis_batch = 0
        cfg.model.model_config.val_open_loop = False
        cfg.model.model_config.val_closed_loop = False
        cfg.model.model_config.trajtok_root = str(
            Path(args.trajtok_root).expanduser().resolve()
        )
        sampling = cfg.model.model_config.validation_rollout_sampling
        sampling.num_k = args.sampling_num_k
        sampling.temp = args.sampling_temp
        cfg.model.model_config.decoder.endpoint_interpolation.is_active = False
    return cfg


def load_model(cfg, checkpoint_path, device):
    import hydra
    import torch

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing checkpoint keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Unexpected checkpoint keys: {len(unexpected_keys)}")
    model.to(device)
    model.eval()
    return model


def get_single_scene(datamodule, split, scene_index, scenario_id, device):
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader

    stage = "validate" if split == "val" else "test"
    datamodule.setup(stage)
    dataset = datamodule.val_dataset if split == "val" else datamodule.test_dataset
    raw_paths = list(dataset.raw_paths)
    if scenario_id is not None:
        scenario_stem = Path(scenario_id).stem
        matches = [
            index
            for index, raw_path in enumerate(raw_paths)
            if Path(raw_path).stem == scenario_stem
        ]
        if not matches:
            raise FileNotFoundError(
                f"scenario-id {scenario_id} was not found in split {split}"
            )
        scene_index = matches[0]
    if scene_index < 0 or scene_index >= len(dataset):
        raise IndexError(
            f"scene-index {scene_index} is out of range [0, {len(dataset) - 1}]"
        )

    dataloader = DataLoader(Subset(dataset, [scene_index]), batch_size=1, shuffle=False)
    data = next(iter(dataloader)).to(device)
    return data, scene_index


def get_scenario_id(data):
    scenario_id = data["scenario_id"]
    if isinstance(scenario_id, (list, tuple)):
        scenario_id = scenario_id[0]
    return str(scenario_id)


def set_endpoint_interpolation_active(model, is_active):
    from omegaconf import open_dict

    interpolator = model.encoder.agent_encoder.endpoint_interpolator
    config = interpolator.config
    if isinstance(config, dict):
        config["is_active"] = bool(is_active)
        return
    with open_dict(config):
        config.is_active = bool(is_active)


def reset_sampling_seed(seed):
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_rollouts(model, data, num_rollouts, seed):
    import torch

    reset_sampling_seed(seed)
    tokenized_map, tokenized_agent = model.token_processor(data)
    rollout_states = []
    rollout_tokens = []
    with torch.no_grad():
        for _ in range(num_rollouts):
            pred = model.encoder.inference(
                tokenized_map,
                tokenized_agent,
                model.validation_rollout_sampling,
            )
            rollout_states.append(
                torch.cat(
                    [
                        pred["pred_traj_10hz"],
                        pred["pred_z_10hz"].unsqueeze(-1),
                        pred["pred_head_10hz"].unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )
            rollout_tokens.append(pred["pred_idx"])
    return torch.stack(rollout_states), torch.stack(rollout_tokens)


def save_rollout(path, states, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "simulated_states": states.detach().cpu().numpy(),
        "agent_id": data["agent"]["id"].detach().cpu().numpy(),
        "agent_type": data["agent"]["type"].detach().cpu().numpy(),
        "scenario_id": get_scenario_id(data),
    }
    with path.open("wb") as output_file:
        pickle.dump(result, output_file)


def future_gt_states(data, num_historical_steps, num_future_steps):
    position = data["agent"]["position"].detach().cpu().numpy().astype(np.float32)
    heading = data["agent"]["heading"].detach().cpu().numpy().astype(np.float32)
    valid = data["agent"]["valid_mask"].detach().cpu().numpy().astype(bool)
    start = int(num_historical_steps)
    end = min(start + int(num_future_steps), position.shape[1])
    output = np.full(
        (position.shape[0], int(num_future_steps), 4), np.nan, dtype=np.float32
    )
    if end <= start:
        return output
    n_step = end - start
    output[:, :n_step, :3] = position[:, start:end, :3]
    output[:, :n_step, 3] = heading[:, start:end]
    future = output[:, :n_step]
    future[~valid[:, start:end]] = np.nan
    output[:, :n_step] = future
    return output


def history_gt_states(data, num_historical_steps):
    position = data["agent"]["position"].detach().cpu().numpy().astype(np.float32)
    heading = data["agent"]["heading"].detach().cpu().numpy().astype(np.float32)
    valid = data["agent"]["valid_mask"].detach().cpu().numpy().astype(bool)
    end = min(int(num_historical_steps), position.shape[1])
    output = np.full(
        (position.shape[0], int(num_historical_steps), 4),
        np.nan,
        dtype=np.float32,
    )
    if end <= 0:
        return output
    output[:, :end, :3] = position[:, :end, :3]
    output[:, :end, 3] = heading[:, :end]
    history = output[:, :end]
    history[~valid[:, :end]] = np.nan
    output[:, :end] = history
    return output


def motion_split_summary(model, data, raw_states, token_stride=5):
    import torch

    with torch.no_grad():
        _, tokenized_agent = model.token_processor(data)
    decoder = model.encoder.agent_encoder
    interpolator = decoder.endpoint_interpolator
    step_current_2hz = (decoder.num_historical_steps - 1) // decoder.shift
    start_pos = tokenized_agent["gt_pos"][:, step_current_2hz - 1]

    endpoint_indices = np.arange(
        token_stride - 1, raw_states.shape[-2], token_stride
    )
    endpoint_pos = raw_states[..., endpoint_indices, :2]
    n_rollout, n_agent, n_segment, _ = endpoint_pos.shape
    flat_endpoint = torch.as_tensor(endpoint_pos.reshape(-1, n_segment, 2))
    flat_start = (
        start_pos.detach()
        .cpu()
        .unsqueeze(0)
        .expand(n_rollout, -1, -1)
        .reshape(-1, 2)
    )
    flat_type = (
        tokenized_agent["type"]
        .detach()
        .cpu()
        .unsqueeze(0)
        .expand(n_rollout, -1)
        .reshape(-1)
    )
    segment_speed_t = interpolator._endpoint_segment_speed(flat_start, flat_endpoint)
    segment_speed = segment_speed_t.numpy().reshape(n_rollout, n_agent, n_segment)
    min_speed = segment_speed.min(axis=-1)
    max_speed = segment_speed.max(axis=-1)

    control_pos = np.concatenate(
        [
            flat_start.numpy().reshape(n_rollout, n_agent, 1, 2),
            endpoint_pos,
        ],
        axis=-2,
    )
    endpoint_span = np.linalg.norm(
        control_pos.max(axis=-2) - control_pos.min(axis=-2), axis=-1
    )

    if bool(interpolator._get("low_speed_reconstruction", False)):
        static_mask = interpolator._static_agent_mask(
            flat_start,
            flat_endpoint,
            segment_speed_t,
            flat_type,
        )
        low_speed_mask = (
            interpolator._low_speed_agent_mask(segment_speed_t) & ~static_mask
        )
        static_mask = static_mask.numpy().reshape(n_rollout, n_agent)
        low_speed_mask = low_speed_mask.numpy().reshape(n_rollout, n_agent)
        mode = np.where(
            static_mask,
            "static_reconstruction",
            np.where(
                low_speed_mask,
                "low_speed_reconstruction",
                "endpoint_interpolation",
            ),
        )
    else:
        moving_agent = interpolator._moving_agent_mask(segment_speed_t)
        moving_segment = interpolator._moving_segment_mask(segment_speed_t)
        all_segments = moving_segment.all(dim=-1)
        any_segments = moving_segment.any(dim=-1)
        mode_flat = np.where(
            (~moving_agent).numpy(),
            "raw_token_expansion",
            np.where(
                all_segments.numpy(),
                "endpoint_interpolation",
                np.where(any_segments.numpy(), "mixed", "raw_token_expansion"),
            ),
        )
        mode = mode_flat.reshape(n_rollout, n_agent)

    unique, counts = np.unique(mode, return_counts=True)
    return {
        "mode": mode,
        "min_segment_speed_mps": min_speed,
        "max_segment_speed_mps": max_speed,
        "endpoint_span_m": endpoint_span,
        "counts": {str(key): int(value) for key, value in zip(unique, counts)},
    }


def endpoint_delta_summary(raw_states, post_states, token_stride=5):
    endpoint_indices = list(
        range(token_stride - 1, raw_states.shape[-2], token_stride)
    )
    if not endpoint_indices:
        return {
            "mean_m": 0.0,
            "max_m": 0.0,
            "heading_mean_rad": 0.0,
            "heading_max_rad": 0.0,
            "indices": [],
        }
    position_delta = (
        raw_states[..., endpoint_indices, :2]
        - post_states[..., endpoint_indices, :2]
    )
    position_delta = np.linalg.norm(position_delta, axis=-1)
    heading_delta = np.abs(
        (
            raw_states[..., endpoint_indices, 3]
            - post_states[..., endpoint_indices, 3]
            + np.pi
        )
        % (2 * np.pi)
        - np.pi
    )
    return {
        "mean_m": float(position_delta.mean()),
        "max_m": float(position_delta.max()),
        "heading_mean_rad": float(heading_delta.mean()),
        "heading_max_rad": float(heading_delta.max()),
        "indices": endpoint_indices,
    }


def mean_abs_kinematics(plotter, states, dt):
    kinematics = plotter.compute_kinematics(states, dt)
    return {
        key: float(np.nanmean(np.abs(values)))
        for key, values in kinematics.items()
    }


def write_selected_csv(path, plotter, raw_states, post_states, gt_states, dt):
    raw_kinematics = plotter.compute_kinematics(raw_states, dt)
    post_kinematics = plotter.compute_kinematics(post_states, dt)
    gt_kinematics = plotter.compute_kinematics(gt_states, dt)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time_s"]
    for prefix in ("raw", "post_interp", "gt"):
        fieldnames.extend(
            [
                f"{prefix}_x_m",
                f"{prefix}_y_m",
                f"{prefix}_z_m",
                f"{prefix}_heading_rad",
                f"{prefix}_heading_unwrapped_rad",
                f"{prefix}_linear_speed_mps",
                f"{prefix}_linear_acceleration_mps2",
                f"{prefix}_angular_speed_radps",
                f"{prefix}_angular_acceleration_radps2",
            ]
        )

    state_sets = {"raw": raw_states, "post_interp": post_states, "gt": gt_states}
    kinematic_sets = {
        "raw": raw_kinematics,
        "post_interp": post_kinematics,
        "gt": gt_kinematics,
    }
    unwrapped_headings = {
        prefix: np.unwrap(states[:, 3]) for prefix, states in state_sets.items()
    }
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(raw_states.shape[0]):
            row = {"time_s": float((step + 1) * dt)}
            for prefix, states in state_sets.items():
                kinematics = kinematic_sets[prefix]
                row.update(
                    {
                        f"{prefix}_x_m": float(states[step, 0]),
                        f"{prefix}_y_m": float(states[step, 1]),
                        f"{prefix}_z_m": float(states[step, 2]),
                        f"{prefix}_heading_rad": float(states[step, 3]),
                        f"{prefix}_heading_unwrapped_rad": float(
                            unwrapped_headings[prefix][step]
                        ),
                        f"{prefix}_linear_speed_mps": float(
                            kinematics["linear_speed"][step]
                        ),
                        f"{prefix}_linear_acceleration_mps2": float(
                            kinematics["linear_acceleration"][step]
                        ),
                        f"{prefix}_angular_speed_radps": float(
                            kinematics["angular_speed"][step]
                        ),
                        f"{prefix}_angular_acceleration_radps2": float(
                            kinematics["angular_acceleration"][step]
                        ),
                    }
                )
            writer.writerow(row)


def resolve_outputs(output_dir, scenario_id, rollout_index, agent_id):
    root = Path(output_dir)
    stem = f"{scenario_id}_rollout{rollout_index}_agent{agent_id}"
    return {
        "figure": root / "images" / f"{stem}.png",
        "csv": root / "tables" / f"{stem}.csv",
        "raw_pkl": root / "pkls" / f"{stem}_raw.pkl",
        "post_pkl": root / "pkls" / f"{stem}_post_interp.pkl",
        "summary": root / "summaries" / f"{stem}.json",
    }


def main():
    args = parse_args()

    import hydra
    import torch
    from omegaconf import OmegaConf

    torch.set_float32_matmul_precision("high")
    plotter = load_trajtok_plotter(args.trajtok_root)
    device = resolve_device(args.device)
    cfg = build_cfg(args)
    print(f"Using device: {device}")
    print(f"TrajTok plotter: {Path(args.trajtok_root) / 'tools' / 'visualize_reconstruction.py'}")
    print(f"Data config:\n{OmegaConf.to_yaml(cfg.data)}")

    datamodule = hydra.utils.instantiate(cfg.data)
    model = load_model(cfg, args.ckpt_path, device)
    data, resolved_scene_index = get_single_scene(
        datamodule,
        args.split,
        args.scene_index,
        args.scenario_id,
        device,
    )
    scenario_id = get_scenario_id(data)
    print(
        f"Loaded scenario: split={args.split}, scene_index={resolved_scene_index}, "
        f"scenario_id={scenario_id}"
    )

    set_endpoint_interpolation_active(model, False)
    raw_states_t, raw_tokens = generate_rollouts(
        model, data, args.num_rollouts, args.seed
    )
    set_endpoint_interpolation_active(model, True)
    post_states_t, post_tokens = generate_rollouts(
        model, data, args.num_rollouts, args.seed
    )
    if not torch.equal(raw_tokens, post_tokens):
        mismatch = int((raw_tokens != post_tokens).sum().item())
        raise RuntimeError(
            "Raw and post-interpolation token paths differ despite seed reset: "
            f"{mismatch} token indices mismatch. Comparison aborted."
        )
    print("Token path check: identical")

    raw_states = raw_states_t.detach().cpu().numpy().astype(np.float32)
    post_states = post_states_t.detach().cpu().numpy().astype(np.float32)
    if args.rollout_index < 0 or args.rollout_index >= raw_states.shape[0]:
        raise ValueError(
            f"rollout-index {args.rollout_index} is out of range "
            f"[0, {raw_states.shape[0] - 1}]"
        )
    agent_ids = data["agent"]["id"].detach().cpu().numpy()
    agent_types = data["agent"]["type"].long().detach().cpu().numpy()
    raw_rollout = raw_states[args.rollout_index]
    post_rollout = post_states[args.rollout_index]
    split_summary = motion_split_summary(model, data, raw_states)

    candidate_mask = None
    if (
        args.select_motion_mode != "any"
        and args.agent_id is None
        and args.agent_index is None
    ):
        candidate_mask = (
            split_summary["mode"][args.rollout_index]
            == args.select_motion_mode
        )
        if not np.any(candidate_mask):
            raise ValueError(
                f"No agents with mode={args.select_motion_mode} in "
                f"scenario={scenario_id}, rollout={args.rollout_index}."
            )
    agent_index = plotter.choose_agent_index(
        args,
        agent_ids,
        raw_rollout,
        post_rollout,
        args.dt,
        candidate_mask=candidate_mask,
    )
    selected_agent_id = int(agent_ids[agent_index])
    selected_mode = str(split_summary["mode"][args.rollout_index, agent_index])
    selected_min_speed = float(
        split_summary["min_segment_speed_mps"][args.rollout_index, agent_index]
    )
    selected_max_speed = float(
        split_summary["max_segment_speed_mps"][args.rollout_index, agent_index]
    )
    selected_span = float(
        split_summary["endpoint_span_m"][args.rollout_index, agent_index]
    )

    num_historical_steps = cfg.model.model_config.decoder.num_historical_steps
    gt_future = future_gt_states(data, num_historical_steps, raw_states.shape[-2])
    gt_history = history_gt_states(data, num_historical_steps)
    outputs = resolve_outputs(
        args.output_dir,
        scenario_id,
        args.rollout_index,
        selected_agent_id,
    )

    title = (
        f"CatK raw vs post_interp | split={args.split}, scenario={scenario_id}, "
        f"rollout={args.rollout_index}, agent_id={selected_agent_id}, "
        f"mode={selected_mode}, min/max_seg_speed="
        f"{selected_min_speed:.3f}/{selected_max_speed:.3f}m/s, "
        f"span={selected_span:.3f}m"
    )
    plotter.plot_reconstruction(
        raw_states=raw_rollout[agent_index],
        recon_states=post_rollout[agent_index],
        output_path=outputs["figure"],
        title=title,
        dt=args.dt,
        recon_label=selected_mode,
        token_stride=5,
        show_heading_controls=True,
        gt_states=gt_future[agent_index],
        history_states=gt_history[agent_index],
    )
    plotter.print_summary(
        agent_index=agent_index,
        agent_id=selected_agent_id,
        agent_type=agent_types[agent_index],
        raw_states=raw_rollout[agent_index],
        recon_states=post_rollout[agent_index],
        dt=args.dt,
        output_path=outputs["figure"],
    )
    write_selected_csv(
        outputs["csv"],
        plotter,
        raw_rollout[agent_index],
        post_rollout[agent_index],
        gt_future[agent_index],
        args.dt,
    )
    save_rollout(outputs["raw_pkl"], raw_states_t, data)
    save_rollout(outputs["post_pkl"], post_states_t, data)

    selected_raw_kinematics = mean_abs_kinematics(
        plotter, raw_rollout[agent_index], args.dt
    )
    selected_post_kinematics = mean_abs_kinematics(
        plotter, post_rollout[agent_index], args.dt
    )
    summary = {
        "ckpt_path": str(Path(args.ckpt_path).expanduser().resolve()),
        "split": args.split,
        "scene_index": resolved_scene_index,
        "scenario_id": scenario_id,
        "seed": args.seed,
        "sampling_num_k": args.sampling_num_k,
        "sampling_temp": args.sampling_temp,
        "num_rollouts": args.num_rollouts,
        "rollout_index": args.rollout_index,
        "token_paths_identical": True,
        "selected_agent": {
            "index": agent_index,
            "id": selected_agent_id,
            "type": TYPE_NAMES.get(int(agent_types[agent_index]), "unknown"),
            "mode": selected_mode,
            "min_segment_speed_mps": selected_min_speed,
            "max_segment_speed_mps": selected_max_speed,
            "endpoint_span_m": selected_span,
        },
        "motion_mode_counts": split_summary["counts"],
        "endpoint_delta_all_agents": endpoint_delta_summary(raw_states, post_states),
        "endpoint_delta_selected_agent": endpoint_delta_summary(
            raw_rollout[agent_index], post_rollout[agent_index]
        ),
        "mean_abs_kinematics_selected_agent": {
            "raw": selected_raw_kinematics,
            "post_interp": selected_post_kinematics,
        },
        "mean_abs_kinematics_all_agents": {
            "raw": mean_abs_kinematics(plotter, raw_states, args.dt),
            "post_interp": mean_abs_kinematics(plotter, post_states, args.dt),
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["summary"].parent.mkdir(parents=True, exist_ok=True)
    with outputs["summary"].open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print(f"Saved per-step CSV: {outputs['csv']}")
    print(f"Saved raw rollout: {outputs['raw_pkl']}")
    print(f"Saved post-interpolation rollout: {outputs['post_pkl']}")
    print(f"Saved summary: {outputs['summary']}")
    print("Mean absolute kinematics, selected agent (raw -> post_interp):")
    for key in selected_raw_kinematics:
        print(
            f"  {key}: {selected_raw_kinematics[key]:.6f} -> "
            f"{selected_post_kinematics[key]:.6f}"
        )


if __name__ == "__main__":
    main()
