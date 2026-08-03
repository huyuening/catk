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

import math
from pathlib import Path

import hydra
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from src.smart.metrics import (
    CrossEntropy,
    FastWOSACMetrics,
    TokenCls,
    WOSACMetrics,
    WOSACSubmission,
    minADE,
)
from src.smart.datasets.text_prompts import flatten_batched_prompts
from src.smart.modules.smart_decoder import SMARTDecoder
from src.smart.tokens.token_processor import TokenProcessor
from src.smart.utils.finetune import (
    set_model_for_finetuning,
    set_model_for_text_control,
)
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts


class SMART(LightningModule):

    def __init__(self, model_config) -> None:
        super(SMART, self).__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop
        history_dynamics = model_config.get("history_dynamics", None)
        future_token_dynamics = model_config.get(
            "future_token_dynamics",
            None,
        )
        text_control = model_config.get("text_control", None)
        self.text_control_active = bool(
            text_control is not None
            and text_control.get("is_active", False)
        )
        if self.text_control_active and bool(model_config.finetune):
            raise ValueError(
                "model_config.finetune and text control cannot both be enabled"
            )
        self.token_processor = TokenProcessor(
            **model_config.token_processor,
            history_dynamics=history_dynamics,
            future_token_dynamics=future_token_dynamics,
        )

        self.encoder = SMARTDecoder(
            **model_config.decoder,
            n_token_agent=self.token_processor.n_token_agent,
            history_dynamics=history_dynamics,
            future_token_dynamics=future_token_dynamics,
            text_control=text_control,
        )
        if self.text_control_active:
            self.text_control_trainable_names = set_model_for_text_control(
                self.encoder
            )
        else:
            self.text_control_trainable_names = []
            set_model_for_finetuning(self.encoder, model_config.finetune)

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)
        self.wosac_backend = model_config.get("wosac_backend", "official")
        self.wosac_metrics_version = str(
            model_config.get("wosac_metrics_version", "2024")
        )
        if self.wosac_backend == "fast":
            self.wosac_metrics = FastWOSACMetrics(
                "val_closed",
                version=self.wosac_metrics_version,
                gt_scenario_dir=model_config.get("fast_wosac_gt_dir"),
                require_preprocessed_gt=model_config.get(
                    "fast_wosac_require_preprocessed_gt",
                    False,
                ),
            )
        elif self.wosac_backend == "official":
            if self.wosac_metrics_version != "2024":
                raise ValueError(
                    "CatK's official WOSAC backend only supports version 2024. "
                    "Use wosac_backend=fast for WOSAC 2025."
                )
            self.wosac_metrics = WOSACMetrics("val_closed")
        else:
            raise ValueError(f"Unknown WOSAC backend: {self.wosac_backend}")
        self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)
        self.training_loss = CrossEntropy(**model_config.training_loss)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling

    @staticmethod
    def _scenario_context(data) -> str:
        try:
            scenario_ids = data["scenario_id"]
        except (KeyError, TypeError):
            return "<unknown scenario>"
        return str(scenario_ids)

    def _prepare_text_control(self, data):
        if not self.text_control_active:
            return None

        agent_data = data["agent"]
        scenario_context = self._scenario_context(data)
        try:
            prompts = flatten_batched_prompts(agent_data["text_prompt"])
            prompt_mask = agent_data["text_prompt_mask"]
            agent_ids = agent_data["id"]
        except KeyError as exc:
            raise KeyError(
                f"missing text-control field {exc.args[0]!r} for "
                f"{scenario_context}"
            ) from exc

        if not isinstance(prompt_mask, torch.Tensor):
            raise TypeError(
                f"text_prompt_mask must be a tensor for {scenario_context}"
            )
        if prompt_mask.dtype != torch.bool:
            raise ValueError(
                f"text_prompt_mask must be Boolean for {scenario_context}"
            )
        mask = prompt_mask.reshape(-1).clone()
        n_agent = int(agent_ids.shape[0])
        if len(prompts) != n_agent or mask.numel() != n_agent:
            raise ValueError(
                "text prompt alignment mismatch for "
                f"{scenario_context}: prompts={len(prompts)}, "
                f"mask={mask.numel()}, agents={n_agent}"
            )

        if "train_mask" in agent_data:
            train_mask = agent_data["train_mask"]
            if not isinstance(train_mask, torch.Tensor):
                raise TypeError(
                    f"train_mask must be a tensor for {scenario_context}"
                )
            if train_mask.dtype != torch.bool:
                raise ValueError(
                    f"train_mask must be Boolean for {scenario_context}"
                )
            train_mask = train_mask.reshape(-1).to(mask.device)
            if train_mask.numel() != n_agent:
                raise ValueError(
                    "train_mask alignment mismatch for "
                    f"{scenario_context}: mask={train_mask.numel()}, "
                    f"agents={n_agent}"
                )
            mask &= train_mask

        if self.training:
            fraction = (
                mask.float().mean()
                if mask.numel()
                else torch.zeros((), device=mask.device)
            )
            self.log(
                "train/text_control_fraction",
                fraction,
                on_step=True,
                batch_size=1,
            )

        return self.encoder.encode_text_control(
            prompts,
            mask,
            mask.device,
            training=self.training,
        )

    def _add_text_control_ddp_anchor(self, loss: torch.Tensor) -> torch.Tensor:
        if not self.text_control_active:
            return loss
        anchor = loss.new_zeros(())
        for parameter in self.parameters():
            if parameter.requires_grad:
                anchor = anchor + parameter.reshape(-1)[0] * 0.0
        return loss + anchor

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        encoded_text_control = self._prepare_text_control(data)
        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(
                tokenized_map,
                tokenized_agent,
                encoded_text_control=encoded_text_control,
            )
        else:
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                sampling_scheme=self.training_rollout_sampling,
                encoded_text_control=encoded_text_control,
            )

        loss = self.training_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
            token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
            gt_idx=tokenized_agent["gt_idx"][:, 2:],  # [n_agent, 16]
            train_mask=data["agent"]["train_mask"],  # [n_agent]
            current_epoch=self.current_epoch,
        )
        loss = self._add_text_control_ddp_anchor(loss)
        self.log("train/loss", loss, on_step=True, batch_size=1)

        return loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        encoded_text_control = self._prepare_text_control(data)

        # ! open-loop vlidation
        if self.val_open_loop:
            pred = self.encoder(
                tokenized_map,
                tokenized_agent,
                encoded_text_control=encoded_text_control,
            )
            loss = self.training_loss(
                **pred,
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_agent, 2]
                token_traj=tokenized_agent["token_traj"],  # [n_agent, n_token, 4, 2]
                gt_idx=tokenized_agent["gt_idx"][:, 2:],  # [n_agent, 16]
            )

            self.TokenCls.update(
                # action that goes from [(10->15), ..., (85->90)]
                pred=pred["next_token_logits"],  # [n_agent, 16, n_token]
                pred_valid=pred["next_token_valid"],  # [n_agent, 16]
                target=tokenized_agent["gt_idx"][:, 2:],
                target_valid=tokenized_agent["valid_mask"][:, 2:],
            )
            self.log(
                "val_open/acc",
                self.TokenCls,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)

        # ! closed-loop vlidation
        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self.n_rollout_closed_val):
                pred = self.encoder.inference(
                    tokenized_map,
                    tokenized_agent,
                    self.validation_rollout_sampling,
                    encoded_text_control=encoded_text_control,
                )
                pred_traj.append(pred["pred_traj_10hz"])
                pred_z.append(pred["pred_z_10hz"])
                pred_head.append(pred["pred_head_10hz"])

            pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
            pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
            pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]

            # ! WOSAC
            scenario_rollouts = None
            if self.wosac_submission.is_active:  # ! save WOSAC submission
                self.wosac_submission.update(
                    scenario_id=data["scenario_id"],
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                    global_rank=self.global_rank,
                )
                _gpu_dict_sync = self.wosac_submission.compute()
                if self.global_rank == 0:
                    for k in _gpu_dict_sync.keys():  # single gpu fix
                        if type(_gpu_dict_sync[k]) is list:
                            _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
                    scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
                    self.wosac_submission.aggregate_rollouts(scenario_rollouts)
                self.wosac_submission.reset()

            else:  # ! compute metrics, disable if save WOSAC submission
                self.minADE.update(
                    pred=pred_traj,
                    target=data["agent"]["position"][
                        :, self.num_historical_steps :, : pred_traj.shape[-1]
                    ],
                    target_valid=data["agent"]["valid_mask"][
                        :, self.num_historical_steps :
                    ],
                )

                # WOSAC metrics
                should_compute_wosac = (
                    self.n_batch_wosac_metric < 0
                    or batch_idx < self.n_batch_wosac_metric
                )
                if should_compute_wosac:
                    if self.wosac_backend == "fast":
                        # CatK keeps agents first; TrajTok Fast WOSAC expects
                        # [n_rollout, n_agent, n_step, (x, y, z, yaw)].
                        simulated_states = torch.cat(
                            [
                                pred_traj,
                                pred_z.unsqueeze(-1),
                                pred_head.unsqueeze(-1),
                            ],
                            dim=-1,
                        ).permute(1, 0, 2, 3).contiguous()
                        self.wosac_metrics.update(
                            scenario_files=data["tfrecord_path"],
                            scenario_ids=data["scenario_id"],
                            agent_id=data["agent"]["id"],
                            agent_batch=data["agent"]["batch"],
                            simulated_states=simulated_states,
                        )
                    else:
                        device = pred_traj.device
                        scenario_rollouts = get_scenario_rollouts(
                            scenario_id=get_scenario_id_int_tensor(
                                data["scenario_id"], device
                            ),
                            agent_id=data["agent"]["id"],
                            agent_batch=data["agent"]["batch"],
                            pred_traj=pred_traj,
                            pred_z=pred_z,
                            pred_head=pred_head,
                        )
                        self.wosac_metrics.update(
                            data["tfrecord_path"], scenario_rollouts
                        )

                if (
                    self.wosac_backend == "fast"
                    and self.global_rank == 0
                    and batch_idx < self.n_vis_batch
                ):
                    scenario_rollouts = get_scenario_rollouts(
                        scenario_id=get_scenario_id_int_tensor(
                            data["scenario_id"], pred_traj.device
                        ),
                        agent_id=data["agent"]["id"],
                        agent_batch=data["agent"]["batch"],
                        pred_traj=pred_traj,
                        pred_z=pred_z,
                        pred_head=pred_head,
                    )

            # ! visualization
            if self.global_rank == 0 and batch_idx < self.n_vis_batch:
                if scenario_rollouts is not None:
                    for _i_sc in range(self.n_vis_scenario):
                        _vis = VisWaymo(
                            scenario_path=data["tfrecord_path"][_i_sc],
                            save_dir=self.video_dir
                            / f"batch_{batch_idx:02d}-scenario_{_i_sc:02d}",
                        )
                        _vis.save_video_scenario_rollout(
                            scenario_rollouts[_i_sc], self.n_vis_rollout
                        )
                        for _path in _vis.video_paths:
                            self.logger.log_video(
                                "/".join(_path.split("/")[-3:]), [_path]
                            )

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_wosac_metrics = self.wosac_metrics.compute()
                epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
                if self.global_rank == 0:
                    epoch_wosac_metrics["epoch"] = (
                        self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    )
                    metric_lines = [
                        f"WOSAC {self.wosac_metrics_version} validation metrics:"
                    ]
                    for metric_name, metric_value in sorted(
                        epoch_wosac_metrics.items()
                    ):
                        if isinstance(metric_value, torch.Tensor):
                            metric_value = metric_value.detach().cpu().item()
                        if isinstance(metric_value, float):
                            metric_lines.append(f"  {metric_name}: {metric_value:.8f}")
                        else:
                            metric_lines.append(f"  {metric_name}: {metric_value}")
                    self.print("\n".join(metric_lines))
                    if self.logger is not None:
                        self.logger.log_metrics(epoch_wosac_metrics)

                self.wosac_metrics.reset()
                self.minADE.reset()

            if self.global_rank == 0:
                if self.wosac_submission.is_active:
                    self.wosac_submission.save_sub_file()

    def configure_optimizers(self):
        trainable_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters")
        optimizer = torch.optim.AdamW(trainable_parameters, lr=self.lr)

        def lr_lambda(current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return (
                    self.lr_min_ratio
                    + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
                )
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1.0
                + math.cos(
                    math.pi
                    * min(
                        1.0,
                        (current_step - self.lr_warmup_steps)
                        / (self.lr_total_steps - self.lr_warmup_steps),
                    )
                )
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        encoded_text_control = self._prepare_text_control(data)

        # ! only closed-loop vlidation
        pred_traj, pred_z, pred_head = [], [], []
        for _ in range(self.n_rollout_closed_val):
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                self.validation_rollout_sampling,
                encoded_text_control=encoded_text_control,
            )
            pred_traj.append(pred["pred_traj_10hz"])
            pred_z.append(pred["pred_z_10hz"])
            pred_head.append(pred["pred_head_10hz"])

        pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
        pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
        pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]

        # ! WOSAC submission save
        self.wosac_submission.update(
            scenario_id=data["scenario_id"],
            agent_id=data["agent"]["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            global_rank=self.global_rank,
        )
        _gpu_dict_sync = self.wosac_submission.compute()
        if self.global_rank == 0:
            for k in _gpu_dict_sync.keys():  # single gpu fix
                if type(_gpu_dict_sync[k]) is list:
                    _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
            scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
            self.wosac_submission.aggregate_rollouts(scenario_rollouts)
        self.wosac_submission.reset()

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.wosac_submission.save_sub_file()
