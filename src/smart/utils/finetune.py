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

import torch

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def set_model_for_text_control(model: torch.nn.Module) -> list[str]:
    """Freeze CatK and expose only the ECoSim text-control parameters."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    adapter = model.agent_encoder.text_control_adapter
    if adapter is None:
        raise ValueError("text control is active but no adapter was constructed")
    adapter.unfreeze_control_parameters()

    names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not names:
        raise RuntimeError("text-control training has no trainable parameters")
    log.info("Training only %d text-control parameter tensors", len(names))
    return names


def set_model_for_finetuning(model: torch.nn.Module, finetune: bool) -> None:
    def _unfreeze(module: torch.nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad = True

    if finetune:
        for p in model.parameters():
            p.requires_grad = False

        try:
            _unfreeze(model.agent_encoder.token_predict_head)
            log.info("Unfreezing token_predict_head")
        except:
            log.info("No token_predict_head in model.agent_encoder")

        try:
            _unfreeze(model.agent_encoder.gmm_logits_head)
            _unfreeze(model.agent_encoder.gmm_pose_head)
            # _unfreeze(model.agent_encoder.gmm_gmm_covpose_head)
            log.info("Unfreezing gmm heads")
        except:
            log.info("No gmm_logits_head in model.agent_encoder")

        _unfreeze(model.agent_encoder.t_attn_layers)
        _unfreeze(model.agent_encoder.pt2a_attn_layers)
        _unfreeze(model.agent_encoder.a2a_attn_layers)
