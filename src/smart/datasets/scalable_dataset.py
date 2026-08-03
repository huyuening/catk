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

import pickle
from pathlib import Path
from typing import Callable, List, Optional

import torch
from torch_geometric.data import Dataset

from src.smart.datasets.text_prompts import ECoSimTagPromptStore
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class MultiDataset(Dataset):
    def __init__(
        self,
        raw_dir: str,
        transform: Callable,
        tfrecord_dir: Optional[str] = None,
        text_prompt_root: Optional[str] = None,
        text_mapping_path: Optional[str] = None,
        text_split: str = "auto",
        tag_prompt_subdir: str = "auto",
        tag_sentence_style: str = "ordered",
        text_key: str = "text_prompt",
        text_mask_key: str = "text_prompt_mask",
    ) -> None:
        raw_dir = Path(raw_dir)
        self._raw_dir = raw_dir
        self._raw_paths = [p.as_posix() for p in sorted(raw_dir.glob("*"))]
        self._num_samples = len(self._raw_paths)

        self._tfrecord_dir = Path(tfrecord_dir) if tfrecord_dir is not None else None
        self._text_key = str(text_key)
        self._text_mask_key = str(text_mask_key)
        self._text_prompt_store = None
        if text_prompt_root is not None:
            resolved_split = self._resolve_text_split(text_split)
            self._text_prompt_store = ECoSimTagPromptStore(
                root=text_prompt_root,
                mapping_path=text_mapping_path,
                split=resolved_split,
                tag_subdir=tag_prompt_subdir,
                sentence_style=tag_sentence_style,
            )

        log.info("Length of {} dataset is ".format(raw_dir) + str(self._num_samples))
        super(MultiDataset, self).__init__(
            transform=transform, pre_transform=None, pre_filter=None
        )

    @property
    def raw_paths(self) -> List[str]:
        return self._raw_paths

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int):
        with open(self.raw_paths[idx], "rb") as handle:
            data = pickle.load(handle)

        if self._tfrecord_dir is not None:
            data["tfrecord_path"] = (
                self._tfrecord_dir / (data["scenario_id"] + ".tfrecords")
            ).as_posix()
        if self._text_prompt_store is not None:
            agent = data["agent"]
            role = agent["role"]
            if role.ndim != 2 or role.shape[1] < 1:
                raise ValueError(
                    "agent.role must have shape [n_agent, n_role] to resolve ego"
                )
            ego_indices = torch.where(role[:, 0].bool())[0]
            if ego_indices.numel() > 1:
                raise ValueError("agent.role identifies more than one ego agent")
            ego_id = (
                None
                if ego_indices.numel() == 0
                else int(agent["id"][int(ego_indices[0])].item())
            )
            prompts, prompt_mask = self._text_prompt_store.prompts_for(
                scenario_id=str(data["scenario_id"]),
                agent_ids=agent["id"].tolist(),
                agent_types=agent["type"].tolist(),
                ego_id=ego_id,
            )
            agent[self._text_key] = prompts
            agent[self._text_mask_key] = prompt_mask
        return data

    def _resolve_text_split(self, configured: str) -> str:
        split = str(configured).lower()
        if split != "auto":
            return split
        directory_name = self._raw_dir.name.lower()
        if "train" in directory_name:
            return "train"
        if "val" in directory_name:
            return "val"
        raise ValueError(
            "text_split='auto' requires a training or validation raw directory, "
            f"got {self._raw_dir}"
        )
