from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch


TAG_RE = re.compile(
    r"^(?P<tag>\w+)\((?P<agents>[^ ]+) at (?P<ts>\d+)-(?P<te>\d+)\)$"
)
TAG_PHRASES = {
    "Parked": "parked",
    "Straight": "moving straight",
    "LeftTurn": "turning left",
    "RightTurn": "turning right",
    "LeftLaneChange": "changing lanes left",
    "RightLaneChange": "changing lanes right",
    "Accelerate": "accelerating",
    "Decelerate": "decelerating",
    "Stopping": "slowing down to a stop",
    "KeepSpeed": "keeping a steady speed",
}
TAG_PRIORITY = {
    "LeftTurn": 1,
    "RightTurn": 1,
    "LeftLaneChange": 1,
    "RightLaneChange": 1,
    "Straight": 2,
    "Accelerate": 3,
    "Decelerate": 3,
    "Stopping": 3,
    "KeepSpeed": 3,
    "Parked": 4,
}
TAG_EXCLUSION_GROUPS = {
    "Parked": list(TAG_PHRASES.keys() - {"Parked"}),
    "Straight": [
        "LeftTurn",
        "RightTurn",
        "Parked",
        "LeftLaneChange",
        "RightLaneChange",
    ],
    "LeftTurn": [
        "RightTurn",
        "Straight",
        "LeftLaneChange",
        "RightLaneChange",
    ],
    "RightTurn": [
        "LeftTurn",
        "Straight",
        "LeftLaneChange",
        "RightLaneChange",
    ],
    "LeftLaneChange": [
        "RightLaneChange",
        "Straight",
        "LeftTurn",
        "RightTurn",
        "Parked",
    ],
    "RightLaneChange": [
        "LeftLaneChange",
        "Straight",
        "LeftTurn",
        "RightTurn",
        "Parked",
    ],
    "Accelerate": ["Parked"],
    "Decelerate": ["Parked"],
    "Stopping": ["Parked"],
    "KeepSpeed": ["Parked"],
}
TAG_INTEGRATE_TOLERANCE = 5
TAG_MIN_DURATION = 10
AGENT_TYPE_LABELS = {0: "vehicle", 1: "pedestrian", 2: "cyclist"}


def scene_bucket(scene_id: str) -> int:
    """Return the deterministic ECoSim tag bucket for a scene identifier."""

    match = re.search(r"(\d+)$", str(scene_id))
    if match:
        return int(match.group(1)) % 100
    digest = hashlib.sha256(str(scene_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 100


def flatten_batched_prompts(value: object) -> List[str]:
    """Flatten PyG's nested string collation in graph and agent order."""

    flattened: List[str] = []

    def visit(item: object) -> None:
        if isinstance(item, str):
            flattened.append(item)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        raise TypeError(
            "text prompts must contain only string leaves, "
            f"got {type(item).__name__}"
        )

    visit(value)
    return flattened


@dataclass(frozen=True)
class ActionInterval:
    agent_id: str
    action: str
    start: int
    end: int


def _as_id(value: object) -> str:
    if hasattr(value, "item"):
        value = value.item()
    return str(value)


class ECoSimTagPromptStore:
    """Load ECoSim action tags and align ordered prompts to CatK agents."""

    def __init__(
        self,
        *,
        root: str | Path,
        mapping_path: str | Path | None,
        split: str,
        tag_subdir: str = "auto",
        sentence_style: str = "ordered",
    ) -> None:
        self.root = Path(root)
        self.split = str(split).lower()
        if self.split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if sentence_style != "ordered":
            raise ValueError("sentence_style must be 'ordered'")
        self.tag_subdir = (
            "waymo_train_v_action"
            if tag_subdir == "auto" and self.split == "train"
            else "waymo_val_v_action"
            if tag_subdir == "auto"
            else str(tag_subdir)
        )
        self.mapping: Optional[Dict[str, object]] = None
        if mapping_path is not None:
            loaded = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("scenario mapping must be a JSON object")
            self.mapping = {str(key): value for key, value in loaded.items()}
        self._cache: Dict[Tuple[Path, Optional[str]], Tuple[ActionInterval, ...]] = {}

    def prompts_for(
        self,
        *,
        scenario_id: str,
        agent_ids: Sequence[int],
        agent_types: Sequence[int],
        ego_id: int | None,
    ) -> tuple[list[str], torch.Tensor]:
        if len(agent_ids) != len(agent_types):
            raise ValueError("agent_ids and agent_types must have equal length")

        empty = [""] * len(agent_ids)
        if self.mapping is None:
            scene_id: Optional[str] = str(scenario_id)
        else:
            mapped = self.mapping.get(str(scenario_id))
            scene_id = None if mapped is None else str(mapped)
        if scene_id is None:
            return empty, torch.zeros(len(agent_ids), dtype=torch.bool)

        tag_path = (
            self.root
            / "tag_prompts"
            / self.tag_subdir
            / "tags"
            / str(scene_bucket(scene_id))
            / f"{scene_id}.json"
        )
        if not tag_path.is_file():
            return empty, torch.zeros(len(agent_ids), dtype=torch.bool)

        normalized_ego = None if ego_id is None else _as_id(ego_id)
        intervals = self._read_and_process(tag_path, ego_id=normalized_ego)
        prompt_by_id = self._ordered_sentences(
            intervals,
            agent_ids=agent_ids,
            agent_types=agent_types,
        )
        prompts = [prompt_by_id.get(_as_id(agent_id), "") for agent_id in agent_ids]
        mask = torch.tensor([bool(prompt) for prompt in prompts], dtype=torch.bool)
        return prompts, mask

    def _read_and_process(
        self, tag_path: Path, *, ego_id: Optional[str]
    ) -> Tuple[ActionInterval, ...]:
        cache_key = (tag_path, ego_id)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        raw_entries = json.loads(tag_path.read_text(encoding="utf-8"))
        if not isinstance(raw_entries, list):
            raise ValueError(f"tag file must contain a JSON list: {tag_path}")

        parsed: List[ActionInterval] = []
        for entry in raw_entries:
            if not isinstance(entry, str):
                continue
            match = TAG_RE.fullmatch(entry.strip())
            if match is None:
                continue
            action = match.group("tag")
            if action.endswith("Temporal"):
                action = action[: -len("Temporal")]
            if action not in TAG_PHRASES:
                continue
            start = int(match.group("ts"))
            end = int(match.group("te"))
            if end <= start:
                continue
            for raw_agent in match.group("agents").split(","):
                agent_id = raw_agent.strip()
                if not agent_id:
                    continue
                if agent_id == "ego":
                    if ego_id is None:
                        continue
                    agent_id = ego_id
                parsed.append(ActionInterval(agent_id, action, start, end))

        merged = self._merge_equal_actions(parsed)
        long_enough = [
            interval
            for interval in merged
            if interval.end - interval.start >= TAG_MIN_DURATION
        ]
        resolved = self._resolve_conflicts(long_enough)
        processed = tuple(self._drop_parked_when_multiple(resolved))
        if len(self._cache) >= 2048:
            self._cache.clear()
        self._cache[cache_key] = processed
        return processed

    @staticmethod
    def _merge_equal_actions(
        intervals: Sequence[ActionInterval],
    ) -> List[ActionInterval]:
        grouped: Dict[Tuple[str, str], List[ActionInterval]] = {}
        for interval in intervals:
            grouped.setdefault((interval.agent_id, interval.action), []).append(
                interval
            )

        merged: List[ActionInterval] = []
        for (agent_id, action), values in grouped.items():
            ordered = sorted(values, key=lambda value: (value.start, value.end))
            current_start = ordered[0].start
            current_end = ordered[0].end
            for value in ordered[1:]:
                if value.start <= current_end + TAG_INTEGRATE_TOLERANCE:
                    current_end = max(current_end, value.end)
                else:
                    merged.append(
                        ActionInterval(agent_id, action, current_start, current_end)
                    )
                    current_start, current_end = value.start, value.end
            merged.append(ActionInterval(agent_id, action, current_start, current_end))
        return merged

    @staticmethod
    def _resolve_conflicts(
        intervals: Sequence[ActionInterval],
    ) -> List[ActionInterval]:
        by_agent: Dict[str, List[ActionInterval]] = {}
        for interval in intervals:
            by_agent.setdefault(interval.agent_id, []).append(interval)

        resolved: List[ActionInterval] = []
        for agent_id, agent_intervals in by_agent.items():
            current: List[ActionInterval] = []
            for new in sorted(
                agent_intervals,
                key=lambda value: (
                    value.start,
                    TAG_PRIORITY.get(value.action, 99),
                    value.action,
                ),
            ):
                new_start = new.start
                new_end = new.end
                adjusted: List[ActionInterval] = []
                for existing in current:
                    existing_start = existing.start
                    existing_end = existing.end
                    overlap = max(existing_start, new_start) < min(
                        existing_end, new_end
                    )
                    conflicts = new.action in TAG_EXCLUSION_GROUPS.get(
                        existing.action, []
                    )
                    if overlap and conflicts:
                        existing_priority = TAG_PRIORITY.get(existing.action, 99)
                        new_priority = TAG_PRIORITY.get(new.action, 99)
                        if existing_priority < new_priority:
                            new_start = max(new_start, existing_end)
                        elif new_priority < existing_priority:
                            existing_end = min(existing_end, new_start)
                        else:
                            existing_end = min(existing_end, new_start)
                    if existing_start < existing_end:
                        adjusted.append(
                            ActionInterval(
                                agent_id,
                                existing.action,
                                existing_start,
                                existing_end,
                            )
                        )
                current = adjusted
                if new_start < new_end:
                    current.append(
                        ActionInterval(agent_id, new.action, new_start, new_end)
                    )

            current.sort(key=lambda value: (value.start, value.end, value.action))
            coalesced: List[ActionInterval] = []
            for interval in current:
                if (
                    coalesced
                    and coalesced[-1].action == interval.action
                    and interval.start <= coalesced[-1].end
                ):
                    previous = coalesced[-1]
                    coalesced[-1] = ActionInterval(
                        agent_id,
                        interval.action,
                        previous.start,
                        max(previous.end, interval.end),
                    )
                else:
                    coalesced.append(interval)
            resolved.extend(coalesced)
        return resolved

    @staticmethod
    def _drop_parked_when_multiple(
        intervals: Sequence[ActionInterval],
    ) -> List[ActionInterval]:
        by_agent: Dict[str, List[ActionInterval]] = {}
        for interval in intervals:
            by_agent.setdefault(interval.agent_id, []).append(interval)
        kept: List[ActionInterval] = []
        for values in by_agent.values():
            if len(values) == 1:
                kept.extend(values)
            else:
                kept.extend(value for value in values if value.action != "Parked")
        return kept

    @staticmethod
    def _ordered_sentences(
        intervals: Sequence[ActionInterval],
        *,
        agent_ids: Sequence[int],
        agent_types: Sequence[int],
    ) -> Dict[str, str]:
        type_by_id = {
            _as_id(agent_id): int(agent_type)
            for agent_id, agent_type in zip(agent_ids, agent_types)
        }
        grouped: Dict[str, Dict[int, List[str]]] = {}
        for interval in intervals:
            if interval.agent_id not in type_by_id:
                continue
            grouped.setdefault(interval.agent_id, {}).setdefault(
                interval.start, []
            ).append(interval.action)

        prompts: Dict[str, str] = {}
        for agent_id, start_groups in grouped.items():
            type_label = AGENT_TYPE_LABELS.get(type_by_id[agent_id], "vehicle")
            sentences: List[str] = []
            for start in sorted(start_groups):
                actions = sorted(
                    set(start_groups[start]),
                    key=lambda action: (TAG_PRIORITY.get(action, 99), action),
                )
                phrases = [TAG_PHRASES[action] for action in actions]
                if len(phrases) == 1:
                    action_text = phrases[0]
                elif len(phrases) == 2:
                    action_text = f"{phrases[0]} and {phrases[1]}"
                else:
                    action_text = f"{', '.join(phrases[:-1])}, and {phrases[-1]}"
                prefix = "The" if not sentences else "Then, the"
                sentences.append(
                    f"{prefix} target {type_label} is {action_text}."
                )
            if sentences:
                prompts[agent_id] = " ".join(sentences)
        return prompts


__all__ = [
    "ActionInterval",
    "ECoSimTagPromptStore",
    "flatten_batched_prompts",
    "scene_bucket",
]
