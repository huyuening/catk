# Original-Vocabulary Reconstructed Transition Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fixed transition lookup whose token IDs come from original CatK trajectories and vocabulary while its `[a_lon, angular_speed, a_lat]` values come from aligned full-trajectory reconstructions.

**Architecture:** Add a focused paired-cache dataset that joins scenarios by filename/scenario ID and agents by `agent.id`, then extend the existing offline builder with a paired mode. The paired mode tokenizes only the original batch, computes dynamics only from the aligned reconstructed batch, and writes the existing pair-conditioned tensor under a new provenance identifier bound to the original vocabulary SHA-256.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric, NumPy, Hydra/OmegaConf, Bash, `unittest`.

## Global Constraints

- Use only the WOMD training split; no validation or testing directory may be accepted.
- Preserve the existing single-cache builder and disabled-by-default CatK configuration.
- Preserve lookup shape `[3, n_token, n_token, 3]`, feature order `["a_lon", "angular_speed", "a_lat"]`, `float16` storage, and default shrinkage count `8.0`.
- Match original token IDs with deterministic CatK `num_k=1` tokenization against `agent_vocab_555_s2.pkl`.
- Compute dynamics from reconstructed 91-frame `position`, `heading`, and `valid_mask` only.
- Align scenarios by filename and `scenario_id`; align agents by unique `agent.id`, never by unverified array position.
- Use artifact source `raw_tokens_reconstructed_dynamics`.
- Keep runtime causal: the model reads only the fixed lookup and known or selected token IDs.
- Do not modify or stage the user-owned `scripts/cache_womd.sh`, `.DS_Store` files, or `.codex_work/`.

---

## File Structure

- Create `src/smart/tokens/paired_transition_dataset.py`: paired cache discovery, scenario validation, agent-ID alignment, and paired sample loading.
- Modify `src/smart/tokens/build_transition_dynamics.py`: paired builder API, dual-input CLI dispatch, and hybrid summary/artifact generation.
- Modify `src/smart/tokens/transition_dynamics_artifact.py`: declare and validate the hybrid source identifier.
- Modify `src/smart/tokens/token_processor.py`: report the expanded source allowlist without changing lookup loading.
- Create `configs/experiment/pre_bc_history_future_token_dynamics_hybrid.yaml`: original-vocabulary pre-BC with hybrid lookup provenance.
- Create `configs/experiment/clsft_history_future_token_dynamics_hybrid.yaml`: original-vocabulary CLSFT with hybrid lookup provenance.
- Create `configs/experiment/inference_history_future_token_dynamics_hybrid.yaml`: original-vocabulary validation with hybrid lookup provenance.
- Create `scripts/build_original_vocab_reconstructed_dynamics.sh`: concise environment-variable wrapper around the paired builder.
- Modify `docs/training-trajectory-transition-dynamics.md`: document lookup generation and training/validation commands.
- Modify `tests/test_transition_dynamics.py`: artifact-source round-trip and rejection coverage.
- Create `tests/test_paired_transition_dataset.py`: cache-set, scenario, ID, order, type, and shape alignment coverage.
- Modify `tests/test_build_transition_dynamics.py`: paired builder semantics, provenance, CLI, and legacy-regression coverage.
- Modify `tests/test_future_token_dynamics_configs.py`: hybrid YAML and Hydra-composition coverage.
- Create `tests/test_build_transition_dynamics_script.py`: Bash syntax/default/override command coverage.

---

### Task 1: Add Hybrid Artifact Provenance

**Files:**
- Modify: `src/smart/tokens/transition_dynamics_artifact.py:31-34`
- Modify: `src/smart/tokens/token_processor.py:18-24, 145-150`
- Modify: `tests/test_transition_dynamics.py:330-535`

**Interfaces:**
- Produces: `HYBRID_SOURCE: str = "raw_tokens_reconstructed_dynamics"`.
- Produces: `VALID_SOURCES == ("raw", "reconstructed", HYBRID_SOURCE)`.
- Consumes: existing `make_transition_dynamics_artifact` and `load_transition_dynamics_artifact` APIs without signature changes.

- [ ] **Step 1: Write the failing artifact round-trip test**

Add the constant import and this test to
`TransitionDynamicsArtifactTest`:

```python
from src.smart.tokens.transition_dynamics_artifact import (
    HYBRID_SOURCE,
    load_transition_dynamics_artifact,
    make_transition_dynamics_artifact,
    save_transition_dynamics_artifact,
)

def test_hybrid_source_round_trip_is_explicit(self):
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        vocabulary = root / "agent_vocab.pkl"
        vocabulary.write_bytes(b"original-vocabulary")
        artifact = make_transition_dynamics_artifact(
            np.zeros((3, 2, 2, 3), dtype=np.float16),
            vocabulary_path=vocabulary,
            source=HYBRID_SOURCE,
            dt=0.1,
            clipping_limits=(15.0, 3.0, 15.0),
            shrinkage_count=8.0,
            statistics={
                "assignment_source": "raw",
                "dynamics_source": "reconstructed",
            },
        )
        output = save_transition_dynamics_artifact(
            root / "hybrid.pt",
            artifact,
            vocabulary_path=vocabulary,
        )

        loaded = load_transition_dynamics_artifact(
            output,
            vocabulary_path=vocabulary,
            expected_source=HYBRID_SOURCE,
            expected_n_token=2,
        )

        self.assertEqual(tuple(loaded.shape), (3, 2, 2, 3))
        with self.assertRaisesRegex(ValueError, "source"):
            load_transition_dynamics_artifact(
                output,
                vocabulary_path=vocabulary,
                expected_source="raw",
                expected_n_token=2,
            )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest \
  tests.test_transition_dynamics.TransitionDynamicsArtifactTest.test_hybrid_source_round_trip_is_explicit
```

Expected: import failure for `HYBRID_SOURCE`, or artifact creation rejects the
hybrid source.

- [ ] **Step 3: Add the source constant and dynamic error message**

In `transition_dynamics_artifact.py`:

```python
FORMAT_VERSION = 1
FEATURE_ORDER = ("a_lon", "angular_speed", "a_lat")
HYBRID_SOURCE = "raw_tokens_reconstructed_dynamics"
VALID_SOURCES = ("raw", "reconstructed", HYBRID_SOURCE)
```

In `token_processor.py`, retain the imported `VALID_SOURCES` tuple and replace
the hard-coded error message with:

```python
if source not in TRANSITION_DYNAMICS_SOURCES:
    raise ValueError(
        "future_token_dynamics.source must be one of "
        f"{TRANSITION_DYNAMICS_SOURCES}"
    )
```

- [ ] **Step 4: Run artifact tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_transition_dynamics
```

Expected: all transition-dynamics tests pass, including raw/reconstructed
source mismatch tests.

- [ ] **Step 5: Commit the provenance change**

```bash
git add \
  src/smart/tokens/transition_dynamics_artifact.py \
  src/smart/tokens/token_processor.py \
  tests/test_transition_dynamics.py
git commit -m "feat: identify hybrid transition dynamics artifacts"
```

---

### Task 2: Pair and Align Original/Reconstructed Cache Samples

**Files:**
- Create: `src/smart/tokens/paired_transition_dataset.py`
- Create: `tests/test_paired_transition_dataset.py`

**Interfaces:**
- Produces: `align_reconstructed_cache(assignment: Mapping, reconstructed: Mapping, *, context: str) -> dict`.
- Produces: `PairedTransitionDataset(assignment_dir: str | Path, dynamics_dir: str | Path, transform: Callable)`.
- Produces: `PairedTransitionDataset.__getitem__(index: int) -> tuple[Any, Any]`, ordered as `(assignment_sample, aligned_dynamics_sample)`.
- Consumes: pickle cache dictionaries containing `scenario_id`, `current_time_index`, and `agent` stores.

- [ ] **Step 1: Write failing alignment tests**

Create `tests/test_paired_transition_dataset.py` with a cache helper and the
order-sensitive test:

```python
import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.smart.tokens.paired_transition_dataset import (
    PairedTransitionDataset,
)


def make_cache(
    scenario_id,
    agent_ids,
    agent_types,
    x_accelerations,
    *,
    reconstructed,
):
    time = torch.arange(91, dtype=torch.float64) * 0.1
    position = torch.zeros(len(agent_ids), 91, 3, dtype=torch.float64)
    for index, acceleration in enumerate(x_accelerations):
        position[index, :, 0] = 0.5 * acceleration * time.square()
    return {
        "scenario_id": scenario_id,
        "current_time_index": 10,
        "agent": {
            "num_nodes": len(agent_ids),
            "id": torch.tensor(agent_ids, dtype=torch.long),
            "type": torch.tensor(agent_types, dtype=torch.long),
            "position": position,
            "heading": torch.zeros(len(agent_ids), 91, dtype=torch.float64),
            "valid_mask": torch.ones(len(agent_ids), 91, dtype=torch.bool),
            "trajectory_reconstructed": torch.full(
                (len(agent_ids),),
                reconstructed,
                dtype=torch.bool,
            ),
        },
    }


class PairedTransitionDatasetTest(unittest.TestCase):
    def write_cache(self, path, value):
        with path.open("wb") as stream:
            pickle.dump(value, stream)

    def test_reconstructed_agents_are_aligned_by_id(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment_dir = root / "original"
            dynamics_dir = root / "reconstructed"
            assignment_dir.mkdir()
            dynamics_dir.mkdir()
            self.write_cache(
                assignment_dir / "scene.pkl",
                make_cache(
                    "scene",
                    [11, 22],
                    [0, 1],
                    [1.0, 2.0],
                    reconstructed=False,
                ),
            )
            self.write_cache(
                dynamics_dir / "scene.pkl",
                make_cache(
                    "scene",
                    [22, 11],
                    [1, 0],
                    [8.0, 4.0],
                    reconstructed=True,
                ),
            )

            dataset = PairedTransitionDataset(
                assignment_dir=assignment_dir,
                dynamics_dir=dynamics_dir,
                transform=lambda value: value,
            )
            assignment, dynamics = dataset[0]

            self.assertEqual(assignment["agent"]["id"].tolist(), [11, 22])
            self.assertEqual(dynamics["agent"]["id"].tolist(), [11, 22])
            self.assertAlmostEqual(
                float(dynamics["agent"]["position"][0, 10, 0]),
                2.0,
            )
            self.assertAlmostEqual(
                float(dynamics["agent"]["position"][1, 10, 0]),
                4.0,
            )
```

Add three failure tests in the same class using `assertRaisesRegex`:

```python
def test_file_sets_must_match(self):
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        assignment_dir = root / "original"
        dynamics_dir = root / "reconstructed"
        assignment_dir.mkdir()
        dynamics_dir.mkdir()
        self.write_cache(
            assignment_dir / "only-original.pkl",
            make_cache("scene", [1], [0], [1.0], reconstructed=False),
        )
        with self.assertRaisesRegex(ValueError, "file sets"):
            PairedTransitionDataset(
                assignment_dir=assignment_dir,
                dynamics_dir=dynamics_dir,
                transform=lambda value: value,
            )

def test_duplicate_or_missing_agent_ids_are_rejected(self):
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        assignment_dir = root / "original"
        dynamics_dir = root / "reconstructed"
        assignment_dir.mkdir()
        dynamics_dir.mkdir()
        self.write_cache(
            assignment_dir / "scene.pkl",
            make_cache("scene", [1, 1], [0, 0], [1.0, 1.0], reconstructed=False),
        )
        self.write_cache(
            dynamics_dir / "scene.pkl",
            make_cache("scene", [1, 2], [0, 0], [2.0, 2.0], reconstructed=True),
        )
        dataset = PairedTransitionDataset(
            assignment_dir=assignment_dir,
            dynamics_dir=dynamics_dir,
            transform=lambda value: value,
        )
        with self.assertRaisesRegex(ValueError, "unique agent.id"):
            dataset[0]

def test_agent_type_mismatch_is_rejected_after_id_alignment(self):
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        assignment_dir = root / "original"
        dynamics_dir = root / "reconstructed"
        assignment_dir.mkdir()
        dynamics_dir.mkdir()
        self.write_cache(
            assignment_dir / "scene.pkl",
            make_cache("scene", [1], [0], [1.0], reconstructed=False),
        )
        self.write_cache(
            dynamics_dir / "scene.pkl",
            make_cache("scene", [1], [2], [2.0], reconstructed=True),
        )
        dataset = PairedTransitionDataset(
            assignment_dir=assignment_dir,
            dynamics_dir=dynamics_dir,
            transform=lambda value: value,
        )
        with self.assertRaisesRegex(ValueError, "agent.type"):
            dataset[0]
```

- [ ] **Step 2: Run the new test module and verify RED**

Run:

```bash
python -m unittest tests.test_paired_transition_dataset
```

Expected: import failure because `paired_transition_dataset.py` does not exist.

- [ ] **Step 3: Implement deterministic paired loading**

Create `src/smart/tokens/paired_transition_dataset.py` with these public
interfaces and validation structure:

```python
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_AGENT_FIELDS = (
    "id",
    "type",
    "position",
    "heading",
    "valid_mask",
    "trajectory_reconstructed",
)


def _cache_paths(directory: str | Path, *, label: str) -> dict[str, Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    paths = {
        path.name: path
        for path in sorted(directory.glob("*.pkl"))
        if path.is_file()
    }
    if not paths:
        raise ValueError(f"{label} directory contains no .pkl cache files: {directory}")
    return paths


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _unique_ids(agent: Mapping, *, context: str) -> np.ndarray:
    if "id" not in agent:
        raise ValueError(f"{context}: agent.id is required")
    ids = _numpy(agent["id"])
    if ids.ndim != 1:
        raise ValueError(f"{context}: agent.id must be one-dimensional")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{context}: agent.id values must be unique")
    return ids.astype(np.int64, copy=False)


def _take_agents(value, order: np.ndarray, *, context: str):
    if isinstance(value, torch.Tensor):
        if value.ndim < 1:
            raise ValueError(f"{context}: agent field must have an agent axis")
        index = torch.as_tensor(order, dtype=torch.long, device=value.device)
        return value.index_select(0, index)
    array = np.asarray(value)
    if array.ndim < 1:
        raise ValueError(f"{context}: agent field must have an agent axis")
    return array[order]


def align_reconstructed_cache(
    assignment: Mapping,
    reconstructed: Mapping,
    *,
    context: str,
) -> dict:
    if assignment.get("scenario_id") != reconstructed.get("scenario_id"):
        raise ValueError(f"{context}: scenario_id mismatch")
    if (
        "current_time_index" in assignment
        and "current_time_index" in reconstructed
        and assignment["current_time_index"] != reconstructed["current_time_index"]
    ):
        raise ValueError(f"{context}: current_time_index mismatch")

    assignment_agent = assignment.get("agent")
    reconstructed_agent = reconstructed.get("agent")
    if not isinstance(assignment_agent, Mapping) or not isinstance(
        reconstructed_agent, Mapping
    ):
        raise ValueError(f"{context}: both caches require an agent store")
    for field in REQUIRED_AGENT_FIELDS:
        if field not in reconstructed_agent:
            raise ValueError(f"{context}: reconstructed agent.{field} is required")

    assignment_ids = _unique_ids(assignment_agent, context=f"{context} assignment")
    reconstructed_ids = _unique_ids(
        reconstructed_agent,
        context=f"{context} reconstructed",
    )
    if set(assignment_ids.tolist()) != set(reconstructed_ids.tolist()):
        raise ValueError(f"{context}: agent.id sets differ")
    reconstructed_index = {
        int(agent_id): index
        for index, agent_id in enumerate(reconstructed_ids.tolist())
    }
    order = np.asarray(
        [reconstructed_index[int(agent_id)] for agent_id in assignment_ids],
        dtype=np.int64,
    )
    aligned_agent = {
        field: _take_agents(
            reconstructed_agent[field],
            order,
            context=f"{context} reconstructed agent.{field}",
        )
        for field in REQUIRED_AGENT_FIELDS
    }
    aligned_agent["num_nodes"] = len(assignment_ids)

    assignment_type = _numpy(assignment_agent["type"]).astype(np.int64, copy=False)
    aligned_type = _numpy(aligned_agent["type"]).astype(np.int64, copy=False)
    if not np.array_equal(assignment_type, aligned_type):
        raise ValueError(f"{context}: aligned agent.type values differ")

    assignment_position = _numpy(assignment_agent["position"])
    aligned_position = _numpy(aligned_agent["position"])
    assignment_heading = _numpy(assignment_agent["heading"])
    aligned_heading = _numpy(aligned_agent["heading"])
    assignment_valid = _numpy(assignment_agent["valid_mask"])
    aligned_valid = _numpy(aligned_agent["valid_mask"])
    if assignment_position.shape != aligned_position.shape:
        raise ValueError(f"{context}: agent.position shapes differ")
    if assignment_heading.shape != aligned_heading.shape:
        raise ValueError(f"{context}: agent.heading shapes differ")
    if assignment_valid.shape != aligned_valid.shape:
        raise ValueError(f"{context}: agent.valid_mask shapes differ")

    result = {
        "scenario_id": assignment["scenario_id"],
        "agent": aligned_agent,
    }
    if "current_time_index" in assignment:
        result["current_time_index"] = assignment["current_time_index"]
    return result


class PairedTransitionDataset(Dataset):
    def __init__(
        self,
        assignment_dir: str | Path,
        dynamics_dir: str | Path,
        transform: Callable[[Mapping], Any],
    ) -> None:
        assignment_paths = _cache_paths(
            assignment_dir,
            label="assignment training",
        )
        dynamics_paths = _cache_paths(
            dynamics_dir,
            label="dynamics training",
        )
        if set(assignment_paths) != set(dynamics_paths):
            raise ValueError("assignment and dynamics cache file sets differ")
        self._pairs = [
            (assignment_paths[name], dynamics_paths[name])
            for name in sorted(assignment_paths)
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int):
        assignment_path, dynamics_path = self._pairs[index]
        with assignment_path.open("rb") as stream:
            assignment = pickle.load(stream)
        with dynamics_path.open("rb") as stream:
            reconstructed = pickle.load(stream)
        dynamics = align_reconstructed_cache(
            assignment,
            reconstructed,
            context=assignment_path.name,
        )
        return self.transform(assignment), self.transform(dynamics)
```

Before finalizing this task, ensure `assignment_agent` explicitly requires
`id`, `type`, `position`, `heading`, and `valid_mask`, so malformed assignment
caches raise a contextual `ValueError` rather than a bare `KeyError`.

- [ ] **Step 4: Add scenario/shape failure coverage and verify GREEN**

Add tests that change the reconstructed `scenario_id`, remove one assignment
field, and truncate reconstructed position/heading/validity to 90 frames.
Assert messages contain `scenario_id`, the missing field name, and `shapes
differ`, respectively. When PyTorch Geometric is installed, add this collation
test so the real loader contract is exercised:

```python
def test_paired_samples_collate_as_two_batches(self):
    try:
        from torch_geometric.data import HeteroData
        from torch_geometric.loader import DataLoader
    except ModuleNotFoundError:
        self.skipTest("PyTorch Geometric is not installed")
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        assignment_dir = root / "original"
        dynamics_dir = root / "reconstructed"
        assignment_dir.mkdir()
        dynamics_dir.mkdir()
        self.write_cache(
            assignment_dir / "scene.pkl",
            make_cache("scene", [1], [0], [1.0], reconstructed=False),
        )
        self.write_cache(
            dynamics_dir / "scene.pkl",
            make_cache("scene", [1], [0], [2.0], reconstructed=True),
        )
        dataset = PairedTransitionDataset(
            assignment_dir=assignment_dir,
            dynamics_dir=dynamics_dir,
            transform=HeteroData,
        )
        assignment_batch, dynamics_batch = next(
            iter(DataLoader(dataset, batch_size=1, shuffle=False))
        )
        self.assertEqual(assignment_batch["agent"]["id"].tolist(), [1])
        self.assertEqual(dynamics_batch["agent"]["id"].tolist(), [1])
```

Run:

```bash
python -m unittest tests.test_paired_transition_dataset
```

Expected: all paired-dataset tests pass.

- [ ] **Step 5: Commit the paired dataset**

```bash
git add \
  src/smart/tokens/paired_transition_dataset.py \
  tests/test_paired_transition_dataset.py
git commit -m "feat: align original and reconstructed training caches"
```

---

### Task 3: Extend the Offline Builder with Paired Semantics

**Files:**
- Modify: `src/smart/tokens/build_transition_dynamics.py:18-290`
- Modify: `tests/test_build_transition_dynamics.py:15-407`

**Interfaces:**
- Produces: `build_paired_transition_dynamics(assignment_training_dir, dynamics_training_dir, agent_token_file, output, *, map_token_file="map_traj_token5.pkl", batch_size=8, num_workers=8, max_scenarios=None, shrinkage_count=8.0) -> Path`.
- Consumes: `PairedTransitionDataset` from Task 2.
- Consumes: `HYBRID_SOURCE` from Task 1.
- Preserves: existing `build_transition_dynamics -> Path`.
- Preserves: `accumulate_tokenized_batch` with cache source `"raw"` or `"reconstructed"`; paired mode calls it with reconstructed data and `source="reconstructed"`.

- [ ] **Step 1: Write a failing paired-builder data-boundary test**

Extend imports:

```python
from src.smart.tokens.build_transition_dynamics import (
    build_paired_transition_dynamics,
)
from src.smart.tokens.transition_dynamics_artifact import HYBRID_SOURCE
```

Add a test using fake runtime components. The assignment trajectory has
`2 m/s²`, the reconstructed trajectory has `6 m/s²`, and the fake tokenizer
returns token pair `(0, 1)` only when it receives the assignment object:

```python
def test_paired_builder_uses_original_tokens_and_reconstructed_values(self):
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        root = Path(temp_dir)
        assignment_dir = root / "original"
        dynamics_dir = root / "reconstructed"
        assignment_dir.mkdir()
        dynamics_dir.mkdir()
        (assignment_dir / "scene.pkl").write_bytes(b"assignment")
        (dynamics_dir / "scene.pkl").write_bytes(b"dynamics")
        vocabulary = root / "agent.pkl"
        token_all = {
            name: np.zeros((2, 6, 4, 2), dtype=np.float32)
            for name in ("veh", "ped", "cyc")
        }
        with vocabulary.open("wb") as stream:
            pickle.dump({"token_all": token_all}, stream)

        time = torch.arange(91, dtype=torch.float64) * 0.1
        assignment = TransitionDynamicsBatchTest._constant_acceleration_batch()
        assignment["agent"]["id"] = torch.tensor([10])
        assignment["agent"]["trajectory_reconstructed"] = torch.tensor([False])
        reconstructed = TransitionDynamicsBatchTest._constant_acceleration_batch()
        reconstructed["agent"]["id"] = torch.tensor([10])
        reconstructed["agent"]["position"][0, :, 0] = 3.0 * time.square()
        reconstructed["agent"]["trajectory_reconstructed"] = torch.tensor([True])
        tokenized = {
            "type": torch.tensor([0]),
            "gt_idx": torch.tensor([[0, 1]]),
            "valid_mask": torch.tensor([[True, True]]),
        }

        class FakePairedDataset:
            def __init__(self, assignment_dir, dynamics_dir, transform):
                self.items = [(assignment, reconstructed)]

            def __len__(self):
                return len(self.items)

        class FakeLoader:
            def __init__(self, dataset, **kwargs):
                self.dataset = dataset

            def __iter__(self):
                return iter(self.dataset.items)

        class FakeProcessor:
            def __init__(self, **kwargs):
                pass

            def eval(self):
                return self

            def tokenize_agent(self, data):
                self.last_data = data
                self.asserted_assignment = data is assignment
                return tokenized

        processor = FakeProcessor()
        runtime = SimpleNamespace(
            PairedTransitionDataset=FakePairedDataset,
            DataLoader=FakeLoader,
            HeteroData=dict,
            TokenProcessor=lambda **kwargs: processor,
            Subset=lambda dataset, indices: dataset,
        )
        output = root / "hybrid.pt"
        with patch(
            "src.smart.tokens.build_transition_dynamics._load_runtime_components",
            return_value=runtime,
        ):
            build_paired_transition_dynamics(
                assignment_training_dir=assignment_dir,
                dynamics_training_dir=dynamics_dir,
                agent_token_file=vocabulary,
                output=output,
                batch_size=1,
                num_workers=0,
            )

        self.assertTrue(processor.asserted_assignment)
        table = load_transition_dynamics_artifact(
            output,
            vocabulary_path=vocabulary,
            expected_source=HYBRID_SOURCE,
            expected_n_token=2,
        )
        self.assertAlmostEqual(float(table[0, 0, 1, 0]), 6.0, delta=2e-3)
        summary = json.loads(output.with_suffix(".summary.json").read_text())
        self.assertEqual(summary["assignment_source"], "raw")
        self.assertEqual(summary["dynamics_source"], "reconstructed")
        self.assertEqual(summary["source"], HYBRID_SOURCE)
        self.assertEqual(summary["aligned_agents"], 1)
```

The final `self.assertTrue(processor.asserted_assignment)` proves the fake
processor received the original assignment object; no assertion method is
needed on `FakeProcessor`.

- [ ] **Step 2: Run the paired-builder test and verify RED**

Run:

```bash
python -m unittest \
  tests.test_build_transition_dynamics.TransitionDynamicsCliTest.test_paired_builder_uses_original_tokens_and_reconstructed_values
```

Expected: import failure because `build_paired_transition_dynamics` is absent.

- [ ] **Step 3: Load the paired dataset through runtime components**

Extend `_load_runtime_components()`:

```python
from src.smart.tokens.paired_transition_dataset import PairedTransitionDataset

return SimpleNamespace(
    MultiDataset=MultiDataset,
    PairedTransitionDataset=PairedTransitionDataset,
    DataLoader=DataLoader,
    HeteroData=HeteroData,
    TokenProcessor=TokenProcessor,
    Subset=Subset,
)
```

Keep `VALID_CACHE_SOURCES = ("raw", "reconstructed")` local to the builder.
Import artifact provenance independently:

```python
from src.smart.tokens.transition_dynamics_artifact import (
    HYBRID_SOURCE,
    make_transition_dynamics_artifact,
    save_transition_dynamics_artifact,
    vocabulary_sha256,
)

VALID_CACHE_SOURCES = ("raw", "reconstructed")
```

Use `VALID_CACHE_SOURCES` in `build_transition_dynamics`,
`validate_source_provenance`, and `accumulate_tokenized_batch`. The hybrid
identifier is an artifact source, never a direct trajectory-cache source.

- [ ] **Step 4: Implement the paired builder**

Add this function before `main`:

```python
def build_paired_transition_dynamics(
    assignment_training_dir: str | Path,
    dynamics_training_dir: str | Path,
    agent_token_file: str | Path,
    output: str | Path,
    *,
    map_token_file: str | Path = "map_traj_token5.pkl",
    batch_size: int = 8,
    num_workers: int = 8,
    max_scenarios: int | None = None,
    shrinkage_count: float = 8.0,
) -> Path:
    assignment_training_dir = Path(assignment_training_dir)
    dynamics_training_dir = Path(dynamics_training_dir)
    agent_token_file = Path(agent_token_file)
    output = Path(output)
    for directory, label in (
        (assignment_training_dir, "assignment training"),
        (dynamics_training_dir, "dynamics training"),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {directory}")
    if not agent_token_file.is_file():
        raise FileNotFoundError(
            f"agent vocabulary does not exist: {agent_token_file}"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if max_scenarios is not None and max_scenarios < 1:
        raise ValueError("max_scenarios must be positive when provided")
    if not np.isfinite(shrinkage_count) or shrinkage_count <= 0.0:
        raise ValueError("shrinkage_count must be finite and positive")

    agent_token_file = agent_token_file.resolve()
    isolated_fallback = _load_isolated_fallback(agent_token_file)
    n_token = int(isolated_fallback.shape[1])
    accumulator = TransitionDynamicsAccumulator(
        n_agent_types=3,
        n_token=n_token,
    )
    runtime = _load_runtime_components()
    dataset = runtime.PairedTransitionDataset(
        assignment_dir=assignment_training_dir,
        dynamics_dir=dynamics_training_dir,
        transform=lambda value: runtime.HeteroData(value),
    )
    scenario_count = len(dataset)
    if max_scenarios is not None and max_scenarios < scenario_count:
        scenario_count = max_scenarios
        dataset = runtime.Subset(dataset, range(scenario_count))
    loader = runtime.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    sampling = SimpleNamespace(num_k=1, temp=1.0)
    processor = runtime.TokenProcessor(
        map_token_file=str(map_token_file),
        agent_token_file=str(agent_token_file),
        map_token_sampling=sampling,
        agent_token_sampling=sampling,
        history_dynamics={"is_active": False},
        future_token_dynamics={"is_active": False},
    )
    processor.eval()

    scan_statistics = {
        "candidate_occurrences": 0,
        "accepted_occurrences": 0,
        "skipped_occurrences": 0,
    }
    aligned_agents = 0
    for assignment_batch, dynamics_batch in _progress(
        loader,
        description="paired training transition dynamics",
    ):
        assignment_snapshot = {
            "agent": _snapshot_agent_store(assignment_batch["agent"])
        }
        validate_source_provenance(
            assignment_snapshot["agent"],
            "raw",
            context="assignment training batch",
        )
        dynamics_snapshot = {
            "agent": _snapshot_agent_store(dynamics_batch["agent"])
        }
        tokenized_agent = processor.tokenize_agent(assignment_batch)
        batch_statistics = accumulate_tokenized_batch(
            accumulator,
            dynamics_snapshot,
            tokenized_agent,
            source="reconstructed",
        )
        aligned_agents += int(len(dynamics_snapshot["agent"]["position"]))
        for key in scan_statistics:
            scan_statistics[key] += int(batch_statistics[key])

    values, coverage_statistics = accumulator.finalize(
        isolated_fallback,
        shrinkage_count=shrinkage_count,
    )
    summary = {
        "source": HYBRID_SOURCE,
        "assignment_source": "raw",
        "dynamics_source": "reconstructed",
        "scenarios": int(scenario_count),
        "aligned_agents": int(aligned_agents),
        "vocabulary_sha256": vocabulary_sha256(agent_token_file),
        "vocabulary_size": n_token,
        **scan_statistics,
        **coverage_statistics,
    }
    artifact = make_transition_dynamics_artifact(
        values,
        vocabulary_path=agent_token_file,
        source=HYBRID_SOURCE,
        dt=0.1,
        clipping_limits=(15.0, 3.0, 15.0),
        shrinkage_count=shrinkage_count,
        statistics=summary,
    )
    result = save_transition_dynamics_artifact(
        output,
        artifact,
        vocabulary_path=agent_token_file,
    )
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
```

The current `_snapshot_agent_store` already retains the four fields needed by
raw provenance checks and reconstructed dynamics. Do not add `agent.id` to the
snapshot because identity alignment is completed and tested inside
`PairedTransitionDataset`.

- [ ] **Step 5: Extend the CLI with exclusive input modes**

Change parser input declarations:

```python
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--training-dir")
input_group.add_argument("--assignment-training-dir")
parser.add_argument("--dynamics-training-dir")
parser.add_argument(
    "--source",
    choices=("raw", "reconstructed", HYBRID_SOURCE),
)
```

Dispatch in `main`:

```python
parser = build_parser()
args = parser.parse_args(argv)
common = {
    "agent_token_file": args.agent_token_file,
    "output": args.output,
    "map_token_file": args.map_token_file,
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,
    "max_scenarios": args.max_scenarios,
    "shrinkage_count": args.shrinkage_count,
}
if args.assignment_training_dir is not None:
    if args.dynamics_training_dir is None:
        parser.error(
            "--dynamics-training-dir is required with "
            "--assignment-training-dir"
        )
    if args.source not in (None, HYBRID_SOURCE):
        parser.error(
            "paired mode source must be "
            f"{HYBRID_SOURCE}"
        )
    output = build_paired_transition_dynamics(
        assignment_training_dir=args.assignment_training_dir,
        dynamics_training_dir=args.dynamics_training_dir,
        **common,
    )
else:
    if args.dynamics_training_dir is not None:
        parser.error(
            "--dynamics-training-dir requires --assignment-training-dir"
        )
    source = args.source or "raw"
    if source == HYBRID_SOURCE:
        parser.error(
            f"{HYBRID_SOURCE} requires paired training directories"
        )
    output = build_transition_dynamics(
        training_dir=args.training_dir,
        source=source,
        **common,
    )
print(f"Transition dynamics artifact: {output}")
print(f"Summary: {output.with_suffix('.summary.json')}")
```

- [ ] **Step 6: Add CLI negative tests**

Add tests that call `main(argument_list)` and assert `SystemExit.code == 2`
for:

```text
--assignment-training-dir without --dynamics-training-dir
--training-dir together with --dynamics-training-dir
--training-dir with --source raw_tokens_reconstructed_dynamics
--assignment-training-dir with --source raw
```

Update the help test to require both paired directory flags while continuing
to reject validation/test flags.

- [ ] **Step 7: Run builder and dataset tests and verify GREEN**

Run:

```bash
python -m unittest \
  tests.test_paired_transition_dataset \
  tests.test_build_transition_dynamics
```

Expected: paired behavior and every legacy single-cache test pass.

- [ ] **Step 8: Commit the paired builder**

```bash
git add \
  src/smart/tokens/build_transition_dynamics.py \
  tests/test_build_transition_dynamics.py
git commit -m "feat: build dynamics from paired training caches"
```

---

### Task 4: Add Original-Vocabulary Hybrid Experiment Configurations

**Files:**
- Create: `configs/experiment/pre_bc_history_future_token_dynamics_hybrid.yaml`
- Create: `configs/experiment/clsft_history_future_token_dynamics_hybrid.yaml`
- Create: `configs/experiment/inference_history_future_token_dynamics_hybrid.yaml`
- Modify: `tests/test_future_token_dynamics_configs.py:7-124`

**Interfaces:**
- Produces Hydra experiments:
  - `pre_bc_history_future_token_dynamics_hybrid`
  - `clsft_history_future_token_dynamics_hybrid`
  - `inference_history_future_token_dynamics_hybrid`
- Preserves inherited original vocabulary `agent_vocab_555_s2.pkl`.
- Overrides only `future_token_dynamics.source`.

- [ ] **Step 1: Write failing YAML structure/composition tests**

Extend the class constants:

```python
HYBRID_SOURCE = "raw_tokens_reconstructed_dynamics"
HYBRID_EXPERIMENTS = {
    f"{name}_hybrid": name
    for name in RAW_EXPERIMENTS
}
```

Add:

```python
def test_hybrid_experiments_keep_original_vocabulary(self):
    for experiment, parent in self.HYBRID_EXPERIMENTS.items():
        with self.subTest(experiment=experiment):
            config = self._load_experiment(experiment)
            self.assertEqual(config["defaults"], [parent, "_self_"])
            model_config = config["model"]["model_config"]
            self.assertEqual(
                model_config["future_token_dynamics"],
                {"source": self.HYBRID_SOURCE},
            )
            self.assertNotIn("token_processor", model_config)
```

Include `list(self.HYBRID_EXPERIMENTS)` in the Hydra composition loop and map
expectations explicitly:

```python
if experiment.endswith("_reconstructed"):
    expected_source = "reconstructed"
    expected_vocabulary = "agent_vocab_reconstructed.pkl"
elif experiment.endswith("_hybrid"):
    expected_source = self.HYBRID_SOURCE
    expected_vocabulary = "agent_vocab_555_s2.pkl"
else:
    expected_source = "raw"
    expected_vocabulary = "agent_vocab_555_s2.pkl"
```

- [ ] **Step 2: Run config tests and verify RED**

Run:

```bash
python -m unittest tests.test_future_token_dynamics_configs
```

Expected: file-not-found failures for the three hybrid YAML files.

- [ ] **Step 3: Create the three minimal Hydra fragments**

For pre-BC:

```yaml
# @package _global_

defaults:
  - pre_bc_history_future_token_dynamics
  - _self_

model:
  model_config:
    future_token_dynamics:
      source: raw_tokens_reconstructed_dynamics
```

For CLSFT, use parent `clsft_history_future_token_dynamics`. For inference,
use parent `inference_history_future_token_dynamics`. Keep the remaining YAML
content identical to the pre-BC fragment.

- [ ] **Step 4: Run config tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_future_token_dynamics_configs
```

Expected: all raw, reconstructed, and hybrid YAML and Hydra composition tests
pass.

- [ ] **Step 5: Commit the experiment family**

```bash
git add \
  configs/experiment/pre_bc_history_future_token_dynamics_hybrid.yaml \
  configs/experiment/clsft_history_future_token_dynamics_hybrid.yaml \
  configs/experiment/inference_history_future_token_dynamics_hybrid.yaml \
  tests/test_future_token_dynamics_configs.py
git commit -m "feat: configure hybrid transition dynamics experiments"
```

---

### Task 5: Add the One-Command Builder Wrapper

**Files:**
- Create: `scripts/build_original_vocab_reconstructed_dynamics.sh`
- Create: `tests/test_build_transition_dynamics_script.py`

**Interfaces:**
- Consumes required environment variable: `RECON_OUTPUT`.
- Consumes optional environment variables: `CATK_ROOT`, `VOCAB_FILE`,
  `LOOKUP_FILE`, `BATCH_SIZE`, `NUM_WORKERS`, `SHRINKAGE_COUNT`,
  `MAX_SCENARIOS`, and `PYTHON_BIN`.
- Produces default output:
  `$RECON_OUTPUT/agent_transition_dynamics_original_vocab_reconstructed.pt`.

- [ ] **Step 1: Write the failing wrapper behavior test**

Create `tests/test_build_transition_dynamics_script.py`:

```python
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class BuildTransitionDynamicsScriptTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    SCRIPT = ROOT / "scripts" / "build_original_vocab_reconstructed_dynamics.sh"

    def test_wrapper_has_valid_bash_and_expands_default_paths(self):
        subprocess.run(["bash", "-n", str(self.SCRIPT)], check=True)
        with TemporaryDirectory() as temp_dir:
            reconstruction_root = Path(temp_dir) / "reconstruction"
            environment = os.environ.copy()
            environment.update(
                {
                    "CATK_ROOT": str(self.ROOT),
                    "RECON_OUTPUT": str(reconstruction_root),
                    "PYTHON_BIN": "/bin/echo",
                    "BATCH_SIZE": "3",
                    "NUM_WORKERS": "4",
                    "SHRINKAGE_COUNT": "5",
                    "MAX_SCENARIOS": "6",
                }
            )
            result = subprocess.run(
                ["bash", str(self.SCRIPT)],
                check=True,
                text=True,
                capture_output=True,
                env=environment,
            )
        arguments = result.stdout
        self.assertIn("-m src.smart.tokens.build_transition_dynamics", arguments)
        self.assertIn(
            f"--assignment-training-dir "
            f"{reconstruction_root}/datasets/original/training",
            arguments,
        )
        self.assertIn(
            f"--dynamics-training-dir "
            f"{reconstruction_root}/datasets/reconstructed/training",
            arguments,
        )
        self.assertIn("agent_vocab_555_s2.pkl", arguments)
        self.assertIn("--batch-size 3", arguments)
        self.assertIn("--num-workers 4", arguments)
        self.assertIn("--shrinkage-count 5", arguments)
        self.assertIn("--max-scenarios 6", arguments)
```

Add a second test that omits `RECON_OUTPUT` and asserts non-zero exit plus a
message containing `RECON_OUTPUT`.

- [ ] **Step 2: Run the wrapper test and verify RED**

Run:

```bash
python -m unittest tests.test_build_transition_dynamics_script
```

Expected: Bash reports the wrapper file does not exist.

- [ ] **Step 3: Implement the Bash wrapper**

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATK_ROOT="${CATK_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
: "${RECON_OUTPUT:?Set RECON_OUTPUT to the batch reconstruction output directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
VOCAB_FILE="${VOCAB_FILE:-${CATK_ROOT}/src/smart/tokens/agent_vocab_555_s2.pkl}"
LOOKUP_FILE="${LOOKUP_FILE:-${RECON_OUTPUT}/agent_transition_dynamics_original_vocab_reconstructed.pt}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SHRINKAGE_COUNT="${SHRINKAGE_COUNT:-8.0}"

COMMAND=(
  "${PYTHON_BIN}"
  -m src.smart.tokens.build_transition_dynamics
  --assignment-training-dir "${RECON_OUTPUT}/datasets/original/training"
  --dynamics-training-dir "${RECON_OUTPUT}/datasets/reconstructed/training"
  --agent-token-file "${VOCAB_FILE}"
  --output "${LOOKUP_FILE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --shrinkage-count "${SHRINKAGE_COUNT}"
)
if [[ -n "${MAX_SCENARIOS:-}" ]]; then
  COMMAND+=(--max-scenarios "${MAX_SCENARIOS}")
fi

cd "${CATK_ROOT}"
"${COMMAND[@]}"
```

Make it executable:

```bash
chmod +x scripts/build_original_vocab_reconstructed_dynamics.sh
```

- [ ] **Step 4: Run wrapper tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_build_transition_dynamics_script
```

Expected: syntax, default path expansion, overrides, and missing-variable
failure tests pass.

- [ ] **Step 5: Commit the wrapper**

```bash
git add \
  scripts/build_original_vocab_reconstructed_dynamics.sh \
  tests/test_build_transition_dynamics_script.py
git commit -m "feat: add hybrid lookup build script"
```

---

### Task 6: Document Build, Training, CLSFT, and Validation Commands

**Files:**
- Modify: `docs/training-trajectory-transition-dynamics.md:18-155`

**Interfaces:**
- Documents: paired lookup build command.
- Documents: hybrid pre-BC, CLSFT, and validation experiment names.
- Documents: runtime `CACHE_ROOT` remains the normal CatK cache with
  `history_dynamics`; `RECON_OUTPUT` is read only by the offline builder.

- [ ] **Step 1: Add the hybrid workflow section**

Insert after the reconstructed-vocabulary builder section:

````markdown
### Original vocabulary with reconstructed transition values

This hybrid keeps every token assignment in the original CatK vocabulary.
The paired builder matches tokens from `datasets/original/training`, aligns
the same agents in `datasets/reconstructed/training`, and calculates the three
dynamics values from the reconstructed trajectories.

```bash
export CATK_ROOT=/root/workspace/catk
export RECON_OUTPUT=/mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1

bash scripts/build_original_vocab_reconstructed_dynamics.sh
```

The default artifact is:

```text
$RECON_OUTPUT/agent_transition_dynamics_original_vocab_reconstructed.pt
```
````

- [ ] **Step 2: Add exact runtime commands**

Document pre-BC:

```bash
export CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact
export LOOKUP_FILE=/mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1/agent_transition_dynamics_original_vocab_reconstructed.pt
export MY_EXPERIMENT=pre_bc_history_future_token_dynamics_hybrid
export MY_TASK_NAME=pre_bc_history_future_token_dynamics_hybrid_b200

bash scripts/train.sh \
  ckpt_path=null \
  model.model_config.future_token_dynamics.lookup_file="$LOOKUP_FILE"
```

Document CLSFT:

```bash
PRE_BC_CKPT=/path/to/pre_bc/checkpoints/last.ckpt \
LOOKUP_FILE=/mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1/agent_transition_dynamics_original_vocab_reconstructed.pt \
MY_EXPERIMENT=clsft_history_future_token_dynamics_hybrid \
MY_TASK_NAME=clsft_history_future_token_dynamics_hybrid_b200 \
bash scripts/train.sh \
  ckpt_path="$PRE_BC_CKPT" \
  model.model_config.future_token_dynamics.lookup_file="$LOOKUP_FILE"
```

Document full validation:

```bash
CATK_CKPT=/path/to/checkpoints/last.ckpt \
CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact \
python run.py \
  experiment=inference_history_future_token_dynamics_hybrid \
  model.model_config.future_token_dynamics.lookup_file=/mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1/agent_transition_dynamics_original_vocab_reconstructed.pt \
  trainer.limit_val_batches=1.0 \
  task_name=hybrid_transition_dynamics_full
```

- [ ] **Step 3: Check documentation formatting**

Run:

```bash
git diff --check -- docs/training-trajectory-transition-dynamics.md
```

Expected: no whitespace errors.

- [ ] **Step 4: Commit the documentation**

```bash
git add docs/training-trajectory-transition-dynamics.md
git commit -m "docs: explain original-vocab reconstructed dynamics"
```

---

### Task 7: Run Integrated Regression Verification

**Files:**
- Verify only; modify a task-owned file only when a failing test identifies a
  defect in Tasks 1-6.

**Interfaces:**
- Verifies all new public APIs, artifact provenance, Hydra configurations,
  wrapper behavior, and legacy lookup behavior together.

- [ ] **Step 1: Run the focused feature suite**

```bash
python -m unittest \
  tests.test_transition_dynamics \
  tests.test_paired_transition_dataset \
  tests.test_build_transition_dynamics \
  tests.test_future_token_dynamics_configs \
  tests.test_build_transition_dynamics_script
```

Expected: all tests pass.

- [ ] **Step 2: Run neighboring dynamics/decoder regressions**

```bash
python -m unittest \
  tests.test_future_token_dynamics \
  tests.test_history_dynamics \
  tests.test_agent_preprocessing
```

Expected: all neighboring tests pass or explicitly skip unavailable optional
dependencies.

- [ ] **Step 3: Compile changed Python modules and validate Bash**

```bash
python -m compileall -q \
  src/smart/tokens/build_transition_dynamics.py \
  src/smart/tokens/paired_transition_dataset.py \
  src/smart/tokens/transition_dynamics_artifact.py \
  src/smart/tokens/token_processor.py \
  tests/test_build_transition_dynamics.py \
  tests/test_paired_transition_dataset.py

bash -n scripts/build_original_vocab_reconstructed_dynamics.sh
```

Expected: both commands exit zero with no syntax errors.

- [ ] **Step 4: Verify the exact patch scope**

```bash
git status --short
git diff --check
git log --oneline -7
```

Expected: task-owned changes are committed; the only remaining dirty paths are
the user's pre-existing `scripts/cache_womd.sh`, `.DS_Store` files, and
`.codex_work/`.

- [ ] **Step 5: Perform a local CLI help smoke test**

```bash
python -m src.smart.tokens.build_transition_dynamics --help
```

Expected: help lists `--training-dir`, `--assignment-training-dir`,
`--dynamics-training-dir`, `--agent-token-file`, and `--output`, with no
validation/test data options.
