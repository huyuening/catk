# Embedded Fast WOSAC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CatK provide Fast WOSAC 2024/2025 and generate compatible validation ground truth without a TrajTok checkout or `TRAJTOK_ROOT`.

**Architecture:** Vendor the authorized TrajTok metric backend as a private package under CatK's metrics layer, then replace dynamic filesystem imports with direct internal imports. Extend CatK's existing validation preprocessing path to write the same `validation_gt` pickle representation while preserving all current metric keys, strict-loading behavior, and existing artifacts.

**Tech Stack:** Python 3.11, PyTorch, TorchMetrics, TensorFlow TFRecord I/O, Waymo Open Dataset protos, Hydra/OmegaConf, pytest.

## Global Constraints

- Source the backend from TrajTok commit `5920c89e26b62e8337512c253ab59efee995a496`.
- Keep the copied numeric backend byte-equivalent to the authorized source wherever package placement does not require an import adjustment.
- Support exactly WOSAC versions `"2024"` and `"2025"`.
- Preserve all existing `val_closed/wosac/*` and `val_closed/wosac_likelihood/*` metric keys.
- Preserve compatibility with existing TrajTok-generated `validation_gt/<scenario_id>.pkl` files.
- Preserve strict failure when required validation GT is absent or malformed.
- Never consult `/root/workspace/TrajTok`, `TRAJTOK_ROOT`, or an external `sys.path` entry at runtime.
- Keep `model.model_config.trajtok_root` temporarily accepted as a deprecated, ignored Hydra field with value `null`.
- Generate validation GT only for the `validation` preprocessing split.
- Do not migrate TrajTok training, tokenization, visualization, or `fast_eval_offline.py`.
- Preserve unrelated user changes in the dirty main worktree; create an isolated worktree at execution time.

## File Structure

New files:

```text
src/smart/metrics/fast_wosac_backend/
├── __init__.py
├── NOTICE.md
├── scenario_gt_converter.py
└── fast_sim_agents_metrics/
    ├── __init__.py
    ├── challenge_2024_config.textproto
    ├── challenge_2025_sim_agents_config.textproto
    ├── estimators.py
    ├── interaction_features.py
    ├── map_metric_features.py
    ├── metric_features.py
    ├── metrics.py
    ├── traffic_light_features.py
    └── trajectory_features.py
```

New tests:

```text
tests/test_embedded_fast_wosac_backend.py
tests/test_fast_wosac_internal_integration.py
tests/test_validation_gt_preprocessing.py
```

Modified files:

```text
src/smart/metrics/fast_wosac_metrics.py
src/smart/model/smart.py
src/data_preprocess.py
configs/model/smart.yaml
configs/experiment/inference.yaml
tests/test_fast_wosac_strict_integration.py
tests/test_training_fast_wosac_config.py
README.md
```

---

### Task 1: Vendor the authorized Fast WOSAC backend

**Files:**
- Create: `src/smart/metrics/fast_wosac_backend/__init__.py`
- Create: `src/smart/metrics/fast_wosac_backend/NOTICE.md`
- Create: `src/smart/metrics/fast_wosac_backend/scenario_gt_converter.py`
- Create: `src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics/__init__.py`
- Create: `src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics/challenge_2024_config.textproto`
- Create: `src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics/challenge_2025_sim_agents_config.textproto`
- Create: `src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics/estimators.py`
- Create: `src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics/interaction_features.py`
- Create: `src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics/map_metric_features.py`
- Create: `src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics/metric_features.py`
- Create: `src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics/metrics.py`
- Create: `src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics/traffic_light_features.py`
- Create: `src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics/trajectory_features.py`
- Test: `tests/test_embedded_fast_wosac_backend.py`

**Interfaces:**
- Consumes: authorized sources at `/Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool`.
- Produces: module `src.smart.metrics.fast_wosac_backend.fast_sim_agents_metrics.metrics`.
- Produces: `extract_gt_scenario(scenario: scenario_pb2.Scenario, device: str | torch.device = "cpu") -> dict`.
- Produces: `gt_scenario_to_device(value: object, device: str | torch.device) -> object`.

- [ ] **Step 1: Write the failing embedded-backend smoke tests**

Create `tests/test_embedded_fast_wosac_backend.py`:

```python
import importlib
from pathlib import Path

import pytest
from google.protobuf import text_format
from waymo_open_dataset.protos import sim_agents_metrics_pb2


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "src/smart/metrics/fast_wosac_backend"


def test_embedded_backend_exports_metric_entry_points():
    metrics = importlib.import_module(
        "src.smart.metrics.fast_wosac_backend."
        "fast_sim_agents_metrics.metrics"
    )
    converter = importlib.import_module(
        "src.smart.metrics.fast_wosac_backend.scenario_gt_converter"
    )

    assert callable(metrics.compute_scenario_metrics_for_bundle)
    assert callable(metrics.aggregate_metrics_to_buckets)
    assert callable(converter.extract_gt_scenario)
    assert callable(converter.gt_scenario_to_device)


@pytest.mark.parametrize(
    "filename",
    (
        "challenge_2024_config.textproto",
        "challenge_2025_sim_agents_config.textproto",
    ),
)
def test_embedded_wosac_configs_parse(filename):
    path = BACKEND / "fast_sim_agents_metrics" / filename
    config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
    text_format.Parse(path.read_text(encoding="utf-8"), config)

    assert config.ByteSize() > 0
```

- [ ] **Step 2: Run the tests and verify the expected RED failure**

Run:

```bash
python -m pytest tests/test_embedded_fast_wosac_backend.py -q
```

Expected: FAIL because
`src.smart.metrics.fast_wosac_backend` and its config files do not exist.

- [ ] **Step 3: Copy only the authorized runtime backend**

Create the target directory, then copy the following source files without
formatting or behavioral edits:

```bash
mkdir -p src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics

cp \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/scenario_gt_converter.py \
  src/smart/metrics/fast_wosac_backend/scenario_gt_converter.py

cp \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/fast_sim_agents_metrics/__init__.py \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/fast_sim_agents_metrics/challenge_2024_config.textproto \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/fast_sim_agents_metrics/challenge_2025_sim_agents_config.textproto \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/fast_sim_agents_metrics/estimators.py \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/fast_sim_agents_metrics/interaction_features.py \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/fast_sim_agents_metrics/map_metric_features.py \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/fast_sim_agents_metrics/metric_features.py \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/fast_sim_agents_metrics/metrics.py \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/fast_sim_agents_metrics/traffic_light_features.py \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/fast_sim_agents_metrics/trajectory_features.py \
  src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics/
```

Do not copy `fast_eval_offline.py` or any `__pycache__` files.

- [ ] **Step 4: Add the package marker and provenance notice**

Create `src/smart/metrics/fast_wosac_backend/__init__.py`:

```python
"""CatK-embedded Fast WOSAC backend sourced from TrajTok."""
```

Create `src/smart/metrics/fast_wosac_backend/NOTICE.md`:

```markdown
# TrajTok Fast WOSAC source notice

These files embed the Fast WOSAC evaluator from:

- Repository: https://github.com/Thinklab-SJTU/TrajTok
- Commit: 5920c89e26b62e8337512c253ab59efee995a496
- Original path: `wosac_fast_eval_tool`

The source is included with explicit authorization from this CatK fork's
owner. The evaluator is unofficial and is intended for rapid local
evaluation. Official challenge results must use the official WOSAC server.
```

- [ ] **Step 5: Verify source parity before running GREEN**

Run:

```bash
diff -u \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/scenario_gt_converter.py \
  src/smart/metrics/fast_wosac_backend/scenario_gt_converter.py

diff -ru \
  --exclude='__pycache__' \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/fast_sim_agents_metrics \
  src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics
```

Expected: both commands exit `0` with no output.

- [ ] **Step 6: Run the tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_embedded_fast_wosac_backend.py -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit the embedded backend**

```bash
git add \
  src/smart/metrics/fast_wosac_backend \
  tests/test_embedded_fast_wosac_backend.py
git commit -m "feat: embed TrajTok fast WOSAC backend"
```

---

### Task 2: Replace external discovery with internal imports

**Files:**
- Modify: `src/smart/metrics/fast_wosac_metrics.py:15-330`
- Modify: `src/smart/model/smart.py:73-87`
- Modify: `configs/model/smart.yaml:15-18`
- Modify: `configs/experiment/inference.yaml:10-21`
- Modify: `tests/test_fast_wosac_strict_integration.py`
- Modify: `tests/test_training_fast_wosac_config.py`
- Create: `tests/test_fast_wosac_internal_integration.py`

**Interfaces:**
- Consumes: Task 1 internal `fast_metrics`, `extract_gt_scenario`, and `gt_scenario_to_device`.
- Produces: `FastWOSACMetrics(prefix, version="2025", gt_scenario_dir=None, require_preprocessed_gt=False)` with no `trajtok_root` parameter.
- Produces: deprecated Hydra field `model.model_config.trajtok_root: null`, accepted but ignored.

- [ ] **Step 1: Write failing runtime-independence tests**

Create `tests/test_fast_wosac_internal_integration.py`:

```python
import inspect
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

from src.smart.metrics.fast_wosac_metrics import FastWOSACMetrics


ROOT = Path(__file__).resolve().parents[1]
COMMON_METRICS = (
    "metametric",
    "average_displacement_error",
    "min_average_displacement_error",
    "linear_speed_likelihood",
    "linear_acceleration_likelihood",
    "angular_speed_likelihood",
    "angular_acceleration_likelihood",
    "distance_to_nearest_object_likelihood",
    "collision_indication_likelihood",
    "time_to_collision_likelihood",
    "distance_to_road_edge_likelihood",
    "offroad_indication_likelihood",
    "simulated_collision_rate",
    "simulated_offroad_rate",
)


def test_fast_metric_constructs_without_external_trajtok(tmp_path):
    before = list(sys.path)
    with patch.dict(
        os.environ,
        {"TRAJTOK_ROOT": str(tmp_path / "missing-trajtok")},
    ):
        metric = FastWOSACMetrics(
            prefix="val_closed",
            version="2025",
            gt_scenario_dir=None,
            require_preprocessed_gt=False,
        )

    assert metric.version == "2025"
    assert list(sys.path) == before
    assert "trajtok_root" not in inspect.signature(
        FastWOSACMetrics
    ).parameters


def test_fast_metric_preserves_2024_and_2025_metric_sets():
    metric_2024 = FastWOSACMetrics(prefix="val", version="2024")
    metric_2025 = FastWOSACMetrics(prefix="val", version="2025")

    assert tuple(metric_2024.metric_names) == COMMON_METRICS
    assert tuple(metric_2025.metric_names) == COMMON_METRICS + (
        "traffic_light_violation_likelihood",
        "simulated_traffic_light_violation_rate",
    )


def test_malformed_2025_gt_points_to_catk_preprocessing(tmp_path):
    gt_dir = tmp_path / "validation_gt"
    gt_dir.mkdir()
    with (gt_dir / "scenario-1.pkl").open("wb") as handle:
        pickle.dump({"scenario_id": "scenario-1"}, handle)
    metric = FastWOSACMetrics(
        prefix="val",
        version="2025",
        gt_scenario_dir=str(gt_dir),
        require_preprocessed_gt=True,
    )

    with pytest.raises(
        KeyError,
        match=r"CatK's current src\.data_preprocess",
    ):
        metric.update(
            scenario_files=["unused.tfrecord"],
            scenario_ids=["scenario-1"],
            agent_id=torch.tensor([101]),
            agent_batch=torch.tensor([0]),
            simulated_states=torch.zeros(1, 1, 80, 4),
        )


def test_model_config_keeps_only_ignored_compatibility_value():
    config = yaml.safe_load(
        (ROOT / "configs/model/smart.yaml").read_text(encoding="utf-8")
    )
    assert config["model_config"]["trajtok_root"] is None


def test_inference_config_does_not_resolve_trajtok_environment(tmp_path):
    hydra = pytest.importorskip("hydra")
    with patch.dict(
        os.environ,
        {"TRAJTOK_ROOT": str(tmp_path / "missing-trajtok")},
    ):
        with hydra.initialize_config_dir(
            config_dir=str(ROOT / "configs"),
            version_base=None,
        ):
            config = hydra.compose(
                config_name="run.yaml",
                overrides=["experiment=inference"],
            )

    assert config.model.model_config.trajtok_root is None
```

Add this behavioral wiring test to
`tests/test_fast_wosac_strict_integration.py`:

```python
def test_smart_does_not_forward_external_trajtok_root(self):
    tree = ast.parse((ROOT / "src/smart/model/smart.py").read_text())
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FastWOSACMetrics"
    )
    self.assertNotIn(
        "trajtok_root",
        {keyword.arg for keyword in call.keywords},
    )
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python -m pytest \
  tests/test_fast_wosac_internal_integration.py \
  tests/test_fast_wosac_strict_integration.py \
  tests/test_training_fast_wosac_config.py \
  -q
```

Expected: FAIL because the constructor still requires `trajtok_root`, mutates
`sys.path`, and Hydra still resolves `TRAJTOK_ROOT`.

- [ ] **Step 3: Replace dynamic imports in the adapter**

In `src/smart/metrics/fast_wosac_metrics.py`:

1. Remove `import sys`.
2. Keep `Path` because the embedded textproto path is resolved from the
   internal metrics module.
3. Add:

```python
from src.smart.metrics.fast_wosac_backend.fast_sim_agents_metrics import (
    metrics as fast_metrics,
)
from src.smart.metrics.fast_wosac_backend.scenario_gt_converter import (
    extract_gt_scenario,
    gt_scenario_to_device,
)
```

4. Delete `_load_trajtok_modules`.
5. Change the constructor signature to:

```python
def __init__(
    self,
    prefix: str,
    version: str = "2025",
    gt_scenario_dir: str | None = None,
    require_preprocessed_gt: bool = False,
) -> None:
```

6. Delete `self.trajtok_root`.
7. Resolve the textproto from the internal module:

```python
config_path = Path(fast_metrics.__file__).resolve().parent / config_name
```

8. In `update`, use the directly imported functions:

```python
gt_scenario = self._load_scenario(
    scenario_file,
    expected_id,
    extract_gt_scenario,
)
gt_scenario = gt_scenario_to_device(gt_scenario, device=device)
scenario_metrics = fast_metrics.compute_scenario_metrics_for_bundle(
    self.wosac_config,
    gt_scenario,
    prediction,
    self.version,
)
```

9. In `compute`, call
   `fast_metrics.aggregate_metrics_to_buckets(...)` directly.
10. Change the malformed-2025-GT error suffix to:

```python
"Regenerate validation_gt with CatK's current src.data_preprocess."
```

- [ ] **Step 4: Stop SMART from forwarding an external root**

Change the `FastWOSACMetrics` call in `src/smart/model/smart.py` from:

```python
trajtok_root=model_config.get(
    "trajtok_root", "/root/workspace/TrajTok"
),
```

to no `trajtok_root` keyword.

- [ ] **Step 5: Make the old Hydra key inert**

In `configs/model/smart.yaml`, replace:

```yaml
trajtok_root: ${oc.env:TRAJTOK_ROOT,/root/workspace/TrajTok}
```

with:

```yaml
# Deprecated compatibility field; embedded Fast WOSAC ignores this value.
trajtok_root: null
```

Delete the `trajtok_root` override from
`configs/experiment/inference.yaml`. The inherited `null` key keeps old
command-line overrides valid.

Extend `tests/test_training_fast_wosac_config.py` with:

```python
def test_fast_wosac_has_no_external_runtime_root(self):
    model = self._load("configs/model/smart.yaml")["model_config"]
    inference = self._load("configs/experiment/inference.yaml")

    self.assertIsNone(model["trajtok_root"])
    self.assertNotIn(
        "trajtok_root",
        inference["model"]["model_config"],
    )
```

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
python -m pytest \
  tests/test_embedded_fast_wosac_backend.py \
  tests/test_fast_wosac_internal_integration.py \
  tests/test_fast_wosac_strict_integration.py \
  tests/test_training_fast_wosac_config.py \
  -q
```

Expected: all tests PASS without a TrajTok checkout.

- [ ] **Step 7: Commit internal integration**

```bash
git add \
  src/smart/metrics/fast_wosac_metrics.py \
  src/smart/model/smart.py \
  configs/model/smart.yaml \
  configs/experiment/inference.yaml \
  tests/test_fast_wosac_internal_integration.py \
  tests/test_fast_wosac_strict_integration.py \
  tests/test_training_fast_wosac_config.py
git commit -m "refactor: use embedded fast WOSAC backend"
```

---

### Task 3: Generate validation GT during CatK preprocessing

**Files:**
- Modify: `src/data_preprocess.py:15-490`
- Create: `tests/test_validation_gt_preprocessing.py`

**Interfaces:**
- Consumes: Task 1
  `src.smart.metrics.fast_wosac_backend.scenario_gt_converter.extract_gt_scenario`.
- Changes: `wm2argo(..., output_dir_gt: Path | None = None, ...)`.
- Produces: `<output_dir>/validation_gt/<scenario_id>.pkl` for validation only.

- [ ] **Step 1: Write a real synthetic Waymo scenario fixture**

Create `tests/test_validation_gt_preprocessing.py`:

```python
import pickle
from pathlib import Path

import pytest


tf = pytest.importorskip("tensorflow")
from waymo_open_dataset.protos import scenario_pb2  # noqa: E402

from src.data_preprocess import batch_process9s_transformer  # noqa: E402
from src.smart.metrics.fast_wosac_backend.scenario_gt_converter import (  # noqa: E402
    extract_gt_scenario,
)


def _scenario() -> scenario_pb2.Scenario:
    scenario = scenario_pb2.Scenario(
        scenario_id="embedded-fast-wosac-test",
        current_time_index=10,
        sdc_track_index=0,
    )
    scenario.timestamps_seconds.extend([step * 0.1 for step in range(91)])
    scenario.objects_of_interest.append(101)

    track = scenario.tracks.add(id=101, object_type=1)
    for step in range(91):
        state = track.states.add()
        state.center_x = float(step) * 0.5
        state.center_y = 0.0
        state.center_z = 0.0
        state.length = 4.5
        state.width = 2.0
        state.height = 1.6
        state.heading = 0.0
        state.velocity_x = 5.0
        state.velocity_y = 0.0
        state.valid = True
    scenario.tracks_to_predict.add(track_index=0)

    road_edge = scenario.map_features.add(id=201)
    road_edge.road_edge.type = 1
    for x, y in ((-10.0, -3.0), (50.0, -3.0)):
        point = road_edge.road_edge.polyline.add()
        point.x = x
        point.y = y

    lane = scenario.map_features.add(id=301)
    lane.lane.type = 2
    for x, y in ((-10.0, 0.0), (20.0, 0.0), (50.0, 0.0)):
        point = lane.lane.polyline.add()
        point.x = x
        point.y = y

    for _ in range(91):
        dynamic_state = scenario.dynamic_map_states.add()
        dynamic_state.lane_states.add(lane=301, state=4)

    return scenario


def _write_shard(path: Path, scenario: scenario_pb2.Scenario) -> None:
    path.parent.mkdir(parents=True)
    with tf.io.TFRecordWriter(str(path)) as writer:
        writer.write(scenario.SerializeToString())
```

- [ ] **Step 2: Add the failing converter and preprocessing tests**

Append:

```python
def test_embedded_converter_produces_wosac_2025_fields():
    gt = extract_gt_scenario(_scenario())

    assert gt["scenario_id"] == "embedded-fast-wosac-test"
    assert gt["tracks"].shape == (1, 91, 9)
    assert gt["track_masks"].shape == (1, 91)
    assert gt["object_ids"].tolist() == [101]
    assert gt["lane_ids"] == [301]
    assert len(gt["traffic_signals"]) == 91
    assert {
        "scenario_id",
        "tracks",
        "track_masks",
        "object_ids",
        "object_types",
        "road_edges",
        "predict_index",
        "sim_agent_ids",
        "lane_ids",
        "lane_polylines",
        "traffic_signals",
    }.issubset(gt)


def test_validation_preprocessing_writes_compatible_gt(tmp_path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_shard(
        input_root / "validation" / "validation.tfrecord-00000-of-00001",
        _scenario(),
    )

    batch_process9s_transformer(
        input_dir=input_root,
        output_dir=output_root,
        split="validation",
        num_workers=1,
    )

    gt_path = (
        output_root
        / "validation_gt"
        / "embedded-fast-wosac-test.pkl"
    )
    with gt_path.open("rb") as handle:
        gt = pickle.load(handle)

    assert gt["scenario_id"] == "embedded-fast-wosac-test"
    assert gt["tracks"].shape == (1, 91, 9)
    assert (
        output_root
        / "validation_tfrecords_splitted"
        / "embedded-fast-wosac-test.tfrecords"
    ).is_file()
    assert (
        output_root
        / "validation"
        / "embedded-fast-wosac-test.pkl"
    ).is_file()


@pytest.mark.parametrize("split", ("training", "testing"))
def test_non_validation_preprocessing_does_not_write_validation_gt(
    tmp_path,
    split,
):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _write_shard(
        input_root / split / f"{split}.tfrecord-00000-of-00001",
        _scenario(),
    )

    batch_process9s_transformer(
        input_dir=input_root,
        output_dir=output_root,
        split=split,
        num_workers=1,
    )

    assert not (output_root / "validation_gt").exists()
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
python -m pytest tests/test_validation_gt_preprocessing.py -q
```

Expected: the converter-only test passes, while
`test_validation_preprocessing_writes_compatible_gt` fails because CatK does
not create `validation_gt`.

- [ ] **Step 4: Wire the embedded converter into preprocessing**

Add this import to `src/data_preprocess.py`:

```python
from src.smart.metrics.fast_wosac_backend.scenario_gt_converter import (
    extract_gt_scenario,
)
```

Add `output_dir_gt=None` to `wm2argo` immediately after
`output_dir_tfrecords_splitted`:

```python
def wm2argo(
    file_path,
    split,
    output_dir,
    output_dir_tfrecords_splitted,
    output_dir_gt=None,
    history_dynamics_filter_strength=None,
    history_dynamics_max_gap_frames=None,
):
```

After writing the split TFRecord, add:

```python
if output_dir_gt is not None:
    gt = extract_gt_scenario(scenario)
    with (output_dir_gt / f"{scenario_id}.pkl").open("wb") as handle:
        pickle.dump(gt, handle)
```

In `batch_process9s_transformer`, initialize and populate the directory:

```python
output_dir_tfrecords_splitted = None
output_dir_gt = None
if split == "validation":
    output_dir_tfrecords_splitted = (
        output_dir / "validation_tfrecords_splitted"
    )
    output_dir_tfrecords_splitted.mkdir(exist_ok=True, parents=True)
    output_dir_gt = output_dir / "validation_gt"
    output_dir_gt.mkdir(exist_ok=True, parents=True)
```

Pass it through the worker partial:

```python
output_dir_gt=output_dir_gt,
```

- [ ] **Step 5: Run the preprocessing tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_validation_gt_preprocessing.py -q
```

Expected: all four test cases PASS.

- [ ] **Step 6: Run existing preprocessing regressions**

Run:

```bash
python -m pytest \
  tests/test_agent_preprocessing.py \
  tests/test_history_dynamics.py \
  tests/test_preprocessed_scenario_gt.py \
  -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit validation GT generation**

```bash
git add \
  src/data_preprocess.py \
  tests/test_validation_gt_preprocessing.py
git commit -m "feat: generate fast WOSAC validation ground truth"
```

---

### Task 4: Update user-facing documentation

**Files:**
- Modify: `README.md:195-265`

**Interfaces:**
- Consumes: Tasks 1-3 completed runtime and preprocessing behavior.
- Produces: deployment and evaluation instructions requiring only CatK.

- [ ] **Step 1: Rewrite the training-time Fast WOSAC description**

In `README.md`, replace references to “TrajTok Fast WOSAC” with
“CatK-embedded Fast WOSAC, sourced from TrajTok”. Keep the existing 10%
validation, 32-rollout, `K=48`, and strict-GT statements unchanged.

- [ ] **Step 2: Rewrite the standalone Fast WOSAC section**

Document these exact behaviors:

```text
- No sibling TrajTok checkout is required.
- No TRAJTOK_ROOT variable is required.
- Existing validation_gt artifacts remain compatible.
- python -m src.data_preprocess --split validation generates validation_gt.
- FAST_WOSAC_GT_DIR remains the optional ground-truth path override.
- The embedded evaluator supports WOSAC 2024 and 2025.
```

Remove this line from the optional command:

```bash
TRAJTOK_ROOT=/path/to/TrajTok \
```

Add provenance:

```markdown
The embedded evaluator is sourced from the TrajTok Fast WOSAC implementation
at commit
[`5920c89`](https://github.com/Thinklab-SJTU/TrajTok/commit/5920c89e26b62e8337512c253ab59efee995a496).
It is an unofficial local evaluator; use the official WOSAC server for final
challenge results.
```

- [ ] **Step 3: Verify documentation and runtime references**

Run:

```bash
rg -n \
  'TRAJTOK_ROOT|/root/workspace/TrajTok|sibling checkout' \
  README.md \
  configs/model/smart.yaml \
  configs/experiment/inference.yaml \
  src/smart/metrics/fast_wosac_metrics.py \
  src/smart/model/smart.py
```

Expected: no active runtime path or instruction remains. The only allowed
`trajtok_root` occurrence is the deprecated `null` compatibility key and its
comment in `configs/model/smart.yaml`.

- [ ] **Step 4: Run the focused suite after documentation changes**

Run:

```bash
python -m pytest \
  tests/test_embedded_fast_wosac_backend.py \
  tests/test_fast_wosac_internal_integration.py \
  tests/test_fast_wosac_strict_integration.py \
  tests/test_training_fast_wosac_config.py \
  tests/test_validation_gt_preprocessing.py \
  -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md
git commit -m "docs: document embedded fast WOSAC workflow"
```

---

### Task 5: Verify source parity and the complete CatK repository

**Files:**
- Verify: all files changed by Tasks 1-4

**Interfaces:**
- Consumes: complete embedded Fast WOSAC implementation.
- Produces: verification evidence for source parity, standalone operation,
  focused behavior, and repository-wide regression safety.

- [ ] **Step 1: Re-run authorized-source parity checks**

Run:

```bash
diff -u \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/scenario_gt_converter.py \
  src/smart/metrics/fast_wosac_backend/scenario_gt_converter.py

diff -ru \
  --exclude='__pycache__' \
  /Users/huyuening/PycharmProjects/TrajTok/wosac_fast_eval_tool/fast_sim_agents_metrics \
  src/smart/metrics/fast_wosac_backend/fast_sim_agents_metrics
```

Expected: both commands exit `0` with no output.

- [ ] **Step 2: Verify resolved Hydra config ignores a missing TrajTok path**

Run:

```bash
TRAJTOK_ROOT=/definitely/missing/TrajTok \
python -m src.run \
  --cfg job \
  --resolve \
  experiment=pre_bc_history_dynamics \
  trainer=ddp \
  task_name=embedded_fast_wosac_config_check \
  | rg -n 'wosac_backend|wosac_metrics_version|trajtok_root'
```

Expected output includes:

```text
wosac_backend: fast
wosac_metrics_version: '2025'
trajtok_root: null
```

The command must not access or require `/definitely/missing/TrajTok`.

- [ ] **Step 3: Compile the changed Python modules**

Run:

```bash
python -m compileall -q \
  src/smart/metrics/fast_wosac_backend \
  src/smart/metrics/fast_wosac_metrics.py \
  src/smart/model/smart.py \
  src/data_preprocess.py
```

Expected: exit `0` with no syntax errors.

- [ ] **Step 4: Run the focused test suite**

Run:

```bash
python -m pytest \
  tests/test_embedded_fast_wosac_backend.py \
  tests/test_fast_wosac_internal_integration.py \
  tests/test_fast_wosac_strict_integration.py \
  tests/test_training_fast_wosac_config.py \
  tests/test_validation_gt_preprocessing.py \
  tests/test_preprocessed_scenario_gt.py \
  -q
```

Expected: all tests PASS.

- [ ] **Step 5: Run the complete test suite**

Run:

```bash
python -m pytest -q
```

Expected: all runnable tests PASS; dependency-gated tests may report SKIPPED,
but no test may FAIL or ERROR.

- [ ] **Step 6: Check repository hygiene**

Run:

```bash
git diff --check
git status --short
git log --oneline -5
```

Expected:

- `git diff --check` exits `0`;
- only intended feature files are changed or committed in the isolated
  worktree;
- unrelated main-worktree files such as `scripts/cache_womd.sh` are untouched;
- the latest commits correspond to Tasks 1-4.
