# Training-Time Fast WOSAC Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every `pre_bc` and `clsft` epoch run strict TrajTok Fast WOSAC 2025 evaluation on a deterministic 10% validation prefix while retaining open-loop validation.

**Architecture:** Add a dependency-light preprocessed-GT store that owns strict-versus-fallback loading semantics, then connect it to `FastWOSACMetrics`. Configure the two base training experiments with the shared validation protocol so all derived experiments inherit it without changing global model or standalone inference defaults.

**Tech Stack:** Python 3.11, pathlib/pickle, PyTorch Lightning, TorchMetrics, Hydra/OmegaConf, YAML, unittest/pytest.

## Global Constraints

- `pre_bc` and `clsft` validate after every epoch.
- Validation uses a deterministic loader prefix of `0.1`.
- Open-loop loss and accuracy remain enabled.
- Closed-loop validation uses TrajTok Fast WOSAC version 2025.
- Every processed validation batch contributes to WOSAC metrics.
- Validation uses 32 rollouts with `topk_prob`, `K=48`, and temperature `1.0`.
- The default GT directory is `/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario/validation_gt`.
- `FAST_WOSAC_GT_DIR` overrides the default GT directory.
- Training experiments require preprocessed GT and never fall back to raw TFRecords.
- Standalone inference and other configurations retain their current fallback behavior.
- Preserve all unrelated worktree changes.

---

### Task 1: Isolate strict preprocessed-GT loading

**Files:**
- Create: `src/smart/metrics/preprocessed_scenario_gt.py`
- Create: `tests/test_preprocessed_scenario_gt.py`

**Interfaces:**
- Produces: `PreprocessedScenarioGT(directory: str | Path | None, *, required: bool = False)`
- Produces: `PreprocessedScenarioGT.directory -> Path | None`
- Produces: `PreprocessedScenarioGT.load(scenario_id: str) -> dict | None`
- Consumers: `FastWOSACMetrics` in Task 2

- [ ] **Step 1: Write the failing store tests**

Create `tests/test_preprocessed_scenario_gt.py` with dependency-free tests:

```python
import importlib.util
import pickle
import tempfile
import unittest
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "src/smart/metrics/preprocessed_scenario_gt.py"
)
SPEC = importlib.util.spec_from_file_location(
    "preprocessed_scenario_gt",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
PreprocessedScenarioGT = MODULE.PreprocessedScenarioGT


class PreprocessedScenarioGTTest(unittest.TestCase):
    def test_required_store_rejects_missing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "validation_gt"
            with self.assertRaisesRegex(
                FileNotFoundError,
                "strict Fast WOSAC.*validation_gt",
            ):
                PreprocessedScenarioGT(missing, required=True)

    def test_required_store_rejects_missing_scenario(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = PreprocessedScenarioGT(temporary, required=True)
            with self.assertRaisesRegex(
                FileNotFoundError,
                "scenario-123.*pkl",
            ):
                store.load("scenario-123")

    def test_store_loads_valid_dictionary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scenario-123.pkl"
            expected = {"scenario_id": "scenario-123", "tracks": []}
            with path.open("wb") as file:
                pickle.dump(expected, file)
            store = PreprocessedScenarioGT(temporary, required=True)
            self.assertEqual(store.load("scenario-123"), expected)

    def test_optional_store_keeps_fallback_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "validation_gt"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store = PreprocessedScenarioGT(missing, required=False)
            self.assertIsNone(store.directory)
            self.assertIsNone(store.load("scenario-123"))
            self.assertTrue(
                any("falling back" in str(item.message) for item in caught)
            )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest tests/test_preprocessed_scenario_gt.py -q
```

Expected: collection fails with `FileNotFoundError` for
`src/smart/metrics/preprocessed_scenario_gt.py`.

- [ ] **Step 3: Implement the minimal GT store**

Create `src/smart/metrics/preprocessed_scenario_gt.py`:

```python
from __future__ import annotations

import pickle
import warnings
from pathlib import Path


class PreprocessedScenarioGT:
    def __init__(
        self,
        directory: str | Path | None,
        *,
        required: bool = False,
    ) -> None:
        self.required = bool(required)
        self.directory = (
            Path(directory).expanduser().resolve() if directory else None
        )
        if self.directory is not None and self.directory.is_dir():
            return
        if self.required:
            target = (
                "<not configured>"
                if self.directory is None
                else str(self.directory)
            )
            raise FileNotFoundError(
                "strict Fast WOSAC requires a validation_gt directory: "
                f"{target}"
            )
        if self.directory is not None:
            warnings.warn(
                "Fast WOSAC GT directory does not exist; falling back to "
                f"per-scenario TFRecords: {self.directory}",
                stacklevel=2,
            )
        self.directory = None

    def load(self, scenario_id: str) -> dict | None:
        if self.directory is None:
            return None
        path = self.directory / f"{scenario_id}.pkl"
        if not path.is_file():
            if self.required:
                raise FileNotFoundError(
                    "strict Fast WOSAC requires preprocessed GT for "
                    f"scenario {scenario_id}: {path}"
                )
            return None
        with path.open("rb") as file:
            scenario = pickle.load(file)
        if hasattr(scenario, "value"):
            scenario = scenario.value
        if not isinstance(scenario, dict):
            raise TypeError(
                "Fast WOSAC GT must be a dict, got "
                f"{type(scenario).__name__}: {path}"
            )
        return scenario
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_preprocessed_scenario_gt.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the isolated loader**

```bash
git add \
  src/smart/metrics/preprocessed_scenario_gt.py \
  tests/test_preprocessed_scenario_gt.py
git commit -m "feat: add strict preprocessed WOSAC GT loader"
```

---

### Task 2: Connect strict GT semantics to Fast WOSAC

**Files:**
- Modify: `src/smart/metrics/fast_wosac_metrics.py`
- Modify: `src/smart/model/smart.py`
- Modify: `configs/model/smart.yaml`
- Create: `tests/test_fast_wosac_strict_integration.py`

**Interfaces:**
- Consumes: `PreprocessedScenarioGT` from Task 1
- Changes: `FastWOSACMetrics(..., require_preprocessed_gt: bool = False)`
- Adds config: `model.model_config.fast_wosac_require_preprocessed_gt`
- Preserves: optional GT lookup followed by raw-TFRecord fallback when strict mode is false

- [ ] **Step 1: Write failing integration-contract tests**

Create `tests/test_fast_wosac_strict_integration.py`. Use AST so the tests run
in the lightweight local environment where TorchMetrics and Waymo may be
absent, while Task 1 covers the runtime loading behavior:

```python
import ast
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class FastWOSACStrictIntegrationTest(unittest.TestCase):
    def test_fast_metric_constructor_exposes_required_gt_flag(self):
        tree = ast.parse(
            (ROOT / "src/smart/metrics/fast_wosac_metrics.py").read_text()
        )
        metric_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "FastWOSACMetrics"
        )
        constructor = next(
            node
            for node in metric_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        arguments = [argument.arg for argument in constructor.args.args]
        self.assertIn("require_preprocessed_gt", arguments)

    def test_smart_forwards_model_required_gt_flag(self):
        tree = ast.parse(
            (ROOT / "src/smart/model/smart.py").read_text()
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "FastWOSACMetrics"
        ]
        self.assertEqual(len(calls), 1)
        keyword = next(
            item
            for item in calls[0].keywords
            if item.arg == "require_preprocessed_gt"
        )
        self.assertIn(
            "fast_wosac_require_preprocessed_gt",
            ast.unparse(keyword.value),
        )

    def test_model_default_preserves_optional_gt_behavior(self):
        config = yaml.safe_load(
            (ROOT / "configs/model/smart.yaml").read_text()
        )
        self.assertFalse(
            config["model_config"][
                "fast_wosac_require_preprocessed_gt"
            ]
        )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python -m pytest tests/test_fast_wosac_strict_integration.py -q
```

Expected: all three assertions fail because the constructor argument, SMART
forwarding, and model default do not exist.

- [ ] **Step 3: Integrate the store into `FastWOSACMetrics`**

In `src/smart/metrics/fast_wosac_metrics.py`:

```python
from src.smart.metrics.preprocessed_scenario_gt import (
    PreprocessedScenarioGT,
)
```

Extend the constructor:

```python
def __init__(
    self,
    prefix: str,
    trajtok_root: str,
    version: str = "2025",
    gt_scenario_dir: str | None = None,
    require_preprocessed_gt: bool = False,
) -> None:
```

Replace the existing directory-resolution block with:

```python
self.preprocessed_gt = PreprocessedScenarioGT(
    gt_scenario_dir,
    required=require_preprocessed_gt,
)
self.gt_scenario_dir = self.preprocessed_gt.directory
```

At the start of `_load_scenario`, replace the inline pickle loading with:

```python
gt_scenario = self.preprocessed_gt.load(scenario_id)
if gt_scenario is not None:
    return gt_scenario
```

Leave the existing TFRecord extraction below this branch unchanged. It is
reachable only when the store is optional and returns `None`.

- [ ] **Step 4: Forward the model option**

In the `FastWOSACMetrics` call in `src/smart/model/smart.py`, add:

```python
require_preprocessed_gt=model_config.get(
    "fast_wosac_require_preprocessed_gt",
    False,
),
```

In `configs/model/smart.yaml`, add beside `fast_wosac_gt_dir`:

```yaml
fast_wosac_require_preprocessed_gt: false
```

- [ ] **Step 5: Run focused integration and store tests**

Run:

```bash
python -m pytest \
  tests/test_preprocessed_scenario_gt.py \
  tests/test_fast_wosac_strict_integration.py \
  -q
```

Expected: `7 passed`.

- [ ] **Step 6: Compile the modified runtime modules**

Run:

```bash
python -m py_compile \
  src/smart/metrics/preprocessed_scenario_gt.py \
  src/smart/metrics/fast_wosac_metrics.py \
  src/smart/model/smart.py
```

Expected: exit code `0`.

- [ ] **Step 7: Commit strict Fast WOSAC integration**

```bash
git add \
  configs/model/smart.yaml \
  src/smart/metrics/fast_wosac_metrics.py \
  src/smart/model/smart.py \
  tests/test_fast_wosac_strict_integration.py
git commit -m "feat: require preprocessed GT for training WOSAC"
```

---

### Task 3: Make the protocol default for pre-BC and CLSFT

**Files:**
- Modify: `configs/paths/default.yaml`
- Modify: `configs/experiment/pre_bc.yaml`
- Modify: `configs/experiment/clsft.yaml`
- Create: `tests/test_training_fast_wosac_config.py`

**Interfaces:**
- Adds: `paths.validation_gt_dir`
- Consumes: `model.model_config.fast_wosac_require_preprocessed_gt`
- Produces: the same resolved training validation protocol for both base experiments
- Preserves: standalone `inference.yaml`'s existing `paths.validation_gt_dir` override

- [ ] **Step 1: Write failing raw-YAML and Hydra-composition tests**

Create `tests/test_training_fast_wosac_config.py`:

```python
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = (
    "/mnt/pfs/waymo_motion_1_3_0/"
    "preprocessed_scenario/validation_gt"
)


class TrainingFastWOSACConfigTest(unittest.TestCase):
    def _load(self, relative_path):
        return yaml.safe_load(
            (ROOT / relative_path).read_text(encoding="utf-8")
        )

    def _assert_raw_protocol(self, name):
        config = self._load(f"configs/experiment/{name}.yaml")
        trainer = config["trainer"]
        model = config["model"]["model_config"]
        sampling = model["validation_rollout_sampling"]
        self.assertEqual(trainer["limit_val_batches"], 0.1)
        self.assertEqual(trainer["check_val_every_n_epoch"], 1)
        self.assertTrue(model["val_open_loop"])
        self.assertTrue(model["val_closed_loop"])
        self.assertEqual(model["wosac_backend"], "fast")
        self.assertEqual(model["wosac_metrics_version"], "2025")
        self.assertTrue(model["fast_wosac_require_preprocessed_gt"])
        self.assertEqual(model["n_batch_wosac_metric"], -1)
        self.assertEqual(model["n_rollout_closed_val"], 32)
        self.assertEqual(sampling["criterium"], "topk_prob")
        self.assertEqual(sampling["num_k"], 48)
        self.assertEqual(sampling["temp"], 1.0)
        self.assertEqual(
            model["fast_wosac_gt_dir"],
            "${paths.validation_gt_dir}",
        )

    def test_pre_bc_protocol(self):
        self._assert_raw_protocol("pre_bc")

    def test_clsft_protocol(self):
        self._assert_raw_protocol("clsft")

    def test_shared_path_uses_environment_with_deployment_default(self):
        config = self._load("configs/paths/default.yaml")
        self.assertEqual(
            config["validation_gt_dir"],
            "${oc.env:FAST_WOSAC_GT_DIR," + DEFAULT_GT + "}",
        )

    def test_derived_experiments_inherit_protocol(self):
        try:
            import hydra
        except ModuleNotFoundError:
            self.skipTest("Hydra is not installed")
        with hydra.initialize_config_dir(
            config_dir=str(ROOT / "configs"),
            version_base=None,
        ):
            for experiment in (
                "pre_bc_history_dynamics",
                "pre_bc_trajtok_original",
                "clsft_history_dynamics",
            ):
                config = hydra.compose(
                    config_name="run.yaml",
                    overrides=[f"experiment={experiment}"],
                )
                self.assertEqual(config.trainer.limit_val_batches, 0.1)
                self.assertEqual(
                    config.model.model_config.wosac_backend,
                    "fast",
                )
                self.assertTrue(
                    config.model.model_config
                    .fast_wosac_require_preprocessed_gt
                )
                self.assertEqual(
                    config.model.model_config
                    .validation_rollout_sampling.num_k,
                    48,
                )

    def test_environment_overrides_gt_directory(self):
        try:
            import hydra
        except ModuleNotFoundError:
            self.skipTest("Hydra is not installed")
        with patch.dict(
            os.environ,
            {"FAST_WOSAC_GT_DIR": "/tmp/custom-validation-gt"},
        ):
            with hydra.initialize_config_dir(
                config_dir=str(ROOT / "configs"),
                version_base=None,
            ):
                config = hydra.compose(
                    config_name="run.yaml",
                    overrides=["experiment=pre_bc"],
                )
            self.assertEqual(
                config.paths.validation_gt_dir,
                "/tmp/custom-validation-gt",
            )
```

- [ ] **Step 2: Run the config tests and verify RED**

Run:

```bash
python -m pytest tests/test_training_fast_wosac_config.py -q
```

Expected: `pre_bc`, `clsft`, and shared-path assertions fail because the new
protocol is not configured. Hydra-dependent tests may be skipped locally.

- [ ] **Step 3: Add the shared validation-GT path**

Append to `configs/paths/default.yaml`:

```yaml
validation_gt_dir: ${oc.env:FAST_WOSAC_GT_DIR,/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario/validation_gt}
```

Do not remove the existing `inference.yaml` path override; standalone inference
must keep deriving its fallback path from `CACHE_ROOT`.

- [ ] **Step 4: Apply the training protocol to both base experiments**

Add the following under `model.model_config` in both
`configs/experiment/pre_bc.yaml` and `configs/experiment/clsft.yaml`:

```yaml
n_rollout_closed_val: 32
n_batch_wosac_metric: -1
val_open_loop: true
val_closed_loop: true
wosac_backend: fast
wosac_metrics_version: "2025"
fast_wosac_gt_dir: ${paths.validation_gt_dir}
fast_wosac_require_preprocessed_gt: true
validation_rollout_sampling:
  criterium: topk_prob
  num_k: 48
  temp: 1.0
```

Keep `pre_bc` at:

```yaml
trainer:
  limit_val_batches: 0.1
  check_val_every_n_epoch: 1
```

Change the corresponding `clsft` value from `50` to `0.1`.

- [ ] **Step 5: Run config and existing composition tests**

Run:

```bash
python -m pytest \
  tests/test_training_fast_wosac_config.py \
  tests/test_spatial_aware_loss.py \
  tests/test_future_token_dynamics_configs.py \
  -q
```

Expected: all available tests pass; Hydra-only tests are allowed to skip only
when Hydra is not installed.

- [ ] **Step 6: Commit the training defaults**

```bash
git add \
  configs/paths/default.yaml \
  configs/experiment/pre_bc.yaml \
  configs/experiment/clsft.yaml \
  tests/test_training_fast_wosac_config.py
git commit -m "feat: run fast WOSAC during every training epoch"
```

---

### Task 4: Document and verify the complete workflow

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents: default protocol, strict GT requirement, path override, cost, and opt-out overrides

- [ ] **Step 1: Update the training documentation**

In the README training section, add:

````markdown
### Training-time Fast WOSAC validation

`pre_bc` and `clsft` validate after every epoch on a deterministic 10% prefix
of the validation loader. Open-loop loss/accuracy remain enabled, and
closed-loop validation uses TrajTok Fast WOSAC 2025 with 32 rollouts and
inference `K=48`.

The evaluator strictly reads preprocessed scenario dictionaries from
`/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario/validation_gt`. Override
the location when necessary:

```bash
FAST_WOSAC_GT_DIR=/path/to/validation_gt \
MY_EXPERIMENT=pre_bc \
bash scripts/train.sh
```

A missing directory or scenario artifact is an error; training-time
validation never falls back to raw TFRecords. Each epoch evaluates roughly
4,400 scenarios, so this is substantially slower than the previous limited
validation. Override `trainer.limit_val_batches` for a different fraction, or
set `model.model_config.val_closed_loop=false` to retain only open-loop
validation.
````

- [ ] **Step 2: Run focused regression tests**

Run:

```bash
python -m pytest \
  tests/test_preprocessed_scenario_gt.py \
  tests/test_fast_wosac_strict_integration.py \
  tests/test_training_fast_wosac_config.py \
  tests/test_spatial_aware_loss.py \
  tests/test_future_token_dynamics_configs.py \
  -q
```

Expected: all runnable tests pass with no failures.

- [ ] **Step 3: Run the complete repository test suite**

Run:

```bash
python -m pytest -q
```

Expected: exit code `0`; dependency-gated tests may report explicit skips.

- [ ] **Step 4: Verify syntax, YAML, and patch cleanliness**

Run:

```bash
python -m py_compile \
  src/smart/metrics/preprocessed_scenario_gt.py \
  src/smart/metrics/fast_wosac_metrics.py \
  src/smart/model/smart.py
python - <<'PY'
from pathlib import Path
import yaml

for path in (
    "configs/paths/default.yaml",
    "configs/model/smart.yaml",
    "configs/experiment/pre_bc.yaml",
    "configs/experiment/clsft.yaml",
):
    yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    print("OK", path)
PY
git diff --check
```

Expected: every command exits `0`, each YAML file prints `OK`, and
`git diff --check` produces no output.

- [ ] **Step 5: Review only in-scope changes**

Run:

```bash
git status --short
git diff -- \
  README.md \
  configs/paths/default.yaml \
  configs/model/smart.yaml \
  configs/experiment/pre_bc.yaml \
  configs/experiment/clsft.yaml \
  src/smart/metrics/preprocessed_scenario_gt.py \
  src/smart/metrics/fast_wosac_metrics.py \
  src/smart/model/smart.py \
  tests/test_preprocessed_scenario_gt.py \
  tests/test_fast_wosac_strict_integration.py \
  tests/test_training_fast_wosac_config.py
```

Confirm that unrelated existing files such as `scripts/cache_womd.sh`,
presentation artifacts, `.DS_Store`, and `.codex_work/` were neither staged
nor modified by this implementation.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain training-time fast WOSAC validation"
```
