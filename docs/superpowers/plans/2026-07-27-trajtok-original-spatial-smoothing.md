# TrajTok Original Spatial-Smoothing Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CatK-compatible pre-BC loss mode that reproduces TrajTok commit `5920c89`'s original `gt_idx`-centered, twice-divided spatial smoothing formula without changing existing CatK experiments or checkpoints.

**Architecture:** Add a separate probability-target helper for the original TrajTok math, then select it inside CatK's existing `CrossEntropy` metric through an explicit string mode. CatK's `SMART` model passes the already-available 16 future `gt_idx` values to the loss, while a new Hydra experiment selects the mode without changing the current `pre_bc` default.

**Tech Stack:** Python 3, PyTorch, TorchMetrics, Lightning, Hydra/OmegaConf, YAML, `unittest`.

## Global Constraints

- Preserve the original TrajTok double division: normalize `proj` by `proj_sum`, then divide by the same original `proj_sum` again.
- Center spatial distances on the contour selected by `gt_idx`, not on CatK's raw ground-truth endpoint.
- Do not clamp or renormalize the original-mode probability target.
- Keep `raw_gt_normalized` as the default spatial mode for all existing CatK experiments.
- Keep `spatial_aware_smoothing=false` behavior unchanged.
- Keep CLSFT, vocabulary files, model parameters, state-dict keys, and checkpoint loading unchanged.
- Reuse CatK's shared 2048-class head and per-agent type-appropriate `token_traj`; do not introduce TrajTok's type-split heads.
- Preserve all unrelated user-owned working-tree changes and stage only the files named in each task.

---

## File Structure

- Modify `src/smart/metrics/utils.py`: own the standalone, auditable TrajTok-original probability formula.
- Modify `src/smart/metrics/cross_entropy.py`: validate and select `raw_gt_normalized` versus `trajtok_original`.
- Modify `src/smart/model/smart.py`: pass aligned future `gt_idx` values into training and open-loop validation loss calls.
- Modify `configs/model/smart.yaml`: define the backward-compatible default mode.
- Create `configs/experiment/pre_bc_trajtok_original.yaml`: expose a concise plain pre-BC experiment.
- Modify `tests/test_spatial_aware_loss.py`: provide reference-parity, loss-routing, validation, and configuration regression coverage.

---

### Task 1: Add the Original TrajTok Probability Target

**Files:**
- Modify: `tests/test_spatial_aware_loss.py:31-185`
- Modify: `src/smart/metrics/utils.py:49-97`

**Interfaces:**
- Consumes: `gt_idx: Tensor[n_agent, n_step]`, `token_traj: Tensor[n_agent, n_token, 4, 2]`, and `label_smoothing: float`.
- Produces: `get_prob_targets_trajtok_original(gt_idx: Tensor, token_traj: Tensor, label_smoothing: float) -> Tensor[n_agent, n_step, n_token]`.

- [ ] **Step 1: Write a failing reference-parity test**

Add a helper lookup and reference implementation to
`tests/test_spatial_aware_loss.py`:

```python
def _get_trajtok_original_helper():
    return getattr(METRIC_UTILS, "get_prob_targets_trajtok_original")


def _trajtok_original_reference(gt_idx, token_traj, label_smoothing):
    gt_token_traj = torch.gather(
        token_traj,
        dim=1,
        index=gt_idx.unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, -1, 4, 2),
    )
    dists = torch.norm(
        gt_token_traj[:, :, None, :, :]
        - token_traj[:, None, :, :, :],
        dim=-1,
    ).mean(-1)
    target_mask = torch.nn.functional.one_hot(
        gt_idx,
        num_classes=token_traj.shape[1],
    ).bool()
    probability = torch.zeros(
        gt_idx.shape[0],
        gt_idx.shape[1],
        token_traj.shape[1],
        device=gt_idx.device,
    )
    probability[target_mask] = 1.0 - label_smoothing
    projection = 1.0 / ((0.0001 + dists) ** 2)
    projection = projection * (~target_mask).int()
    projection_sum = projection.sum(dim=-1, keepdim=True)
    projection = projection / projection_sum
    probability += projection / projection_sum * label_smoothing
    return probability
```

Add these tests after `SpatialAwareTargetTest`:

```python
class TrajTokOriginalTargetTest(unittest.TestCase):
    def test_matches_trajtok_5920c89_reference(self):
        token_traj = SpatialAwareTargetTest._contours(
            [0.0, 1.0, 4.0]
        )
        gt_idx = torch.tensor([[0]], dtype=torch.long)

        actual = _get_trajtok_original_helper()(
            gt_idx=gt_idx,
            token_traj=token_traj,
            label_smoothing=0.1,
        )
        expected = _trajtok_original_reference(
            gt_idx,
            token_traj,
            0.1,
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_retains_second_division_and_non_unit_mass(self):
        token_traj = SpatialAwareTargetTest._contours(
            [0.0, 1.0, 4.0]
        )
        probability = _get_trajtok_original_helper()(
            gt_idx=torch.tensor([[0]], dtype=torch.long),
            token_traj=token_traj,
            label_smoothing=0.1,
        )
        projection_sum = (
            1.0 / (1.0001**2) + 1.0 / (4.0001**2)
        )
        expected_mass = 0.9 + 0.1 / projection_sum

        self.assertAlmostEqual(
            float(probability[0, 0, 0]),
            0.9,
            places=6,
        )
        self.assertAlmostEqual(
            float(probability.sum()),
            expected_mass,
            places=6,
        )
        self.assertFalse(
            torch.allclose(probability.sum(), torch.tensor(1.0))
        )

    def test_selected_gt_idx_changes_spatial_center(self):
        token_traj = SpatialAwareTargetTest._contours(
            [0.0, 1.0, 4.0]
        )
        helper = _get_trajtok_original_helper()

        centered_at_zero = helper(
            torch.tensor([[0]]),
            token_traj,
            0.1,
        )
        centered_at_one = helper(
            torch.tensor([[1]]),
            token_traj,
            0.1,
        )

        self.assertAlmostEqual(
            float(centered_at_zero[0, 0, 0]),
            0.9,
            places=6,
        )
        self.assertAlmostEqual(
            float(centered_at_one[0, 0, 1]),
            0.9,
            places=6,
        )
        self.assertFalse(torch.equal(centered_at_zero, centered_at_one))
```

- [ ] **Step 2: Run the focused tests and verify the expected failure**

Run:

```bash
python -m unittest \
  tests.test_spatial_aware_loss.TrajTokOriginalTargetTest \
  -v
```

Expected: `ERROR` with
`AttributeError: module 'catk_test_metric_utils' has no attribute 'get_prob_targets_trajtok_original'`.

- [ ] **Step 3: Implement the exact original formula**

Add this separate helper below
`get_prob_targets_spatial_aware_smoothing` in
`src/smart/metrics/utils.py`:

```python
@torch.no_grad()
def get_prob_targets_trajtok_original(
    gt_idx: Tensor,  # [n_agent, n_step]
    token_traj: Tensor,  # [n_agent, n_token, 4, 2]
    label_smoothing: float,
) -> Tensor:  # [n_agent, n_step, n_token]
    gt_token_traj = torch.gather(
        token_traj,
        dim=1,
        index=gt_idx.unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, -1, 4, 2),
    )
    dists = torch.norm(
        gt_token_traj[:, :, None, :, :]
        - token_traj[:, None, :, :, :],
        dim=-1,
    ).mean(-1)
    closest_token_mask = one_hot(
        gt_idx,
        num_classes=token_traj.shape[1],
    ).bool()
    prob_target = torch.zeros(
        gt_idx.shape[0],
        gt_idx.shape[1],
        token_traj.shape[1],
        device=gt_idx.device,
    )
    prob_target[closest_token_mask] = 1.0 - label_smoothing
    proj = 1.0 / ((0.0001 + dists) ** 2)
    proj = proj * (~closest_token_mask).int()
    proj_sum = proj.sum(dim=-1, keepdim=True)
    proj = proj / proj_sum
    prob_target += proj / proj_sum * label_smoothing
    return prob_target
```

Do not reuse CatK's raw-GT helper, mask `proj_sum` with a clamp, add a
single-token fallback, or normalize `prob_target`.

- [ ] **Step 4: Run the original-target and existing target tests**

Run:

```bash
python -m unittest \
  tests.test_spatial_aware_loss.TrajTokOriginalTargetTest \
  tests.test_spatial_aware_loss.SpatialAwareTargetTest \
  -v
```

Expected: all eight target tests pass; the existing raw-GT normalized tests
remain unchanged.

- [ ] **Step 5: Commit the helper**

```bash
git add \
  src/smart/metrics/utils.py \
  tests/test_spatial_aware_loss.py
git commit -m "feat: add TrajTok original spatial targets"
```

---

### Task 2: Route the Compatibility Mode Through CatK Loss

**Files:**
- Modify: `tests/test_spatial_aware_loss.py:38-283`
- Modify: `src/smart/metrics/cross_entropy.py:21-138`
- Modify: `src/smart/model/smart.py:108-140`

**Interfaces:**
- Consumes: Task 1's
  `get_prob_targets_trajtok_original(gt_idx, token_traj, label_smoothing)`.
- Produces: `CrossEntropy(..., spatial_aware_smoothing_mode: str = "raw_gt_normalized")` and optional `update(..., gt_idx: Optional[Tensor] = None)`.
- Produces: both training and open-loop validation pass
  `tokenized_agent["gt_idx"][:, 2:]` to the loss.

- [ ] **Step 1: Make the metric tests executable without TorchMetrics**

Replace the current skip-based `_load_cross_entropy` setup with a minimal test
stub when TorchMetrics is unavailable:

```python
def _install_torchmetrics_stub():
    torchmetrics_module = types.ModuleType("torchmetrics")
    metric_module = types.ModuleType("torchmetrics.metric")

    class Metric(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def add_state(self, name, default, dist_reduce_fx=None):
            self.register_buffer(name, default.clone())

        def forward(self, *args, **kwargs):
            self.update(*args, **kwargs)
            return self.compute()

    metric_module.Metric = Metric
    torchmetrics_module.metric = metric_module
    sys.modules["torchmetrics"] = torchmetrics_module
    sys.modules["torchmetrics.metric"] = metric_module


def _load_cross_entropy():
    if importlib.util.find_spec("torchmetrics") is None:
        _install_torchmetrics_stub()

    package_name = "catk_test_metrics"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "src/smart/metrics")]
    sys.modules[package_name] = package

    for module_name in ("utils", "cross_entropy"):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name,
            ROOT / f"src/smart/metrics/{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)

    return sys.modules[f"{package_name}.cross_entropy"].CrossEntropy
```

Remove the `@unittest.skipIf` decorator because the stub supplies the exact
`Metric` behavior used by these unit tests.

- [ ] **Step 2: Write failing mode-selection tests**

Change `_metric` and `_inputs`:

```python
@staticmethod
def _metric(
    spatial_aware_smoothing,
    spatial_aware_smoothing_mode="raw_gt_normalized",
):
    metric = CrossEntropy(
        use_gt_raw=True,
        gt_thresh_scale_length=-1.0,
        label_smoothing=0.1,
        rollout_as_gt=False,
        spatial_aware_smoothing=spatial_aware_smoothing,
        spatial_aware_smoothing_mode=spatial_aware_smoothing_mode,
    )
    metric.eval()
    return metric
```

Add the aligned index to `_inputs()`:

```python
"gt_idx": torch.zeros(n_agent, n_step, dtype=torch.long),
```

Add these tests:

```python
def test_trajtok_original_mode_uses_original_soft_target(self):
    metric = self._metric(
        spatial_aware_smoothing=True,
        spatial_aware_smoothing_mode="trajtok_original",
    )
    inputs = self._inputs()

    metric.update(**inputs)

    probability = _trajtok_original_reference(
        inputs["gt_idx"],
        inputs["token_traj"],
        0.1,
    )
    expected = torch.nn.functional.cross_entropy(
        inputs["next_token_logits"].transpose(1, 2),
        probability.transpose(1, 2),
        reduction="none",
        label_smoothing=0.0,
    ).mean()
    self.assertTrue(torch.allclose(metric.compute(), expected))

def test_trajtok_original_mode_requires_gt_idx(self):
    metric = self._metric(
        spatial_aware_smoothing=True,
        spatial_aware_smoothing_mode="trajtok_original",
    )
    inputs = self._inputs()
    inputs.pop("gt_idx")

    with self.assertRaisesRegex(ValueError, "gt_idx"):
        metric.update(**inputs)

def test_unknown_spatial_mode_is_rejected(self):
    with self.assertRaisesRegex(
        ValueError,
        "spatial_aware_smoothing_mode",
    ):
        self._metric(
            spatial_aware_smoothing=True,
            spatial_aware_smoothing_mode="unknown",
        )
```

- [ ] **Step 3: Run the metric tests and verify the expected failure**

Run:

```bash
python -m unittest \
  tests.test_spatial_aware_loss.CrossEntropySelectionTest \
  -v
```

Expected: `ERROR` because `CrossEntropy.__init__` does not yet accept
`spatial_aware_smoothing_mode`.

- [ ] **Step 4: Implement mode validation and routing**

Import the new helper in `src/smart/metrics/cross_entropy.py`:

```python
from .utils import (
    get_euclidean_targets,
    get_prob_targets,
    get_prob_targets_spatial_aware_smoothing,
    get_prob_targets_trajtok_original,
)
```

Extend the constructor:

```python
def __init__(
    self,
    use_gt_raw: bool,
    gt_thresh_scale_length: float,
    label_smoothing: float,
    rollout_as_gt: bool,
    spatial_aware_smoothing: bool = False,
    spatial_aware_smoothing_mode: str = "raw_gt_normalized",
) -> None:
```

After validating `label_smoothing`, validate and store the mode:

```python
valid_spatial_modes = {
    "raw_gt_normalized",
    "trajtok_original",
}
if spatial_aware_smoothing_mode not in valid_spatial_modes:
    raise ValueError(
        "spatial_aware_smoothing_mode must be one of "
        f"{sorted(valid_spatial_modes)}, got "
        f"{spatial_aware_smoothing_mode!r}"
    )
self.spatial_aware_smoothing_mode = spatial_aware_smoothing_mode
```

Add the optional argument after `token_traj` in `update`:

```python
gt_idx: Optional[Tensor] = None,  # [n_agent, 16]
```

Replace the current spatial branch with:

```python
if self.spatial_aware_smoothing:
    if self.spatial_aware_smoothing_mode == "trajtok_original":
        if gt_idx is None:
            raise ValueError(
                "gt_idx is required when "
                "spatial_aware_smoothing_mode='trajtok_original'"
            )
        prob_target = get_prob_targets_trajtok_original(
            gt_idx=gt_idx,
            token_traj=token_traj,
            label_smoothing=self.label_smoothing,
        )
    else:
        prob_target = get_prob_targets_spatial_aware_smoothing(
            target=euclidean_target,
            token_agent_shape=token_agent_shape,
            token_traj=token_traj,
            label_smoothing=self.label_smoothing,
        )
    builtin_label_smoothing = 0.0
else:
    prob_target = get_prob_targets(
        target=euclidean_target,
        token_agent_shape=token_agent_shape,
        token_traj=token_traj,
    )
    builtin_label_smoothing = self.label_smoothing
```

Keep `euclidean_target_valid` and the current mask calculation unchanged so
CatK's valid transition set remains identical.

- [ ] **Step 5: Pass the future token indices from SMART**

Add this keyword to both loss calls in `src/smart/model/smart.py`:

```python
gt_idx=tokenized_agent["gt_idx"][:, 2:],  # [n_agent, 16]
```

The training call becomes:

```python
loss = self.training_loss(
    **pred,
    token_agent_shape=tokenized_agent["token_agent_shape"],
    token_traj=tokenized_agent["token_traj"],
    gt_idx=tokenized_agent["gt_idx"][:, 2:],
    train_mask=data["agent"]["train_mask"],
    current_epoch=self.current_epoch,
)
```

The open-loop validation call uses the same `gt_idx` keyword and omits
`train_mask`, preserving its current behavior.

- [ ] **Step 6: Run metric and source compilation tests**

Run:

```bash
python -m unittest \
  tests.test_spatial_aware_loss.CrossEntropySelectionTest \
  -v
python -m compileall -q \
  src/smart/metrics \
  src/smart/model/smart.py
```

Expected: all five metric-selection tests pass, including the two existing
raw-GT and uniform-smoothing regressions; compilation exits zero.

- [ ] **Step 7: Commit the loss routing**

```bash
git add \
  src/smart/metrics/cross_entropy.py \
  src/smart/model/smart.py \
  tests/test_spatial_aware_loss.py
git commit -m "feat: wire TrajTok original loss mode"
```

---

### Task 3: Add the Backward-Compatible Hydra Experiment

**Files:**
- Modify: `tests/test_spatial_aware_loss.py:286-323`
- Modify: `configs/model/smart.yaml:105-111`
- Create: `configs/experiment/pre_bc_trajtok_original.yaml`

**Interfaces:**
- Consumes: Task 2's
  `CrossEntropy(..., spatial_aware_smoothing_mode="trajtok_original")`.
- Produces: resolved configuration
  `model.model_config.training_loss.spatial_aware_smoothing_mode`.
- Produces: `experiment=pre_bc_trajtok_original`.

- [ ] **Step 1: Write failing configuration tests**

Add these assertions to `SpatialAwareConfigTest`:

```python
def test_base_defaults_to_current_raw_gt_mode(self):
    config = self._load("configs/model/smart.yaml")
    self.assertEqual(
        config["model_config"]["training_loss"][
            "spatial_aware_smoothing_mode"
        ],
        "raw_gt_normalized",
    )

def test_trajtok_original_experiment_only_overrides_mode(self):
    config = self._load(
        "configs/experiment/pre_bc_trajtok_original.yaml"
    )
    self.assertEqual(config["defaults"], ["pre_bc", "_self_"])
    self.assertEqual(
        config["model"]["model_config"]["training_loss"],
        {"spatial_aware_smoothing_mode": "trajtok_original"},
    )

def test_trajtok_original_experiment_composes(self):
    try:
        import hydra
    except ModuleNotFoundError:
        self.skipTest("Hydra is not installed in this test environment")

    with hydra.initialize_config_dir(
        config_dir=str(ROOT / "configs"),
        version_base=None,
    ):
        config = hydra.compose(
            config_name="run.yaml",
            overrides=["experiment=pre_bc_trajtok_original"],
        )

    loss_config = config.model.model_config.training_loss
    self.assertTrue(loss_config.spatial_aware_smoothing)
    self.assertEqual(
        loss_config.spatial_aware_smoothing_mode,
        "trajtok_original",
    )
    self.assertEqual(loss_config.label_smoothing, 0.1)
```

Extend the existing base/CLSFT regression to assert that CLSFT remains
spatially disabled without defining a conflicting experiment-local mode:

```python
def test_base_and_clsft_keep_spatial_smoothing_disabled(self):
    base = self._load("configs/model/smart.yaml")
    clsft = self._load("configs/experiment/clsft.yaml")
    base_loss = base["model_config"]["training_loss"]
    clsft_loss = clsft["model"]["model_config"]["training_loss"]

    self.assertFalse(base_loss["spatial_aware_smoothing"])
    self.assertEqual(
        base_loss["spatial_aware_smoothing_mode"],
        "raw_gt_normalized",
    )
    self.assertFalse(clsft_loss["spatial_aware_smoothing"])
    self.assertNotIn("spatial_aware_smoothing_mode", clsft_loss)
```

In `test_trajtok_original_experiment_composes`, compose CLSFT in the same
Hydra context and verify the inherited resolved values:

```python
clsft_config = hydra.compose(
    config_name="run.yaml",
    overrides=["experiment=clsft"],
)
clsft_loss = clsft_config.model.model_config.training_loss
self.assertFalse(clsft_loss.spatial_aware_smoothing)
self.assertEqual(
    clsft_loss.spatial_aware_smoothing_mode,
    "raw_gt_normalized",
)
```

- [ ] **Step 2: Run the config tests and verify the expected failure**

Run:

```bash
python -m unittest \
  tests.test_spatial_aware_loss.SpatialAwareConfigTest \
  -v
```

Expected: failures for the absent base key and absent
`pre_bc_trajtok_original.yaml`.

- [ ] **Step 3: Add the default mode and experiment file**

Append the mode under `training_loss` in `configs/model/smart.yaml`:

```yaml
  training_loss:
    use_gt_raw: true
    gt_thresh_scale_length: -1.0
    label_smoothing: 0.1
    rollout_as_gt: false
    spatial_aware_smoothing: false
    spatial_aware_smoothing_mode: raw_gt_normalized
```

Create `configs/experiment/pre_bc_trajtok_original.yaml`:

```yaml
# @package _global_

# CatK pre-BC with the original TrajTok spatial-smoothing formula.
defaults:
  - pre_bc
  - _self_

model:
  model_config:
    training_loss:
      spatial_aware_smoothing_mode: trajtok_original
```

Do not edit `pre_bc.yaml`, `clsft.yaml`, or any history/future-dynamics
experiment. Their inheritance and explicit overrides already provide the
required backward compatibility.

- [ ] **Step 4: Run configuration and focused loss tests**

Run:

```bash
python -m unittest \
  tests.test_spatial_aware_loss \
  -v
```

Expected: every test passes. Hydra composition may be reported as skipped only
when Hydra is absent from the local Python environment.

On an environment with project dependencies, also run:

```bash
python -m src.run \
  --cfg job \
  --resolve \
  experiment=pre_bc_trajtok_original \
  task_name=trajtok_original_config_check
```

Expected resolved values:

```text
spatial_aware_smoothing: true
spatial_aware_smoothing_mode: trajtok_original
label_smoothing: 0.1
```

- [ ] **Step 5: Commit the experiment**

```bash
git add \
  configs/model/smart.yaml \
  configs/experiment/pre_bc_trajtok_original.yaml \
  tests/test_spatial_aware_loss.py
git commit -m "feat: add TrajTok original pre-BC experiment"
```

---

### Task 4: Run Adjacent Regression and Compatibility Verification

**Files:**
- Verify only; no additional production files should be introduced.

**Interfaces:**
- Consumes: Tasks 1-3 as one complete feature.
- Produces: evidence that current CatK modes, dynamics experiments, source
  syntax, and checkpoint parameter structure remain compatible.

- [ ] **Step 1: Run the focused and adjacent unit suites**

Run:

```bash
python -m unittest \
  tests.test_spatial_aware_loss \
  tests.test_history_dynamics \
  tests.test_future_token_dynamics \
  tests.test_future_token_dynamics_configs \
  -v
```

Expected: all runnable tests pass; dependency-gated tests may skip with their
existing explicit skip reason.

- [ ] **Step 2: Run source and YAML checks**

Run:

```bash
python -m compileall -q src tests
python -c 'import pathlib,yaml; [yaml.safe_load(p.read_text()) for p in pathlib.Path("configs").rglob("*.yaml")]'
git diff --check HEAD~3..HEAD
```

Expected: all commands exit zero.

- [ ] **Step 3: Verify only intended files changed**

Run:

```bash
git diff --name-only HEAD~3..HEAD
git status --short --branch
```

Expected feature files:

```text
configs/experiment/pre_bc_trajtok_original.yaml
configs/model/smart.yaml
src/smart/metrics/cross_entropy.py
src/smart/metrics/utils.py
src/smart/model/smart.py
tests/test_spatial_aware_loss.py
```

Existing unrelated user-owned changes may still appear in `git status`; they
must remain unstaged and unmodified by this implementation.

- [ ] **Step 4: Confirm the two experiment entry points**

Plain CatK pre-BC:

```bash
MY_EXPERIMENT=pre_bc_trajtok_original \
MY_TASK_NAME=pre_bc_trajtok_original_b200 \
bash scripts/train.sh \
  ckpt_path=null \
  model.model_config.val_closed_loop=false
```

Any existing pre-BC family, including the hybrid dynamics experiment:

```bash
MY_EXPERIMENT=pre_bc_history_future_token_dynamics_hybrid \
MY_TASK_NAME=pre_bc_history_future_token_dynamics_hybrid_trajtok_original_b200 \
bash scripts/train.sh \
  ckpt_path=null \
  model.model_config.future_token_dynamics.lookup_file="$LOOKUP_FILE" \
  model.model_config.training_loss.spatial_aware_smoothing_mode=trajtok_original \
  model.model_config.val_closed_loop=false
```

Expected: both configurations instantiate the same CatK model parameter
structure; only the selected training target formula differs.

- [ ] **Step 5: Record final verification without an empty commit**

Run:

```bash
git log --oneline -5
git status --short --branch
```

Expected: the design commit, this implementation-plan commit, and the three
feature commits are present. Do not create a verification-only commit when no
files changed.
