# Raw-GT Spatial-Aware Label Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CatK pre-BC use normalized, endpoint-only spatial label smoothing centered on the raw ground-truth contour with `epsilon=0.1`.

**Architecture:** Add one pure target-construction helper beside the existing one-hot helper, then select between the helpers inside CatK's existing `CrossEntropy` metric. Preserve the current rollout-relative Euclidean target and all masks. Enable the new path only through pre-BC experiment configuration, without changing model parameters or checkpoint keys.

**Tech Stack:** Python 3.11, PyTorch 2.4, TorchMetrics 1.4, Hydra/OmegaConf configuration, `unittest`.

## Global Constraints

- Supervise only the final four-corner contour of each 0.5-second token.
- Center neighbor probabilities on the raw ground-truth contour, not on the quantized target token.
- Use `label_smoothing=0.1` for pre-BC: target mass `0.9`, normalized non-target mass `0.1`.
- Keep the legacy loss bit-for-bit on its existing code path when spatial smoothing is disabled.
- Enable spatial smoothing for `pre_bc` and inherited `pre_bc_history_dynamics`.
- Keep the base model and `clsft` on the legacy path.
- Do not change model parameter shapes, vocabulary data, token matching, or checkpoint keys.
- Do not stage or modify the user's existing `scripts/cache_womd.sh`, `.DS_Store`, or `.codex_work/` changes.

---

### Task 1: Raw-GT spatial target construction

**Files:**
- Create: `tests/test_spatial_aware_loss.py`
- Modify: `src/smart/metrics/utils.py:23-46`

**Interfaces:**
- Consumes:
  - `target: Tensor` with shape `[n_agent, n_step, 3]`
  - `token_agent_shape: Tensor` with shape `[n_agent, 2]`
  - `token_traj: Tensor` with shape `[n_agent, n_token, 4, 2]`
  - `label_smoothing: float`
- Produces:
  - `get_prob_targets_spatial_aware_smoothing(...) -> Tensor`
  - Output shape `[n_agent, n_step, n_token]`, on the target device and in the target dtype

- [ ] **Step 1: Write failing target-distribution tests**

Create `tests/test_spatial_aware_loss.py` with:

```python
import unittest

import torch

from src.smart.metrics.utils import (
    get_prob_targets,
    get_prob_targets_spatial_aware_smoothing,
)


class SpatialAwareTargetTest(unittest.TestCase):
    @staticmethod
    def _contours(center_x):
        center_x = torch.as_tensor(center_x, dtype=torch.float32)
        half_length = 2.0
        half_width = 1.0
        corners = torch.stack(
            (
                torch.stack((center_x + half_length, torch.full_like(center_x, half_width)), dim=-1),
                torch.stack((center_x + half_length, torch.full_like(center_x, -half_width)), dim=-1),
                torch.stack((center_x - half_length, torch.full_like(center_x, -half_width)), dim=-1),
                torch.stack((center_x - half_length, torch.full_like(center_x, half_width)), dim=-1),
            ),
            dim=-2,
        )
        return corners.unsqueeze(0)

    @staticmethod
    def _shape():
        return torch.tensor([[2.0, 4.0]], dtype=torch.float32)

    def test_distribution_is_normalized_and_reserves_point_one_for_neighbors(self):
        target = torch.tensor([[[0.2, 0.0, 0.0]]], dtype=torch.float32)
        token_traj = self._contours([0.0, 1.0, 4.0])

        probability = get_prob_targets_spatial_aware_smoothing(
            target=target,
            token_agent_shape=self._shape(),
            token_traj=token_traj,
            label_smoothing=0.1,
        )

        self.assertTrue(torch.allclose(probability.sum(dim=-1), torch.ones(1, 1)))
        self.assertAlmostEqual(float(probability[0, 0, 0]), 0.9, places=6)
        self.assertAlmostEqual(float(probability[0, 0, 1:].sum()), 0.1, places=6)
        self.assertGreater(float(probability[0, 0, 1]), float(probability[0, 0, 2]))

    def test_neighbors_are_ranked_from_raw_gt_not_quantized_target(self):
        target = torch.tensor([[[0.2, 0.0, 0.0]]], dtype=torch.float32)
        # Token 0 is the hard target. Relative to token 0, token 2 is closer
        # than token 1; relative to raw GT, token 1 is closer than token 2.
        token_traj = self._contours([0.0, 0.55, -0.45])

        probability = get_prob_targets_spatial_aware_smoothing(
            target=target,
            token_agent_shape=self._shape(),
            token_traj=token_traj,
            label_smoothing=0.1,
        )

        self.assertAlmostEqual(float(probability[0, 0, 0]), 0.9, places=6)
        self.assertGreater(float(probability[0, 0, 1]), float(probability[0, 0, 2]))

    def test_zero_smoothing_matches_legacy_one_hot_target(self):
        target = torch.tensor([[[0.2, 0.0, 0.0]]], dtype=torch.float32)
        token_traj = self._contours([0.0, 1.0, 4.0])

        legacy = get_prob_targets(target, self._shape(), token_traj)
        spatial = get_prob_targets_spatial_aware_smoothing(
            target=target,
            token_agent_shape=self._shape(),
            token_traj=token_traj,
            label_smoothing=0.0,
        )

        self.assertTrue(torch.equal(spatial, legacy))

    def test_single_token_vocabulary_returns_one_hot(self):
        probability = get_prob_targets_spatial_aware_smoothing(
            target=torch.tensor([[[0.2, 0.0, 0.0]]], dtype=torch.float32),
            token_agent_shape=self._shape(),
            token_traj=self._contours([0.0]),
            label_smoothing=0.1,
        )

        self.assertTrue(torch.equal(probability, torch.ones(1, 1, 1)))

    def test_invalid_smoothing_is_rejected(self):
        target = torch.tensor([[[0.2, 0.0, 0.0]]], dtype=torch.float32)
        token_traj = self._contours([0.0, 1.0])

        for value in (-0.1, 1.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "label_smoothing"):
                    get_prob_targets_spatial_aware_smoothing(
                        target=target,
                        token_agent_shape=self._shape(),
                        token_traj=token_traj,
                        label_smoothing=value,
                    )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and verify RED**

Run in the CatK environment:

```bash
python -m unittest tests.test_spatial_aware_loss -v
```

Expected: import failure stating that
`get_prob_targets_spatial_aware_smoothing` does not exist.

- [ ] **Step 3: Implement the minimal spatial target helper**

Add this function after `get_prob_targets` in `src/smart/metrics/utils.py`:

```python
@torch.no_grad()
def get_prob_targets_spatial_aware_smoothing(
    target: Tensor,
    token_agent_shape: Tensor,
    token_traj: Tensor,
    label_smoothing: float,
) -> Tensor:
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(
            "label_smoothing must be in [0, 1), "
            f"got {label_smoothing}"
        )
    if token_traj.shape[1] < 1:
        raise ValueError("token_traj must contain at least one token")

    contour = cal_polygon_contour(
        target[..., :2],
        target[..., 2],
        token_agent_shape[:, None, :],
    )
    distances = torch.norm(
        contour.unsqueeze(2) - token_traj[:, None, :, :, :],
        dim=-1,
    ).mean(-1)
    target_token_index = distances.argmin(-1)
    target_mask = one_hot(
        target_token_index,
        num_classes=token_traj.shape[1],
    ).bool()
    one_hot_target = target_mask.to(target.dtype)

    if label_smoothing == 0.0 or token_traj.shape[1] == 1:
        return one_hot_target

    work_distances = distances.float()
    neighbor_weight = (work_distances + 1.0e-4).pow(-2)
    neighbor_weight = neighbor_weight.masked_fill(target_mask, 0.0)
    neighbor_probability = neighbor_weight / neighbor_weight.sum(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(neighbor_weight.dtype).tiny)

    probability = (neighbor_probability * label_smoothing).to(target.dtype)
    probability = probability.masked_fill(target_mask, 1.0 - label_smoothing)
    return probability
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
python -m unittest tests.test_spatial_aware_loss -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit the target helper**

```bash
git add tests/test_spatial_aware_loss.py src/smart/metrics/utils.py
git commit -m "feat: add raw-GT spatial token targets"
```

---

### Task 2: Loss selection and experiment configuration

**Files:**
- Modify: `tests/test_spatial_aware_loss.py`
- Modify: `src/smart/metrics/cross_entropy.py:21-106`
- Modify: `configs/model/smart.yaml:94-99`
- Modify: `configs/experiment/pre_bc.yaml:7-18`
- Modify: `configs/experiment/clsft.yaml:22-27`

**Interfaces:**
- Consumes:
  - `spatial_aware_smoothing: bool` in `model.model_config.training_loss`
  - The helper from Task 1
- Produces:
  - Legacy one-hot plus uniform smoothing when disabled
  - Raw-GT spatial targets plus zero built-in smoothing when enabled
  - Explicit configuration inheritance for pre-BC and CLSFT

- [ ] **Step 1: Add failing integration and configuration tests**

Append these imports to `tests/test_spatial_aware_loss.py`:

```python
from pathlib import Path
from unittest.mock import patch

import yaml

from src.smart.metrics.cross_entropy import CrossEntropy
```

Append these test classes before the `if __name__ == "__main__":` block:

```python
class CrossEntropySelectionTest(unittest.TestCase):
    @staticmethod
    def _metric(spatial_aware_smoothing):
        metric = CrossEntropy(
            use_gt_raw=True,
            gt_thresh_scale_length=-1.0,
            label_smoothing=0.1,
            rollout_as_gt=False,
            spatial_aware_smoothing=spatial_aware_smoothing,
        )
        metric.eval()
        return metric

    @staticmethod
    def _update(metric):
        n_agent, n_step, n_token = 1, 16, 3
        positions = torch.zeros(n_agent, 18, 2)
        headings = torch.zeros(n_agent, 18)
        valid_18 = torch.ones(n_agent, 18, dtype=torch.bool)
        metric.update(
            next_token_logits=torch.zeros(n_agent, n_step, n_token),
            next_token_valid=torch.ones(n_agent, n_step, dtype=torch.bool),
            pred_pos=positions,
            pred_head=headings,
            pred_valid=valid_18,
            gt_pos_raw=positions,
            gt_head_raw=headings,
            gt_valid_raw=valid_18,
            gt_pos=positions,
            gt_head=headings,
            gt_valid=valid_18,
            token_agent_shape=torch.tensor([[2.0, 4.0]]),
            token_traj=SpatialAwareTargetTest._contours([0.0, 1.0, 4.0]),
        )

    def test_spatial_path_disables_builtin_uniform_smoothing(self):
        spatial_target = torch.tensor(
            [[[0.9, 0.08, 0.02]] * 16],
            dtype=torch.float32,
        )
        with (
            patch(
                "src.smart.metrics.cross_entropy."
                "get_prob_targets_spatial_aware_smoothing",
                return_value=spatial_target,
            ) as spatial_helper,
            patch(
                "src.smart.metrics.cross_entropy.cross_entropy",
                return_value=torch.ones(1, 16),
            ) as cross_entropy_mock,
        ):
            self._update(self._metric(spatial_aware_smoothing=True))

        spatial_helper.assert_called_once()
        self.assertEqual(
            cross_entropy_mock.call_args.kwargs["label_smoothing"],
            0.0,
        )

    def test_legacy_path_retains_builtin_uniform_smoothing(self):
        one_hot_target = torch.tensor(
            [[[1.0, 0.0, 0.0]] * 16],
            dtype=torch.float32,
        )
        with (
            patch(
                "src.smart.metrics.cross_entropy.get_prob_targets",
                return_value=one_hot_target,
            ) as legacy_helper,
            patch(
                "src.smart.metrics.cross_entropy.cross_entropy",
                return_value=torch.ones(1, 16),
            ) as cross_entropy_mock,
        ):
            self._update(self._metric(spatial_aware_smoothing=False))

        legacy_helper.assert_called_once()
        self.assertEqual(
            cross_entropy_mock.call_args.kwargs["label_smoothing"],
            0.1,
        )


class SpatialAwareConfigTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    @classmethod
    def _load(cls, relative_path):
        with (cls.ROOT / relative_path).open(encoding="utf-8") as file:
            return yaml.safe_load(file)

    def test_pre_bc_enables_spatial_smoothing(self):
        config = self._load("configs/experiment/pre_bc.yaml")
        self.assertTrue(
            config["model"]["model_config"]["training_loss"][
                "spatial_aware_smoothing"
            ]
        )

    def test_history_dynamics_inherits_pre_bc(self):
        config = self._load(
            "configs/experiment/pre_bc_history_dynamics.yaml"
        )
        self.assertIn("pre_bc", config["defaults"])

    def test_base_and_clsft_keep_spatial_smoothing_disabled(self):
        base = self._load("configs/model/smart.yaml")
        clsft = self._load("configs/experiment/clsft.yaml")
        self.assertFalse(
            base["model_config"]["training_loss"][
                "spatial_aware_smoothing"
            ]
        )
        self.assertFalse(
            clsft["model"]["model_config"]["training_loss"][
                "spatial_aware_smoothing"
            ]
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m unittest tests.test_spatial_aware_loss -v
```

Expected failures:

- `CrossEntropy` rejects the new `spatial_aware_smoothing` argument.
- Configuration dictionaries do not contain the new key.

- [ ] **Step 3: Integrate the spatial helper into `CrossEntropy`**

Change the import in `src/smart/metrics/cross_entropy.py` to:

```python
from .utils import (
    get_euclidean_targets,
    get_prob_targets,
    get_prob_targets_spatial_aware_smoothing,
)
```

Extend `CrossEntropy.__init__`:

```python
def __init__(
    self,
    use_gt_raw: bool,
    gt_thresh_scale_length: float,
    label_smoothing: float,
    rollout_as_gt: bool,
    spatial_aware_smoothing: bool = False,
) -> None:
    super().__init__()
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(
            "label_smoothing must be in [0, 1), "
            f"got {label_smoothing}"
        )
    self.use_gt_raw = use_gt_raw
    self.gt_thresh_scale_length = gt_thresh_scale_length
    self.label_smoothing = label_smoothing
    self.rollout_as_gt = rollout_as_gt
    self.spatial_aware_smoothing = spatial_aware_smoothing
```

Replace the current `prob_target` construction and cross-entropy call with:

```python
if self.spatial_aware_smoothing:
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

loss = cross_entropy(
    next_token_logits.transpose(1, 2),
    prob_target.transpose(1, 2),
    reduction="none",
    label_smoothing=builtin_label_smoothing,
)
```

Keep all surrounding target construction and masks unchanged.

- [ ] **Step 4: Configure pre-BC and legacy paths**

Add to `configs/model/smart.yaml` under `training_loss`:

```yaml
    spatial_aware_smoothing: false
```

Add to `configs/experiment/pre_bc.yaml` under
`model.model_config`:

```yaml
    training_loss:
      spatial_aware_smoothing: true
```

Add to `configs/experiment/clsft.yaml` under its existing
`training_loss` mapping:

```yaml
      spatial_aware_smoothing: false
```

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_spatial_aware_loss -v
```

Expected: 10 tests pass.

- [ ] **Step 6: Commit loss integration and configuration**

```bash
git add \
  tests/test_spatial_aware_loss.py \
  src/smart/metrics/cross_entropy.py \
  configs/model/smart.yaml \
  configs/experiment/pre_bc.yaml \
  configs/experiment/clsft.yaml
git commit -m "feat: use raw-GT spatial smoothing in pre-BC"
```

---

### Task 3: Regression and configuration verification

**Files:**
- Verify only; no production files added

**Interfaces:**
- Consumes: all implementation and configuration from Tasks 1–2
- Produces: evidence that the loss is backward compatible and the repository remains healthy

- [ ] **Step 1: Run the complete unit-test suite**

Run in the CatK environment:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected: all existing tests plus the 10 new tests pass.

- [ ] **Step 2: Resolve the pre-BC configurations with Hydra**

Run:

```bash
python -m src.run --cfg job --resolve \
  experiment=pre_bc \
  task_name=spatial_config_check \
  | sed -n '/training_loss:/,/finetune:/p'

python -m src.run --cfg job --resolve \
  experiment=pre_bc_history_dynamics \
  task_name=spatial_history_config_check \
  | sed -n '/training_loss:/,/finetune:/p'

python -m src.run --cfg job --resolve \
  experiment=clsft \
  ckpt_path=/tmp/config-check.ckpt \
  task_name=clsft_config_check \
  | sed -n '/training_loss:/,/finetune:/p'
```

Expected:

- `pre_bc`: `label_smoothing: 0.1`,
  `spatial_aware_smoothing: true`.
- `pre_bc_history_dynamics`: the same loss settings and
  `history_dynamics.is_active: true`.
- `clsft`: `label_smoothing: 0.0`,
  `spatial_aware_smoothing: false`.

- [ ] **Step 3: Check formatting and repository scope**

Run:

```bash
git diff --check
git status --short
git diff --stat HEAD~2..HEAD
```

Expected:

- No whitespace errors.
- User-owned changes remain present but uncommitted and untouched.
- Implementation commits contain only the new test, loss helper, loss
  integration, and configuration files.

- [ ] **Step 4: Record final verification without another code commit**

Report the exact test count, resolved configuration values, and any
environment-limited checks. Do not claim a check passed unless its command
completed successfully.
