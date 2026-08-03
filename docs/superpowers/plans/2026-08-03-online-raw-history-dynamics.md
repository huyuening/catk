# Online Raw History Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-BC experiment that retains CatK's three history-dynamics inputs but calculates them online from ordinary cached history with direct backward finite differences and no additional trajectory reconstruction.

**Architecture:** A tensor-only estimator in `history_dynamics.py` calculates two causal body-frame dynamics vectors from the unmodified batch tensors at history endpoints 5 and 10. `TokenProcessor` selects either the existing cached reconstructed fields or the new online estimator through an explicit backward-compatible mode. A dedicated Hydra experiment enables online mode while preserving all existing defaults and checkpoint parameter shapes.

**Tech Stack:** Python 3.11 on the deployment machine, PyTorch, Hydra/OmegaConf, PyTorch Geometric, `unittest`, YAML.

## Global Constraints

- Use ordinary CatK cache `position`, `heading`, and `valid_mask`; do not read WOMD TFRecords in the training loop.
- Perform no additional gap filling, smoothing, outlier filtering, polynomial fitting, heading cleaning, or trajectory reconstruction in `online_raw` mode.
- Calculate only the two observed history-token vectors at frames 5 and 10 with `dt = 0.1` seconds.
- Return values in `[a_longitudinal, angular_speed, a_lateral]` order and clip their absolute values to `[15, 3, 15]`.
- Mark a token invalid unless positions at `t-2`, `t-1`, and `t` and headings at `t-1` and `t` are valid and finite; invalid values are zero.
- Keep `cached_reconstructed` as the default and preserve strict missing-cache-field errors in that mode.
- Ignore cached `history_dynamics` fields in `online_raw` mode.
- Preserve original CatK, reconstructed-history, loss, vocabulary, rollout, and WOSAC behavior.
- Do not stage or modify unrelated user-owned working-tree changes.

---

### Task 1: Tensor-only raw history dynamics estimator

**Files:**
- Modify: `tests/test_history_dynamics.py`
- Modify: `src/smart/tokens/history_dynamics.py`

**Interfaces:**
- Consumes: `position: Tensor[n_agent, n_step, >=2]`, `heading: Tensor[n_agent, n_step]`, and `valid_mask: Tensor[n_agent, n_step]` from an ordinary CatK batch.
- Produces: `estimate_raw_history_dynamics(...) -> tuple[Tensor, Tensor]`, containing `values: Tensor[n_agent, 2, 3]` and `valid: BoolTensor[n_agent, 2]`.

- [ ] **Step 1: Write the failing tests for direct finite differences**

Extend the imports and add these tests to `tests/test_history_dynamics.py`:

```python
import torch

from src.smart.tokens.history_dynamics import (
    estimate_raw_history_dynamics,
    extract_history_dynamics,
)


class RawHistoryDynamicsTest(unittest.TestCase):
    @staticmethod
    def _history(x, y=None, heading=None):
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.zeros_like(x) if y is None else torch.as_tensor(y)
        position = torch.zeros(1, len(x), 3, dtype=torch.float32)
        position[0, :, :2] = torch.stack((x, y), dim=-1)
        if heading is None:
            heading = torch.zeros(1, len(x), dtype=torch.float32)
        else:
            heading = torch.as_tensor(heading, dtype=torch.float32).view(1, -1)
        valid = torch.ones(1, len(x), dtype=torch.bool)
        return position, heading, valid

    def test_quadratic_motion_returns_endpoint_body_acceleration(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        acceleration = 2.0
        x = 3.0 * time + 0.5 * acceleration * time.square()
        position, heading, valid = self._history(x)

        values, feature_valid = estimate_raw_history_dynamics(
            position, heading, valid
        )

        self.assertEqual(tuple(values.shape), (1, 2, 3))
        torch.testing.assert_close(feature_valid, torch.ones(1, 2, dtype=torch.bool))
        torch.testing.assert_close(
            values[0, :, 0], torch.full((2,), acceleration), atol=2e-4, rtol=0
        )
        torch.testing.assert_close(values[0, :, 1:], torch.zeros(2, 2))

    def test_constant_velocity_has_zero_dynamics(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        position, heading, valid = self._history(4.0 * time)

        values, feature_valid = estimate_raw_history_dynamics(
            position, heading, valid
        )

        torch.testing.assert_close(values, torch.zeros(1, 2, 3), atol=2e-4, rtol=0)
        self.assertTrue(feature_valid.all())
```

- [ ] **Step 2: Run the core estimator tests and verify RED**

Run:

```bash
python -m unittest \
  tests.test_history_dynamics.RawHistoryDynamicsTest.test_quadratic_motion_returns_endpoint_body_acceleration \
  tests.test_history_dynamics.RawHistoryDynamicsTest.test_constant_velocity_has_zero_dynamics -v
```

Expected: import failure because `estimate_raw_history_dynamics` does not exist.

- [ ] **Step 3: Add failing tests for wraparound, validity, and clipping**

Add to `RawHistoryDynamicsTest`:

```python
    def test_heading_difference_wraps_across_pi(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        heading = torch.zeros(11)
        heading[4], heading[5] = torch.pi - 0.01, -torch.pi + 0.01
        heading[9], heading[10] = torch.pi - 0.01, -torch.pi + 0.01
        position, heading, valid = self._history(time, heading=heading)

        values, feature_valid = estimate_raw_history_dynamics(
            position, heading, valid
        )

        torch.testing.assert_close(
            values[0, :, 1], torch.full((2,), 0.2), atol=2e-4, rtol=0
        )
        self.assertTrue(feature_valid.all())

    def test_invalid_or_nonfinite_support_is_zero_and_masked(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        position, heading, valid = self._history(time.square())
        valid[0, 3] = False
        position[0, 9, 0] = float("nan")

        values, feature_valid = estimate_raw_history_dynamics(
            position, heading, valid
        )

        torch.testing.assert_close(feature_valid, torch.zeros(1, 2, dtype=torch.bool))
        torch.testing.assert_close(values, torch.zeros(1, 2, 3))

    def test_feature_ranges_are_clipped(self):
        time = torch.arange(11, dtype=torch.float32) * 0.1
        x = 50.0 * time.square()
        heading = torch.arange(11, dtype=torch.float32) * 1.0
        position, heading, valid = self._history(x, heading=heading)

        values, feature_valid = estimate_raw_history_dynamics(
            position, heading, valid
        )

        self.assertTrue(feature_valid.all())
        self.assertTrue(torch.all(values.abs() <= torch.tensor([15.0, 3.0, 15.0])))
        torch.testing.assert_close(values[0, :, 1], torch.full((2,), 3.0))

    def test_malformed_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "position"):
            estimate_raw_history_dynamics(
                torch.zeros(11),
                torch.zeros(1, 11),
                torch.ones(1, 11, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "dt"):
            estimate_raw_history_dynamics(
                torch.zeros(1, 11, 3),
                torch.zeros(1, 11),
                torch.ones(1, 11, dtype=torch.bool),
                dt=0.0,
            )
```

- [ ] **Step 4: Implement the minimal vectorized estimator**

Add `torch` imports and the following public function to
`src/smart/tokens/history_dynamics.py`. Keep `extract_history_dynamics`
unchanged.

```python
import torch
from torch import Tensor


@torch.no_grad()
def estimate_raw_history_dynamics(
    position: Tensor,
    heading: Tensor,
    valid_mask: Tensor,
    *,
    num_historical_steps: int = 11,
    token_shift_steps: int = 5,
    dt: float = 0.1,
    max_abs_longitudinal_accel_mps2: float = 15.0,
    max_abs_angular_speed_radps: float = 3.0,
    max_abs_lateral_accel_mps2: float = 15.0,
) -> tuple[Tensor, Tensor]:
    if position.ndim != 3 or position.size(-1) < 2:
        raise ValueError("position must have shape [n_agent, n_step, >=2]")
    if heading.shape != position.shape[:2] or valid_mask.shape != position.shape[:2]:
        raise ValueError("heading and valid_mask must match position[:2]")
    if num_historical_steps > position.size(1) or num_historical_steps < 3:
        raise ValueError("num_historical_steps must fit the available trajectory")
    if token_shift_steps < 2 or (num_historical_steps - 1) % token_shift_steps:
        raise ValueError("token_shift_steps must produce endpoints with 3-frame support")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    limits_tuple = (
        max_abs_longitudinal_accel_mps2,
        max_abs_angular_speed_radps,
        max_abs_lateral_accel_mps2,
    )
    if any(not np.isfinite(value) or value <= 0 for value in limits_tuple):
        raise ValueError("dynamics clipping limits must be finite and positive")

    output_dtype = position.dtype
    compute_dtype = (
        torch.float32
        if output_dtype in (torch.float16, torch.bfloat16)
        else output_dtype
    )
    endpoints = torch.arange(
        token_shift_steps,
        num_historical_steps,
        token_shift_steps,
        device=position.device,
    )
    xy = position[..., :2].to(compute_dtype)
    theta = heading.to(device=position.device, dtype=compute_dtype)
    valid = valid_mask.to(device=position.device, dtype=torch.bool)

    p0, p1, p2 = xy[:, endpoints - 2], xy[:, endpoints - 1], xy[:, endpoints]
    theta_previous, theta_current = theta[:, endpoints - 1], theta[:, endpoints]
    feature_valid = (
        valid[:, endpoints - 2]
        & valid[:, endpoints - 1]
        & valid[:, endpoints]
        & torch.isfinite(p0).all(dim=-1)
        & torch.isfinite(p1).all(dim=-1)
        & torch.isfinite(p2).all(dim=-1)
        & torch.isfinite(theta_previous)
        & torch.isfinite(theta_current)
    )

    velocity_previous = (p1 - p0) / dt
    velocity_current = (p2 - p1) / dt
    acceleration = (velocity_current - velocity_previous) / dt
    delta_heading = torch.atan2(
        torch.sin(theta_current - theta_previous),
        torch.cos(theta_current - theta_previous),
    )
    angular_speed = delta_heading / dt
    cosine, sine = torch.cos(theta_current), torch.sin(theta_current)
    longitudinal = acceleration[..., 0] * cosine + acceleration[..., 1] * sine
    lateral = -acceleration[..., 0] * sine + acceleration[..., 1] * cosine
    values = torch.stack((longitudinal, angular_speed, lateral), dim=-1)
    limits = values.new_tensor(limits_tuple)
    values = torch.maximum(torch.minimum(values, limits), -limits)
    values = torch.where(feature_valid.unsqueeze(-1), values, torch.zeros_like(values))
    return values.to(output_dtype), feature_valid
```

- [ ] **Step 5: Run estimator tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_history_dynamics -v
```

Expected: all existing reconstructed-history tests and new raw estimator tests pass.

- [ ] **Step 6: Commit the estimator**

```bash
git add src/smart/tokens/history_dynamics.py tests/test_history_dynamics.py
git commit -m "feat: calculate raw history dynamics online"
```

---

### Task 2: Backward-compatible TokenProcessor mode routing

**Files:**
- Modify: `tests/test_future_token_dynamics.py`
- Modify: `src/smart/tokens/token_processor.py`

**Interfaces:**
- Consumes: `estimate_raw_history_dynamics(...)` from Task 1 and the existing `history_dynamics` DictConfig.
- Produces: `TokenProcessor.history_dynamics_mode: str` and `_prepare_history_dynamics(...) -> tuple[Tensor, Tensor]` used before heading cleaning or token-boundary extrapolation.

- [ ] **Step 1: Write failing mode-routing tests**

Add this class after `FutureTokenDynamicsTokenProcessorTest` in
`tests/test_future_token_dynamics.py`:

```python
class OnlineRawHistoryDynamicsTokenProcessorTest(unittest.TestCase):
    @staticmethod
    def _processor(mode):
        processor = TokenProcessor.__new__(TokenProcessor)
        torch.nn.Module.__init__(processor)
        processor.history_dynamics_active = True
        processor.history_dynamics_mode = mode
        processor.shift = 5
        return processor

    @staticmethod
    def _inputs():
        time = torch.arange(11, dtype=torch.float32) * 0.1
        position = torch.zeros(1, 11, 3)
        position[0, :, 0] = time.square()
        heading = torch.zeros(1, 11)
        valid = torch.ones(1, 11, dtype=torch.bool)
        return position, heading, valid

    def test_online_mode_works_without_cached_fields(self):
        processor = self._processor("online_raw")
        position, heading, valid = self._inputs()

        values, feature_valid = processor._prepare_history_dynamics(
            {}, position=position, heading=heading, valid=valid
        )

        torch.testing.assert_close(values[0, :, 0], torch.full((2,), 2.0), atol=2e-4, rtol=0)
        self.assertTrue(feature_valid.all())

    def test_online_mode_ignores_conflicting_cached_fields(self):
        processor = self._processor("online_raw")
        position, heading, valid = self._inputs()
        cached = {
            "history_dynamics": torch.full((1, 2, 3), 999.0),
            "history_dynamics_valid": torch.zeros(1, 2, dtype=torch.bool),
        }

        values, feature_valid = processor._prepare_history_dynamics(
            cached, position=position, heading=heading, valid=valid
        )

        self.assertFalse(torch.eq(values, 999.0).any())
        self.assertTrue(feature_valid.all())

    def test_cached_mode_keeps_strict_missing_field_error(self):
        processor = self._processor("cached_reconstructed")
        position, heading, valid = self._inputs()

        with self.assertRaisesRegex(KeyError, "history_dynamics.*missing"):
            processor._prepare_history_dynamics(
                {}, position=position, heading=heading, valid=valid
            )

    def test_cached_mode_returns_existing_fields(self):
        processor = self._processor("cached_reconstructed")
        position, heading, valid = self._inputs()
        expected_values = torch.ones(1, 2, 3)
        expected_valid = torch.tensor([[True, False]])

        values, feature_valid = processor._prepare_history_dynamics(
            {
                "history_dynamics": expected_values,
                "history_dynamics_valid": expected_valid,
            },
            position=position,
            heading=heading,
            valid=valid,
        )

        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(feature_valid, expected_valid)
```

- [ ] **Step 2: Run routing tests and verify RED**

Run:

```bash
python -m unittest tests.test_future_token_dynamics.OnlineRawHistoryDynamicsTokenProcessorTest -v
```

Expected: failure because `_prepare_history_dynamics` does not exist.

- [ ] **Step 3: Implement mode parsing and routing**

In `src/smart/tokens/token_processor.py`:

1. Import `estimate_raw_history_dynamics`.
2. Add `HISTORY_DYNAMICS_MODES = ("cached_reconstructed", "online_raw")`.
3. Parse `history_dynamics.mode` before vocabulary initialization and raise a
   `ValueError` listing both supported modes for any unknown value.
4. Add the following method:

```python
    def _prepare_history_dynamics(
        self,
        agent_data,
        *,
        position: Tensor,
        heading: Tensor,
        valid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.history_dynamics_mode == "online_raw":
            return estimate_raw_history_dynamics(
                position=position,
                heading=heading,
                valid_mask=valid,
                num_historical_steps=11,
                token_shift_steps=self.shift,
                dt=0.1,
            )

        required = ("history_dynamics", "history_dynamics_valid")
        missing = [key for key in required if key not in agent_data]
        if missing:
            raise KeyError(
                "history_dynamics mode 'cached_reconstructed' is active but "
                f"the CatK cache is missing {missing}. Regenerate every split "
                "with src.data_preprocess --history_dynamics_filter_strength strong."
            )
        return (
            agent_data["history_dynamics"].to(dtype=position.dtype),
            agent_data["history_dynamics_valid"].bool(),
        )
```

5. In `tokenize_agent`, call `_prepare_history_dynamics` immediately after
   reading `valid`, `heading`, and `pos`, before `_clean_heading` and
   `_extrapolate_agent_to_prev_token_step`. Store the returned tensors in local
   variables and attach them to `tokenized_agent` after the dictionary is
   constructed.
6. Remove the old inline cached-field-only branch.

- [ ] **Step 4: Add and verify invalid-mode behavior**

Add to `OnlineRawHistoryDynamicsTokenProcessorTest`:

```python
    def test_unknown_mode_fails_before_vocabulary_loading(self):
        sampling = SimpleNamespace(num_k=1, temp=1.0)
        with self.assertRaisesRegex(ValueError, "cached_reconstructed.*online_raw"):
            TokenProcessor(
                map_token_file="does-not-exist.pkl",
                agent_token_file="does-not-exist.pkl",
                map_token_sampling=sampling,
                agent_token_sampling=sampling,
                history_dynamics={"is_active": True, "mode": "unknown"},
                future_token_dynamics={"is_active": False},
            )
```

Run:

```bash
python -m unittest tests.test_future_token_dynamics.OnlineRawHistoryDynamicsTokenProcessorTest -v
```

Expected: all mode-routing tests pass.

- [ ] **Step 5: Run TokenProcessor regression tests**

Run:

```bash
python -m unittest tests.test_future_token_dynamics -v
```

Expected: all future-token and history-mode tests pass.

- [ ] **Step 6: Commit routing behavior**

```bash
git add src/smart/tokens/token_processor.py tests/test_future_token_dynamics.py
git commit -m "feat: route online raw history dynamics"
```

---

### Task 3: Hydra experiment and user-facing command

**Files:**
- Create: `configs/experiment/pre_bc_history_dynamics_online_raw.yaml`
- Create: `tests/test_online_history_dynamics_config.py`
- Modify: `configs/model/smart.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `history_dynamics.mode` implemented in Task 2.
- Produces: `experiment=pre_bc_history_dynamics_online_raw`, with active online history dynamics and unchanged pre-BC protocol.

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_online_history_dynamics_config.py`:

```python
import unittest
from pathlib import Path

import yaml


class OnlineHistoryDynamicsConfigTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    @classmethod
    def _load(cls, relative_path):
        return yaml.safe_load((cls.ROOT / relative_path).read_text(encoding="utf-8"))

    def test_base_configuration_defaults_to_cached_reconstruction(self):
        config = self._load("configs/model/smart.yaml")
        history = config["model_config"]["history_dynamics"]
        self.assertFalse(history["is_active"])
        self.assertEqual(history["mode"], "cached_reconstructed")

    def test_online_experiment_enables_only_online_history_mode(self):
        config = self._load(
            "configs/experiment/pre_bc_history_dynamics_online_raw.yaml"
        )
        self.assertEqual(config["defaults"], ["pre_bc", "_self_"])
        self.assertEqual(
            config["model"]["model_config"]["history_dynamics"],
            {"is_active": True, "mode": "online_raw"},
        )

    def test_existing_reconstructed_experiment_keeps_cached_mode(self):
        reconstructed = self._load(
            "configs/experiment/pre_bc_history_dynamics.yaml"
        )
        self.assertNotIn(
            "mode",
            reconstructed["model"]["model_config"]["history_dynamics"],
        )

    def test_online_experiment_composes_with_pre_bc_protocol(self):
        try:
            import hydra
        except ModuleNotFoundError:
            self.skipTest("Hydra is not installed in this test environment")
        with hydra.initialize_config_dir(
            config_dir=str(self.ROOT / "configs"), version_base=None
        ):
            config = hydra.compose(
                config_name="run.yaml",
                overrides=["experiment=pre_bc_history_dynamics_online_raw"],
            )
        self.assertTrue(config.model.model_config.history_dynamics.is_active)
        self.assertEqual(
            config.model.model_config.history_dynamics.mode, "online_raw"
        )
        self.assertEqual(config.trainer.max_epochs, 32)
        self.assertEqual(config.trainer.limit_val_batches, 0.1)
        self.assertEqual(config.model.model_config.wosac_backend, "fast")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run configuration tests and verify RED**

Run:

```bash
python -m unittest tests.test_online_history_dynamics_config -v
```

Expected: failures because the base `mode` and new experiment file do not exist.

- [ ] **Step 3: Add the default mode and dedicated experiment**

Add this key to `configs/model/smart.yaml` under `history_dynamics`:

```yaml
    mode: cached_reconstructed
```

Create `configs/experiment/pre_bc_history_dynamics_online_raw.yaml`:

```yaml
# @package _global_

# Pre-BC with three history dynamics calculated online from ordinary CatK tensors.
defaults:
  - pre_bc
  - _self_

model:
  model_config:
    history_dynamics:
      is_active: true
      mode: online_raw
```

- [ ] **Step 4: Document semantics and the training command**

Update README's “Optional causal history dynamics” section to distinguish:

```text
cached_reconstructed reconstructs raw frames 0--10 during preprocessing and
requires history_dynamics fields in every cache. online_raw instead calculates
the same three input channels at frames 5 and 10 inside TokenProcessor from
ordinary CatK position/heading/valid tensors. It adds no smoothing, fitting,
gap filling, or trajectory reconstruction and ignores cached history-dynamics
fields. Ordinary CatK preprocessing, including its legacy interpolation,
remains unchanged.
```

Add the exact hard-label comparison command:

```bash
MY_EXPERIMENT=pre_bc_history_dynamics_online_raw \
MY_TASK_NAME=pre_bc_history_dynamics_online_raw_hard_ce_b200 \
CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact \
WANDB_OFFLINE=false \
WANDB_ENTITY=huyuening911-beijing-jiaotong-university \
bash scripts/train.sh \
  ckpt_path=null \
  model.model_config.training_loss.spatial_aware_smoothing=false \
  model.model_config.training_loss.label_smoothing=0.0 \
  model.model_config.val_closed_loop=false
```

State explicitly that the exact cache is reused only to hold all unrelated
inputs constant; `online_raw` ignores its reconstructed history-dynamics fields.

- [ ] **Step 5: Run configuration and protocol tests**

Run:

```bash
python -m unittest \
  tests.test_online_history_dynamics_config \
  tests.test_training_fast_wosac_config \
  tests.test_spatial_aware_loss -v
```

Expected: all tests pass; the Hydra-specific test may skip only when Hydra is
not installed in the local test environment.

- [ ] **Step 6: Commit experiment configuration and documentation**

```bash
git add \
  configs/model/smart.yaml \
  configs/experiment/pre_bc_history_dynamics_online_raw.yaml \
  tests/test_online_history_dynamics_config.py \
  README.md
git commit -m "feat: configure online raw history dynamics pretraining"
```

---

### Task 4: Integrated verification

**Files:**
- Verify only; no new production files.

**Interfaces:**
- Consumes: all deliverables from Tasks 1–3.
- Produces: evidence that the new mode works and existing modes remain intact.

- [ ] **Step 1: Run all targeted tests together**

```bash
python -m unittest \
  tests.test_history_dynamics \
  tests.test_future_token_dynamics \
  tests.test_online_history_dynamics_config \
  tests.test_training_fast_wosac_config \
  tests.test_spatial_aware_loss -v
```

Expected: PASS, except explicitly reported dependency-based skips.

- [ ] **Step 2: Compile changed Python modules**

```bash
python -m compileall -q \
  src/smart/tokens/history_dynamics.py \
  src/smart/tokens/token_processor.py \
  tests/test_history_dynamics.py \
  tests/test_future_token_dynamics.py \
  tests/test_online_history_dynamics_config.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Validate configuration on the deployment environment when dependencies are available**

```bash
python -m src.run \
  --cfg job \
  --resolve \
  experiment=pre_bc_history_dynamics_online_raw \
  model.model_config.training_loss.spatial_aware_smoothing=false \
  model.model_config.training_loss.label_smoothing=0.0 \
  model.model_config.val_closed_loop=false \
  | grep -E 'history_dynamics:|mode: online_raw|max_epochs: 32'
```

Expected: resolved configuration includes active history dynamics, `mode:
online_raw`, and 32 pre-BC epochs.

- [ ] **Step 4: Check scope and whitespace**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only task-owned files are changed or committed,
while unrelated pre-existing user files remain untouched.
