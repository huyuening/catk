# CAT-K Unbiased Accelerated Safety Testing Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a non-RL, single-POV risk-tilted importance-sampling evaluator that accelerates ego collision and near-miss generation while estimating the original CAT-K closed-loop event probabilities with an exact, auditable \(p/q\) likelihood ratio.

**Architecture:** Keep CAT-K and its checkpoints unchanged, inject an optional `FixedISController` at the token-sampling boundary, and place proposal construction, event detection, calibration, audit output, and estimators in a new `src/smart/accelerated_testing` package. The normal inference path remains unchanged when the feature is disabled; formal accelerated evaluation uses a separate Hydra experiment, rank-local writers, a rank-zero merge/report stage, and a training-only criticality calibration artifact.

**Tech Stack:** Python 3.11, PyTorch 2.4.1, Lightning 2.4.0, Hydra/OmegaConf 1.3.2, NumPy, stdlib `dataclasses`/`hashlib`/`gzip`/`json`, pytest/unittest, existing embedded Fast WOSAC geometry.

## Global Constraints

- The formal target distribution is the actual CAT-K `topk_prob` behavior distribution after Top-K and temperature; defaults are `num_k=48` and `temp=1.0`.
- Formal accelerated evaluation must reject `topk_prob_sampled_with_dist` and `topk_dist_sampled_with_prob`.
- Tokens outside baseline support remain at \(p=q=0\). Full-vocabulary testing requires both baseline and proposal to use `num_k=2048`.
- Phase 1 controls at most one non-ego vehicle within 60 m per scenario-rollout and never replaces it after locking.
- Proposal defaults are fixed at `epsilon=0.05`, `beta=1.0`, and `z_clip=5.0`.
- The proposal is \(r\propto p\exp(\beta z(R))\) and \(q=(1-\epsilon)p+\epsilon r\); final importance weights are never clipped or self-normalized.
- Probability construction and accumulated log weights use float64 and log-domain operations.
- Risk scoring, proposal construction, online events, and final safety estimates must not consume validation/test future GT.
- The risk function is collision-type-neutral and uses ego Top-8 token expectation, a 1.0 s horizon, and the four equal-weight components from the approved spec.
- Collision means Fast WOSAC rounded-box signed distance `< 0` with `CORNER_ROUNDING_FACTOR=0.7`, checked at 10 Hz.
- Near-miss defaults are gap `<1.0 m`, generic TTO `<1.5 s`, applicable PET `<1.5 s`, or required deceleration `>3.0 m/s²`, and are mutually exclusive with an ego collision.
- Videos may stop at first ego collision, but complete 8 s trajectories remain saved.
- A formal proposal is immutable during evaluation. Calibration uses training data only and writes a hashed artifact.
- Fast WOSAC embedded source files remain unchanged.
- `accelerated_testing.enabled=false` must preserve the current inference and WOSAC path.
- Formal Phase 1 config keeps endpoint interpolation disabled so online event geometry and saved evaluation trajectories use the same 10 Hz token reconstruction.
- DDP ranks write separate temporary shards; only rank zero publishes merged artifacts and `report.json`.

---

## File Structure

### New production files

- `src/smart/accelerated_testing/__init__.py` — stable public exports.
- `src/smart/accelerated_testing/config.py` — immutable configuration dataclasses and formal-run validation.
- `src/smart/accelerated_testing/token_distribution.py` — exact Top-K baseline distribution, stable per-agent uniforms, and inverse-CDF sampling.
- `src/smart/accelerated_testing/proposal.py` — risk tilt, mixture proposal, and selected-token log ratio.
- `src/smart/accelerated_testing/risk.py` — token pose expansion, rounded-box pair distance, generic TTO, composite risk, and criticality.
- `src/smart/accelerated_testing/pov_selector.py` — deterministic candidate selection and one-time POV locking.
- `src/smart/accelerated_testing/events.py` — online ego collision tracking, collision type classification, and near-miss summaries.
- `src/smart/accelerated_testing/calibration.py` — training-only threshold calibration and artifact verification.
- `src/smart/accelerated_testing/ledger.py` — JSON-safe records, hashes, rank-local files, failure handling, and merged artifact index.
- `src/smart/accelerated_testing/estimators.py` — scenario-balanced ordinary IS, cluster bootstrap, ESS, and acceleration report.
- `src/smart/accelerated_testing/controller.py` — per-batch fixed-IS rollout state machine.
- `src/smart/accelerated_testing/runner.py` — Lightning-facing lifecycle, DDP merge, calibration/report publication, and logging values.
- `configs/experiment/accelerated_testing.yaml` — formal held-out fixed-IS evaluation.
- `configs/experiment/accelerated_testing_calibrate.yaml` — training-cache baseline calibration run.
- `scripts/accelerated_test.sh` — reproducible multi-GPU entry point for calibration or evaluation.

### Existing files to modify

- `src/smart/utils/rollout.py:105-185` — reuse exact Top-K distribution construction without changing legacy sampling semantics.
- `src/smart/utils/__init__.py:15-21` — keep existing exports; no accelerated-testing implementation is re-exported here.
- `src/smart/tokens/token_processor.py:240-254` — include stable agent IDs in `tokenized_agent`.
- `src/smart/modules/agent_decoder.py:565-902` — optional controller hook around token selection and 10 Hz observation.
- `src/smart/modules/smart_decoder.py:83-93` — forward the optional controller.
- `src/smart/model/smart.py:39-108,132-307` — configure runner, choose normal versus accelerated validation, and finalize reports.
- `src/run.py:38-105` — pass resolved run metadata and checkpoint path to the model before validation.
- `configs/model/smart.yaml:2-114` — add a disabled-by-default accelerated-testing section.
- `src/utils/vis_waymo.py:278-299` — optional per-rollout first-collision stop frame.
- `README.md:235-275` — document calibration, evaluation, artifacts, and interpretation.

### New tests

```text
tests/accelerated_testing/
├── __init__.py
├── test_token_distribution.py
├── test_config.py
├── test_proposal.py
├── test_risk.py
├── test_pov_selector.py
├── test_events.py
├── test_calibration.py
├── test_ledger.py
├── test_estimators.py
├── test_controller.py
├── test_model_integration.py
├── test_runner.py
├── test_shell_entrypoint.py
├── test_visualization.py
└── test_end_to_end.py
```

---

### Task 1: Exact baseline token distribution and deterministic sampling

**Files:**
- Create: `src/smart/accelerated_testing/__init__.py`
- Create: `src/smart/accelerated_testing/token_distribution.py`
- Create: `tests/accelerated_testing/__init__.py`
- Create: `tests/accelerated_testing/test_token_distribution.py`
- Modify: `src/smart/utils/rollout.py:105-185`

**Interfaces:**
- Produces: `TopKTokenDistribution(token_ids, logits, log_probs)`.
- Produces: `build_topk_token_distribution(logits, num_k, temperature)`.
- Produces: `stable_uniform(global_seed, scenario_id, rollout_id, token_step, agent_id)`.
- Produces: `sample_token_ids_from_uniform(distribution, uniforms)`.
- Preserves: `sample_next_token_traj(...) -> tuple[Tensor, Tensor]`.

- [ ] **Step 1: Write failing distribution and RNG tests**

```python
# tests/accelerated_testing/test_token_distribution.py
import torch
import pytest

from src.smart.accelerated_testing.token_distribution import (
    build_topk_token_distribution,
    sample_token_ids_from_uniform,
    stable_uniform,
)


def test_topk_distribution_matches_catk_definition():
    logits = torch.tensor([[0.0, 2.0, 1.0, -3.0]], dtype=torch.float64)
    result = build_topk_token_distribution(logits, num_k=3, temperature=2.0)
    expected_logits, expected_ids = torch.topk(logits, 3, dim=-1, sorted=False)
    assert torch.equal(result.token_ids, expected_ids)
    torch.testing.assert_close(result.logits, expected_logits / 2.0)
    torch.testing.assert_close(
        result.log_probs,
        torch.log_softmax(expected_logits / 2.0, dim=-1),
    )


def test_inverse_cdf_uses_supplied_uniform():
    dist = build_topk_token_distribution(
        torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float64),
        num_k=3,
        temperature=1.0,
    )
    first = sample_token_ids_from_uniform(dist, torch.tensor([0.0]))
    last = sample_token_ids_from_uniform(dist, torch.tensor([0.999999]))
    assert first.item() == dist.token_ids[0, 0].item()
    assert last.item() == dist.token_ids[0, -1].item()


def test_stable_uniform_is_keyed_and_batch_order_independent():
    key = dict(
        global_seed=817,
        scenario_id="scenario-a",
        rollout_id=7,
        token_step=3,
        agent_id=41,
    )
    assert stable_uniform(**key) == stable_uniform(**key)
    assert stable_uniform(**key) != stable_uniform(**{**key, "agent_id": 42})


@pytest.mark.parametrize("num_k", [0, 5])
def test_invalid_topk_fails(num_k):
    with pytest.raises(ValueError, match="num_k"):
        build_topk_token_distribution(
            torch.zeros(2, 4),
            num_k=num_k,
            temperature=1.0,
        )
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `pytest -q tests/accelerated_testing/test_token_distribution.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.smart.accelerated_testing'`.

- [ ] **Step 3: Implement the exact distribution and stable uniform**

```python
# src/smart/accelerated_testing/token_distribution.py
from dataclasses import dataclass
import hashlib

import torch
from torch import Tensor


@dataclass(frozen=True)
class TopKTokenDistribution:
    token_ids: Tensor
    logits: Tensor
    log_probs: Tensor


def build_topk_token_distribution(
    logits: Tensor,
    *,
    num_k: int,
    temperature: float,
) -> TopKTokenDistribution:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [n_agent, n_token]")
    if num_k <= 0 or num_k > logits.shape[-1]:
        raise ValueError("num_k must be in [1, n_token]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    topk_logits, token_ids = torch.topk(
        logits.detach(), num_k, dim=-1, sorted=False
    )
    scaled = topk_logits / float(temperature)
    return TopKTokenDistribution(
        token_ids=token_ids,
        logits=scaled,
        log_probs=torch.log_softmax(scaled.to(torch.float64), dim=-1),
    )


def stable_uniform(
    *,
    global_seed: int,
    scenario_id: str,
    rollout_id: int,
    token_step: int,
    agent_id: int,
) -> float:
    payload = (
        f"{global_seed}|{scenario_id}|{rollout_id}|"
        f"{token_step}|{agent_id}"
    ).encode("utf-8")
    value = int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "big"
    )
    # Convert the stable 64-bit hash to a binary64 value in [0, 1).
    return (value >> 11) * (2.0 ** -53)


def sample_token_ids_from_uniform(
    distribution: TopKTokenDistribution,
    uniforms: Tensor,
) -> Tensor:
    if uniforms.shape != distribution.log_probs.shape[:-1]:
        raise ValueError("uniforms must have shape [n_agent]")
    cdf = distribution.log_probs.exp().cumsum(dim=-1)
    support_index = torch.searchsorted(
        cdf.contiguous(),
        uniforms.to(device=cdf.device, dtype=cdf.dtype).unsqueeze(-1),
    ).squeeze(-1)
    support_index = support_index.clamp_max(cdf.shape[-1] - 1)
    return distribution.token_ids.gather(1, support_index[:, None]).squeeze(1)
```

Update only the `topk_prob` branch of `sample_next_token_traj` to call
`build_topk_token_distribution`, then continue sampling with
`Categorical(logits=distribution.logits).sample()`. Leave both GT-conditioned
branches unchanged. This preserves their training behavior and makes the
formal baseline definition reusable.

- [ ] **Step 4: Run focused and legacy rollout tests**

Run: `pytest -q tests/accelerated_testing/test_token_distribution.py tests/test_future_token_dynamics.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the distribution boundary**

```bash
git add src/smart/accelerated_testing/__init__.py \
  src/smart/accelerated_testing/token_distribution.py \
  src/smart/utils/rollout.py \
  tests/accelerated_testing/__init__.py \
  tests/accelerated_testing/test_token_distribution.py
git commit -m "feat: expose exact CAT-K token distribution"
```

---

### Task 2: Immutable accelerated-testing configuration

**Files:**
- Create: `src/smart/accelerated_testing/config.py`
- Create: `tests/accelerated_testing/test_config.py`
- Create: `configs/experiment/accelerated_testing.yaml`
- Create: `configs/experiment/accelerated_testing_calibrate.yaml`
- Modify: `configs/model/smart.yaml:20-55`

**Interfaces:**
- Produces: `AcceleratedTestingConfig.from_mapping(mapping)`.
- Produces immutable nested configs: `BaselineConfig`, `POVConfig`, `RiskConfig`, `ProposalConfig`, `EventConfig`, `StatisticsConfig`.
- Produces: `AcceleratedTestingConfig.validate_formal(endpoint_interpolation_active)`.

- [ ] **Step 1: Write failing config and Hydra composition tests**

```python
# tests/accelerated_testing/test_config.py
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir

from src.smart.accelerated_testing.config import AcceleratedTestingConfig

ROOT = Path(__file__).resolve().parents[2]


def test_model_default_is_disabled():
    model = yaml.safe_load((ROOT / "configs/model/smart.yaml").read_text())
    assert model["model_config"]["accelerated_testing"]["enabled"] is False


def test_formal_defaults_match_approved_spec():
    raw = yaml.safe_load(
        (ROOT / "configs/experiment/accelerated_testing.yaml").read_text()
    )["model"]["model_config"]["accelerated_testing"]
    config = AcceleratedTestingConfig.from_mapping(raw)
    assert config.phase == "fixed_is"
    assert config.mode == "evaluate"
    assert config.baseline.num_k == 48
    assert config.baseline.temperature == 1.0
    assert config.proposal.epsilon == 0.05
    assert config.proposal.beta == 1.0
    assert config.pov.candidate_radius_m == 60.0


@pytest.mark.parametrize(
    "criterium",
    [
        "topk_prob_sampled_with_dist",
        "topk_dist_sampled_with_prob",
    ],
)
def test_formal_run_rejects_gt_conditioned_sampling(criterium):
    config = AcceleratedTestingConfig.from_mapping(
        {
            "enabled": True,
            "mode": "evaluate",
            "baseline": {
                "criterium": criterium,
                "num_k": 48,
                "temperature": 1.0,
            },
        }
    )
    with pytest.raises(ValueError, match="future GT"):
        config.validate_formal(endpoint_interpolation_active=False)


def test_formal_run_rejects_endpoint_interpolation():
    config = AcceleratedTestingConfig.from_mapping(
        {"enabled": True, "mode": "evaluate"}
    )
    with pytest.raises(ValueError, match="endpoint interpolation"):
        config.validate_formal(endpoint_interpolation_active=True)


def test_hydra_composes_evaluation_and_calibration_experiments():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "configs"),
    ):
        evaluate = compose(
            config_name="run",
            overrides=["experiment=accelerated_testing"],
        )
        calibrate = compose(
            config_name="run",
            overrides=["experiment=accelerated_testing_calibrate"],
        )
    assert evaluate.action == "validate"
    assert evaluate.model.model_config.accelerated_testing.mode == "evaluate"
    assert evaluate.model.model_config.validation_rollout_sampling.num_k == 48
    assert calibrate.model.model_config.accelerated_testing.mode == "calibrate"
    assert str(calibrate.data.val_raw_dir).endswith("/training")
```

- [ ] **Step 2: Run the config tests and verify they fail**

Run: `pytest -q tests/accelerated_testing/test_config.py`

Expected: import or missing-key failures.

- [ ] **Step 3: Implement frozen dataclasses and exact YAML defaults**

```python
# src/smart/accelerated_testing/config.py
from dataclasses import dataclass, field, fields
import math
from typing import Mapping, Optional


@dataclass(frozen=True)
class BaselineConfig:
    criterium: str = "topk_prob"
    num_k: int = 48
    temperature: float = 1.0


@dataclass(frozen=True)
class POVConfig:
    candidate_radius_m: float = 60.0
    vehicle_only: bool = True
    max_per_rollout: int = 1
    lock_once_selected: bool = True
    criticality_threshold: float = 0.0
    criticality_threshold_source: str = "conservative_fallback"
    criticality_calibration_file: Optional[str] = None


@dataclass(frozen=True)
class RiskConfig:
    ego_top_k: int = 8
    horizon_s: float = 1.0
    internal_dt_s: float = 0.1
    tto_max_s: float = 5.0
    z_clip: float = 5.0
    component_weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)


@dataclass(frozen=True)
class ProposalConfig:
    epsilon: float = 0.05
    beta: float = 1.0
    frozen: bool = True


@dataclass(frozen=True)
class EventConfig:
    near_gap_m: float = 1.0
    near_tto_s: float = 1.5
    near_pet_s: float = 1.5
    near_required_decel_mps2: float = 3.0
    same_direction_max_deg: float = 45.0
    opposing_direction_min_deg: float = 135.0
    contact_axis_margin: float = 0.15


@dataclass(frozen=True)
class StatisticsConfig:
    confidence_level: float = 0.90
    bootstrap_replicates: int = 2000
    target_rhw: float = 0.30
    minimum_ess_fraction: float = 0.10


@dataclass(frozen=True)
class AcceleratedTestingConfig:
    enabled: bool = False
    phase: str = "fixed_is"
    mode: str = "evaluate"
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    pov: POVConfig = field(default_factory=POVConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    proposal: ProposalConfig = field(default_factory=ProposalConfig)
    events: EventConfig = field(default_factory=EventConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object] | None,
    ) -> "AcceleratedTestingConfig":
        raw = dict(mapping or {})
        nested = {
            "baseline": BaselineConfig,
            "pov": POVConfig,
            "risk": RiskConfig,
            "proposal": ProposalConfig,
            "events": EventConfig,
            "statistics": StatisticsConfig,
        }
        allowed = {item.name for item in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown accelerated_testing keys: {sorted(unknown)}")
        values = {}
        for name, nested_type in nested.items():
            child = dict(raw.pop(name, {}) or {})
            child_allowed = {item.name for item in fields(nested_type)}
            child_unknown = set(child) - child_allowed
            if child_unknown:
                raise ValueError(
                    f"unknown accelerated_testing.{name} keys: "
                    f"{sorted(child_unknown)}"
                )
            if name == "risk" and "component_weights" in child:
                child["component_weights"] = tuple(child["component_weights"])
            values[name] = nested_type(**child)
        result = cls(**raw, **values)
        result._validate_ranges()
        return result

    def _validate_ranges(self) -> None:
        if self.phase != "fixed_is":
            raise ValueError("Phase 1 only supports phase=fixed_is")
        if self.mode not in {"calibrate", "evaluate", "baseline"}:
            raise ValueError("mode must be calibrate, evaluate, or baseline")
        if not 0.0 <= self.proposal.epsilon < 1.0:
            raise ValueError("proposal.epsilon must be in [0, 1)")
        if self.proposal.beta < 0.0:
            raise ValueError("proposal.beta must be non-negative")
        if self.baseline.num_k <= 0:
            raise ValueError("baseline.num_k must be positive")
        if self.baseline.temperature <= 0.0:
            raise ValueError("baseline.temperature must be positive")
        if self.risk.ego_top_k <= 0:
            raise ValueError("risk.ego_top_k must be positive")
        if (
            self.risk.horizon_s <= 0.0
            or self.risk.internal_dt_s <= 0.0
            or self.risk.tto_max_s <= 0.0
        ):
            raise ValueError("risk horizons and timestep must be positive")
        if self.risk.z_clip <= 0.0:
            raise ValueError("risk.z_clip must be positive")
        weights = self.risk.component_weights
        if len(weights) != 4 or any(weight < 0.0 for weight in weights):
            raise ValueError("risk.component_weights must have four non-negative values")
        if abs(sum(weights) - 1.0) > 1e-8:
            raise ValueError("risk.component_weights must sum to one")
        if self.pov.max_per_rollout != 1:
            raise ValueError("Phase 1 requires pov.max_per_rollout == 1")
        if not self.pov.vehicle_only or not self.pov.lock_once_selected:
            raise ValueError("Phase 1 requires a locked vehicle-only POV")
        if self.pov.candidate_radius_m <= 0.0:
            raise ValueError("pov.candidate_radius_m must be positive")
        if not math.isfinite(self.pov.criticality_threshold):
            raise ValueError("pov.criticality_threshold must be finite")
        near_thresholds = (
            self.events.near_gap_m,
            self.events.near_tto_s,
            self.events.near_pet_s,
            self.events.near_required_decel_mps2,
        )
        if any(value <= 0.0 for value in near_thresholds):
            raise ValueError("event near-miss thresholds must be positive")
        if not (
            0.0 <= self.events.same_direction_max_deg
            < self.events.opposing_direction_min_deg
            <= 180.0
        ):
            raise ValueError("event heading boundaries are inconsistent")
        if self.events.contact_axis_margin < 0.0:
            raise ValueError("events.contact_axis_margin must be non-negative")
        if not 0.0 < self.statistics.confidence_level < 1.0:
            raise ValueError("statistics.confidence_level must be in (0, 1)")
        if self.statistics.bootstrap_replicates <= 0:
            raise ValueError("statistics.bootstrap_replicates must be positive")
        if self.statistics.target_rhw <= 0.0:
            raise ValueError("statistics.target_rhw must be positive")
        if not 0.0 < self.statistics.minimum_ess_fraction <= 1.0:
            raise ValueError(
                "statistics.minimum_ess_fraction must be in (0, 1]"
            )

    def validate_formal(self, *, endpoint_interpolation_active: bool) -> None:
        self._validate_ranges()
        if self.baseline.criterium != "topk_prob":
            raise ValueError("formal accelerated testing cannot use future GT")
        if self.mode == "evaluate" and not self.proposal.frozen:
            raise ValueError("formal proposal must be frozen")
        calibrated = (
            self.pov.criticality_threshold_source != "conservative_fallback"
        )
        if calibrated and not self.pov.criticality_calibration_file:
            raise ValueError(
                "calibrated threshold requires its calibration artifact"
            )
        if self.mode == "evaluate" and endpoint_interpolation_active:
            raise ValueError(
                "formal accelerated testing requires endpoint interpolation off"
            )
```

`validate_formal` additionally requires `proposal.frozen=true` and endpoint
interpolation disabled in evaluate mode.

Add this disabled-by-default section under `model_config` in
`configs/model/smart.yaml`:

```yaml
accelerated_testing:
  enabled: false
  phase: fixed_is
  mode: evaluate
  baseline:
    criterium: topk_prob
    num_k: 48
    temperature: 1.0
  pov:
    candidate_radius_m: 60.0
    vehicle_only: true
    max_per_rollout: 1
    lock_once_selected: true
    criticality_threshold: 0.0
    criticality_threshold_source: conservative_fallback
    criticality_calibration_file: null
  risk:
    ego_top_k: 8
    horizon_s: 1.0
    internal_dt_s: 0.1
    tto_max_s: 5.0
    z_clip: 5.0
    component_weights: [0.25, 0.25, 0.25, 0.25]
  proposal:
    epsilon: 0.05
    beta: 1.0
    frozen: true
  events:
    near_gap_m: 1.0
    near_tto_s: 1.5
    near_pet_s: 1.5
    near_required_decel_mps2: 3.0
    same_direction_max_deg: 45.0
    opposing_direction_min_deg: 135.0
    contact_axis_margin: 0.15
  statistics:
    confidence_level: 0.90
    bootstrap_replicates: 2000
    target_rhw: 0.30
    minimum_ess_fraction: 0.10
```

`configs/experiment/accelerated_testing.yaml` must inherit the existing
`inference` experiment, override after it with `_self_`, keep
`action: validate`, set `validation_rollout_sampling` to `topk_prob`, K=48,
temperature=1.0, and enable Fast WOSAC-independent accelerated output:

```yaml
# @package _global_
defaults:
  - inference
  - _self_

model:
  model_config:
    n_rollout_closed_val: 32
    accelerated_testing:
      enabled: true
      phase: fixed_is
      mode: evaluate
      baseline:
        criterium: topk_prob
        num_k: 48
        temperature: 1.0
    validation_rollout_sampling:
      criterium: topk_prob
      num_k: 48
      temp: 1.0

action: validate
```

`configs/experiment/accelerated_testing_calibrate.yaml` must inherit that
experiment and replace validation data with the training cache:

```yaml
# @package _global_
defaults:
  - accelerated_testing
  - _self_

data:
  val_raw_dir: ${paths.cache_root}/training
  val_tfrecords_splitted: null
  shuffle: false
model:
  model_config:
    accelerated_testing:
      enabled: true
      mode: calibrate
```

The evaluation config must keep `data.val_raw_dir` on validation and set
`mode: evaluate`.

- [ ] **Step 4: Run config tests**

Run: `pytest -q tests/accelerated_testing/test_config.py tests/test_training_fast_wosac_config.py`

Expected: all tests pass and existing training configs remain unchanged.

- [ ] **Step 5: Commit configuration**

```bash
git add src/smart/accelerated_testing/config.py \
  configs/model/smart.yaml \
  configs/experiment/accelerated_testing.yaml \
  configs/experiment/accelerated_testing_calibrate.yaml \
  tests/accelerated_testing/test_config.py
git commit -m "feat: configure fixed-IS safety evaluation"
```

---

### Task 3: Risk-tilted proposal and exact selected-token ratio

**Files:**
- Create: `src/smart/accelerated_testing/proposal.py`
- Create: `tests/accelerated_testing/test_proposal.py`
- Modify: `src/smart/accelerated_testing/__init__.py`

**Interfaces:**
- Consumes: `TopKTokenDistribution`.
- Produces: `ProposalDistribution(log_r, log_q)`.
- Produces: `build_risk_tilted_proposal(baseline, risk, epsilon, beta, z_clip)`.
- Produces: `selected_log_ratio(baseline, proposal, selected_token_ids)`.

- [ ] **Step 1: Write failing proposal tests**

```python
# tests/accelerated_testing/test_proposal.py
import torch

from src.smart.accelerated_testing.proposal import (
    build_risk_tilted_proposal,
    selected_log_ratio,
)
from src.smart.accelerated_testing.token_distribution import (
    build_topk_token_distribution,
)


def _baseline():
    return build_topk_token_distribution(
        torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float64),
        num_k=3,
        temperature=1.0,
    )


def test_zero_epsilon_is_exact_baseline():
    baseline = _baseline()
    proposal = build_risk_tilted_proposal(
        baseline,
        torch.tensor([[0.0, 1.0, 4.0]], dtype=torch.float64),
        epsilon=0.0,
        beta=1.0,
        z_clip=5.0,
    )
    torch.testing.assert_close(proposal.log_q, baseline.log_probs)


def test_zero_beta_is_exact_baseline():
    baseline = _baseline()
    proposal = build_risk_tilted_proposal(
        baseline,
        torch.tensor([[0.0, 1.0, 4.0]], dtype=torch.float64),
        epsilon=0.4,
        beta=0.0,
        z_clip=5.0,
    )
    torch.testing.assert_close(proposal.log_r, baseline.log_probs)
    torch.testing.assert_close(proposal.log_q, baseline.log_probs)


def test_mixture_preserves_support_and_normalization():
    baseline = _baseline()
    proposal = build_risk_tilted_proposal(
        baseline,
        torch.tensor([[0.0, 1.0, 4.0]], dtype=torch.float64),
        epsilon=0.05,
        beta=1.0,
        z_clip=5.0,
    )
    q = proposal.log_q.exp()
    p = baseline.log_probs.exp()
    torch.testing.assert_close(q.sum(-1), torch.ones(1, dtype=torch.float64))
    assert torch.all(q >= 0.95 * p)
    chosen = baseline.token_ids[:, 0]
    ratio = selected_log_ratio(baseline, proposal, chosen)
    torch.testing.assert_close(
        ratio,
        baseline.log_probs[:, 0] - proposal.log_q[:, 0],
    )
```

- [ ] **Step 2: Run proposal tests and verify failure**

Run: `pytest -q tests/accelerated_testing/test_proposal.py`

Expected: import fails because `proposal.py` does not exist.

- [ ] **Step 3: Implement log-domain proposal math**

```python
# src/smart/accelerated_testing/proposal.py
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .token_distribution import TopKTokenDistribution


@dataclass(frozen=True)
class ProposalDistribution:
    log_r: Tensor
    log_q: Tensor


def build_risk_tilted_proposal(
    baseline: TopKTokenDistribution,
    risk: Tensor,
    *,
    epsilon: float,
    beta: float,
    z_clip: float,
) -> ProposalDistribution:
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1)")
    if beta < 0.0 or z_clip <= 0.0:
        raise ValueError("beta must be non-negative and z_clip positive")
    log_p = baseline.log_probs.to(torch.float64)
    risk = risk.to(torch.float64)
    if risk.shape != log_p.shape or not torch.isfinite(risk).all():
        raise ValueError("risk must be finite and match baseline support")
    p = log_p.exp()
    mean = (p * risk).sum(dim=-1, keepdim=True)
    variance = (p * (risk - mean).square()).sum(dim=-1, keepdim=True)
    z = torch.where(
        variance.sqrt() < 1e-6,
        torch.zeros_like(risk),
        (risk - mean) / variance.sqrt().clamp_min(1e-6),
    ).clamp(-z_clip, z_clip)
    tilted = log_p + float(beta) * z
    log_r = tilted - torch.logsumexp(tilted, dim=-1, keepdim=True)
    if epsilon == 0.0:
        log_q = log_p
    else:
        log_q = torch.logaddexp(
            log_p + math.log1p(-epsilon),
            log_r + math.log(epsilon),
        )
    return ProposalDistribution(log_r=log_r, log_q=log_q)
```

Implement the selected-token ratio as:

```python
def selected_log_ratio(
    baseline: TopKTokenDistribution,
    proposal: ProposalDistribution,
    selected_token_ids: Tensor,
) -> Tensor:
    matches = baseline.token_ids == selected_token_ids[:, None]
    if not bool(matches.any(dim=-1).all()):
        raise ValueError("selected token is outside baseline support")
    support_index = matches.to(torch.int64).argmax(dim=-1)
    log_p = baseline.log_probs.gather(
        1, support_index[:, None]
    ).squeeze(1)
    log_q = proposal.log_q.gather(
        1, support_index[:, None]
    ).squeeze(1)
    return log_p.to(torch.float64) - log_q.to(torch.float64)
```

- [ ] **Step 4: Run proposal tests**

Run: `pytest -q tests/accelerated_testing/test_proposal.py`

Expected: all tests pass.

- [ ] **Step 5: Commit proposal**

```bash
git add src/smart/accelerated_testing/__init__.py \
  src/smart/accelerated_testing/proposal.py \
  tests/accelerated_testing/test_proposal.py
git commit -m "feat: add risk-tilted token proposal"
```

---

### Task 4: Vectorized token risk and criticality

**Files:**
- Create: `src/smart/accelerated_testing/risk.py`
- Create: `tests/accelerated_testing/test_risk.py`

**Interfaces:**
- Consumes: global current poses, shapes, baseline support IDs/probabilities, and local `[n_token,6,4,2]` contours.
- Produces: `TokenPoseBatch(centers, headings)` with `[K,11,2]` and `[K,11]` for 0.0–1.0 s.
- Produces: `rounded_box_signed_distance(boxes_a, boxes_b)`.
- Produces: `constant_velocity_time_to_overlap(...)`.
- Produces: `compute_expected_token_risk(...) -> Tensor[K_pov]`.
- Produces: `compute_criticality(risk, pov_log_probs) -> Tensor`.

- [ ] **Step 1: Write failing geometry and risk tests**

```python
# tests/accelerated_testing/test_risk.py
import inspect
import torch

from src.smart.accelerated_testing.risk import (
    compute_criticality,
    compute_expected_token_risk,
    constant_velocity_time_to_overlap,
    rounded_box_signed_distance,
)
from src.smart.metrics.fast_wosac_backend.fast_sim_agents_metrics import (
    interaction_features,
)


def test_pair_distance_matches_fast_wosac_nearest_distance():
    boxes = torch.tensor(
        [[[
            [[0.0, 0.0, 0.0, 4.8, 2.0, 1.5, 0.0]],
            [[6.0, 0.0, 0.0, 4.8, 2.0, 1.5, 0.0]],
        ]]],
        dtype=torch.float64,
    ).reshape(1, 2, 1, 7)
    valid = torch.ones(1, 2, 1, dtype=torch.bool)
    nearest = interaction_features.compute_distance_to_nearest_object(
        boxes=boxes,
        valid=valid,
        evaluated_object_mask=torch.tensor([True, False]),
    )[0, 0, 0]
    pair = rounded_box_signed_distance(boxes[0, 0], boxes[0, 1])[0]
    torch.testing.assert_close(pair, nearest)


def test_closing_token_has_higher_risk_than_diverging_token():
    result = compute_expected_token_risk(
        ego_centers=torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
        ego_headings=torch.zeros(1, 2),
        ego_log_probs=torch.zeros(1),
        pov_centers=torch.tensor(
            [
                [[8.0, 0.0], [5.0, 0.0]],
                [[8.0, 0.0], [10.0, 0.0]],
            ]
        ),
        pov_headings=torch.zeros(2, 2),
        ego_shape=torch.tensor([4.8, 2.0, 1.5]),
        pov_shape=torch.tensor([4.8, 2.0, 1.5]),
        dt_s=0.5,
        tto_max_s=5.0,
        component_weights=(0.25, 0.25, 0.25, 0.25),
    )
    assert result[0] > result[1]


def test_generic_time_to_overlap_handles_closing_and_diverging_motion():
    closing = constant_velocity_time_to_overlap(
        center_a=torch.tensor([0.0, 0.0]),
        velocity_a=torch.tensor([2.0, 0.0]),
        heading_a=torch.tensor(0.0),
        shape_a=torch.tensor([4.8, 2.0, 1.5]),
        center_b=torch.tensor([10.0, 0.0]),
        velocity_b=torch.tensor([-2.0, 0.0]),
        heading_b=torch.tensor(0.0),
        shape_b=torch.tensor([4.8, 2.0, 1.5]),
        dt_s=0.1,
        max_s=5.0,
    )
    diverging = constant_velocity_time_to_overlap(
        center_a=torch.tensor([0.0, 0.0]),
        velocity_a=torch.tensor([-2.0, 0.0]),
        heading_a=torch.tensor(0.0),
        shape_a=torch.tensor([4.8, 2.0, 1.5]),
        center_b=torch.tensor([10.0, 0.0]),
        velocity_b=torch.tensor([2.0, 0.0]),
        heading_b=torch.tensor(0.0),
        shape_b=torch.tensor([4.8, 2.0, 1.5]),
        dt_s=0.1,
        max_s=5.0,
    )
    assert 0.0 < closing <= 5.0
    assert torch.isinf(diverging)


def test_criticality_is_action_sensitivity():
    risk = torch.tensor([0.1, 0.9], dtype=torch.float64)
    log_p = torch.log(torch.tensor([0.8, 0.2], dtype=torch.float64))
    expected = 0.9 - (0.8 * 0.1 + 0.2 * 0.9)
    torch.testing.assert_close(compute_criticality(risk, log_p), expected)


def test_risk_api_has_no_future_gt_argument():
    names = inspect.signature(compute_expected_token_risk).parameters
    assert not any("gt" in name.lower() for name in names)
```

- [ ] **Step 2: Run risk tests and verify failure**

Run: `pytest -q tests/accelerated_testing/test_risk.py`

Expected: import fails because `risk.py` does not exist.

- [ ] **Step 3: Implement pose expansion and risk components**

Use the existing functions
`get_upright_3d_box_corners`,
`minkowski_sum_of_box_and_box_points`, and
`signed_distance_from_point_to_convex_polygon` from the embedded backend;
do not edit that backend. The core composite must be:

```python
def _risk_components(
    signed_distance: Tensor,
    time_to_overlap: Tensor,
    closing_speed: Tensor,
) -> Tensor:
    required_decel = torch.where(
        closing_speed > 0,
        closing_speed.square()
        / (2.0 * signed_distance.clamp_min(0.1)),
        torch.zeros_like(closing_speed),
    )
    overlap = torch.sigmoid(-signed_distance / 0.25)
    gap = torch.exp(-signed_distance.clamp_min(0.0) / 1.0)
    tto = torch.where(
        torch.isfinite(time_to_overlap),
        torch.exp(-time_to_overlap / 1.5),
        torch.zeros_like(time_to_overlap),
    )
    brake = torch.sigmoid((required_decel - 3.0) / 0.5)
    return torch.stack((overlap, gap, tto, brake), dim=-1)


def compute_criticality(risk: Tensor, pov_log_probs: Tensor) -> Tensor:
    p = pov_log_probs.to(torch.float64).exp()
    return risk.max() - (p * risk.to(torch.float64)).sum()
```

`token_support_to_global_poses` must:

1. gather only support token contours;
2. transform them from the agent frame with current position/heading;
3. derive center by averaging four corners and heading from corner 3 to corner 0;
4. keep token frames 0.0–0.5 s;
5. use the last two centers for constant terminal velocity and append frames
   0.6–1.0 s while keeping terminal heading fixed.

`compute_expected_token_risk` must broadcast all POV support tokens against
the normalized ego Top-8 support, take the maximum risk over horizon frames,
and then take the ego probability-weighted expectation.

- [ ] **Step 4: Run risk tests**

Run: `pytest -q tests/accelerated_testing/test_risk.py tests/test_embedded_fast_wosac_backend.py`

Expected: all tests pass and embedded-backend parity remains intact.

- [ ] **Step 5: Commit risk scoring**

```bash
git add src/smart/accelerated_testing/risk.py \
  tests/accelerated_testing/test_risk.py
git commit -m "feat: score token-level collision risk"
```

---

### Task 5: Deterministic single-POV selection

**Files:**
- Create: `src/smart/accelerated_testing/pov_selector.py`
- Create: `tests/accelerated_testing/test_pov_selector.py`

**Interfaces:**
- Produces: `POVSelection(track_id, agent_index, locked, disabled)`.
- Produces: `SinglePOVSelector.update(...) -> POVSelection`.
- Consumes one scenario-rollout at a time; the controller owns one selector per scenario.

- [ ] **Step 1: Write failing locking tests**

```python
# tests/accelerated_testing/test_pov_selector.py
import torch

from src.smart.accelerated_testing.pov_selector import SinglePOVSelector


def _state():
    return dict(
        track_ids=torch.tensor([100, 20, 10, 30]),
        agent_types=torch.tensor([0, 0, 0, 1]),
        ego_mask=torch.tensor([True, False, False, False]),
        valid=torch.tensor([True, True, True, True]),
        shapes=torch.tensor(
            [
                [4.8, 2.0, 1.5],
                [4.8, 2.0, 1.5],
                [4.8, 2.0, 1.5],
                [0.8, 0.8, 1.7],
            ]
        ),
        positions=torch.tensor(
            [[0.0, 0.0], [10.0, 0.0], [10.0, 1.0], [5.0, 0.0]]
        ),
        criticality=torch.tensor([0.0, 0.7, 0.7, 1.0]),
    )


def test_tie_breaks_by_smallest_track_id_and_locks_once():
    selector = SinglePOVSelector(radius_m=60.0, threshold=0.5)
    first = selector.update(**_state())
    assert first.track_id == 10
    changed = _state()
    changed["criticality"] = torch.tensor([0.0, 2.0, 0.1, 3.0])
    second = selector.update(**changed)
    assert second.track_id == 10


def test_invalid_locked_pov_disables_without_replacement():
    selector = SinglePOVSelector(radius_m=60.0, threshold=0.5)
    selector.update(**_state())
    changed = _state()
    changed["valid"][2] = False
    result = selector.update(**changed)
    assert result.track_id == 10
    assert result.agent_index is None
    assert result.disabled is True


def test_locked_pov_leaving_radius_disables_without_replacement():
    selector = SinglePOVSelector(radius_m=60.0, threshold=0.5)
    selector.update(**_state())
    changed = _state()
    changed["positions"][2] = torch.tensor([61.0, 0.0])
    result = selector.update(**changed)
    assert result.track_id == 10
    assert result.agent_index is None
    assert result.disabled is True
```

- [ ] **Step 2: Run selector tests and verify failure**

Run: `pytest -q tests/accelerated_testing/test_pov_selector.py`

Expected: import fails because `pov_selector.py` does not exist.

- [ ] **Step 3: Implement selection and stable-ID lookup**

```python
# src/smart/accelerated_testing/pov_selector.py
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class POVSelection:
    track_id: Optional[int]
    agent_index: Optional[int]
    locked: bool
    disabled: bool


class SinglePOVSelector:
    def __init__(self, *, radius_m: float, threshold: float) -> None:
        self.radius_m = float(radius_m)
        self.threshold = float(threshold)
        self._track_id: Optional[int] = None
        self._disabled = False

    def _eligible(
        self,
        *,
        index: int,
        ego_index: int,
        agent_types: Tensor,
        valid: Tensor,
        shapes: Tensor,
        positions: Tensor,
    ) -> bool:
        distance = torch.linalg.vector_norm(
            positions[index] - positions[ego_index]
        )
        shape_ok = bool(
            torch.isfinite(shapes[index, :2]).all().item()
            and (shapes[index, :2] > 0).all().item()
        )
        return bool(
            valid[index].item()
            and (agent_types[index] == 0).item()
            and (distance <= self.radius_m).item()
            and shape_ok
        )

    def update(
        self,
        *,
        track_ids: Tensor,
        agent_types: Tensor,
        ego_mask: Tensor,
        valid: Tensor,
        shapes: Tensor,
        positions: Tensor,
        criticality: Tensor,
    ) -> POVSelection:
        ego_index = torch.where(ego_mask)[0].item()
        if self._track_id is not None:
            if self._disabled:
                return POVSelection(self._track_id, None, True, True)
            matches = torch.where(track_ids == self._track_id)[0]
            if len(matches) != 1 or not self._eligible(
                index=int(matches[0].item()),
                ego_index=ego_index,
                agent_types=agent_types,
                valid=valid,
                shapes=shapes,
                positions=positions,
            ):
                self._disabled = True
                return POVSelection(self._track_id, None, True, True)
            return POVSelection(
                self._track_id,
                int(matches[0].item()),
                True,
                self._disabled,
            )
        distance = torch.linalg.vector_norm(
            positions - positions[ego_index], dim=-1
        )
        shape_ok = (
            torch.isfinite(shapes[:, :2]).all(dim=-1)
            & (shapes[:, :2] > 0).all(dim=-1)
        )
        candidate = (
            (~ego_mask)
            & valid
            & (agent_types == 0)
            & (distance <= self.radius_m)
            & shape_ok
            & (criticality >= self.threshold)
        )
        indices = torch.where(candidate)[0].tolist()
        if not indices:
            return POVSelection(None, None, False, False)
        index = min(
            indices,
            key=lambda idx: (
                -float(criticality[idx].item()),
                int(track_ids[idx].item()),
            ),
        )
        self._track_id = int(track_ids[index].item())
        return POVSelection(self._track_id, index, True, False)
```

- [ ] **Step 4: Run selector tests**

Run: `pytest -q tests/accelerated_testing/test_pov_selector.py`

Expected: all tests pass.

- [ ] **Step 5: Commit selector**

```bash
git add src/smart/accelerated_testing/pov_selector.py \
  tests/accelerated_testing/test_pov_selector.py
git commit -m "feat: lock one deterministic challenge vehicle"
```

---

### Task 6: Ego collision, collision type, and near-miss events

**Files:**
- Create: `src/smart/accelerated_testing/events.py`
- Create: `tests/accelerated_testing/test_events.py`

**Interfaces:**
- Consumes: `AgentTokenFrames(track_ids, centers, headings, shapes, valid, ego_mask)` for one scenario.
- Produces: `CollisionEvent(time_s, frame_index, partner_id, collision_type, simultaneous_partner_ids)`.
- Produces: `SafetyEventSummary(collision, initial_overlap, near_gap, near_tto, near_pet, near_required_decel, near_miss_union, pet_applicable)`.
- Produces: `EgoSafetyEventTracker.observe_token(...)` and `.finalize()`.
- Produces the geometry-only test seam
  `EgoSafetyEventTracker.observe_signed_distances(time_s, partner_ids, distances)`;
  production inference calls `observe_token`, which computes those distances
  from reconstructed boxes before delegating to the same state transition.

- [ ] **Step 1: Write failing event and classification tests**

```python
# tests/accelerated_testing/test_events.py
import torch

from src.smart.accelerated_testing.events import (
    EgoSafetyEventTracker,
    classify_collision,
    compute_post_encroachment_time,
    summarize_near_miss,
)


def test_strict_zero_distance_is_not_collision():
    tracker = EgoSafetyEventTracker.default()
    tracker.observe_signed_distances(
        time_s=torch.tensor([0.1]),
        partner_ids=torch.tensor([9]),
        distances=torch.tensor([[0.0]]),
    )
    assert tracker.finalize().collision is None


def test_initial_overlap_must_separate_before_recollision():
    tracker = EgoSafetyEventTracker.default()
    tracker.observe_signed_distances(
        time_s=torch.tensor([0.0, 0.1, 0.2]),
        partner_ids=torch.tensor([9]),
        distances=torch.tensor([[-0.1], [0.2], [-0.1]]),
    )
    summary = tracker.finalize()
    assert summary.initial_overlap is True
    assert summary.collision.time_s == 0.2


def test_collision_classes_have_auditable_boundaries():
    assert classify_collision(
        heading_delta_deg=5.0,
        normalized_longitudinal=0.9,
        normalized_lateral=0.1,
        closing_speed=2.0,
    ) == "rear_end"
    assert classify_collision(
        heading_delta_deg=5.0,
        normalized_longitudinal=0.1,
        normalized_lateral=0.9,
        closing_speed=2.0,
    ) == "sideswipe"
    assert classify_collision(
        heading_delta_deg=90.0,
        normalized_longitudinal=0.7,
        normalized_lateral=0.7,
        closing_speed=2.0,
    ) == "angle"
    assert classify_collision(
        heading_delta_deg=175.0,
        normalized_longitudinal=0.9,
        normalized_lateral=0.1,
        closing_speed=2.0,
    ) == "head_on"
    assert classify_collision(
        heading_delta_deg=5.0,
        normalized_longitudinal=0.5,
        normalized_lateral=0.5,
        closing_speed=2.0,
    ) == "other_or_unknown"


def test_near_miss_is_mutually_exclusive_with_collision():
    tracker = EgoSafetyEventTracker.default()
    tracker.observe_signed_distances(
        time_s=torch.tensor([0.1, 0.2]),
        partner_ids=torch.tensor([9]),
        distances=torch.tensor([[0.5], [-0.1]]),
    )
    summary = tracker.finalize()
    assert summary.collision is not None
    assert summary.near_miss_union is False


def test_simultaneous_collision_tie_breaks_by_track_id():
    tracker = EgoSafetyEventTracker.default()
    tracker.observe_signed_distances(
        time_s=torch.tensor([0.0, 0.1]),
        partner_ids=torch.tensor([9, 7]),
        distances=torch.tensor([[0.2, 0.2], [-0.1, -0.1]]),
    )
    collision = tracker.finalize().collision
    assert collision.partner_id == 7
    assert collision.simultaneous_partner_ids == (7, 9)


def test_simultaneous_collision_prefers_deepest_overlap():
    tracker = EgoSafetyEventTracker.default()
    tracker.observe_signed_distances(
        time_s=torch.tensor([0.0, 0.1]),
        partner_ids=torch.tensor([9, 7]),
        distances=torch.tensor([[0.2, 0.2], [-0.2, -0.1]]),
    )
    assert tracker.finalize().collision.partner_id == 9


def test_each_near_miss_condition_and_strict_boundary():
    gap = summarize_near_miss(
        collision_present=False,
        min_signed_distance=0.5,
        min_tto_s=float("inf"),
        min_pet_s=None,
        max_required_decel_mps2=0.0,
    )
    assert gap.near_gap and gap.near_miss_union

    tto = summarize_near_miss(
        collision_present=False,
        min_signed_distance=1.0,
        min_tto_s=1.49,
        min_pet_s=None,
        max_required_decel_mps2=3.0,
    )
    assert tto.near_tto and not tto.near_required_decel

    pet = summarize_near_miss(
        collision_present=False,
        min_signed_distance=1.0,
        min_tto_s=1.5,
        min_pet_s=1.49,
        max_required_decel_mps2=3.01,
    )
    assert pet.pet_applicable and pet.near_pet
    assert pet.near_required_decel

    boundary = summarize_near_miss(
        collision_present=False,
        min_signed_distance=1.0,
        min_tto_s=1.5,
        min_pet_s=1.5,
        max_required_decel_mps2=3.0,
    )
    assert boundary.near_miss_union is False


def test_pet_reports_not_applicable_without_cross_time_overlap():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 4.8, 2.0, 1.5, 0.0],
            [20.0, 0.0, 0.0, 4.8, 2.0, 1.5, 0.0],
        ]
    )
    crossing = torch.flip(boxes, dims=(0,))
    pet, applicable = compute_post_encroachment_time(
        boxes,
        crossing,
        dt_s=0.1,
    )
    assert applicable is True
    assert pet == 0.1

    separated = boxes + torch.tensor([50.0, 0.0, 0.0, 0, 0, 0, 0])
    pet, applicable = compute_post_encroachment_time(
        boxes,
        separated,
        dt_s=0.1,
    )
    assert pet is None
    assert applicable is False
```

- [ ] **Step 2: Run event tests and verify failure**

Run: `pytest -q tests/accelerated_testing/test_events.py`

Expected: import fails because `events.py` does not exist.

- [ ] **Step 3: Implement the online event tracker**

```python
# src/smart/accelerated_testing/events.py
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


COLLISION_TYPES = (
    "rear_end",
    "sideswipe",
    "angle",
    "head_on",
    "other_or_unknown",
)


def classify_collision(
    *,
    heading_delta_deg: float,
    normalized_longitudinal: float,
    normalized_lateral: float,
    closing_speed: float,
    axis_margin: float = 0.15,
) -> str:
    if closing_speed <= 0:
        return "other_or_unknown"
    longitudinal = (
        normalized_longitudinal - normalized_lateral >= axis_margin
    )
    lateral = normalized_lateral - normalized_longitudinal >= axis_margin
    if heading_delta_deg <= 45.0 and longitudinal:
        return "rear_end"
    if heading_delta_deg <= 45.0 and lateral:
        return "sideswipe"
    if 45.0 < heading_delta_deg < 135.0:
        return "angle"
    if heading_delta_deg >= 135.0 and longitudinal:
        return "head_on"
    return "other_or_unknown"
```

Implement the full tracker with these exact rules:

- process frame 0 only for the first token and frames 1–5 for later tokens;
- keep a suppression bit per initially overlapping ego-partner pair;
- release suppression after one frame with signed distance `>=0`;
- exclude still-suppressed initial-overlap pairs from all near-miss minima;
- choose the most negative signed distance at the first collision frame, then
  smallest `track_id` on ties;
- freeze the first collision and classification;
- derive collision-frame velocities from the immediately preceding valid
  10 Hz frame; if unavailable, centers are degenerate, or dimensions are
  invalid, classify as `other_or_unknown`;
- compute generic TTO by frozen-heading constant-velocity extrapolation on a
  0.1 s grid up to 5.0 s;
- compute PET after rollout as the minimum positive \(|i-j|\times0.1\) for
  cross-time ego/partner vehicle envelopes whose rounded-box signed distance
  is `<0`; set `pet_applicable=false` when no cross-time overlap exists;
- make near-miss false whenever an ego collision exists;
- otherwise expose all four booleans and their union.

`summarize_near_miss` must use strict comparisons
(`0 <= signed gap < 1.0`,
`TTO < 1.5`, applicable `PET < 1.5`, required deceleration `> 3.0`) and
return every subcondition even when the union is false. If collision is
present, it clears all near-miss booleans. `compute_post_encroachment_time`
must use the same rounded-box signed-distance predicate as online collision
detection and must ignore same-time pairs (`i == j`).

Use `rounded_box_signed_distance` from `risk.py`; do not change Fast WOSAC.

- [ ] **Step 4: Run all event and risk tests**

Run: `pytest -q tests/accelerated_testing/test_events.py tests/accelerated_testing/test_risk.py`

Expected: all tests pass.

- [ ] **Step 5: Commit event detection**

```bash
git add src/smart/accelerated_testing/events.py \
  tests/accelerated_testing/test_events.py
git commit -m "feat: detect ego safety-critical events"
```

---

### Task 7: Training-only criticality threshold calibration

**Files:**
- Create: `src/smart/accelerated_testing/calibration.py`
- Create: `tests/accelerated_testing/test_calibration.py`

**Interfaces:**
- Produces: `CalibrationRollout(scenario_id, rollout_id, step_criticality, first_collision_time_s)`.
- Produces: `CalibrationProvenance`.
- Produces: `CriticalityCalibrationArtifact`.
- Produces: `calibrate_criticality_threshold(rollouts, provenance, precursor_window_s=3.0, min_collision_rollouts=100, target_recall=0.99)`.
- Produces: `save_calibration_artifact(path, artifact)` and `load_calibration_artifact(path, expected_sha256)`.

- [ ] **Step 1: Write failing threshold-selection tests**

```python
# tests/accelerated_testing/test_calibration.py
from pathlib import Path

import pytest

from src.smart.accelerated_testing.calibration import (
    CalibrationProvenance,
    CalibrationRollout,
    calibrate_criticality_threshold,
    load_calibration_artifact,
    save_calibration_artifact,
)


def _provenance():
    return CalibrationProvenance(
        source_split="training",
        checkpoint_sha256="checkpoint-sha",
        vocabulary_sha256="vocabulary-sha",
        risk_config_sha256="risk-config-sha",
        scenario_list_sha256="training-scenarios-sha",
    )


def test_selects_highest_threshold_with_required_precursor_recall():
    rollouts = []
    for index in range(100):
        peak = 0.9 if index < 99 else 0.4
        rollouts.append(
            CalibrationRollout(
                scenario_id=f"s-{index}",
                rollout_id=0,
                step_criticality=(0.0, peak, 0.0, 0.0),
                first_collision_time_s=1.0,
            )
        )
    artifact = calibrate_criticality_threshold(
        rollouts,
        provenance=_provenance(),
    )
    assert artifact.threshold == 0.9
    assert artifact.precursor_recall == 0.99


def test_insufficient_collisions_uses_conservative_zero():
    artifact = calibrate_criticality_threshold(
        [
            CalibrationRollout(
                scenario_id="s",
                rollout_id=0,
                step_criticality=(0.2, 0.3),
                first_collision_time_s=0.8,
            )
        ],
        provenance=_provenance(),
    )
    assert artifact.threshold == 0.0
    assert artifact.status == "conservative_fallback"


def test_artifact_hash_is_verified(tmp_path: Path):
    artifact = calibrate_criticality_threshold(
        [],
        provenance=_provenance(),
    )
    path = tmp_path / "criticality.json"
    digest = save_calibration_artifact(path, artifact)
    assert load_calibration_artifact(path, expected_sha256=digest) == artifact
    with pytest.raises(ValueError, match="SHA-256"):
        load_calibration_artifact(path, expected_sha256="0" * 64)
```

- [ ] **Step 2: Run calibration tests and verify failure**

Run: `pytest -q tests/accelerated_testing/test_calibration.py`

Expected: import fails because `calibration.py` does not exist.

- [ ] **Step 3: Implement deterministic threshold calibration**

```python
# src/smart/accelerated_testing/calibration.py
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class CalibrationRollout:
    scenario_id: str
    rollout_id: int
    step_criticality: tuple[float, ...]
    first_collision_time_s: Optional[float]


@dataclass(frozen=True)
class CalibrationProvenance:
    source_split: str
    checkpoint_sha256: str
    vocabulary_sha256: str
    risk_config_sha256: str
    scenario_list_sha256: str


@dataclass(frozen=True)
class CriticalityCalibrationArtifact:
    threshold: float
    status: str
    collision_rollouts: int
    precursor_recall: float
    retained_step_fraction: float
    precursor_window_s: float
    target_recall: float
    provenance: CalibrationProvenance
```

For each collision rollout, use token steps whose start time `0.5*step` lies
in `[collision_time-3.0, collision_time)`. Candidate thresholds are descending
unique finite criticality values plus 0.0. Among thresholds covering at least
99% of collision rollouts, select the highest threshold whose retained
fraction of all token steps lies in `[0.01, 0.05]`. If none meets the sparsity
band, preserve recall and select the highest recall-valid threshold, recording
status `recall_only` and the actual retained fraction. If fewer than 100
collision rollouts exist, return threshold 0.0 and status
`conservative_fallback`. Serialize JSON with sorted keys and compact
separators before computing SHA-256.
Reject calibration unless `provenance.source_split == "training"`. During
formal evaluation, load the exact artifact named in
`criticality_calibration_file`, verify its file SHA-256 equals
`criticality_threshold_source`, and require its checkpoint, vocabulary, and
risk-config hashes to match the current run.

- [ ] **Step 4: Run calibration tests**

Run: `pytest -q tests/accelerated_testing/test_calibration.py`

Expected: all tests pass.

- [ ] **Step 5: Commit calibration**

```bash
git add src/smart/accelerated_testing/calibration.py \
  tests/accelerated_testing/test_calibration.py
git commit -m "feat: calibrate critical-state threshold"
```

---

### Task 8: Auditable ledger and DDP-safe artifact shards

**Files:**
- Create: `src/smart/accelerated_testing/ledger.py`
- Create: `tests/accelerated_testing/test_ledger.py`

**Interfaces:**
- Produces: `StepLedgerRecord`, `RolloutSummaryRecord`, and `FailureRecord`.
- Produces: `RunArtifactWriter(root, rank, world_size)`.
- Produces: `replay_log_weight(step_records)`.
- Produces: `sha256_file(path)` and `sha256_json(value)`.
- Produces root `trajectories.pt` as a sharded-v1 index, not an in-memory concatenation.

- [ ] **Step 1: Write failing serialization, replay, and merge tests**

```python
# tests/accelerated_testing/test_ledger.py
import gzip
import json
from pathlib import Path

import torch

from src.smart.accelerated_testing.ledger import (
    RunArtifactWriter,
    StepLedgerRecord,
    replay_log_weight,
)


def _record(step, increment):
    return StepLedgerRecord(
        scenario_id="s",
        rollout_id=2,
        arm="proposal",
        global_seed=817,
        token_step=step,
        future_frame_start=5 * step,
        future_frame_end=5 * (step + 1),
        pov_track_id=9,
        support_token_ids=(1, 4),
        log_p=(-0.2, -1.7),
        risk=(0.1, 0.9),
        log_r=(-1.0, -0.4),
        log_q=(-0.3, -1.4),
        selected_token_id=1,
        criticality=0.8,
        epsilon=0.05,
        beta=1.0,
        log_ratio_increment=increment,
        cumulative_log_weight=sum([increment] * (step + 1)),
        first_collision_in_token=False,
        rng_key=f"817|s|2|{step}|9",
    )


def test_replay_reconstructs_log_weight():
    records = [_record(0, 0.1), _record(1, 0.1)]
    assert abs(replay_log_weight(records) - 0.2) < 1e-10


def test_rank_merge_preserves_every_jsonl_record(tmp_path: Path):
    for rank in (0, 1):
        writer = RunArtifactWriter(tmp_path, rank=rank, world_size=2)
        writer.write_step(_record(rank, 0.1))
        writer.write_trajectory_shard(
            batch_idx=rank,
            payload={
                "rank": rank,
                "scenario_id": ["s"],
                "rollout_id": [rank],
                "arm": ["proposal"],
                "trajectory": torch.zeros(1, 80, 3),
            },
        )
        writer.close_rank()
    RunArtifactWriter.merge_rank_outputs(tmp_path, world_size=2)
    with gzip.open(tmp_path / "step_ledger.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert {(row["token_step"], row["pov_track_id"]) for row in rows} == {
        (0, 9),
        (1, 9),
    }
    index = torch.load(tmp_path / "trajectories.pt", weights_only=True)
    assert index["format"] == "catk-accelerated-testing-sharded-v1"
    assert len(index["shards"]) == 2
    assert all(
        len(shard["sha256"]) == 64
        and all(len(row["sha256"]) == 64 for row in shard["rows"])
        for shard in index["shards"]
    )
```

Add two adjacent merge tests using the same
`(arm, scenario_id, rollout_id)` on ranks 0 and 1: byte-identical summary,
ledger, and trajectory rows must merge once and appear in
`ddp_padding_duplicates_removed`; changing one trajectory coordinate must
raise `ValueError("conflicting duplicate rollout")`.

- [ ] **Step 2: Run ledger tests and verify failure**

Run: `pytest -q tests/accelerated_testing/test_ledger.py`

Expected: import fails because `ledger.py` does not exist.

- [ ] **Step 3: Implement JSON-safe records and rank-local writers**

```python
# src/smart/accelerated_testing/ledger.py
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class StepLedgerRecord:
    scenario_id: str
    rollout_id: int
    arm: str
    global_seed: int
    token_step: int
    future_frame_start: int
    future_frame_end: int
    pov_track_id: int
    support_token_ids: tuple[int, ...]
    log_p: tuple[float, ...]
    risk: tuple[float, ...]
    log_r: tuple[float, ...]
    log_q: tuple[float, ...]
    selected_token_id: int
    criticality: float
    epsilon: float
    beta: float
    log_ratio_increment: float
    cumulative_log_weight: float
    first_collision_in_token: bool
    rng_key: str


@dataclass(frozen=True)
class RolloutSummaryRecord:
    scenario_id: str
    rollout_id: int
    arm: str
    pov_selected: bool
    pov_track_id: Optional[int]
    collision: bool
    first_collision_time_s: Optional[float]
    collision_partner_id: Optional[int]
    collision_type: Optional[str]
    simultaneous_partner_ids: tuple[int, ...]
    initial_overlap: bool
    near_gap: bool
    near_tto: bool
    near_pet: bool
    near_required_decel: bool
    near_miss_union: bool
    pet_applicable: bool
    final_log_weight: float
    critical_steps: int
    trajectory_shard: str
    trajectory_row: int


@dataclass(frozen=True)
class FailureRecord:
    scenario_id: str
    rollout_id: int
    arm: str
    rank: int
    reason: str


def replay_log_weight(records: Sequence[StepLedgerRecord]) -> float:
    total = 0.0
    for record in sorted(records, key=lambda item: item.token_step):
        total += record.log_ratio_increment
        if abs(total - record.cumulative_log_weight) > 1e-10:
            raise ValueError("ledger cumulative_log_weight is inconsistent")
    return total
```

`RunArtifactWriter` must create
`.rank_shards/rank-{rank:03d}/`, stream JSONL rather than retain records in
memory, close every handle before a barrier, and never allow two ranks to
write the same path. `merge_rank_outputs` must:

- concatenate gzip members into root `step_ledger.jsonl.gz`;
- concatenate summary/failure JSONL in deterministic rank order;
- leave trajectory tensors in
  `trajectory_shards/rank-{rank:03d}-batch-{batch:06d}.pt`;
- write root `trajectories.pt` containing only format, relative shard paths,
  shard SHA-256 values, and per-trajectory content SHA-256 values;
- fail if a rank completion marker is missing.

At merge, key summaries by `(arm, scenario_id, rollout_id)`. Lightning may pad
validation shards with an exact duplicate scenario. Remove such a duplicate
only when its summary, ledger replay, and trajectory tensor SHA-256 are
identical; record every removed key in
`manifest.json["ddp_padding_duplicates_removed"]`. Conflicting duplicates are
a hard failure. This deterministic padding cleanup is not an event-dependent
rollout exclusion.

`future_frame_start` is inclusive and `future_frame_end` is exclusive in the
80-frame future tensor; for token step `t` they are `5*t` and `5*(t+1)`.
`rollout_summary.jsonl` references the complete 8 s trajectory through
`trajectory_shard` and `trajectory_row`; merging must verify that every
reference resolves and that every referenced trajectory has 80 future frames.

- [ ] **Step 4: Run ledger tests**

Run: `pytest -q tests/accelerated_testing/test_ledger.py`

Expected: all tests pass.

- [ ] **Step 5: Commit ledger**

```bash
git add src/smart/accelerated_testing/ledger.py \
  tests/accelerated_testing/test_ledger.py
git commit -m "feat: persist accelerated-testing audit ledger"
```

---

### Task 9: Scenario-balanced IS estimators and acceleration diagnostics

**Files:**
- Create: `src/smart/accelerated_testing/estimators.py`
- Create: `tests/accelerated_testing/test_estimators.py`

**Interfaces:**
- Consumes merged `RolloutSummaryRecord` values from baseline or fixed-IS runs.
- Produces: `EventEstimate`.
- Produces:
  `estimate_event(records, event_field, confidence_level, bootstrap_replicates, seed, target_rhw=0.30)`.
- Produces: `estimate_conditional_type_composition(...)` marked descriptive.
- Produces:
  `build_acceleration_report(baseline_records, proposal_records, baseline_elapsed_s, proposal_elapsed_s, config)`.

At load time, derive each record's ordinary importance weight as
`exp(final_log_weight)` in float64; baseline rows must have
`final_log_weight==0`. Reject rather than clip any non-finite derived weight.

- [ ] **Step 1: Write failing estimator and toy-model tests**

```python
# tests/accelerated_testing/test_estimators.py
import math

from src.smart.accelerated_testing.estimators import (
    estimate_event,
    effective_sample_size,
)


def test_scenarios_are_equal_weight_with_unequal_rollout_counts():
    records = [
        {"scenario_id": "a", "weight": 1.0, "collision": True},
        {"scenario_id": "a", "weight": 1.0, "collision": True},
        {"scenario_id": "a", "weight": 1.0, "collision": True},
        {"scenario_id": "b", "weight": 1.0, "collision": False},
    ]
    estimate = estimate_event(
        records,
        event_field="collision",
        confidence_level=0.90,
        bootstrap_replicates=200,
        seed=817,
    )
    assert estimate.value == 0.5


def test_ess_matches_definition():
    weights = [1.0, 2.0, 3.0]
    expected = sum(weights) ** 2 / sum(value * value for value in weights)
    assert effective_sample_size(weights) == expected


def test_ordinary_is_recovers_analytic_rare_event_probability():
    # P(event)=0.01, Q(event)=0.20.
    records = []
    for index in range(1000):
        event = index < 200
        weight = 0.01 / 0.20 if event else 0.99 / 0.80
        records.append(
            {
                "scenario_id": f"s-{index}",
                "weight": weight,
                "collision": event,
            }
        )
    estimate = estimate_event(
        records,
        event_field="collision",
        confidence_level=0.90,
        bootstrap_replicates=200,
        seed=817,
    )
    assert math.isclose(estimate.value, 0.01, abs_tol=1e-12)
```

- [ ] **Step 2: Run estimator tests and verify failure**

Run: `pytest -q tests/accelerated_testing/test_estimators.py`

Expected: import fails because `estimators.py` does not exist.

- [ ] **Step 3: Implement ordinary IS and hierarchical bootstrap**

```python
# src/smart/accelerated_testing/estimators.py
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EventEstimate:
    value: float
    lower: float
    upper: float
    analytic_standard_error: float
    relative_half_width: Optional[float]
    raw_rate: float
    ess: float
    ess_fraction: float
    event_ess: float
    mean_weight: float
    mean_weight_error: float
    max_weight: float
    weight_cv: float
    insufficient_events: bool
    underpowered: bool


def _scenario_balanced_value(records, event_field):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["scenario_id"]].append(record)
    scenario_values = []
    for rows in grouped.values():
        scenario_values.append(
            sum(row["weight"] * bool(row[event_field]) for row in rows)
            / len(rows)
        )
    return sum(scenario_values) / len(scenario_values)


def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=np.float64)
    return float(weights.sum() ** 2 / np.square(weights).sum())
```

For each of 2000 default bootstrap replicates, sample scenario IDs with
replacement and then sample that scenario's rollout rows with replacement.
Use percentile bounds at 5% and 95%. Compute absolute type rates with the same
function. Compute the analytic standard error as the sample standard deviation
of the per-scenario means of `weight * event`, divided by `sqrt(S)`. Compute
`event_ess` from the nonzero `weight * event` terms. Set `underpowered=true`
when the estimate is zero or its 90% relative half-width exceeds
`target_rhw`; this applies independently to every rare collision type. Put
conditional type composition in a separate field
`descriptive_conditional_type_composition` and never label it unbiased.
Compute `mean_weight` with the same scenario-equal two-level averaging (first
within scenario, then across scenarios), and expose `mean_weight_error` as
`mean_weight - 1.0`; do not replace event estimates with normalized weights.

Build precision curves from deterministic total sample sizes
`start, 2*start, 4*start, ..., N`, with
`start=max(32, number_of_scenarios)`; the first size whose bootstrap RHW is
`<=0.30` is the required rollout count. At each total size, allocate rows as
evenly as possible across scenarios, assigning
the remainder in SHA-256 order of scenario ID, and within each scenario take
the prefix in numeric rollout-ID order. Return `None` when the data do not
contain that balanced prefix or no size reaches the target.
Compute:

```text
statistical_acceleration =
    baseline_required_rollouts / proposal_required_rollouts
wall_clock_acceleration =
    baseline_required_rollouts * baseline_seconds_per_rollout
    / (proposal_required_rollouts * proposal_seconds_per_rollout)
```

Use a scenario-cluster bootstrap of the baseline-minus-proposal estimates for
the 90% difference interval. For each event independently,
`acceleration_success=true` only when proposal
RHW is at most 0.30, proposal ESS/N is at least 0.10, both acceleration ratios
exceed one, that difference interval contains zero, and neither arm has a
failure or underpowered target event. A calibration artifact whose status is
`conservative_fallback` (or a config using that source directly) always forces
`acceleration_success=false` with reason
`uncalibrated_criticality_threshold`. If any required rollout count is
undefined, report both the ratio and success claim as unavailable. Do not emit
clipped or self-normalized estimates in Phase 1.

The report must call the same ordinary-IS estimator for `collision`, each of
the five mutually exclusive first-collision type indicators, `near_gap`,
`near_tto`, `near_pet`, `near_required_decel`, and `near_miss_union`.
`initial_overlap` and `pet_applicable` are reported as diagnostics, not folded
into either safety event.

- [ ] **Step 4: Run estimator tests**

Run: `pytest -q tests/accelerated_testing/test_estimators.py`

Expected: all tests pass.

- [ ] **Step 5: Commit estimators**

```bash
git add src/smart/accelerated_testing/estimators.py \
  tests/accelerated_testing/test_estimators.py
git commit -m "feat: estimate unbiased accelerated-test risk"
```

---

### Task 10: Compose the fixed-IS rollout controller

**Files:**
- Create: `src/smart/accelerated_testing/controller.py`
- Create: `tests/accelerated_testing/test_controller.py`
- Modify: `src/smart/accelerated_testing/__init__.py`

**Interfaces:**
- Consumes every component from Tasks 1–8.
- Produces: `ControllerContext`.
- Produces: `TokenStepInput`.
- Produces: `TokenStepDecision(token_ids, local_contours)`.
- Produces: `FixedISController.sample_step(step_input)`.
- Produces: `FixedISController.observe_step(token_step, global_contours, valid)`.
- Produces: `FixedISController.finalize() -> tuple[list[StepLedgerRecord], list[RolloutSummaryRecord], list[CalibrationRollout]]`.

- [ ] **Step 1: Write failing multi-scenario controller tests**

```python
# tests/accelerated_testing/test_controller.py
import torch

from src.smart.accelerated_testing.config import AcceleratedTestingConfig
from src.smart.accelerated_testing.controller import (
    ControllerContext,
    FixedISController,
    TokenStepInput,
)


def _contours(center_x):
    center = torch.stack((center_x, torch.zeros_like(center_x)), dim=-1)
    offsets = torch.tensor(
        [[2.4, 1.0], [2.4, -1.0], [-2.4, -1.0], [-2.4, 1.0]]
    )
    return center[:, None, :] + offsets[None, :, :]


def _synthetic_step_input():
    time = torch.linspace(0.0, 1.0, 6)
    ego_tokens = torch.stack(
        (_contours(2.0 * time), _contours(3.0 * time), _contours(time))
    )
    pov_tokens = torch.stack(
        (_contours(torch.zeros_like(time)), _contours(-6.0 * time), _contours(2.0 * time))
    )
    return TokenStepInput(
        token_step=0,
        logits=torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]),
        positions=torch.tensor([[0.0, 0.0], [8.0, 0.0]]),
        headings=torch.zeros(2),
        valid=torch.ones(2, dtype=torch.bool),
        agent_ids=torch.tensor([100, 10]),
        agent_batch=torch.zeros(2, dtype=torch.long),
        agent_types=torch.zeros(2, dtype=torch.long),
        ego_mask=torch.tensor([True, False]),
        shapes=torch.tensor([[4.8, 2.0, 1.5], [4.8, 2.0, 1.5]]),
        token_traj_all=torch.stack((ego_tokens, pov_tokens)),
    )


def test_only_locked_pov_uses_q_and_weight_is_exact():
    step_input = _synthetic_step_input()
    config = AcceleratedTestingConfig.from_mapping(
        {
            "enabled": True,
            "mode": "evaluate",
            "pov": {"criticality_threshold": 0.0},
            "proposal": {"epsilon": 0.05, "beta": 1.0, "frozen": True},
        }
    )
    controller = FixedISController(
        config=config,
        context=ControllerContext(
            scenario_ids=("s0",),
            rollout_id=0,
            global_seed=817,
            arm="proposal",
        ),
    )
    decision = controller.sample_step(step_input)
    assert decision.token_ids.shape == step_input.logits.shape[:1]
    assert decision.pov_track_ids == (step_input.agent_ids[1].item(),)
    changed = [
        row for row in decision.audit_rows if row.log_ratio_increment != 0.0
    ]
    assert len(changed) <= 1


def test_calibration_mode_never_changes_p():
    step_input = _synthetic_step_input()
    config = AcceleratedTestingConfig.from_mapping(
        {"enabled": True, "mode": "calibrate"}
    )
    controller = FixedISController(
        config=config,
        context=ControllerContext(
            scenario_ids=("s0",),
            rollout_id=0,
            global_seed=817,
            arm="baseline",
        ),
    )
    decision = controller.sample_step(step_input)
    assert all(row.log_ratio_increment == 0.0 for row in decision.audit_rows)


def test_controller_input_has_no_future_gt_fields():
    assert not any(
        "gt" in name.lower() for name in TokenStepInput.__dataclass_fields__
    )
```

- [ ] **Step 2: Run controller tests and verify failure**

Run: `pytest -q tests/accelerated_testing/test_controller.py`

Expected: import fails because `controller.py` does not exist.

- [ ] **Step 3: Implement the batched controller state machine**

```python
# src/smart/accelerated_testing/controller.py
from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from .ledger import StepLedgerRecord


@dataclass(frozen=True)
class ControllerContext:
    scenario_ids: tuple[str, ...]
    rollout_id: int
    global_seed: int
    arm: str


@dataclass(frozen=True)
class TokenStepInput:
    token_step: int
    logits: Tensor
    positions: Tensor
    headings: Tensor
    valid: Tensor
    agent_ids: Tensor
    agent_batch: Tensor
    agent_types: Tensor
    ego_mask: Tensor
    shapes: Tensor
    token_traj_all: Tensor


@dataclass(frozen=True)
class TokenStepDecision:
    token_ids: Tensor
    local_contours: Tensor
    pov_track_ids: tuple[Optional[int], ...]
    audit_rows: tuple[StepLedgerRecord, ...]
```

`FixedISController` must create one `SinglePOVSelector` and one
`EgoSafetyEventTracker` per scenario in `ControllerContext`. For every step:

1. validate `arm in {"baseline", "proposal"}` and construct baseline `p` for
   every agent;
2. for proposal or calibration rollouts, compute ego Top-8 and candidate POV
   risks only within each scenario; skip risk scoring in a pure baseline arm;
3. call each selector with per-agent criticality only when risk scoring ran;
4. apply `q` only when `arm=="proposal"` and to a locked, valid POV whose
   current \(C_i\) meets threshold;
5. use `stable_uniform` for every agent's sample under either `p` or `q`;
6. add `log_p-log_q` only for the modified POV;
7. store support vectors only for actual `q!=p` steps;
8. in calibration mode or the baseline arm, always sample `p`; calibration
   additionally retains per-step maximum criticality;
9. after `observe_step` finds first ego collision, force later `q=p`;
10. raise immediately on non-finite values, malformed ego masks, or support
    violations. Before sampling, require `exp(log_p)`, `exp(log_r)`, and
    `exp(log_q)` to sum to one within `1e-6`, and require every baseline
    support entry to have finite `log_p` and `log_q`.

The controller must split by `agent_batch` without assuming equal agent counts
or a particular scenario order.

- [ ] **Step 4: Run controller tests**

Run: `pytest -q tests/accelerated_testing/test_controller.py tests/accelerated_testing/test_proposal.py tests/accelerated_testing/test_pov_selector.py`

Expected: all tests pass.

- [ ] **Step 5: Commit controller**

```bash
git add src/smart/accelerated_testing/__init__.py \
  src/smart/accelerated_testing/controller.py \
  tests/accelerated_testing/test_controller.py
git commit -m "feat: orchestrate fixed-IS CAT-K rollouts"
```

---

### Task 11: Inject the controller into CAT-K inference

**Files:**
- Modify: `src/smart/tokens/token_processor.py:240-254`
- Modify: `src/smart/modules/agent_decoder.py:565-902`
- Modify: `src/smart/modules/smart_decoder.py:83-93`
- Create: `tests/accelerated_testing/test_model_integration.py`

**Interfaces:**
- Consumes: optional `rollout_controller: FixedISController | None`.
- Produces module helper:
  `select_next_token_for_inference(legacy_kwargs, rollout_controller, step_input)`.
- Preserves: all existing call sites when controller is `None`.
- Adds to accelerated output: `out_dict["accelerated_testing"]`.

- [ ] **Step 1: Write failing hook and legacy-path tests**

```python
# tests/accelerated_testing/test_model_integration.py
from unittest.mock import Mock, patch

import torch

from src.smart.modules.agent_decoder import select_next_token_for_inference
from src.smart.tokens.token_processor import TokenProcessor


class FakeBatch(dict):
    num_graphs = 1


def test_token_processor_exposes_stable_agent_ids():
    processor = TokenProcessor.__new__(TokenProcessor)
    torch.nn.Module.__init__(processor)
    processor.shift = 5
    processor.history_dynamics_active = False
    processor.future_token_dynamics_active = False
    processor.eval()
    tokens = torch.zeros(2, 3, 6, 4, 2)
    processor._get_agent_shape_and_token_traj = Mock(
        return_value=(torch.ones(2, 2), tokens, tokens[:, :, -1])
    )
    processor._clean_heading = Mock(side_effect=lambda valid, heading: heading)
    processor._extrapolate_agent_to_prev_token_step = Mock(
        side_effect=lambda valid, pos, heading, velocity: (
            valid,
            pos,
            heading,
            velocity,
        )
    )
    processor._match_agent_token = Mock(return_value={})
    for name in ("veh", "ped", "cyc"):
        setattr(processor, f"agent_token_all_{name}", torch.zeros(3, 6, 4, 2))
    batch = FakeBatch(
        agent={
            "id": torch.tensor([11, 12]),
            "type": torch.tensor([0, 0]),
            "shape": torch.ones(2, 3),
            "role": torch.tensor([[True, False, False], [False, False, True]]),
            "batch": torch.zeros(2, dtype=torch.long),
            "valid_mask": torch.ones(2, 91, dtype=torch.bool),
            "heading": torch.zeros(2, 91),
            "position": torch.zeros(2, 91, 3),
            "velocity": torch.zeros(2, 91, 2),
        }
    )
    agents = processor.tokenize_agent(batch)
    assert torch.equal(agents["id"], batch["agent"]["id"])


def test_optional_sampler_uses_controller_decision():
    controller = Mock()
    expected = torch.tensor([2, 1])
    local_contours = torch.zeros(2, 6, 4, 2)
    controller.sample_step.return_value.token_ids = expected
    controller.sample_step.return_value.local_contours = local_contours
    selected_ids, selected_contours = select_next_token_for_inference(
        legacy_kwargs={},
        rollout_controller=controller,
        step_input=object(),
    )
    assert torch.equal(selected_ids, expected)
    assert selected_contours is local_contours
    controller.sample_step.assert_called_once()


def test_none_controller_keeps_legacy_sampler():
    expected = (torch.tensor([0]), torch.zeros(1, 6, 4, 2))
    with patch(
        "src.smart.modules.agent_decoder.sample_next_token_traj",
        return_value=expected,
    ) as legacy:
        result = select_next_token_for_inference(
            legacy_kwargs={"token_traj": torch.zeros(1, 1, 4, 2)},
            rollout_controller=None,
            step_input=None,
        )
    assert result is expected
    legacy.assert_called_once()
```

- [ ] **Step 2: Run integration tests and verify failure**

Run: `pytest -q tests/accelerated_testing/test_model_integration.py`

Expected: `rollout_controller` is not accepted and `tokenized_agent["id"]` is
missing.

- [ ] **Step 3: Add the optional hook without altering the default path**

Add:

```python
# token_processor.py, tokenized_agent
"id": data["agent"]["id"],
```

Change both decoder signatures to:

```python
def inference(
    self,
    tokenized_map,
    tokenized_agent,
    sampling_scheme,
    rollout_controller=None,
):
```

Inside `SMARTAgentDecoder.inference`, use the current legacy
`sample_next_token_traj` block verbatim when the controller is `None`.
Implement the module helper as:

```python
def select_next_token_for_inference(
    *,
    legacy_kwargs,
    rollout_controller,
    step_input,
):
    if rollout_controller is None:
        return sample_next_token_traj(**legacy_kwargs)
    decision = rollout_controller.sample_step(step_input)
    return decision.token_ids, decision.local_contours
```

The inference loop constructs `TokenStepInput`, calls this helper, transforms
the returned contours to global as today, and then calls:

```python
rollout_controller.observe_step(
    token_step=t,
    global_contours=token_traj_global,
    valid=pred_valid[:, t_now],
)
```

After the loop:

```python
if rollout_controller is not None:
    out_dict["accelerated_testing"] = rollout_controller.finalize()
```

Do not pass `gt_pos_raw`, `gt_head_raw`, or `gt_valid_raw` into the controller.

- [ ] **Step 4: Run integration and model-adjacent tests**

Run: `pytest -q tests/accelerated_testing/test_model_integration.py tests/test_endpoint_interpolation.py tests/test_future_token_dynamics.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the inference hook**

```bash
git add src/smart/tokens/token_processor.py \
  src/smart/modules/agent_decoder.py \
  src/smart/modules/smart_decoder.py \
  tests/accelerated_testing/test_model_integration.py
git commit -m "feat: inject safety proposal into CAT-K inference"
```

---

### Task 12: Runner lifecycle, DDP publication, visualization, and command

**Files:**
- Create: `src/smart/accelerated_testing/runner.py`
- Create: `scripts/accelerated_test.sh`
- Create: `tests/accelerated_testing/test_runner.py`
- Create: `tests/accelerated_testing/test_shell_entrypoint.py`
- Create: `tests/accelerated_testing/test_visualization.py`
- Modify: `src/smart/model/smart.py:39-108,132-307`
- Modify: `src/run.py:38-105`
- Modify: `src/utils/vis_waymo.py:278-299`
- Modify: `README.md:235-275`

**Interfaces:**
- Produces: `AcceleratedTestingRunner.new_controller(...)`.
- Produces: `AcceleratedTestingRunner.record_batch(...)`.
- Produces: `AcceleratedTestingRunner.finalize_rank(...)`.
- Produces: `AcceleratedTestingRunner.publish_rank_zero(...)`.
- Produces: W&B-safe scalar metric dictionary under `accelerated/*`.

- [ ] **Step 1: Write failing runner, DDP, video, and shell tests**

```python
# tests/accelerated_testing/test_runner.py
from pathlib import Path

import pytest

from src.smart.accelerated_testing.runner import AcceleratedTestingRunner


def test_evaluate_mode_has_independent_baseline_and_proposal_arms(
    tmp_path: Path,
):
    runner = AcceleratedTestingRunner.for_test(
        root=tmp_path,
        rank=0,
        world_size=1,
        mode="evaluate",
    )
    assert runner.arms == ("baseline", "proposal")
    assert runner.rng_rollout_id("baseline", 3) == 6
    assert runner.rng_rollout_id("proposal", 3) == 7


def test_runner_publishes_only_after_all_rank_markers(tmp_path: Path):
    runner = AcceleratedTestingRunner.for_test(
        root=tmp_path,
        rank=0,
        world_size=2,
    )
    runner.writer.close_rank()
    with pytest.raises(RuntimeError, match="rank 1"):
        runner.publish_rank_zero()


def test_failed_rollout_prevents_formal_report(tmp_path: Path):
    runner = AcceleratedTestingRunner.for_test(
        root=tmp_path,
        rank=0,
        world_size=1,
    )
    runner.record_failure(
        scenario_id="s",
        rollout_id=0,
        arm="proposal",
        reason="non-finite log_q",
    )
    runner.finalize_rank()
    with pytest.raises(RuntimeError, match="formal evaluation failed"):
        runner.publish_rank_zero()
```

```python
# tests/accelerated_testing/test_shell_entrypoint.py
from pathlib import Path
import subprocess


def test_accelerated_test_script_is_valid_bash():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        ["bash", "-n", str(root / "scripts/accelerated_test.sh")],
        check=True,
    )
```

Add a visualization unit test that passes collision step 50 and asserts the
rollout video frame selection contains exactly `11+50` frames while the
trajectory object still contains 80 future states:

```python
# tests/accelerated_testing/test_visualization.py
import torch

from src.utils.vis_waymo import frames_until_collision


def test_video_stops_at_collision_without_truncating_trajectory():
    images = tuple(range(91))
    trajectory = torch.zeros(1, 80, 3)
    selected = frames_until_collision(
        images,
        step_current=10,
        stop_future_step=50,
    )
    assert len(selected) == 61
    assert trajectory.shape[1] == 80
```

- [ ] **Step 2: Run runner tests and verify failure**

Run: `pytest -q tests/accelerated_testing/test_runner.py tests/accelerated_testing/test_shell_entrypoint.py`

Expected: imports/files fail because runner and script do not exist.

- [ ] **Step 3: Implement runner lifecycle and SMART validation branch**

`SMART.__init__` must parse and store `model_config.accelerated_testing`, but
must not open files there because Lightning has not assigned DDP rank yet.
`configure_accelerated_run_metadata` stores the resolved config and checkpoint
path. In `on_validation_start`, after `trainer.global_rank` and
`trainer.world_size` exist, rank zero initializes the new artifact root at
`Path(HydraConfig.get().runtime.output_dir) / "accelerated_testing"`, all
ranks cross a barrier, and each rank constructs its own writer/runner. Before
opening a writer, require the formal baseline `criterium`, `num_k`, and
`temperature` to equal `validation_rollout_sampling.criterium`, `.num_k`, and
`.temp` exactly; a mismatch changes the target distribution and is a hard
error.

Keep the existing closed-loop validation loop unchanged when the runner is
absent. When it is present, run the arms returned by the runner:

```python
predictions_by_arm = defaultdict(list)
for rollout_idx in range(self.n_rollout_closed_val):
    for arm in self.accelerated_runner.arms:
        controller = self.accelerated_runner.new_controller(
            batch_idx=batch_idx,
            rollout_idx=rollout_idx,
            arm=arm,
            scenario_ids=tuple(data["scenario_id"]),
            agent_ids=tokenized_agent["id"],
            agent_batch=tokenized_agent["batch"],
        )
        self.accelerated_runner.start_timing(arm)
        pred = self.encoder.inference(
            tokenized_map,
            tokenized_agent,
            self.validation_rollout_sampling,
            rollout_controller=controller,
        )
        self.accelerated_runner.stop_timing(arm)
        predictions_by_arm[arm].append(pred)
```

Wrap each batched arm inference in `try/except Exception`. On failure, write
one `FailureRecord` for every scenario in that batch/arm, mark the batch
unpublishable, and continue only if the process remains healthy; never create
a summary row for the missing rollout and never estimate after dropping it.
For CUDA OOM, distributed-collective failure, or another unrecoverable
exception, flush the rank-local failure file and re-raise so the run terminates
without `report.json`.

The arm set is `("baseline",)` in `calibrate` or `baseline` mode and
`("baseline", "proposal")` in `evaluate` mode. A baseline controller always
samples \(p\); a proposal controller follows the frozen fixed-IS policy.
Assign disjoint formal RNG rollout IDs (`2*rollout_idx` for baseline and
`2*rollout_idx+1` for proposal) so both arms are independent while retaining
the approved five-part RNG key. Thus `n_rollout_closed_val=32` means 32
rollouts per arm in evaluate mode.

After stacking each arm, call `runner.record_batch(arm=arm, ...)`. Timing must
bracket only inference plus proposal/event bookkeeping, synchronize CUDA
immediately before and after the bracket, and store elapsed seconds and
rollout counts for each arm. At merge, arm wall time is the maximum elapsed
time across ranks (not the sum), while arm rollout count is summed; both arms
must use the same world size and batch assignment. When acceleration is
enabled, skip minADE/Fast
WOSAC updates; these GT-based realism metrics are not part of the formal
safety estimate. Normal validation remains byte-for-byte on the old branch
when disabled.

At `on_validation_epoch_end`:

1. close rank-local writers;
2. call `self.trainer.strategy.barrier("accelerated-testing-rank-close")`;
3. rank zero merges shards and either writes the calibration artifact or
   `report.json`; evaluate mode passes the merged baseline and proposal rows
   plus their measured times to `build_acceleration_report`;
4. call a second barrier before returning;
5. log only finite scalar values under `accelerated/`.

Calibration mode writes
`accelerated_testing/criticality_calibration.json` and prints both its absolute
path and SHA-256. Baseline mode writes event estimates but never an
acceleration-success claim. Evaluate mode writes both arm estimates and the
comparison report.

Change its OmegaConf import to
`from omegaconf import DictConfig, OmegaConf`, then `src/run.py` must call:

```python
if hasattr(model, "configure_accelerated_run_metadata"):
    model.configure_accelerated_run_metadata(
        checkpoint_path=cfg.get("ckpt_path"),
        resolved_config=OmegaConf.to_container(cfg, resolve=True),
    )
```

before `trainer.validate`. Rank zero writes an immutable `manifest.json`
containing checkpoint SHA-256, agent-vocabulary SHA-256, resolved-config value
and SHA-256, Git commit, dirty state and `git diff` SHA-256, sorted-unique
scenario-list SHA-256,
RNG-key
rule, K/T/epsilon/beta/z-clip/criticality threshold, every event/classification
threshold, calibration-artifact SHA-256, proposal frozen state, and Python,
PyTorch, CUDA, driver, and GPU metadata. Formal evaluation must fail if the
proposal is not frozen, the calibration artifact hash mismatches, an artifact
directory already contains a published manifest, the baseline and proposal
scenario sets differ, or any rank reports a failed rollout.

- [ ] **Step 4: Implement first-collision video truncation and shell command**

Change:

```python
def save_video_scenario_rollout(
    self,
    scenario_rollout,
    n_vis_rollout,
    stop_future_steps=None,
):
```

For rollout `i`, use all frames when its stop is `None`; otherwise write
`frames_until_collision(images, self.step_current, stop_future_steps[i])`,
where the pure helper returns
`images[: step_current + 1 + stop_future_step]`. Do not truncate or mutate the
`ScenarioRollouts` object.

`scripts/accelerated_test.sh` must:

- use `set -euo pipefail`;
- require a readable `CATK_CKPT`;
- require `${CACHE_ROOT}/validation` for evaluate mode;
- require `${CACHE_ROOT}/training` for calibrate mode;
- use `NUM_GPUS`, `MY_EXPERIMENT`, and `MY_TASK_NAME`;
- invoke
  `torchrun --standalone --nproc_per_node="${NUM_GPUS}" -m src.run experiment="${MY_EXPERIMENT}" action=validate`;
- default `MY_EXPERIMENT=accelerated_testing`;
- run `unset WANDB_RUN_ID WANDB_RESUME` before `torchrun`, so it never resumes
  a prior pre-BC/CLSFT W&B run.

README commands:

```bash
export CATK_CKPT=/root/workspace/catk/logs/pre_bc_history_dynamics_trajtok_original_b200/runs/2026-07-27_19-49-13/checkpoints/last.ckpt
export CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact
export NUM_GPUS=8

MY_EXPERIMENT=accelerated_testing_calibrate \
MY_TASK_NAME=accelerated_testing_calibrate \
bash scripts/accelerated_test.sh

MY_EXPERIMENT=accelerated_testing \
MY_TASK_NAME=accelerated_testing_eval \
bash scripts/accelerated_test.sh \
  model.model_config.accelerated_testing.pov.criticality_threshold=0.0 \
  model.model_config.accelerated_testing.pov.criticality_threshold_source=conservative_fallback

# For a claim-bearing run, select the artifact produced by the command above:
export CRITICALITY_CALIBRATION_FILE="$(find /root/workspace/catk/logs/accelerated_testing_calibrate/runs -type f -path '*/accelerated_testing/criticality_calibration.json' -print | sort | tail -n 1)"
test -f "${CRITICALITY_CALIBRATION_FILE}"
export CRITICALITY_CALIBRATION_SHA256="$(sha256sum "${CRITICALITY_CALIBRATION_FILE}" | awk '{print $1}')"
export CRITICALITY_THRESHOLD="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"threshold\"])' "${CRITICALITY_CALIBRATION_FILE}")"

MY_EXPERIMENT=accelerated_testing \
MY_TASK_NAME=accelerated_testing_eval_calibrated \
bash scripts/accelerated_test.sh \
  model.model_config.accelerated_testing.pov.criticality_threshold="${CRITICALITY_THRESHOLD}" \
  model.model_config.accelerated_testing.pov.criticality_threshold_source="${CRITICALITY_CALIBRATION_SHA256}" \
  model.model_config.accelerated_testing.pov.criticality_calibration_file="${CRITICALITY_CALIBRATION_FILE}"
```

Document that the zero threshold is a conservative fallback and that formal
claims should instead pass the numeric threshold and SHA-256 from the training
calibration artifact. Also state explicitly that “unbiased” targets the frozen
CAT-K baseline distribution, not real-world crash frequency; raw proposal
collision-type counts may be skewed, while only the ordinary-IS absolute type
probabilities target the baseline. Conditional type composition remains a
descriptive ratio estimate.

- [ ] **Step 5: Run runner, config, and visualization tests**

Run: `pytest -q tests/accelerated_testing/test_runner.py tests/accelerated_testing/test_shell_entrypoint.py tests/accelerated_testing/test_visualization.py tests/accelerated_testing/test_config.py tests/test_training_fast_wosac_config.py`

Expected: all tests pass.

- [ ] **Step 6: Commit runtime integration**

```bash
git add src/smart/accelerated_testing/runner.py \
  src/smart/model/smart.py \
  src/run.py \
  src/utils/vis_waymo.py \
  configs/model/smart.yaml \
  configs/experiment/accelerated_testing.yaml \
  configs/experiment/accelerated_testing_calibrate.yaml \
  scripts/accelerated_test.sh \
  README.md \
  tests/accelerated_testing/test_runner.py \
  tests/accelerated_testing/test_shell_entrypoint.py \
  tests/accelerated_testing/test_visualization.py
git commit -m "feat: run distributed accelerated safety evaluation"
```

---

### Task 13: End-to-end correctness and full regression gate

**Files:**
- Create: `tests/accelerated_testing/test_end_to_end.py`
- Modify: `src/smart/accelerated_testing/__init__.py`
- Modify: `README.md`

**Interfaces:**
- Verifies the complete Phase 1 public behavior.
- Does not add a second implementation path.

- [ ] **Step 1: Write the end-to-end epsilon-zero and fixed-IS tests**

```python
# tests/accelerated_testing/test_end_to_end.py
from dataclasses import dataclass, replace

import torch

from src.smart.accelerated_testing.estimators import EventEstimate, estimate_event
from src.smart.accelerated_testing.proposal import (
    build_risk_tilted_proposal,
    selected_log_ratio,
)
from src.smart.accelerated_testing.token_distribution import (
    build_topk_token_distribution,
    sample_token_ids_from_uniform,
)


@dataclass(frozen=True)
class ToyRun:
    selected_tokens: tuple[int, ...]
    cumulative_log_weights: tuple[float, ...]
    records: tuple[dict, ...]
    collision_estimate: EventEstimate


def _run_analytic_two_action_model(
    *,
    use_proposal: bool,
    epsilon: float,
    rollouts: int = 4096,
) -> ToyRun:
    # Token 1 is a collision. Under p, P(token 1)=0.01 exactly.
    logits = torch.log(
        torch.tensor([[0.99, 0.01]], dtype=torch.float64)
    ).repeat(rollouts, 1)
    baseline = build_topk_token_distribution(
        logits,
        num_k=2,
        temperature=1.0,
    )
    risk = (baseline.token_ids == 1).to(torch.float64)
    proposal = build_risk_tilted_proposal(
        baseline,
        risk,
        epsilon=epsilon,
        beta=5.0,
        z_clip=5.0,
    )
    sampling_distribution = (
        replace(baseline, log_probs=proposal.log_q)
        if use_proposal
        else baseline
    )
    uniforms = (
        torch.arange(rollouts, dtype=torch.float64) + 0.5
    ) / rollouts
    selected = sample_token_ids_from_uniform(
        sampling_distribution,
        uniforms,
    )
    log_ratio = (
        selected_log_ratio(baseline, proposal, selected)
        if use_proposal
        else torch.zeros(rollouts, dtype=torch.float64)
    )
    records = []
    collision_count = 0
    for index, (token_id, log_weight) in enumerate(
        zip(selected.tolist(), log_ratio.tolist())
    ):
        collision = token_id == 1
        collision_count += int(collision)
        collision_type = (
            "rear_end" if collision_count % 2 == 0 else "angle"
        )
        records.append(
            {
                "scenario_id": f"toy-{index:05d}",
                "weight": float(
                    torch.exp(torch.tensor(log_weight, dtype=torch.float64))
                ),
                "collision": collision,
                "collision_rear_end": collision
                and collision_type == "rear_end",
                "collision_angle": collision and collision_type == "angle",
            }
        )
    estimate = estimate_event(
        records,
        event_field="collision",
        confidence_level=0.90,
        bootstrap_replicates=64,
        seed=817,
    )
    return ToyRun(
        selected_tokens=tuple(selected.tolist()),
        cumulative_log_weights=tuple(log_ratio.tolist()),
        records=tuple(records),
        collision_estimate=estimate,
    )


def test_epsilon_zero_reduces_to_baseline():
    baseline = _run_analytic_two_action_model(
        use_proposal=False,
        epsilon=0.0,
    )
    epsilon_zero = _run_analytic_two_action_model(
        use_proposal=True,
        epsilon=0.0,
    )
    assert epsilon_zero.selected_tokens == baseline.selected_tokens
    assert all(value == 0.0 for value in epsilon_zero.cumulative_log_weights)
    assert (
        epsilon_zero.collision_estimate.value
        == baseline.collision_estimate.value
    )


def test_fixed_is_changes_raw_rate_but_recovers_baseline_probability():
    baseline = _run_analytic_two_action_model(
        use_proposal=False,
        epsilon=0.0,
    )
    proposal = _run_analytic_two_action_model(
        use_proposal=True,
        epsilon=0.20,
    )
    assert (
        proposal.collision_estimate.raw_rate
        > baseline.collision_estimate.raw_rate
    )
    difference = (
        proposal.collision_estimate.value
        - baseline.collision_estimate.value
    )
    assert abs(difference) < 0.002
    assert (
        proposal.collision_estimate.lower
        <= 0.01
        <= proposal.collision_estimate.upper
    )
    assert proposal.collision_estimate.ess_fraction >= 0.10


def test_type_absolute_rates_sum_to_collision_rate():
    result = _run_analytic_two_action_model(
        use_proposal=True,
        epsilon=0.20,
    )
    type_total = sum(
        estimate_event(
            result.records,
            event_field=field,
            confidence_level=0.90,
            bootstrap_replicates=64,
            seed=817,
        ).value
        for field in ("collision_rear_end", "collision_angle")
    )
    assert abs(type_total - result.collision_estimate.value) < 1e-12
```

- [ ] **Step 2: Run the new end-to-end test and fix only integration defects**

Run: `pytest -q tests/accelerated_testing/test_end_to_end.py`

Expected before final fixes: failures identify integration mismatches, not
missing product behavior.

Apply the smallest changes in the owning module; do not duplicate logic inside
the analytic test harness.

- [ ] **Step 3: Run the complete accelerated-testing suite**

Run: `pytest -q tests/accelerated_testing`

Expected: all accelerated-testing tests pass.

- [ ] **Step 4: Run the existing CatK regression suite**

Run: `pytest -q`

Expected: the complete existing suite passes. Environment-dependent tests may
skip for missing optional GPU/Waymo packages, but no previously passing test
may fail.

- [ ] **Step 5: Run static and configuration checks**

Run:

```bash
bash -n scripts/accelerated_test.sh
python -m compileall -q src/smart/accelerated_testing
python -m src.run --cfg job experiment=accelerated_testing \
  ckpt_path=/tmp/nonexistent-but-config-only.ckpt
git diff --check
```

Expected:

- shell and Python syntax checks return zero;
- Hydra prints a resolved config with `topk_prob`, K=48, epsilon=0.05, beta=1.0;
- config printing does not try to open the checkpoint;
- `git diff --check` is clean.

- [ ] **Step 6: Verify the formal artifact contract on a tiny real batch**

On the target training machine, run one validation batch and two rollouts:

```bash
CATK_CKPT=/root/workspace/catk/logs/pre_bc_history_dynamics_trajtok_original_b200/runs/2026-07-27_19-49-13/checkpoints/last.ckpt \
CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact \
NUM_GPUS=1 \
MY_EXPERIMENT=accelerated_testing \
MY_TASK_NAME=accelerated_testing_smoke \
bash scripts/accelerated_test.sh \
  trainer.limit_val_batches=1 \
  model.model_config.n_rollout_closed_val=2
```

Expected output directory contains:

```text
accelerated_testing/
├── manifest.json
├── step_ledger.jsonl.gz
├── rollout_summary.jsonl
├── trajectories.pt
├── trajectory_shards/
├── report.json
└── failures.jsonl
```

Verify `report.json` identifies `conservative_fallback`, all weights are finite,
`trajectories.pt` indexes two baseline plus two proposal trajectories for every
scenario in the batch, and any collision video stops at its first collision
while its tensor trajectory remains 80 frames. The fallback run must set
`acceleration_success=false` with reason
`uncalibrated_criticality_threshold`.

- [ ] **Step 7: Commit the acceptance gate**

```bash
git add tests/accelerated_testing/test_end_to_end.py \
  src/smart/accelerated_testing/__init__.py \
  README.md
git commit -m "test: verify unbiased accelerated safety testing"
```

---

## Final Verification Checklist

- [ ] `accelerated_testing.enabled=false` uses the legacy `Categorical` sampler and current WOSAC validation.
- [ ] Evaluation config rejects both future-GT sampling criteria.
- [ ] Baseline \(p\) exactly reflects configured Top-K and temperature.
- [ ] \(q\) never has less support than \(p\), and no support-only token is injected.
- [ ] Only one vehicle is locked per scenario-rollout and no replacement occurs.
- [ ] Proposal parameters and calibration threshold are frozen and hashed.
- [ ] Every changed action has complete support-level `log_p`, risk, `log_r`, and `log_q`.
- [ ] Ledger replay reconstructs each final log weight within `1e-10`.
- [ ] Collision uses strict signed distance `<0` and first collision only.
- [ ] Initial overlap suppression, simultaneous partners, and all five collision classes are covered.
- [ ] Near-miss subconditions are separately reported and collision-exclusive.
- [ ] Absolute type rates use ordinary IS; conditional type composition is labeled descriptive.
- [ ] Scenario-balanced estimates remain correct with unequal rollout counts.
- [ ] 90% cluster-bootstrap CI, RHW, ESS, weight diagnostics, and acceleration ratio are present.
- [ ] `ESS/N<0.10`, unresolved failures, or insufficient events prevent an acceleration-success claim.
- [ ] DDP ranks never share writable files and rank zero publishes only after every completion marker.
- [ ] Complete 8 s trajectories are retained even when videos stop at collision.
- [ ] Existing Fast WOSAC backend files and numeric parity tests remain unchanged.
- [ ] Full pytest suite passes before handoff.
