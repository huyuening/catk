# Future-Token Dynamics Lookup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Causally condition CatK predictions after the first future token on `[a_lon, angular_speed, a_lat]` derived from the selected six-frame vocabulary token, with matched original- and reconstructed-vocabulary experiments.

**Architecture:** Add a pure Torch lookup builder beside the token code and have `TokenProcessor` register one non-persistent lookup buffer per agent class. Add a small optional conditioner module owned by `SMARTAgentDecoder`; it masks the two history positions during teacher forcing and adds the selected token's dynamics only after rollout logits have selected that token. Pass one shared configuration through `SMART`, `SMARTDecoder`, and `SMARTAgentDecoder`, then expose six experiment configurations that differ only in vocabulary selection within each matched pair.

**Tech Stack:** Python 3.11, PyTorch, Lightning, Hydra/OmegaConf, PyTorch Geometric, `unittest`, YAML.

## Global Constraints

- Future lookup order is exactly `[a_lon, angular_speed, a_lat]`.
- Use six vocabulary frames at 10 Hz and second-order finite differences with `dt=0.1`.
- Compute heading from contour corner 3 to corner 0 and unwrap before differentiation.
- Project acceleration into the endpoint body frame and clip to `[15, 3, 15]`.
- Do not call the trajectory reconstructor while building the lookup.
- Positions 0 and 1 receive no future-token dynamics in open-loop decoding.
- A selected rollout token affects only the following prediction.
- Keep raw-history dynamics unchanged and active in all six new experiments.
- Keep `future_token_dynamics.is_active=false` in the base model so legacy checkpoints retain their parameter keys and behavior.
- The reconstructed experiment variants override only `agent_token_file`.
- Do not stage or modify the user's existing `scripts/cache_womd.sh`, `.DS_Store`, or `.codex_work/` changes.

---

### Task 1: Vocabulary dynamics lookup

**Files:**
- Create: `src/smart/tokens/future_token_dynamics.py`
- Create: `tests/test_future_token_dynamics.py`
- Modify: `src/smart/tokens/token_processor.py`

**Interfaces:**
- Consumes: `token_trajectory: Tensor[n_token, 6, 4, 2]`
- Produces: `build_future_token_dynamics_lookup(token_trajectory, dt=0.1, clipping_limits=(15.0, 3.0, 15.0), context=None) -> Tensor[n_token, 3]`
- Produces: `gather_future_token_dynamics(token_index, agent_type, dynamics_veh, dynamics_ped, dynamics_cyc) -> Tensor[..., 3]`
- Produces tokenized-agent keys `agent_token_dynamics_veh`, `agent_token_dynamics_ped`, and `agent_token_dynamics_cyc` when the branch is active.

- [ ] **Step 1: Write lookup tests that fail because the module is absent**

Create fixtures that convert literal centers and headings into CatK's four-corner convention. Add tests with hand-derived expectations:

```python
def test_constant_acceleration_has_longitudinal_dynamics_only():
    time = torch.arange(6, dtype=torch.float64) * 0.1
    center = torch.stack((2.0 * time + time.square(), torch.zeros_like(time)), -1)
    token = contours(center, torch.zeros(6)).unsqueeze(0)
    lookup = build_future_token_dynamics_lookup(token)
    torch.testing.assert_close(
        lookup,
        torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )
```

Cover constant-radius motion, `-pi/pi` heading crossing, rigid rotation and translation, dtype and token order, class-specific gathering, clipping, invalid shapes, and non-finite input.

- [ ] **Step 2: Run the lookup tests and verify RED**

Run:

```bash
python -m unittest tests.test_future_token_dynamics -v
```

Expected: import failure for `src.smart.tokens.future_token_dynamics`.

- [ ] **Step 3: Implement second-order differentiation, heading unwrap, lookup construction, and class gathering**

Use explicit second-order endpoint formulas matching `numpy.gradient(..., edge_order=2)`:

```python
def _second_order_gradient(values: Tensor, dt: float) -> Tensor:
    gradient = torch.empty_like(values)
    gradient[:, 0] = (-3 * values[:, 0] + 4 * values[:, 1] - values[:, 2]) / (2 * dt)
    gradient[:, -1] = (3 * values[:, -1] - 4 * values[:, -2] + values[:, -3]) / (2 * dt)
    gradient[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2 * dt)
    return gradient
```

Unwrap adjacent heading differences into `[-pi, pi)`, differentiate center twice and heading once, project only the final acceleration with final heading, then clip without changing dtype or device.

- [ ] **Step 4: Run the lookup tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_future_token_dynamics -v
```

Expected: all pure lookup and gathering tests pass.

- [ ] **Step 5: Add failing TokenProcessor buffer tests**

Build a temporary vocabulary with distinct vehicle, pedestrian, and cyclist trajectories. Initialize `TokenProcessor.init_agent_token()` on a lightweight instance and assert:

```python
self.assertEqual(tuple(processor.agent_token_dynamics_veh.shape), (2, 3))
self.assertFalse(
    "agent_token_dynamics_veh" in processor.state_dict()
)
self.assertFalse(
    torch.equal(
        processor.agent_token_dynamics_veh,
        processor.agent_token_dynamics_ped,
    )
)
```

Also assert that the error for malformed or non-finite trajectories contains both the vocabulary path and class name.

- [ ] **Step 6: Run the TokenProcessor tests and verify RED**

Run:

```bash
python -m unittest tests.test_future_token_dynamics.FutureTokenDynamicsTokenProcessorTest -v
```

Expected: missing future-dynamics buffers or constructor argument.

- [ ] **Step 7: Register and expose lookup buffers**

Pass `future_token_dynamics` into `TokenProcessor`, build tables during `init_agent_token()` only when active, register them with `persistent=False`, validate equal class token counts, and place the three buffers in `tokenized_agent`.

- [ ] **Step 8: Run Task 1 tests and commit**

Run:

```bash
python -m unittest tests.test_future_token_dynamics -v
```

Then:

```bash
git add tests/test_future_token_dynamics.py src/smart/tokens/future_token_dynamics.py src/smart/tokens/token_processor.py
git commit -m "feat: derive dynamics from agent vocabulary tokens"
```

---

### Task 2: Causal decoder conditioner

**Files:**
- Create: `src/smart/modules/future_token_dynamics.py`
- Modify: `tests/test_future_token_dynamics.py`
- Modify: `src/smart/modules/agent_decoder.py`
- Modify: `src/smart/modules/smart_decoder.py`
- Modify: `src/smart/model/smart.py`
- Modify: `configs/model/smart.yaml`

**Interfaces:**
- Consumes the three class lookup tables, selected indices, agent types, and the fused token feature.
- Produces `FutureTokenDynamicsConditioner.add_open_loop(...) -> Tensor` and `add_selected(...) -> Tensor`.
- Adds no parameters or persistent buffers when `is_active=false`.

- [ ] **Step 1: Write failing conditioner tests**

Use a three-dimensional hidden feature and replace the active conditioner's learned embedding with `nn.Identity()` so the expected values are literal. Verify:

```python
result = conditioner.add_open_loop(
    feature=torch.zeros(3, 4, 3),
    token_index=indices,
    agent_type=torch.tensor([0, 1, 2]),
    dynamics_veh=veh,
    dynamics_ped=ped,
    dynamics_cyc=cyc,
    num_historical_tokens=2,
)
self.assertTrue(torch.equal(result[:, :2], torch.zeros(3, 2, 3)))
self.assertTrue(torch.equal(result[:, 2], expected_for_indices_at_position_2))
```

Add a rollout test proving `add_selected()` conditions a newly appended feature using the already selected index, plus validation tests for normalization scale. Add an inactive test asserting the exact input object is returned and `state_dict()` is empty.

- [ ] **Step 2: Run conditioner tests and verify RED**

Run:

```bash
python -m unittest tests.test_future_token_dynamics.FutureTokenDynamicsConditionerTest -v
```

Expected: import failure for `src.smart.modules.future_token_dynamics`.

- [ ] **Step 3: Implement the optional conditioner**

When active, construct a dedicated `MLPEmbedding(3, hidden_dim)`, non-persistent scale buffer, and scalar gate. `add_open_loop()` gathers every position but masks embedded positions 0 and 1 to zero before addition. `add_selected()` gathers and adds one selected token vector.

- [ ] **Step 4: Run conditioner tests and verify GREEN**

Run:

```bash
python -m unittest tests.test_future_token_dynamics.FutureTokenDynamicsConditionerTest -v
```

Expected: all conditioner tests pass.

- [ ] **Step 5: Add failing decoder wiring tests**

Exercise the real `SMARTAgentDecoder.agent_token_embedding()` with zero agent layers. Assert that teacher-forced positions 0 and 1 match the branch-disabled feature path, while position 2 changes according to its own selected token. Inspect the rollout code through an execution-level one-step harness that records that logits are computed before `add_selected()` receives `next_token_idx`.

- [ ] **Step 6: Run decoder wiring tests and verify RED**

Run:

```bash
python -m unittest tests.test_future_token_dynamics.FutureTokenDynamicsDecoderTest -v
```

Expected: missing configuration propagation or lookup arguments.

- [ ] **Step 7: Wire the conditioner through the model**

Pass the same `future_token_dynamics` config from `SMART` to both `TokenProcessor` and `SMARTDecoder`, and from `SMARTDecoder` to `SMARTAgentDecoder`. In open-loop `agent_token_embedding()`, call `add_open_loop()` after `fusion_emb`. During rollout, call `add_selected()` on `feat_a_next` only after logits, sampling, token embedding, motion embedding, and fusion are complete.

- [ ] **Step 8: Add the disabled base configuration**

Add under `model.model_config`:

```yaml
future_token_dynamics:
  is_active: false
  normalization_scale: [5.0, 1.0, 5.0]
  initial_gate: 1.0
```

- [ ] **Step 9: Run Task 2 tests and commit**

Run:

```bash
python -m unittest tests.test_future_token_dynamics -v
```

Then:

```bash
git add tests/test_future_token_dynamics.py src/smart/modules/future_token_dynamics.py src/smart/modules/agent_decoder.py src/smart/modules/smart_decoder.py src/smart/model/smart.py configs/model/smart.yaml
git commit -m "feat: condition decoding on selected token dynamics"
```

---

### Task 3: Matched experiment families

**Files:**
- Create: `configs/experiment/pre_bc_history_future_token_dynamics.yaml`
- Create: `configs/experiment/clsft_history_future_token_dynamics.yaml`
- Create: `configs/experiment/inference_history_future_token_dynamics.yaml`
- Create: `configs/experiment/pre_bc_history_future_token_dynamics_reconstructed.yaml`
- Create: `configs/experiment/clsft_history_future_token_dynamics_reconstructed.yaml`
- Create: `configs/experiment/inference_history_future_token_dynamics_reconstructed.yaml`
- Modify: `tests/test_future_token_dynamics.py`

**Interfaces:**
- The unreconstructed files inherit existing `*_history_dynamics`, enable the future branch, and explicitly use `agent_vocab_555_s2.pkl`.
- Each reconstructed file inherits its corresponding unreconstructed future-dynamics file and overrides only `model.model_config.token_processor.agent_token_file`.

- [ ] **Step 1: Write failing configuration tests**

For all six experiment names, compose `configs/run.yaml` through Hydra when available and assert:

```python
self.assertTrue(cfg.model.model_config.history_dynamics.is_active)
self.assertTrue(cfg.model.model_config.future_token_dynamics.is_active)
```

Assert original variants resolve to `agent_vocab_555_s2.pkl`, reconstructed variants resolve to `agent_vocab_reconstructed.pkl`, and each reconstructed YAML fragment contains only `defaults` plus the token-file override.

- [ ] **Step 2: Run configuration tests and verify RED**

Run:

```bash
python -m unittest tests.test_future_token_dynamics.FutureTokenDynamicsConfigTest -v
```

Expected: missing experiment configuration errors.

- [ ] **Step 3: Add the six experiment files**

Use this unreconstructed pattern:

```yaml
# @package _global_
defaults:
  - pre_bc_history_dynamics
  - _self_

model:
  model_config:
    future_token_dynamics:
      is_active: true
    token_processor:
      agent_token_file: agent_vocab_555_s2.pkl
```

Use this reconstructed pattern:

```yaml
# @package _global_
defaults:
  - pre_bc_history_future_token_dynamics
  - _self_

model:
  model_config:
    token_processor:
      agent_token_file: agent_vocab_reconstructed.pkl
```

Apply the same structure to CLSFT and inference.

- [ ] **Step 4: Run configuration tests and commit**

Run:

```bash
python -m unittest tests.test_future_token_dynamics.FutureTokenDynamicsConfigTest -v
```

Then:

```bash
git add configs/experiment/*future_token_dynamics*.yaml tests/test_future_token_dynamics.py
git commit -m "feat: add matched future dynamics experiments"
```

---

### Task 4: Regression and integration verification

**Files:**
- Modify only files already listed if verification exposes a defect.

**Interfaces:**
- Existing experiments continue to compose with `future_token_dynamics.is_active=false`.
- Active lookups are non-persistent; active decoder weights and gate are persistent.

- [ ] **Step 1: Run focused tests**

```bash
python -m unittest tests.test_future_token_dynamics -v
python -m unittest tests.test_history_dynamics tests.test_spatial_aware_loss -v
```

- [ ] **Step 2: Run the complete repository suite**

```bash
python -m unittest discover -s tests -v
```

- [ ] **Step 3: Compose all new configurations in the CatK training environment**

```bash
for EXPERIMENT in \
  pre_bc_history_future_token_dynamics \
  clsft_history_future_token_dynamics \
  inference_history_future_token_dynamics \
  pre_bc_history_future_token_dynamics_reconstructed \
  clsft_history_future_token_dynamics_reconstructed \
  inference_history_future_token_dynamics_reconstructed
do
  python -m src.run --cfg job --resolve \
    experiment="$EXPERIMENT" \
    ckpt_path=/tmp/catk-placeholder.ckpt \
    paths.cache_root=/tmp/catk-placeholder-cache \
    >/dev/null
done
```

Expected: all six compositions exit successfully without instantiating the missing reconstructed vocabulary.

- [ ] **Step 4: Verify legacy and active checkpoint keys**

Instantiate a disabled conditioner and assert its `state_dict()` is empty. Instantiate an active conditioner and assert its state contains the embedding and gate but not the normalization scale. Confirm TokenProcessor lookup buffers do not appear in its `state_dict()`.

- [ ] **Step 5: Inspect the final diff**

```bash
git diff --check
git status --short
git diff --stat HEAD~3..HEAD
```

Confirm `scripts/cache_womd.sh`, `.DS_Store`, and `.codex_work/` were not staged or changed by this implementation.
