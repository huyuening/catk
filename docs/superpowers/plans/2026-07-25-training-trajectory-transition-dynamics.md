# Training-Trajectory Token-Transition Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the isolated six-frame future-token dynamics lookup with a fixed previous/current-token transition lookup built only from complete 91-frame training trajectories.

**Architecture:** A pure NumPy module extracts continuous body-frame dynamics and aggregates class-specific token transitions. A separate offline builder reuses CatK's deterministic token matcher and writes a vocabulary-bound tensor artifact. `TokenProcessor` validates and loads that artifact, while the existing optional decoder conditioner gathers `(previous_token, current_token)` values causally.

**Tech Stack:** Python 3.11, NumPy, PyTorch, PyTorch Geometric, Hydra/OmegaConf, unittest/pytest.

## Global Constraints

- Read complete trajectories only from the explicit training-cache directory.
- Never instantiate or read validation/test data while building the table.
- Preserve feature order `[a_lon, angular_speed, a_lat]`.
- Use 10 Hz timestamps and clipping limits `[15.0, 3.0, 15.0]`.
- Maintain separate raw-cache and full-trajectory-reconstructed artifacts.
- Bind every artifact to the exact agent vocabulary with SHA-256.
- Runtime validation/inference may use matched or selected token IDs only; it must not read occurrence-level future dynamics.
- Position \(t\) may use transition `(token[t-1], token[t])` only to predict `token[t+1]`.
- Positions 0 and 1 retain existing history dynamics and receive zero future-transition dynamics.
- The feature remains disabled by default and legacy configurations remain checkpoint compatible.
- Preserve unrelated worktree changes, including `scripts/cache_womd.sh`, `.DS_Store`, and `.codex_work/`.

---

## File Structure

- Create `src/smart/tokens/transition_dynamics.py`: continuous 91-frame dynamics extraction and streaming dense aggregation.
- Create `src/smart/tokens/transition_dynamics_artifact.py`: vocabulary hashing plus atomic artifact save/load validation.
- Create `src/smart/tokens/build_transition_dynamics.py`: training-only command-line builder and CatK token-matching orchestration.
- Modify `src/smart/tokens/future_token_dynamics.py`: retain isolated-token fallback and add class-specific pair gathering.
- Modify `src/smart/tokens/token_processor.py`: load a validated transition artifact instead of constructing the active lookup from isolated tokens.
- Modify `src/smart/modules/future_token_dynamics.py`: embed previous/current transition dynamics.
- Modify `src/smart/modules/agent_decoder.py`: pass the correct previous token in open-loop and rollout paths.
- Modify `configs/model/smart.yaml`: add disabled-by-default lookup path and source fields.
- Modify six `configs/experiment/*history_future_token_dynamics*.yaml` files: select raw or reconstructed source and require an artifact override.
- Create `tests/test_transition_dynamics.py`: extraction, aggregation, fallback, and artifact tests.
- Create `tests/test_build_transition_dynamics.py`: builder provenance and per-batch accumulation tests.
- Modify `tests/test_future_token_dynamics.py`: pair gathering, runtime loading, and causal decoder alignment tests.
- Modify `tests/test_future_token_dynamics_configs.py`: Hydra/YAML expectations for the six experiment families.

---

### Task 1: Continuous Dynamics Extraction and Pair Aggregation

**Files:**
- Create: `src/smart/tokens/transition_dynamics.py`
- Create: `tests/test_transition_dynamics.py`

**Interfaces:**
- Produces:
  - `FullTrajectoryDynamics(values: np.ndarray, valid: np.ndarray)`
  - `extract_full_trajectory_dynamics(position, heading, valid_mask, *, dt=0.1, clipping_limits=(15.0, 3.0, 15.0)) -> FullTrajectoryDynamics`
  - `TransitionDynamicsAccumulator(n_agent_types: int, n_token: int)`
  - `TransitionDynamicsAccumulator.add(agent_type, previous_token, current_token, values, valid) -> None`
  - `TransitionDynamicsAccumulator.finalize(isolated_fallback, *, shrinkage_count=8.0) -> tuple[np.ndarray, dict]`
- Consumes: NumPy arrays only; no dataset, Hydra, or model dependencies.

- [ ] **Step 1: Write failing extraction tests**

Add tests using complete 91-frame trajectories:

```python
def test_complete_constant_acceleration_trajectory():
    time = np.arange(91, dtype=np.float64) * 0.1
    position = np.column_stack(
        (2.0 * time + time**2, np.zeros_like(time))
    )
    heading = np.zeros(91)
    result = extract_full_trajectory_dynamics(
        position, heading, np.ones(91, dtype=bool)
    )
    np.testing.assert_allclose(result.values[[5, 10, 90], 0], 2.0, atol=1e-8)
    np.testing.assert_allclose(result.values[:, 1:], 0.0, atol=1e-8)
    assert result.valid.all()


def test_endpoint_uses_neighbors_across_token_boundary():
    time = np.arange(91, dtype=np.float64) * 0.1
    position = np.column_stack((time**3, np.zeros_like(time)))
    result = extract_full_trajectory_dynamics(
        position, np.zeros(91), np.ones(91, dtype=bool)
    )
    six_frame = extract_full_trajectory_dynamics(
        position[:6], np.zeros(6), np.ones(6, dtype=bool)
    )
    assert not np.isclose(result.values[5, 0], six_frame.values[5, 0])
```

Also cover a constant-radius path, heading wraparound, disjoint valid runs, an
invalid run shorter than three frames, shape errors, and clipping.

- [ ] **Step 2: Run extraction tests and verify failure**

Run:

```bash
python -m pytest tests/test_transition_dynamics.py -k "trajectory or endpoint" -v
```

Expected: collection fails because `transition_dynamics` does not exist.

- [ ] **Step 3: Implement continuous extraction**

Implement:

```python
@dataclass(frozen=True)
class FullTrajectoryDynamics:
    values: np.ndarray
    valid: np.ndarray


def extract_full_trajectory_dynamics(
    position: np.ndarray,
    heading: np.ndarray,
    valid_mask: np.ndarray,
    *,
    dt: float = 0.1,
    clipping_limits: Sequence[float] = (15.0, 3.0, 15.0),
) -> FullTrajectoryDynamics:
    position = np.asarray(position, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    values = np.zeros((len(position), 3), dtype=np.float32)
    output_valid = np.zeros(len(position), dtype=bool)
    usable = (
        valid_mask
        & np.isfinite(position[:, :2]).all(axis=1)
        & np.isfinite(heading)
    )
    for start, end in _true_runs(usable):
        if end - start < 3:
            continue
        xy = position[start:end, :2]
        theta = np.unwrap(heading[start:end])
        velocity = np.column_stack(
            [
                np.gradient(xy[:, axis], dt, edge_order=2)
                for axis in range(2)
            ]
        )
        acceleration = np.column_stack(
            [
                np.gradient(velocity[:, axis], dt, edge_order=2)
                for axis in range(2)
            ]
        )
        angular_speed = np.gradient(theta, dt, edge_order=2)
        cosine, sine = np.cos(theta), np.sin(theta)
        run_values = np.column_stack(
            (
                acceleration[:, 0] * cosine + acceleration[:, 1] * sine,
                angular_speed,
                -acceleration[:, 0] * sine + acceleration[:, 1] * cosine,
            )
        )
        finite = np.isfinite(run_values).all(axis=1)
        values[start:end][finite] = np.clip(
            run_values[finite],
            -np.asarray(clipping_limits),
            np.asarray(clipping_limits),
        )
        output_valid[start:end][finite] = True
    return FullTrajectoryDynamics(values=values, valid=output_valid)
```

Validate shared step dimensions, finite positive `dt`, and three finite
positive limits. Iterate contiguous valid runs of length at least three. Within
each run, use the same second-order gradient convention as
`history_dynamics.py`, unwrap heading, project acceleration into the body
frame, clip values, and mark only finite derived frames valid.

- [ ] **Step 4: Run extraction tests and verify pass**

Run:

```bash
python -m pytest tests/test_transition_dynamics.py -k "trajectory or endpoint" -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Write failing accumulator tests**

Add:

```python
def test_pair_shrinkage_and_unseen_pair_fallback():
    accumulator = TransitionDynamicsAccumulator(n_agent_types=3, n_token=2)
    accumulator.add(
        agent_type=np.array([0, 0]),
        previous_token=np.array([0, 0]),
        current_token=np.array([1, 1]),
        values=np.array([[4.0, 0.4, 2.0], [6.0, 0.6, 4.0]]),
        valid=np.array([True, True]),
    )
    isolated = np.zeros((3, 2, 3), dtype=np.float64)
    table, stats = accumulator.finalize(isolated, shrinkage_count=2.0)
    np.testing.assert_allclose(table[0, 1, 1], table[0, 0, 1])
    assert stats["observed_pairs"]["veh"] == 1
```

Add separate assertions that:

- class accumulators do not mix;
- a current-token marginal is used when pair count is zero;
- an entirely unseen current token falls back to isolated geometry;
- invalid rows and non-finite values are skipped;
- invalid token IDs are rejected;
- final output has shape `[3, n_token, n_token, 3]` and dtype `float16`.

- [ ] **Step 6: Run accumulator tests and verify failure**

Run:

```bash
python -m pytest tests/test_transition_dynamics.py -k "pair or accumulator or fallback" -v
```

Expected: failure because `TransitionDynamicsAccumulator` is missing.

- [ ] **Step 7: Implement the accumulator**

Use float64 dense sums and int64 counts:

```python
self.pair_sum = np.zeros(
    (n_agent_types, n_token, n_token, 3), dtype=np.float64
)
self.pair_count = np.zeros(
    (n_agent_types, n_token, n_token), dtype=np.int64
)
self.marginal_sum = np.zeros(
    (n_agent_types, n_token, 3), dtype=np.float64
)
self.marginal_count = np.zeros(
    (n_agent_types, n_token), dtype=np.int64
)
```

Use `np.add.at` for repeated indices. In `finalize`, calculate current-token
marginals first, replace unseen marginals with `isolated_fallback`, then apply:

```python
values = (
    self.pair_sum
    + shrinkage_count * marginal[:, None, :, :]
) / (
    self.pair_count[..., None] + shrinkage_count
)
```

Return finite float16 values and JSON-serializable coverage statistics.

- [ ] **Step 8: Run the complete Task 1 test file**

Run:

```bash
python -m pytest tests/test_transition_dynamics.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/smart/tokens/transition_dynamics.py tests/test_transition_dynamics.py
git commit -m "feat: aggregate dynamics from full training trajectories"
```

---

### Task 2: Vocabulary-Bound Artifact I/O

**Files:**
- Create: `src/smart/tokens/transition_dynamics_artifact.py`
- Modify: `tests/test_transition_dynamics.py`

**Interfaces:**
- Consumes: finalized float16 `[3, n_token, n_token, 3]` table from Task 1.
- Produces:
  - `FORMAT_VERSION = 1`
  - `vocabulary_sha256(path: str | Path) -> str`
  - `make_transition_dynamics_artifact(values, *, vocabulary_path, source, dt, clipping_limits, shrinkage_count, statistics) -> dict`
  - `save_transition_dynamics_artifact(path: str | Path, artifact: Mapping, *, vocabulary_path: str | Path) -> Path`
  - `load_transition_dynamics_artifact(path, *, vocabulary_path, expected_source, expected_n_token) -> torch.Tensor`

- [ ] **Step 1: Write failing artifact tests**

Test a temporary vocabulary file and artifact:

```python
def test_artifact_round_trip_is_bound_to_vocabulary(tmp_path):
    vocab = tmp_path / "agent_vocab.pkl"
    vocab.write_bytes(b"vocabulary-a")
    values = np.zeros((3, 2, 2, 3), dtype=np.float16)
    artifact = make_transition_dynamics_artifact(
        values,
        vocabulary_path=vocab,
        source="raw",
        dt=0.1,
        clipping_limits=(15.0, 3.0, 15.0),
        shrinkage_count=8.0,
        statistics={"occurrences": 0},
    )
    output = save_transition_dynamics_artifact(
        tmp_path / "lookup.pt",
        artifact,
        vocabulary_path=vocab,
    )
    loaded = load_transition_dynamics_artifact(
        output,
        vocabulary_path=vocab,
        expected_source="raw",
        expected_n_token=2,
    )
    assert loaded.shape == (3, 2, 2, 3)
    assert loaded.dtype == torch.float16
```

Add failures for a changed vocabulary, source mismatch, unsupported version,
wrong feature order, wrong shape, non-finite values, and a missing file.
Assert that no `.tmp` file remains after a successful save.

- [ ] **Step 2: Run artifact tests and verify failure**

Run:

```bash
python -m pytest tests/test_transition_dynamics.py -k artifact -v
```

Expected: import or name failure for artifact helpers.

- [ ] **Step 3: Implement hashing, construction, atomic save, and safe load**

Hash files in chunks with `hashlib.sha256`. Store only tensors, strings,
numbers, tuples/lists, and dictionaries accepted by
`torch.load(path, weights_only=True)`. Save to
`output.with_suffix(output.suffix + ".tmp")`, flush by closing `torch.save`,
validate the temporary artifact through the public loader, then use
`os.replace`.

Validate:

```python
FEATURE_ORDER = ("a_lon", "angular_speed", "a_lat")
expected_shape = (3, expected_n_token, expected_n_token, 3)
```

Return the `values` tensor only after all metadata and finite checks pass.

- [ ] **Step 4: Run artifact tests**

Run:

```bash
python -m pytest tests/test_transition_dynamics.py -k artifact -v
```

Expected: all artifact tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/smart/tokens/transition_dynamics_artifact.py tests/test_transition_dynamics.py
git commit -m "feat: add vocabulary-bound transition artifacts"
```

---

### Task 3: Training-Only Offline Builder

**Files:**
- Create: `src/smart/tokens/build_transition_dynamics.py`
- Create: `tests/test_build_transition_dynamics.py`
- Modify: `src/smart/tokens/transition_dynamics_artifact.py`

**Interfaces:**
- Consumes:
  - training cache files containing CatK `agent` tensors;
  - one exact agent vocabulary;
  - `TokenProcessor.tokenize_agent` in evaluation mode;
  - Tasks 1 and 2 extraction/aggregation/artifact APIs.
- Produces:
  - `validate_source_provenance(agent_store, source, *, context) -> None`
  - `accumulate_tokenized_batch(accumulator, data, tokenized_agent, *, source) -> dict`
  - `build_transition_dynamics(training_dir, agent_token_file, output, *, source, map_token_file, batch_size, num_workers, max_scenarios, shrinkage_count) -> Path`
  - CLI entry point for `python -m src.smart.tokens.build_transition_dynamics`.

- [ ] **Step 1: Write failing provenance and batch tests**

Construct small synthetic agent stores and tokenized output:

```python
def test_reconstructed_source_requires_marker():
    with pytest.raises(ValueError, match="trajectory_reconstructed"):
        validate_source_provenance(
            {"position": torch.zeros(1, 91, 3)},
            "reconstructed",
            context="scene.pkl",
        )


def test_batch_accumulates_current_endpoint_for_adjacent_pair():
    data = fake_batch_with_one_constant_acceleration_agent()
    tokenized = {
        "type": torch.tensor([0]),
        "gt_idx": torch.tensor([[0, 1, 1]]),
        "valid_mask": torch.tensor([[True, True, True]]),
    }
    accumulator = TransitionDynamicsAccumulator(3, 2)
    stats = accumulate_tokenized_batch(
        accumulator, data, tokenized, source="raw"
    )
    assert accumulator.pair_count[0, 0, 1] == 1
    assert accumulator.pair_count[0, 1, 1] == 1
    assert stats["accepted_occurrences"] == 2
```

Verify raw mode accepts a missing marker or an all-false marker and rejects any
true marker. Verify reconstructed mode requires an all-true marker. Verify the
sampled dynamics align with endpoints 10, 15, … corresponding to current token
positions 1, 2, ….

- [ ] **Step 2: Run focused builder tests and verify failure**

Run:

```bash
python -m pytest tests/test_build_transition_dynamics.py -k "source or batch" -v
```

Expected: module import failure.

- [ ] **Step 3: Implement provenance and batch accumulation**

Before calling the token processor, clone the source position, heading, valid
mask, and type tensors needed for dynamics extraction. In raw mode, apply
`TokenProcessor._clean_heading` to a clone. In reconstructed mode, use the
stored heading directly. For every agent:

```python
full = extract_full_trajectory_dynamics(position, heading, valid)
endpoint_values = full.values[5::5]
endpoint_valid = full.valid[5::5]
pair_valid = (
    tokenized_agent["valid_mask"][:, :-1]
    & tokenized_agent["valid_mask"][:, 1:]
    & endpoint_valid[:, 1:]
)
```

Accumulate previous IDs `gt_idx[:, :-1]`, current IDs `gt_idx[:, 1:]`, and
current endpoint values `endpoint_values[:, 1:]`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_build_transition_dynamics.py -k "source or batch" -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Write failing orchestration and CLI tests**

Patch `MultiDataset`, `DataLoader`, and `TokenProcessor` with small fakes.
Assert:

- only the explicit `training_dir` is passed to `MultiDataset`;
- `shuffle=False` and deterministic ordering are used;
- the processor is put in evaluation mode so `num_k=1`;
- `max_scenarios` truncates before dataset construction or via a deterministic
  subset;
- the output and JSON summary are written;
- missing directories, invalid source strings, and empty datasets fail before
  producing an artifact;
- `parse_args` exposes only `--training-dir`, never validation/test directory
  options.

- [ ] **Step 6: Run orchestration tests and verify failure**

Run:

```bash
python -m pytest tests/test_build_transition_dynamics.py -k "build or cli" -v
```

Expected: failures for missing orchestration functions.

- [ ] **Step 7: Implement builder orchestration and CLI**

Use:

```python
dataset = MultiDataset(
    raw_dir=str(training_dir),
    transform=lambda value: HeteroData(value),
)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False,
)
```

Instantiate `TokenProcessor` with the requested vocabulary, the existing map
token file, `num_k=1` sampling namespaces/configs, and both dynamics branches
disabled. Call `processor.eval()` and `processor.tokenize_agent(batch)`.

Load vocabulary contours once to construct the isolated fallback with
`build_future_token_dynamics_lookup`. Finalize the accumulator, create the
artifact, atomically save it, and write
`<output-stem>.summary.json`.

Provide CLI defaults:

```text
--source raw
--map-token-file map_traj_token5.pkl
--batch-size 8
--num-workers 8
--shrinkage-count 8.0
--max-scenarios omitted
```

- [ ] **Step 8: Run complete builder tests**

Run:

```bash
python -m pytest tests/test_build_transition_dynamics.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Verify CLI help imports successfully**

Run:

```bash
python -m src.smart.tokens.build_transition_dynamics --help
```

Expected: exit 0 and help containing `--training-dir`, `--agent-token-file`,
`--source`, and `--output`.

- [ ] **Step 10: Commit Task 3**

```bash
git add src/smart/tokens/build_transition_dynamics.py src/smart/tokens/transition_dynamics_artifact.py tests/test_build_transition_dynamics.py
git commit -m "feat: build transition dynamics from training cache"
```

---

### Task 4: Runtime Artifact Loading and Pair Gathering

**Files:**
- Modify: `src/smart/tokens/future_token_dynamics.py`
- Modify: `src/smart/tokens/token_processor.py`
- Modify: `tests/test_future_token_dynamics.py`

**Interfaces:**
- Consumes:
  - `load_transition_dynamics_artifact` from Task 2;
  - `future_token_dynamics.lookup_file` and `.source`;
  - loaded table shape `[3, n_token, n_token, 3]`.
- Produces:
  - `gather_transition_dynamics(previous_token_index, current_token_index, agent_type, dynamics_veh, dynamics_ped, dynamics_cyc) -> Tensor`
  - three non-persistent TokenProcessor buffers shaped `[n_token, n_token, 3]`.

- [ ] **Step 1: Write failing pair-gather tests**

Add:

```python
def test_pair_gather_uses_previous_and_current_indices():
    table = torch.arange(2 * 2 * 3).reshape(2, 2, 3).float()
    result = gather_transition_dynamics(
        previous_token_index=torch.tensor([0, 1]),
        current_token_index=torch.tensor([1, 0]),
        agent_type=torch.tensor([0, 1]),
        dynamics_veh=table,
        dynamics_ped=table + 100,
        dynamics_cyc=table + 200,
    )
    torch.testing.assert_close(result[0], table[0, 1])
    torch.testing.assert_close(result[1], table[1, 0] + 100)
```

Cover multi-dimensional token indices, invalid agent types, invalid pair IDs,
wrong lookup shapes, and mismatched devices/dtypes.

- [ ] **Step 2: Run pair-gather tests and verify failure**

Run:

```bash
python -m pytest tests/test_future_token_dynamics.py -k pair_gather -v
```

Expected: missing function failure.

- [ ] **Step 3: Implement pair gathering**

Replace active runtime calls to the single-index gather with a pair-index
function. Keep `build_future_token_dynamics_lookup` because the offline builder
uses it only for never-observed-token fallback.

- [ ] **Step 4: Run pair-gather tests**

Run:

```bash
python -m pytest tests/test_future_token_dynamics.py -k pair_gather -v
```

Expected: pass.

- [ ] **Step 5: Write failing TokenProcessor artifact tests**

Create a two-token vocabulary and matching artifact in a temporary directory.
Patch the module directory or pass absolute paths. Assert:

- active mode requires `lookup_file`;
- source is either `raw` or `reconstructed`;
- the three registered buffers equal artifact class slices;
- each buffer is non-persistent;
- a changed vocabulary hash fails during initialization;
- disabled mode does not require or load an artifact;
- active mode no longer derives the runtime table directly from six-frame
  tokens.

- [ ] **Step 6: Run TokenProcessor tests and verify failure**

Run:

```bash
python -m pytest tests/test_future_token_dynamics.py -k token_processor -v
```

Expected: active mode still builds `[n_token, 3]` isolated lookups.

- [ ] **Step 7: Implement validated loading**

Store the full config as `self.future_token_dynamics_config`. Resolve relative
lookup paths against `src/smart/tokens`, as vocabulary paths are resolved.
After vocabulary class counts are validated, call:

```python
table = load_transition_dynamics_artifact(
    lookup_path,
    vocabulary_path=agent_token_path,
    expected_source=config["source"],
    expected_n_token=token_count,
)
```

Register `table[0]`, `table[1]`, and `table[2]` under the existing buffer names.
Do not persist them in checkpoints.

- [ ] **Step 8: Run runtime-loading tests**

Run:

```bash
python -m pytest tests/test_future_token_dynamics.py -k "pair_gather or token_processor" -v
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 4**

```bash
git add src/smart/tokens/future_token_dynamics.py src/smart/tokens/token_processor.py tests/test_future_token_dynamics.py
git commit -m "feat: load token transition dynamics at runtime"
```

---

### Task 5: Causal Previous/Current Decoder Wiring

**Files:**
- Modify: `src/smart/modules/future_token_dynamics.py`
- Modify: `src/smart/modules/agent_decoder.py`
- Modify: `tests/test_future_token_dynamics.py`

**Interfaces:**
- Consumes: three `[n_token, n_token, 3]` buffers from Task 4.
- Produces:
  - `FutureTokenDynamicsConditioner.add_open_loop` keeps its current lookup
    arguments and gathers `(token[:, t-1], token[:, t])`;
  - `FutureTokenDynamicsConditioner.add_selected` replaces `token_index` with
    explicit `previous_token_index` and `current_token_index` arguments while
    retaining the three class lookup arguments.

- [ ] **Step 1: Rewrite conditioner tests to fail on pair semantics**

Use a lookup whose rows differ by previous token. Assert:

```python
def test_open_loop_uses_previous_current_pair_and_masks_history():
    output = conditioner.add_open_loop(
        feature=torch.zeros(1, 4, hidden_dim),
        token_index=torch.tensor([[0, 1, 0, 1]]),
        agent_type=torch.tensor([0]),
        dynamics_veh=veh,
        dynamics_ped=ped,
        dynamics_cyc=cyc,
        num_historical_tokens=2,
    )
    torch.testing.assert_close(output[:, :2], torch.zeros_like(output[:, :2]))
    assert_embedding_uses_pair(output[:, 2], previous=1, current=0)
    assert_embedding_uses_pair(output[:, 3], previous=0, current=1)
```

Update selected-token tests to require both index arguments and verify the
previous token changes the result while the current token stays fixed.

- [ ] **Step 2: Run conditioner tests and verify failure**

Run:

```bash
python -m pytest tests/test_future_token_dynamics.py -k "open_loop or selected" -v
```

Expected: signature or value failures under the old single-token lookup.

- [ ] **Step 3: Implement conditioner pair semantics**

In open-loop:

```python
previous_index = torch.cat(
    (token_index[:, :1], token_index[:, :-1]),
    dim=1,
)
```

Gather pairs for all positions, then zero positions before
`num_historical_tokens`. Cast gathered float16 values to `feature.dtype` before
normalization and the MLP.

In selected mode, validate equal shapes for `previous_token_index`,
`current_token_index`, and `feature.shape[:1]`.

- [ ] **Step 4: Run conditioner tests**

Run:

```bash
python -m pytest tests/test_future_token_dynamics.py -k "open_loop or selected" -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Write failing rollout-alignment test**

Patch `FutureTokenDynamicsConditioner.add_selected` and run a minimal decoder
rollout. Capture arguments and assert the first call receives:

```python
previous_token_index == tokenized_agent["gt_idx"][:, 1]
current_token_index == first_selected_token
```

Assert each later call receives the prior selected token and the newly selected
token. Assert no `gt_idx` position after index 1 is passed to the conditioner.

- [ ] **Step 6: Run rollout test and verify failure**

Run:

```bash
python -m pytest tests/test_future_token_dynamics.py -k rollout_pair_alignment -v
```

Expected: the old `add_selected` call provides only one token index and does not provide the
previous index.

- [ ] **Step 7: Update agent decoder**

Immediately before assigning the selected token into `pred_idx[:, n_step]`,
retain:

```python
previous_token_idx = pred_idx[:, t_now]
```

Pass:

```python
self.future_token_dynamics.add_selected(
    feature=feat_a_next.squeeze(1),
    previous_token_index=previous_token_idx,
    current_token_index=next_token_idx,
    agent_type=tokenized_agent["type"],
    dynamics_veh=tokenized_agent.get("agent_token_dynamics_veh"),
    dynamics_ped=tokenized_agent.get("agent_token_dynamics_ped"),
    dynamics_cyc=tokenized_agent.get("agent_token_dynamics_cyc"),
)
```

Do not pass future `gt_idx` values.

- [ ] **Step 8: Run all future-dynamics tests**

Run:

```bash
python -m pytest tests/test_future_token_dynamics.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 5**

```bash
git add src/smart/modules/future_token_dynamics.py src/smart/modules/agent_decoder.py tests/test_future_token_dynamics.py
git commit -m "feat: condition decoding on token transitions"
```

---

### Task 6: Configurations, Commands, and Regression Verification

**Files:**
- Modify: `configs/model/smart.yaml`
- Modify: `configs/experiment/pre_bc_history_future_token_dynamics.yaml`
- Modify: `configs/experiment/clsft_history_future_token_dynamics.yaml`
- Modify: `configs/experiment/inference_history_future_token_dynamics.yaml`
- Modify: `configs/experiment/pre_bc_history_future_token_dynamics_reconstructed.yaml`
- Modify: `configs/experiment/clsft_history_future_token_dynamics_reconstructed.yaml`
- Modify: `configs/experiment/inference_history_future_token_dynamics_reconstructed.yaml`
- Modify: `tests/test_future_token_dynamics_configs.py`
- Modify: `README.md` only if it already documents the corresponding history/future dynamics commands; otherwise create `docs/training-trajectory-transition-dynamics.md`.

**Interfaces:**
- Consumes: runtime fields implemented in Tasks 4 and 5.
- Produces: composable raw/reconstructed experiments and copy-paste builder,
  pre-BC, CLSFT, and inference commands.

- [ ] **Step 1: Write failing configuration tests**

Assert:

```python
base["model_config"]["future_token_dynamics"] == {
    "is_active": False,
    "lookup_file": None,
    "source": "raw",
    "normalization_scale": [5.0, 1.0, 5.0],
    "initial_gate": 1.0,
}
```

For raw experiments, assert `is_active=true`, `source=raw`, original
vocabulary, and a null lookup awaiting a CLI override. For reconstructed
experiments, assert `source=reconstructed` and
`agent_vocab_reconstructed.pkl`. Ensure all six preserve history dynamics.

- [ ] **Step 2: Run config tests and verify failure**

Run:

```bash
python -m pytest tests/test_future_token_dynamics_configs.py -v
```

Expected: failures because `lookup_file` and `source` are absent.

- [ ] **Step 3: Update model and experiment YAML**

Add:

```yaml
future_token_dynamics:
  is_active: false
  lookup_file: null
  source: raw
  normalization_scale: [5.0, 1.0, 5.0]
  initial_gate: 1.0
```

Raw experiment files explicitly select `source: raw`. Reconstructed
experiment files override `source: reconstructed`. Do not hard-code a machine
path.

- [ ] **Step 4: Run config tests**

Run:

```bash
python -m pytest tests/test_future_token_dynamics_configs.py -v
```

Expected: pass.

- [ ] **Step 5: Document exact commands**

Document raw and reconstructed table builds:

```bash
python -m src.smart.tokens.build_transition_dynamics \
  --training-dir "$RAW_CACHE/training" \
  --agent-token-file src/smart/tokens/agent_vocab_555_s2.pkl \
  --source raw \
  --output "$RAW_CACHE/agent_transition_dynamics.pt"
```

```bash
python -m src.smart.tokens.build_transition_dynamics \
  --training-dir "$RECON_CACHE/training" \
  --agent-token-file src/smart/tokens/agent_vocab_reconstructed.pkl \
  --source reconstructed \
  --output "$RECON_CACHE/agent_transition_dynamics_reconstructed.pt"
```

Document pre-BC invocation:

```bash
bash scripts/train.sh \
  experiment=pre_bc_history_future_token_dynamics \
  model.model_config.future_token_dynamics.lookup_file="$LOOKUP_FILE"
```

Include matching CLSFT and inference experiment names and state that a
checkpoint and table must use the same vocabulary hash/source.

- [ ] **Step 6: Run focused feature suite**

Run:

```bash
python -m pytest \
  tests/test_transition_dynamics.py \
  tests/test_build_transition_dynamics.py \
  tests/test_future_token_dynamics.py \
  tests/test_future_token_dynamics_configs.py \
  -v
```

Expected: all tests pass.

- [ ] **Step 7: Compose all six Hydra experiments**

For every experiment name, run:

```bash
python -m src.run --cfg job --resolve \
  experiment=<name> \
  model.model_config.future_token_dynamics.lookup_file=/tmp/example.pt \
  task_name=config_check
```

Expected: composition succeeds and resolved config contains the intended
source, lookup path, history dynamics, and agent vocabulary.

- [ ] **Step 8: Run the complete repository suite**

Run:

```bash
python -m pytest -q
```

Expected: all locally runnable tests pass. Record dependency-based skips
separately; do not report them as passes.

- [ ] **Step 9: Run static checks**

Run:

```bash
python -m compileall -q src tests
git diff --check
git status --short
```

Expected: compilation and whitespace checks pass; status contains only this
feature plus the user's pre-existing unrelated changes.

- [ ] **Step 10: Commit Task 6**

```bash
git add \
  configs/model/smart.yaml \
  configs/experiment/pre_bc_history_future_token_dynamics.yaml \
  configs/experiment/clsft_history_future_token_dynamics.yaml \
  configs/experiment/inference_history_future_token_dynamics.yaml \
  configs/experiment/pre_bc_history_future_token_dynamics_reconstructed.yaml \
  configs/experiment/clsft_history_future_token_dynamics_reconstructed.yaml \
  configs/experiment/inference_history_future_token_dynamics_reconstructed.yaml \
  tests/test_future_token_dynamics_configs.py \
  docs/training-trajectory-transition-dynamics.md
git commit -m "docs: configure training trajectory dynamics experiments"
```

---

## Final Verification Checklist

- [ ] Confirm the builder CLI has no validation/test input option.
- [ ] Confirm raw and reconstructed provenance tests pass.
- [ ] Confirm artifact SHA-256 mismatch fails before model construction.
- [ ] Confirm runtime tables are non-persistent buffers.
- [ ] Confirm open-loop positions 0 and 1 are masked.
- [ ] Confirm open-loop position \(t\) uses `(t-1, t)`.
- [ ] Confirm rollout uses previous selected/current selected IDs only.
- [ ] Confirm disabled mode produces no new checkpoint keys beyond the already
      optional conditioner behavior.
- [ ] Confirm focused and full test suites pass.
- [ ] Confirm no generated `.pt` artifact is staged for Git.
- [ ] Confirm unrelated worktree files are untouched.
