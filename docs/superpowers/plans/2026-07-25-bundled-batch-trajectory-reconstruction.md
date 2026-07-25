# Bundled Batch Trajectory Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CatK's complete batch trajectory reconstructor available through `--method batch` without a `WOMD-Traffic-Signal-Data-Improvement` checkout.

**Architecture:** Adapt the audited NumPy/SciPy batch solver into `src.smart.tokens`, where it reuses CatK's bundled geometric filter and WOMD proto-compatible Scenario objects. Extend the existing reconstruction bridge to select bundled filter or bundled batch lazily while preserving an explicitly requested external implementation. Update CLI provenance and documentation so a single CatK checkout is sufficient.

**Tech Stack:** Python 3.11, NumPy, SciPy 1.14.1 (`least_squares` and sparse Jacobians), WOMD Scenario protobufs, unittest/pytest.

## Global Constraints

- Port the complete batch implementation; do not replace it with a reduced smoother.
- Preserve the batch defaults, class-specific behavior, bounds, safety checks, statistics, and fallback behavior.
- Use CatK's bundled `trajectory_filter_reconstructor` as the batch prefilter.
- Optimize each supported continuous trajectory segment as a whole, never as independent six-frame CatK tokens.
- Keep reconstruction training-only, full-trajectory, and vocabulary-only.
- Do not alter normal CatK training/validation/testing caches, model inputs, labels, losses, or runtime inference.
- `--method batch` must work without `--reconstruction-root`.
- An explicit `--reconstruction-root` must remain a backward-compatible external override.
- `optimizer` must continue to require an external reconstruction root.
- Import the bundled batch module lazily so `none` and `filter` do not require SciPy import at module load.
- Retain the PolyForm Noncommercial License 1.0.0 attribution already used by the bundled filter.
- Do not stage or modify the user's existing `scripts/cache_womd.sh`, `.DS_Store`, or `.codex_work/` changes.

---

## File Structure

- Create `src/smart/tokens/trajectory_batch_optimizer.py`: complete two-stage position/heading batch solver and batch statistics.
- Create `tests/test_trajectory_batch_optimizer.py`: numerical and behavioral regression suite for the bundled solver.
- Modify `src/smart/tokens/womd_trajectory_reconstruction.py`: lazy bundled-batch dispatch and updated root validation.
- Modify `tests/test_womd_trajectory_reconstruction.py`: bridge, configuration, parameter-forwarding, and external-override tests.
- Modify `src/smart/tokens/compare_trajectory_token_reconstruction.py`: help text, worker pre-import, and implementation provenance.
- Modify `tests/test_trajectory_token_comparison.py`: bundled/external provenance and parser tests.
- Modify `README.md`: self-contained batch vocabulary command and external optimizer note.

---

### Task 1: Port and Verify the Complete Batch Solver

**Files:**
- Create: `tests/test_trajectory_batch_optimizer.py`
- Create: `src/smart/tokens/trajectory_batch_optimizer.py`

**Interfaces:**
- Consumes:
  - a WOMD-compatible Scenario with `timestamps_seconds` and `tracks`;
  - CatK filter functions from
    `src.smart.tokens.trajectory_filter_reconstructor`.
- Produces:
  - `BatchTrajectoryConfig`
  - `TrackBatchResult`
  - `BatchReconstructionStats`
  - `wosac_acceleration_features(x, y, z, heading, dt_or_time)`
  - `wosac_jerk_features(x, y, z, heading, dt_or_time)`
  - `optimize_track(original_track, reconstructed_track, timestamps, config=None, motion_config=None)`
  - `reconstruct_scenario_agents(scenario, config=None, filter_strength="strong", max_gap_frames=None)`

- [ ] **Step 1: Add the failing bundled-solver regression test**

Create `tests/test_trajectory_batch_optimizer.py` from the audited regression
suite at:

```text
WOMD-Traffic-Signal-Data-Improvement/tests/test_trajectory_batch_optimizer.py
SHA-256: ef0a472b0b07db8d93455cfa463ca3bac6af28708c4e137fa12532e08f6547ff
```

Replace its project-root path manipulation and imports with CatK imports:

```python
from src.smart.tokens import trajectory_batch_optimizer as batch_optimizer
from src.smart.tokens.trajectory_batch_optimizer import (
    BatchTrajectoryConfig,
    optimize_track,
    reconstruct_scenario_agents as batch_reconstruct,
    wosac_acceleration_features,
    wosac_jerk_features,
)
from src.smart.tokens.trajectory_filter_reconstructor import (
    angle_diff,
    compute_kinematic_features,
    reconstruct_scenario_agents as filter_reconstruct,
)

PB2_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "smart"
    / "tokens"
    / "womd_proto"
    / "pb2"
)
if str(PB2_ROOT) not in sys.path:
    sys.path.insert(0, str(PB2_ROOT))
import scenario_pb2
```

Keep the source suite's `build_scenario` fixture and all 24 behavior tests:

- centered WOSAC acceleration and jerk parity;
- nonuniform finite-difference parity;
- use of actual nonuniform timestamps;
- adjacent-frame position/heading sawtooth suppression;
- constant linear/angular acceleration preservation;
- reverse heading preservation;
- low-speed prefilter heading fallback;
- noisy motion-direction protection;
- short sparse pedestrian regularization;
- sustained vehicle forward-heading evidence;
- lateral cyclist heading preservation;
- sparse pedestrian heading behavior;
- internal-gap filling;
- complete preprocessed-heading trust;
- cyclist heading-rate constraint;
- independent position/heading optimization;
- persistent endpoint correction;
- global and sparse-support linear-jerk safety;
- resolved-branch, irregular-time, and sparse-support angular-jerk safety;
- severe along-track timing-jitter smoothing;
- short-track accounting.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest tests/test_trajectory_batch_optimizer.py -v
```

Expected: collection fails with
`ModuleNotFoundError: src.smart.tokens.trajectory_batch_optimizer`.

- [ ] **Step 3: Add the complete bundled solver**

Adapt the audited source:

```text
WOMD-Traffic-Signal-Data-Improvement/src/trajectory_batch_optimizer.py
SHA-256: 63a9d8c26b65b265c6f931749fce9d1b5818f930d6ad1f8527af4e444c007612
```

Prepend CatK's existing attribution form:

```python
# Adapted from WOMD-Traffic-Signal-Data-Improvement for CatK vocabulary-only
# reconstruction. Distributed under the PolyForm Noncommercial License 1.0.0;
# see LICENSE.WOMD_TRAJECTORY_RECONSTRUCTION.txt in this directory.
```

Use only the package-relative filter import:

```python
from .trajectory_filter_reconstructor import (
    ReconstructionStats,
    _interpolate_heading,
    _interpolate_pi_ambiguous_heading,
    _interpolate_vehicle_heading,
    _interpolation_residual_outliers,
    _isolated_endpoint_heading_outliers,
    _keep_sustained_runs,
    _motion_body_heading_observation,
    _percentile,
    _preferred_vehicle_heading_anchor,
    _remove_short_heading_flips,
    _smooth_scalar,
    _smoothing_windows,
    _supports_detailed_pedestrian_heading,
    _vehicle_endpoint_heading_outliers,
    angle_diff,
    config_for_filter_strength,
    reconstruct_scenario_agents as filter_scenario_agents,
    wrap_angle,
)
```

Retain every audited dataclass field and numerical function. In particular,
keep:

```python
@dataclass(frozen=True)
class BatchTrajectoryConfig:
    min_optimization_frames: int = 7
    linear_jerk_weight: float = 1.0
    planar_vector_jerk_weight: float = 2.0
    angular_jerk_weight: float = 1.0
    adjacent_planar_jerk_weight: float = 1.0
    adjacent_angular_jerk_weight: float = 1.0
    max_nfev: int = 120
    ftol: float = 1e-7
    xtol: float = 1e-7
    gtol: float = 1e-7
```

Keep the public reconstruction flow:

```python
def reconstruct_scenario_agents(
    scenario,
    config: BatchTrajectoryConfig | None = None,
    filter_strength: str = "strong",
    max_gap_frames: int | None = None,
):
    config = config or BatchTrajectoryConfig()
    filter_config = config_for_filter_strength(
        filter_strength,
        max_gap_frames=max_gap_frames,
    )
    reconstructed, filter_stats = filter_scenario_agents(
        scenario, filter_config
    )
    stats = BatchReconstructionStats(**asdict(filter_stats))
    # Optimize corresponding original/reconstructed tracks and aggregate the
    # complete TrackBatchResult statistics exactly as in the audited source.
    return reconstructed, stats
```

The implementation between prefiltering and return must remain the audited
position-first, heading-second least-squares solver, including correction
bounds and filter fallback. Do not add imports from the external repository.

- [ ] **Step 4: Run the complete bundled-solver suite**

Run:

```bash
python -m pytest tests/test_trajectory_batch_optimizer.py -v
```

Expected: all 24 tests pass.

- [ ] **Step 5: Verify source independence and syntax**

Run:

```bash
python -m py_compile src/smart/tokens/trajectory_batch_optimizer.py
rg -n "WOMD-Traffic-Signal-Data-Improvement|PycharmProjects" \
  src/smart/tokens/trajectory_batch_optimizer.py
```

Expected: compilation succeeds; the only repository-name occurrence is the
license attribution, and no local absolute path is present.

- [ ] **Step 6: Commit Task 1**

```bash
git add \
  src/smart/tokens/trajectory_batch_optimizer.py \
  tests/test_trajectory_batch_optimizer.py
git commit -m "feat: bundle batch trajectory optimizer"
```

---

### Task 2: Route `batch` Through the Bundled Implementation

**Files:**
- Modify: `tests/test_womd_trajectory_reconstruction.py`
- Modify: `src/smart/tokens/womd_trajectory_reconstruction.py`

**Interfaces:**
- Consumes:
  - `BatchTrajectoryConfig`;
  - bundled `trajectory_batch_optimizer.reconstruct_scenario_agents`;
  - existing optional external reconstruction entry point.
- Produces:
  - `TrajectoryReconstructionConfig(method="batch")` valid without a root;
  - built-in batch dispatch with configurable linear/angular jerk weights;
  - unchanged explicit external override;
  - `TrajectoryReconstructionConfig(method="optimizer")` requiring a root.

- [ ] **Step 1: Change bridge tests to express the new contract**

Replace `test_advanced_reconstruction_requires_project_root` with:

```python
def test_only_optimizer_requires_project_root(self):
    config = TrajectoryReconstructionConfig(method="batch")
    self.assertTrue(config.is_active)
    self.assertIsNone(config.project_root)

    with self.assertRaisesRegex(ValueError, "optimizer"):
        TrajectoryReconstructionConfig(method="optimizer")
```

Add a bundled parameter-forwarding test:

```python
@patch(
    "src.smart.tokens.trajectory_batch_optimizer."
    "reconstruct_scenario_agents"
)
def test_bundled_batch_receives_exposed_weights(self, reconstruct):
    reconstruct.return_value = ("scenario", "stats")
    config = TrajectoryReconstructionConfig(
        method="batch",
        filter_strength="balanced",
        max_gap_frames=4,
        batch_linear_jerk_weight=1.5,
        batch_angular_jerk_weight=2.5,
    )

    result = reconstruct_scenario_agents(object(), config)

    self.assertEqual(result, ("scenario", "stats"))
    call = reconstruct.call_args
    self.assertEqual(call.kwargs["filter_strength"], "balanced")
    self.assertEqual(call.kwargs["max_gap_frames"], 4)
    self.assertEqual(call.kwargs["config"].linear_jerk_weight, 1.5)
    self.assertEqual(call.kwargs["config"].angular_jerk_weight, 2.5)
```

Retain the current isolated-namespace external test so explicit
`project_root` plus `method="batch"` still calls the external wrapper.

- [ ] **Step 2: Run bridge tests and verify RED**

Run:

```bash
python -m pytest tests/test_womd_trajectory_reconstruction.py -v
```

Expected: the no-root batch configuration and built-in dispatch tests fail.

- [ ] **Step 3: Relax only the batch root requirement**

Change configuration validation to:

```python
if self.method == "optimizer" and not self.project_root:
    raise ValueError(
        "--reconstruction-root is required for optimizer trajectory "
        "reconstruction"
    )
```

Update module/configuration docstrings to describe bundled filter and batch
plus the optional external optimizer.

- [ ] **Step 4: Add lazy bundled-batch dispatch**

Before external entry-point loading, add:

```python
if config.method == "batch" and not config.project_root:
    from .trajectory_batch_optimizer import (
        BatchTrajectoryConfig,
        reconstruct_scenario_agents as reconstruct_with_batch,
    )

    return reconstruct_with_batch(
        scenario,
        config=BatchTrajectoryConfig(
            linear_jerk_weight=config.batch_linear_jerk_weight,
            angular_jerk_weight=config.batch_angular_jerk_weight,
        ),
        filter_strength=config.filter_strength,
        max_gap_frames=config.max_gap_frames,
    )
```

Leave `_load_reconstruction_entrypoint` and the final external call unchanged
for any active method with an explicit `project_root`.

- [ ] **Step 5: Run bridge and solver tests**

Run:

```bash
python -m pytest \
  tests/test_womd_trajectory_reconstruction.py \
  tests/test_trajectory_batch_optimizer.py \
  -v
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  src/smart/tokens/womd_trajectory_reconstruction.py \
  tests/test_womd_trajectory_reconstruction.py
git commit -m "feat: dispatch batch reconstruction inside CatK"
```

---

### Task 3: Expose Self-Contained CLI Provenance and Documentation

**Files:**
- Modify: `tests/test_trajectory_token_comparison.py`
- Modify: `src/smart/tokens/compare_trajectory_token_reconstruction.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `args.method` and optional `args.reconstruction_root`.
- Produces:
  - `_reconstruction_implementation(method, reconstruction_root) -> str`;
  - metadata values `catk_bundled_filter`, `catk_bundled_batch`, or `external`;
  - thread pre-import of the selected bundled backend;
  - help and README commands that omit the external root for batch.

- [ ] **Step 1: Add failing CLI provenance tests**

Import `_reconstruction_implementation` and `parse_args` into
`tests/test_trajectory_token_comparison.py`, then add:

```python
def test_reconstruction_provenance_distinguishes_bundled_batch(self):
    self.assertEqual(
        _reconstruction_implementation("filter", None),
        "catk_bundled_filter",
    )
    self.assertEqual(
        _reconstruction_implementation("batch", None),
        "catk_bundled_batch",
    )
    self.assertEqual(
        _reconstruction_implementation("batch", "/external"),
        "external",
    )


def test_parser_accepts_bundled_batch_without_external_root(self):
    with patch.object(
        sys,
        "argv",
        ["compare", "--input-path", "/training", "--method", "batch"],
    ):
        args = parse_args()
    self.assertEqual(args.method, "batch")
    self.assertIsNone(args.reconstruction_root)
```

Also import `_write_output_readme` and assert that a generated bundled-batch
reproduction command contains no external root:

```python
def test_bundled_batch_reproduction_command_omits_external_root(self):
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory)
        args = argparse.Namespace(
            num_clusters=2048,
            write_reconstructed_tfrecord=False,
            vocab_output_dir="src/smart/tokens",
            vocab_output_name="agent_vocab_reconstructed_batch.pkl",
            input_tfrecord="/training",
            reconstruction_root=None,
            method="batch",
            filter_strength="strong",
            num_workers=24,
            worker_backend="process",
        )
        _write_output_readme(output_dir, args)
        reproduction = (output_dir / "README.md").read_text()

    self.assertIn("--method batch", reproduction)
    self.assertNotIn("--reconstruction-root", reproduction)
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
python -m pytest tests/test_trajectory_token_comparison.py -v
```

Expected: import failure because `_reconstruction_implementation` is missing.

- [ ] **Step 3: Implement deterministic provenance**

Add:

```python
def _reconstruction_implementation(
    method: str,
    reconstruction_root: str | None,
) -> str:
    if reconstruction_root:
        return "external"
    if method == "filter":
        return "catk_bundled_filter"
    if method == "batch":
        return "catk_bundled_batch"
    return "external"
```

Replace the inline summary conditional with:

```python
"implementation": _reconstruction_implementation(
    args.method,
    args.reconstruction_root,
),
```

The `optimizer` no-root case is rejected by
`TrajectoryReconstructionConfig` before this metadata is written.

- [ ] **Step 4: Pre-import the correct bundled backend for threads**

Change the thread pre-import branch to:

```python
if reconstruction_root:
    bridge._load_reconstruction_entrypoint(reconstruction_root)
elif args.method == "filter":
    importlib.import_module(
        "src.smart.tokens.trajectory_filter_reconstructor"
    )
elif args.method == "batch":
    importlib.import_module(
        "src.smart.tokens.trajectory_batch_optimizer"
    )
```

This avoids first-import races without creating a mandatory SciPy import for
the filter path.

- [ ] **Step 5: Update CLI help and README**

Change `--reconstruction-root` help to:

```text
Optional WOMD-Traffic-Signal-Data-Improvement checkout. CatK bundles filter
and batch; an external root is required only for optimizer, or may explicitly
override a bundled implementation.
```

Change the README comparison example to the self-contained batch form:

```bash
python -m src.smart.tokens.compare_trajectory_token_reconstruction \
  --input-path /path/to/womd/training \
  --output-dir outputs/trajectory_token_batch \
  --vocab-output-dir src/smart/tokens \
  --vocab-output-name agent_vocab_reconstructed_batch.pkl \
  --method batch \
  --filter-strength strong \
  --num-clusters 2048 \
  --num-workers 24 \
  --worker-backend process
```

State directly below it that `filter` and `batch` are bundled, while
`--reconstruction-root` is needed only for `optimizer` or an explicit external
override.

- [ ] **Step 6: Add a worker-level bundled-batch smoke test**

Import `WorkerConfig`, `_load_scenario_class`, and `_process_scenario_task`.
Create a 91-frame, constant-speed vehicle Scenario:

```python
scenario_class = _load_scenario_class()
scenario = scenario_class()
scenario.scenario_id = "bundled-batch-smoke"
scenario.current_time_index = 10
scenario.sdc_track_index = 0
scenario.timestamps_seconds.extend(np.arange(91) * 0.1)
track = scenario.tracks.add()
track.id = 1
track.object_type = 1
for index in range(91):
    state = track.states.add()
    state.center_x = 0.2 * index
    state.center_y = 0.0
    state.heading = 0.0
    state.length = 4.8
    state.width = 2.0
    state.height = 1.5
    state.velocity_x = 2.0
    state.valid = True
```

Run it through the normal worker with temporary original/reconstructed cache
directories:

```python
config = WorkerConfig(
    reconstruction_root=None,
    method="batch",
    filter_strength="strong",
    max_gap_frames=-1,
    batch_linear_jerk_weight=1.0,
    batch_angular_jerk_weight=1.0,
    serialize_reconstructed=False,
    original_cache_dir=str(original_dir),
    reconstructed_cache_dir=str(reconstructed_dir),
)
result = _process_scenario_task(
    (0, scenario.SerializeToString(), config)
)
self.assertEqual(result["scenario_id"], "bundled-batch-smoke")
self.assertEqual(result["original_agent_count"], 1)
self.assertEqual(result["reconstructed_agent_count"], 1)
self.assertTrue((original_dir / "bundled-batch-smoke.pkl").is_file())
self.assertTrue(
    (reconstructed_dir / "bundled-batch-smoke.pkl").is_file()
)
```

This test must set up `original_dir` and `reconstructed_dir` inside one
`TemporaryDirectory`; it must not set `PYTHONPATH` or create an external
repository path.

- [ ] **Step 7: Run CLI and bridge regression tests**

Run:

```bash
python -m pytest \
  tests/test_trajectory_token_comparison.py \
  tests/test_womd_trajectory_reconstruction.py \
  -v
```

Expected: all selected tests pass.

- [ ] **Step 8: Verify command help**

Run:

```bash
python -m src.smart.tokens.compare_trajectory_token_reconstruction \
  --help
```

Expected: help lists `filter`, `batch`, and `optimizer`; it says batch is
bundled and does not require `--reconstruction-root`.

- [ ] **Step 9: Commit Task 3**

```bash
git add \
  README.md \
  src/smart/tokens/compare_trajectory_token_reconstruction.py \
  tests/test_trajectory_token_comparison.py
git commit -m "docs: expose self-contained batch reconstruction"
```

---

### Task 4: Full Regression and Deployment Audit

**Files:**
- Verify only; no new files.

**Interfaces:**
- Consumes: all outputs from Tasks 1–3.
- Produces: evidence that the branch is self-contained and ready to push.

- [ ] **Step 1: Run all reconstruction-focused tests**

Run:

```bash
python -m pytest \
  tests/test_trajectory_batch_optimizer.py \
  tests/test_womd_trajectory_reconstruction.py \
  tests/test_trajectory_token_comparison.py \
  -v
```

Expected: every focused test passes.

- [ ] **Step 2: Run the complete CatK test suite**

Run:

```bash
python -m pytest tests -q
```

Expected: all tests pass; environment-dependent tests may retain their existing
skip status.

- [ ] **Step 3: Compile the modified package**

Run:

```bash
python -m compileall -q src/smart/tokens
```

Expected: exit status 0.

- [ ] **Step 4: Prove no external checkout is needed**

Run:

```bash
python - <<'PY'
from types import SimpleNamespace

from src.smart.tokens.womd_trajectory_reconstruction import (
    TrajectoryReconstructionConfig,
    reconstruct_scenario_agents,
)

config = TrajectoryReconstructionConfig(method="batch")
scenario = SimpleNamespace(timestamps_seconds=[], tracks=[])
_, stats = reconstruct_scenario_agents(scenario, config)
print(type(stats).__name__, stats.total_tracks)
PY
```

Expected:

```text
BatchReconstructionStats 0
```

No `PYTHONPATH`, environment variable, or directory named
`WOMD-Traffic-Signal-Data-Improvement` is supplied.

- [ ] **Step 5: Audit diff scope and whitespace**

Run:

```bash
git diff --check
git status --short
git log --oneline -4
```

Expected:

- no whitespace errors;
- only the user's pre-existing `scripts/cache_womd.sh`, `.DS_Store`, and
  `.codex_work/` remain outside the intentional commits;
- three implementation commits follow the approved design-document commit.

- [ ] **Step 6: Report the final development-machine command**

Provide:

```bash
cd /root/workspace/catk
git pull

python -m src.smart.tokens.compare_trajectory_token_reconstruction \
  --input-path /path/to/womd/training \
  --output-dir outputs/trajectory_token_batch \
  --vocab-output-dir src/smart/tokens \
  --vocab-output-name agent_vocab_reconstructed_batch.pkl \
  --method batch \
  --filter-strength strong \
  --num-workers 24 \
  --worker-backend process
```

State explicitly that
`/root/workspace/WOMD-Traffic-Signal-Data-Improvement` must not be created for
this command.
