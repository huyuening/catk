# Bundled Batch Trajectory Reconstruction Design

## Goal

Make CatK's `batch` trajectory-reconstruction method fully self-contained.
After this change, a development machine containing only
`/root/workspace/catk` can construct the batch-reconstructed training
vocabulary without creating or installing a separate
`WOMD-Traffic-Signal-Data-Improvement` checkout.

The intended command remains:

```bash
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

No `--reconstruction-root` is required for `--method batch`.

## Scope

- Port the complete batch optimizer into `src/smart/tokens`.
- Use CatK's already bundled geometric filter as the batch prefilter.
- Preserve the current full-trajectory, training-only, vocabulary-only data
  boundary.
- Preserve all existing batch defaults, safety checks, statistics, and
  filter fallbacks.
- Keep the legacy external loading path as an optional compatibility override.
- Keep `optimizer` external; this change does not port the CasADi-based legacy
  optimizer.
- Do not change model inputs, future labels, runtime token matching, training
  loss, or validation logic.
- Do not commit reconstructed caches, TFRecords, vocabularies, or solver
  outputs.

## Approaches Considered

### Exact in-package port

Adapt the existing batch module into CatK, retain its numerical defaults, and
replace its filter import with CatK's bundled filter.

This is selected because it reproduces the already tested method, requires no
new dependency, and minimizes divergence between local and remote results.

### Reduced CatK-specific batch solver

Reimplement only position and heading least-squares objectives in a smaller
module.

This would reduce code size, but it would omit or duplicate mature behavior
around nonuniform timestamps, object-class scaling, reverse motion, sparse
pedestrian tracks, solver bounds, safety rejection, and partial fallback.
Numerical behavior could no longer be described as the same batch method.

### Vendor the complete external project

Copy the external repository or install it as a package/submodule.

This preserves behavior but recreates the deployment dependency the change is
intended to remove, adds unrelated tools and proto code to CatK, and complicates
versioning. It is not selected.

## Bundled Module

Add:

```text
src/smart/tokens/trajectory_batch_optimizer.py
```

The module is an adapted copy of the existing batch implementation and imports
only:

- NumPy;
- `scipy.optimize.least_squares`;
- `scipy.sparse`;
- CatK's bundled `trajectory_filter_reconstructor`.

SciPy is already pinned in `install/requirements.txt`; no environment change is
needed.

The source attribution and PolyForm Noncommercial License 1.0.0 notice will be
placed at the top of the module. The existing
`LICENSE.WOMD_TRAJECTORY_RECONSTRUCTION.txt` covers the adapted filter and batch
reconstruction code.

## Algorithm

For every eligible vehicle, pedestrian, or cyclist track, batch reconstruction
keeps the established two-stage geometry-first procedure.

### Prefilter and gap handling

1. Deep-copy the Scenario through the bundled geometric filter.
2. Fill accepted internal gaps according to `filter_strength` and
   `max_gap_frames`.
3. Use the filtered positions and headings as stable observations and as the
   fallback result.
4. Split processing only at unsupported trajectory gaps. A continuous
   91-frame track is optimized as one continuous segment; it is not optimized
   independently per six-frame CatK token.

### Position optimization

For each supported segment with at least seven frames:

1. Optimize `x` and `y` jointly over the complete segment.
2. Penalize displacement from filtered position observations.
3. Penalize scalar linear jerk and adjacent-frame planar vector jerk.
4. Apply agent-class and short-segment regularization scales.
5. Bound corrections separately for originally observed and filter-filled
   frames.

The `batch_linear_jerk_weight` CLI value maps to the batch configuration's
`linear_jerk_weight`. Other established position defaults remain unchanged.

### Heading optimization

1. Derive motion-aware heading observations from the optimized positions and
   raw/prefilter headings.
2. Preserve reverse-aware vehicle heading and object-class-specific heading
   behavior.
3. Optimize unwrapped heading against heading fidelity, heading-rate fidelity,
   angular jerk, and adjacent-frame angular jerk.
4. Wrap the final heading back to the normal angular range.

The `batch_angular_jerk_weight` CLI value maps to
`angular_jerk_weight`. Other established heading defaults remain unchanged.

### Safety and fallback

Solver failure, non-finite output, excessive trusted-point correction, or a
post-optimization kinematic safety violation must not corrupt a scenario.

- A failed position stage keeps the filtered positions.
- A failed heading stage keeps the filtered headings.
- A rejected segment retains its filtered result.
- Other valid tracks and segments in the scenario continue processing.
- Returned statistics record solver failures, limited segments, optimized
  frames, and before/after acceleration and jerk RMS values.

The batch method therefore always has the bundled filter as its deterministic
best-effort baseline.

## Dispatch and Compatibility

Update `TrajectoryReconstructionConfig` and
`reconstruct_scenario_agents` with the following resolution rules:

| Method | `project_root` absent | `project_root` supplied |
| --- | --- | --- |
| `none` | no-op | no-op; path remains unused |
| `filter` | CatK bundled filter | existing external override |
| `batch` | CatK bundled batch | existing external override |
| `optimizer` | clear configuration error | existing external optimizer |

For bundled batch dispatch:

1. Construct `BatchTrajectoryConfig` with the two exposed jerk weights.
2. Call the bundled batch reconstructor with `filter_strength` and
   `max_gap_frames`.
3. Return the reconstructed Scenario and batch statistics through the existing
   bridge API.

This preserves old explicit external commands while making the normal batch
command independent of the external repository.

`show_solver_warnings` remains an external-optimizer compatibility option and
has no effect on bundled batch, matching current batch behavior.

## CLI and Provenance

Update command help and generated comparison metadata:

- `--reconstruction-root` is optional for `filter` and `batch`;
- it is required only for `optimizer`;
- summary metadata reports `catk_bundled_filter`,
  `catk_bundled_batch`, or `external`;
- generated reproduction commands omit `--reconstruction-root` when bundled
  batch was used;
- README examples show the self-contained batch command.

When process workers are selected, each worker imports the bundled module from
CatK. The parent process may pre-import it for threaded execution, as already
done for the filter. No worker reads an external source directory.

## Data Boundary

The batch method remains an offline vocabulary-source transformation:

- only training TFRecords are reconstructed;
- each available training trajectory may use all 91 frames;
- reconstructed trajectories are segmented into CatK's normal 0.5-second
  vocabulary samples only after full-trajectory reconstruction;
- raw and reconstructed branches retain the same selected agents and shapes;
- reconstructed per-scenario caches and optional TFRecords are analysis
  artifacts only;
- normal CatK training, validation, and testing caches remain untouched.

The final batch vocabulary must be used consistently with its matching
transition-dynamics lookup and experiment configuration. This change does not
silently replace `agent_vocab_555_s2.pkl`.

## Errors

- `method="batch"` without `project_root` must initialize successfully.
- `method="optimizer"` without `project_root` must fail immediately with an
  error that says `--reconstruction-root` is required for optimizer.
- An explicit missing external entry point must retain the current
  `FileNotFoundError`.
- Missing SciPy must fail on selecting bundled batch with a clear import error;
  filter and `none` must remain usable because batch is imported lazily.
- Invalid method, filter strength, or solver configuration continues to fail
  before dataset preprocessing begins.

## Verification

### Ported numerical tests

Adapt the external batch tests to CatK's test environment, including:

- WOSAC-centered acceleration and jerk feature parity;
- nonuniform timestamp support;
- adjacent-frame sawtooth suppression;
- preservation of constant linear/angular acceleration;
- reverse vehicle heading;
- low-speed heading fallback;
- noisy motion-direction protection;
- sparse pedestrian gap handling;
- cyclist heading behavior;
- partial filter fallback and solver-failure accounting.

### Bridge tests

- bundled batch no longer requires `project_root`;
- bundled batch reconstructs an empty Scenario without an external checkout;
- configured linear/angular jerk weights reach `BatchTrajectoryConfig`;
- explicit external batch override still dispatches in an isolated namespace;
- optimizer still requires the external root;
- filter and disabled behavior remain unchanged.

### CLI tests

- argument validation accepts `--method batch` without
  `--reconstruction-root`;
- generated metadata identifies the bundled batch implementation;
- generated reproduction instructions omit an external path;
- a small Scenario smoke test completes through the normal comparison worker.

### Regression verification

Run the focused reconstruction tests, the complete CatK unit suite, import
compilation, and `git diff --check`. No test may require the external
`WOMD-Traffic-Signal-Data-Improvement` directory.

## Deployment Result

After the implementation commit is pulled onto the development machine, the
only required source checkout is:

```text
/root/workspace/catk
```

The batch vocabulary is generated with `--method batch`; neither an environment
variable nor a hidden lookup path will refer to
`/root/workspace/WOMD-Traffic-Signal-Data-Improvement`.
