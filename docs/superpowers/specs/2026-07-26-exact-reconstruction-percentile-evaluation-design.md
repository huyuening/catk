# Exact Reconstruction Percentile Evaluation Design

## Goal

Add a self-contained CatK evaluator that reproduces the established
raw-WOMD-versus-batch-reconstruction metric protocol over the complete training
split. The evaluator must calculate exact first and ninety-ninth percentiles
for both agent-level and frame-level metrics while never serializing a
reconstructed trajectory dataset.

The intended production input is:

```text
/mnt/pfs/waymo_motion_1_3_0/uncompressed/scenario/training
```

The reconstruction settings must be read from the completed comparison run:

```text
/mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1/run_config.json
```

## Scope

The evaluator will:

- read the original WOMD training TFRecords;
- reconstruct each scenario in memory with CatK's bundled batch method;
- use the same raw-valid and reconstructed support policy as the earlier
  reconstruction evaluation;
- calculate agent-level linear-jerk RMS, angular-jerk RMS, support coverage,
  and XY trajectory RMSE;
- calculate frame-level signed linear jerk and signed angular jerk;
- report overall and object-type-specific summaries;
- retain exact count, population mean, population standard deviation, minimum,
  maximum, and full range;
- add exact `p01`, `p99`, and `p99_minus_p01`;
- checkpoint at TFRecord-shard boundaries so an interrupted full evaluation can
  resume without reconstructing completed shards;
- remove temporary scalar-metric buffers after a successful finalization.

The evaluator will not:

- save reconstructed TFRecords, pickle caches, or trajectories;
- consume the vocabulary-comparison caches as the raw baseline;
- change the reconstruction algorithm or its parameters;
- train a vocabulary or model;
- use approximate quantile sketches.

## Why the Original TFRecords Are Required

`datasets/original/training` in the comparison output is a CatK legacy cache.
It expands validity through internal gaps and linearly interpolates trajectory
values. It therefore cannot reproduce the earlier raw-WOMD metric support.

For strict comparability, the evaluator must start from each original WOMD
track and its original validity mask, run the same batch reconstruction again,
measure the pair, and discard the reconstructed scenario immediately.

## Command-Line Interface

The entry point will be:

```bash
python -m src.smart.tokens.evaluate_trajectory_reconstruction \
  --input-path /mnt/pfs/waymo_motion_1_3_0/uncompressed/scenario/training \
  --reconstruction-run-config \
    /mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1/run_config.json \
  --output-dir \
    /mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1/exact_reconstruction_metrics \
  --scratch-dir \
    /mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1/.exact_metric_scratch \
  --workers 24 \
  --resume
```

The CLI will also support:

- `--max-scenarios N` for a smoke test;
- `--progress-every N` for progress refresh frequency;
- one TFRecord file, a training directory, or a quoted shard glob as
  `--input-path`;
- `--keep-scratch` for deliberate metric-buffer auditing.

`--workers` changes only parallel scenario evaluation. It does not alter any
metric definition.

## Reconstruction Configuration

The evaluator will read these keys from `run_config.json`:

- `method`;
- `filter_strength`;
- `max_gap_frames`;
- `batch_linear_jerk_weight`;
- `batch_angular_jerk_weight`.

The production evaluator requires `method == "batch"` and invokes CatK's
bundled implementation. Cluster settings, vocabulary settings, worker counts,
stage selection, output paths, and an external reconstruction checkout are
not used.

The relevant values will be copied into the evaluator's own `run_config.json`
so the metric result remains independently auditable.

## Metric Definitions

### Kinematic stencil

The evaluator will reuse CatK's bundled centered-difference kinematics, which
matches the previous evaluation:

- linear jerk is the centered derivative of linear acceleration derived from
  three-dimensional speed magnitude, in `m/s^3`;
- angular jerk is the centered derivative of angular acceleration derived from
  wrapped heading differences, in `rad/s^3`.

Only finite samples whose required stencil frames are valid contribute.

### Matched support

For each raw/reconstructed track pair:

1. Compute the raw jerk support from the original WOMD validity mask.
2. Compute the reconstructed jerk support.
3. Form their intersection.
4. Accept paired raw and reconstructed jerk values only if every
   raw-computable sample is also reconstructed-computable.

If reconstructed support loses any raw-computable center, paired RMS and frame
samples for that track are omitted rather than silently changing support.
Coverage is still reported.

### Reconstructed full support

The reconstructed-full variants use every finite jerk center supported by the
completed reconstructed trajectory, including filled intervals. These values
are secondary smoothness diagnostics and do not replace the matched comparison.

### Agent-level metrics

For each track:

- `raw_linear_jerk_rms_mps3`;
- `reconstructed_linear_jerk_rms_mps3`;
- `reconstructed_full_linear_jerk_rms_mps3`;
- `linear_jerk_matched_coverage`;
- `raw_angular_jerk_rms_radps3`;
- `reconstructed_angular_jerk_rms_radps3`;
- `reconstructed_full_angular_jerk_rms_radps3`;
- `angular_jerk_matched_coverage`;
- `xy_rmse_m`.

XY RMSE is:

```text
sqrt(mean((x_reconstructed - x_raw)^2
        + (y_reconstructed - y_raw)^2))
```

over frames that are valid and finite in both tracks.

### Frame-level metrics

The frame summary contains:

- raw linear jerk on matched support;
- reconstructed linear jerk on matched support;
- reconstructed linear jerk on full support;
- raw angular jerk on matched support;
- reconstructed angular jerk on matched support;
- reconstructed angular jerk on full support.

Frame jerk remains signed. Consequently, `p01` and `p99` define a robust signed
central range rather than an absolute-magnitude range.

### Scopes

Every summary is emitted for:

- `all`;
- `vehicle`;
- `pedestrian`;
- `cyclist`;
- any additional WOMD object type encountered, under its stable type name.

## Exact Summary Statistics

For every finite metric stream, output:

```text
count
mean
std
min
max
range
p01
p99
p99_minus_p01
```

`std` is the population standard deviation (`ddof=0`). `range` is
`max - min`.

Percentiles use NumPy's linear definition. For a sorted stream of length `n`
and quantile `q`, the fractional index is `(n - 1) * q`; the result linearly
interpolates between the surrounding order statistics. Empty streams report
`count == 0`; their remaining statistics are JSON `null` and blank CSV cells.

## Exact Percentile Storage

Approximate sketches are intentionally excluded. Retaining all frame values
as Python objects or concatenating them in RAM would create excessive
temporary memory amplification, even on a large-memory development machine.

Instead, the evaluator will:

1. Maintain exact mergeable moments in memory.
2. Append every finite scalar value as float64 to a raw per-scope,
   per-metric scratch buffer.
3. Store only object-type buffers during evaluation, avoiding a duplicate
   `all` copy.
4. Memory-map each buffer during finalization.
5. Use in-place multi-index partition to obtain the lower and upper order
   statistics needed by `p01` and `p99`.
6. Build one temporary, one-metric global buffer at a time to calculate the
   `all` scope, then delete it before moving to the next metric.

This is exact relative to the float64 metric values. Expected peak scratch use
for the complete training split is approximately 70--90 GB, depending on valid
support. Scratch contains scalar metric streams only, never trajectory data.

## Parallel Processing and Bounded Memory

Scenarios are reconstructed in a process pool. The parent keeps at most twice
the worker count in flight. Each worker returns:

- per-track scalar agent metrics;
- per-type frame-metric NumPy arrays for that scenario;
- reconstruction counters;
- scenario identity and failure context.

The reconstructed protocol buffer is never returned or serialized. Results
are consumed in deterministic input order so moment merging and output ordering
remain reproducible.

## Shard Checkpointing and Resume

The evaluator processes sorted TFRecord shards in order. At each completed
shard it:

1. flushes and synchronizes all scalar buffers;
2. records their exact byte lengths and sample counts;
3. merges shard moments and reconstruction counters;
4. atomically replaces a scratch checkpoint manifest.

On `--resume`, the evaluator validates:

- the ordered input-shard list and file metadata;
- the reconstruction configuration;
- metric schema and float dtype;
- every recorded scratch-buffer length.

Any bytes beyond the last committed offsets are truncated before work resumes.
Completed shards are skipped. A configuration or input mismatch aborts rather
than mixing incompatible measurements.

A run using `--max-scenarios` is a separate smoke-test identity and cannot
resume into an unrestricted full run.

## Failure Handling

A scenario failure aborts the current shard with its file, record index,
scenario ID when available, and traceback. Previously committed shards remain
resumable. The failed shard is recomputed from its beginning on the next run.

Final CSV and JSON files are written through temporary paths and atomically
renamed only after:

- every selected scenario succeeds;
- sample counts agree with scratch-buffer lengths;
- all moments and percentile values are finite or correctly empty;
- all requested scopes and metric variants are present.

Scratch is removed after successful finalization unless `--keep-scratch` is
set. Scratch is retained after failure to permit resume.

## Final Outputs

The output directory retains only:

```text
agent_summary.csv
frame_jerk_summary.csv
summary.json
reconstruction_summary.json
run_config.json
```

`agent_summary.csv` and `frame_jerk_summary.csv` extend the previous schema with
`p01`, `p99`, and `p99_minus_p01`.

`summary.json` records:

- scenario, failure, and agent counts;
- exact statistic definitions;
- support definitions;
- reconstruction provenance;
- agent and frame summary rows.

`reconstruction_summary.json` aggregates the bundled batch reconstruction
counters and derived rates. `run_config.json` records resolved inputs,
reconstruction settings, worker settings, scratch policy, exact percentile
method, and software provenance.

## Testing

Unit tests will cover:

- exact linear `p01` and `p99` interpolation for odd, even, singleton, and
  empty streams;
- non-finite filtering;
- memory-mapped multi-index partition parity with `numpy.percentile`;
- population-moment merging;
- raw matched-support and reconstructed-full-support behavior;
- signed frame-jerk percentiles;
- XY RMSE;
- overall and object-type scopes;
- reconstruction-configuration loading and validation;
- deterministic TFRecord discovery;
- shard checkpoint commit, partial-append truncation, and resume;
- rejection of changed inputs or reconstruction parameters;
- atomic final output generation;
- absence of reconstructed TFRecord and pickle outputs.

A small synthetic TFRecord integration test will run bundled batch
reconstruction end to end and compare all reported statistics with direct
NumPy calculations.
