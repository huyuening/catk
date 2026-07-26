# Exact Batch-Reconstruction Evaluation

This evaluator reruns CatK's bundled `batch` reconstruction directly from the
original WOMD training TFRecords and compares each reconstructed trajectory
with its raw counterpart. It calculates the complete agent-level and
frame-level metric distributions without writing reconstructed TFRecords,
pickle caches, or trajectory datasets.

## Smoke test

Run a small 16-scenario check before the unrestricted evaluation:

```bash
cd /root/workspace/catk

export WOMD_TRAIN=/mnt/pfs/waymo_motion_1_3_0/uncompressed/scenario/training
export RECON_OUTPUT=/mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1
export METRIC_OUTPUT="$RECON_OUTPUT/exact_reconstruction_metrics_smoke"
export METRIC_SCRATCH="$RECON_OUTPUT/.exact_metric_scratch_smoke"

python -m src.smart.tokens.evaluate_trajectory_reconstruction \
  --input-path "$WOMD_TRAIN" \
  --reconstruction-run-config "$RECON_OUTPUT/run_config.json" \
  --output-dir "$METRIC_OUTPUT" \
  --scratch-dir "$METRIC_SCRATCH" \
  --workers 8 \
  --max-scenarios 16
```

The supplied `run_config.json` must describe a `batch` reconstruction. The
evaluator copies its filter strength, missing-gap limit, linear-jerk weight,
and angular-jerk weight so that the metric run reproduces the current
reconstruction settings.

## Complete training split

The complete run may be started with `--resume`; a new scratch directory is
initialized automatically, while a retained compatible checkpoint is resumed:

```bash
cd /root/workspace/catk

export WOMD_TRAIN=/mnt/pfs/waymo_motion_1_3_0/uncompressed/scenario/training
export RECON_OUTPUT=/mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1
export METRIC_OUTPUT="$RECON_OUTPUT/exact_reconstruction_metrics"
export METRIC_SCRATCH="$RECON_OUTPUT/.exact_metric_scratch"

python -m src.smart.tokens.evaluate_trajectory_reconstruction \
  --input-path "$WOMD_TRAIN" \
  --reconstruction-run-config "$RECON_OUTPUT/run_config.json" \
  --output-dir "$METRIC_OUTPUT" \
  --scratch-dir "$METRIC_SCRATCH" \
  --workers 24 \
  --resume
```

Each completed TFRecord shard is committed atomically. If the process stops,
run the same command again: completed shards are skipped and the interrupted
shard is recomputed from its beginning. Resume is rejected if an input shard's
path, size, or modification time changes; if the reconstruction settings,
metric schema, or `--max-scenarios` value changes; or if the scratch directory
contains unrelated files.

Exact percentiles require disk-backed float64 scalar buffers. Reserve
approximately 70–90 GB for `METRIC_SCRATCH` during the unrestricted run. The
scratch directory is deleted after all final files are written successfully.
It is retained after a failure so the run can resume. Add `--keep-scratch`
only when the scalar buffers and checkpoint need to be audited after success.

## Metrics and support

The evaluator reports every encountered object type and their combined `all`
scope.

- Agent-level linear and angular jerk values are non-negative RMS values, one
  value per eligible agent trajectory.
- Frame-level linear and angular jerk values are signed samples, so their
  distribution preserves acceleration-change direction.
- `xy_rmse` is one agent-level value calculated on frames where both the raw
  and reconstructed XY positions are valid.
- `raw_matched_support` and `reconstructed_matched_support` are paired. To
  avoid comparing different temporal supports, a track contributes paired
  jerk samples only when every raw-computable jerk center is also computable
  after reconstruction.
- `reconstructed_full_support` includes every finite jerk center made valid by
  reconstruction, including supported gap filling. The coverage metrics show
  how much of the raw support was matched.

Every metric row contains:

```text
count, mean, std, min, max, range, p01, p99, p99_minus_p01
```

`std` is the population standard deviation (`ddof=0`). `p01` and `p99` are
exact over the stored float64 values and use NumPy's linear interpolation
definition; they are not estimated from a sketch or sample.

## Final files

`METRIC_OUTPUT` contains exactly:

- `agent_summary.csv`: agent RMS jerk, support coverage, and XY RMSE.
- `frame_jerk_summary.csv`: signed raw/reconstructed frame jerk distributions.
- `summary.json`: the same metric rows plus support and statistical metadata.
- `reconstruction_summary.json`: accumulated batch-reconstruction counters
  and derived track/segment rates.
- `run_config.json`: resolved input metadata, reconstruction settings, and
  evaluation options.

No reconstructed trajectory data is serialized. The temporary scratch area
contains only scalar metric streams and the shard checkpoint.
