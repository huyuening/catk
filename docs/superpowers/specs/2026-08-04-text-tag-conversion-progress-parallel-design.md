# Text Tag Conversion Progress and Parallelism Design

**Date:** 2026-08-04

## Objective

Add visible progress reporting and bounded multiprocessing to the existing
WOMD action-row to ECoSim tag converter. The change must preserve the current
tag semantics, output layout, deterministic mapping, streaming input behavior,
and train/validation-only data boundary.

The production input is one approximately 48 GB compressed CSV for the
training split. The implementation must not preload that file or materialize
an uncompressed copy.

## Command-Line Contract

`src.smart.datasets.build_text_control_tags` gains two options:

```text
--workers INTEGER          default: 80, minimum: 1
--progress-every INTEGER   default: 1000, minimum: 1
```

The shell wrapper exposes matching environment variables:

```text
TAG_WORKERS                default: 80
TAG_PROGRESS_EVERY         default: 1000
```

Operators can use `TAG_WORKERS=1` to retain serial execution when diagnosing
an input or filesystem problem.

## Processing Architecture

The main process remains the sole CSV reader. It sequentially decompresses
each input file, validates required columns, and groups contiguous rows by
`scenario_id`. This preserves the current low-memory behavior and detects a
scenario that reappears after another scenario.

For each complete scenario, the main process validates its `global_index` and
`dataset_current_time_index`, checks scene-ID ownership, records the eventual
scenario mapping, and submits one self-contained task to a
`ProcessPoolExecutor`.

Each worker:

1. derives the existing direction and longitudinal tags;
2. builds the existing half-open action intervals;
3. writes the scenario JSON through the existing atomic temporary-file rename;
4. returns its row count and emitted-tag count to the main process.

At most `workers * 2` tasks may be outstanding. Once the limit is reached,
the reader waits for at least one worker to finish before reading more
scenarios. With the default configuration, no more than 160 complete scenarios
are waiting in worker queues, which bounds memory independently of dataset
size.

Parallel completion order must not affect artifacts. Each scenario owns a
different JSON path, and the final mapping JSON remains key-sorted. The mapping
file is written only after every worker completes successfully.

## Progress Output

The main process writes progress to standard error with immediate flushing so
it remains visible in an interactive terminal, `tmux`, or a redirected log.
It prints an initial configuration line, one line after every configured
number of completed scenarios, and one final line.

Each progress line contains:

```text
split, completed scenarios, submitted scenarios, completed source rows,
emitted tags, pending tasks, scene rate, elapsed wall time
```

Example:

```text
[text-tags train] completed=10000 submitted=10142 rows=28351420
tags=3872105 pending=142 rate=42.7 scenes/s elapsed=00:03:54
```

No percentage or ETA is claimed because the compressed CSV does not contain a
reliable total scenario count and obtaining one would require a full extra
pass. The completed counters and rates are exact.

## Failure Behavior

- Invalid worker counts and progress intervals fail before input processing.
- A worker exception is re-raised in the main process; pending work is
  cancelled where possible.
- The final mapping file is not written after a failed conversion.
- Scenario JSON files completed before a failure may remain, matching the
  converter's existing resumability characteristics; each file is atomic and
  can safely be overwritten by a later full rerun.
- The `test` split remains rejected, and no testing/runtime future labels are
  introduced.

## Performance Boundary

Multiprocessing parallelizes interval construction and the large number of
small JSON writes. Reading, gzip decompression, CSV parsing, and scenario
grouping remain sequential because the current input is a single gzip stream.
Consequently, 80 workers are an operator-selected concurrency default, not a
promise of 80-fold speedup. `TAG_WORKERS` allows lowering concurrency if the
parallel filesystem shows metadata contention.

## Verification

Tests will establish:

1. CLI and shell defaults are 80 workers and a 1000-scenario reporting period.
2. Serial and multiprocessing conversions produce identical tag JSON and
   mapping JSON artifacts.
3. A one-scenario reporting period emits initial, periodic, and final progress
   with the required counters.
4. Invalid concurrency arguments fail before conversion.
5. Existing tag derivation, interval, split-rejection, and output-layout tests
   continue to pass.
