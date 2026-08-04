# Text Tag Conversion Progress and Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact progress counters and bounded 80-process scenario conversion to the WOMD-to-ECoSim tag builder without changing generated labels.

**Architecture:** One main process continues to stream and group the single gzip input. Complete scenarios are submitted to a bounded `ProcessPoolExecutor`; workers build intervals and atomically write independent JSON files, while the main process owns validation, mappings, counters, and stderr progress output.

**Tech Stack:** Python standard library (`argparse`, `concurrent.futures`, `csv`, `gzip`, `time`), Bash, `unittest`.

## Global Constraints

- Default to exactly 80 workers and permit an explicit serial fallback with one worker.
- Allow at most `workers * 2` outstanding scenario tasks.
- Report progress every 1000 completed scenarios by default.
- Keep input reading and scenario grouping streaming and single-process.
- Preserve current tag rules, half-open intervals, paths, atomic writes, sorted mapping, and train/validation-only enforcement.
- Write progress to standard error with immediate flushing.
- Add no third-party dependency or uncompressed intermediate file.
- Do not touch unrelated user changes already present in the worktree.

---

### Task 1: Bounded multiprocessing and exact progress reporting

**Files:**
- Modify: `tests/test_build_text_control_tags.py`
- Modify: `src/smart/datasets/build_text_control_tags.py`

**Interfaces:**
- Consumes: existing `build_intervals`, `_iter_rows`, `_iter_scenarios`, and `_write_json`.
- Produces: `convert_action_rows(..., workers: int = 80, progress_every: int = 1000, progress_stream: TextIO | None = None) -> Dict[str, str]`.
- Produces CLI flags `--workers` and `--progress-every`.
- Worker operation returns `(row_count: int, tag_count: int)` and never owns the scenario mapping.

- [ ] **Step 1: Write failing tests**

Add `io` and `subprocess` imports. Make the existing CLI layout test explicitly pass `--workers 1`, then add:

```python
def test_parser_defaults_to_eighty_workers_and_one_thousand_scenarios(self):
    args = BUILD_TAGS.build_parser().parse_args(
        [
            "--input", str(self.input_path),
            "--output-root", str(self.output_root),
            "--split", "train",
            "--mapping-output", str(self.mapping_path),
        ]
    )
    self.assertEqual(args.workers, 80)
    self.assertEqual(args.progress_every, 1000)


def test_invalid_parallel_settings_fail_before_reading_input(self):
    common = {
        "input_paths": [self.root / "missing.csv.gz"],
        "output_root": self.output_root,
        "split": "train",
        "mapping_output": self.mapping_path,
    }
    with self.assertRaisesRegex(ValueError, "workers"):
        BUILD_TAGS.convert_action_rows(**common, workers=0)
    with self.assertRaisesRegex(ValueError, "progress_every"):
        BUILD_TAGS.convert_action_rows(**common, progress_every=0)


def test_serial_conversion_reports_exact_progress(self):
    rows = []
    for scenario_index in range(2):
        rows.extend(
            BuildTextControlTagsTest.row(
                frame,
                "LEFT_TURN",
                scenario_id=f"waymo-{scenario_index}",
                global_index=17 + scenario_index,
                track_id=100 + scenario_index,
            )
            for frame in range(11, 31)
        )
    self.write_rows(rows)
    progress = io.StringIO()
    BUILD_TAGS.convert_action_rows(
        input_paths=[self.input_path],
        output_root=self.output_root,
        split="train",
        mapping_output=self.mapping_path,
        workers=1,
        progress_every=1,
        progress_stream=progress,
    )
    output = progress.getvalue()
    self.assertIn("status=start", output)
    self.assertIn("workers=1", output)
    self.assertIn("completed=1", output)
    self.assertIn("status=complete", output)
    self.assertIn("completed=2", output)
    self.assertIn("rows=40", output)
    self.assertIn("pending=0", output)
```

Add helpers and a real subprocess comparison. Invoke the file directly rather than `python -m` because the local lightweight test environment does not install `torch_geometric`, which the package initializer imports:

```python
def cli_args(self, output_root, mapping_path, *, workers):
    return [
        sys.executable,
        str(MODULE_PATH),
        "--input", str(self.input_path),
        "--output-root", str(output_root),
        "--split", "train",
        "--mapping-output", str(mapping_path),
        "--workers", str(workers),
        "--progress-every", "1",
    ]


@staticmethod
def artifact_bytes(root):
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_parallel_and_serial_cli_outputs_are_identical(self):
    rows = []
    for scenario_index in range(6):
        rows.extend(
            BuildTextControlTagsTest.row(
                frame,
                "RIGHT_LANE_CHANGE" if scenario_index % 2 else "LEFT_TURN",
                scenario_id=f"waymo-{scenario_index}",
                global_index=100 + scenario_index,
                track_id=1000 + scenario_index,
            )
            for frame in range(11, 31)
        )
    self.write_rows(rows)
    serial_root = self.root / "serial"
    parallel_root = self.root / "parallel"
    serial = subprocess.run(
        self.cli_args(serial_root, serial_root / "mapping.json", workers=1),
        cwd=MODULE_PATH.parents[3],
        text=True,
        capture_output=True,
        check=False,
    )
    parallel = subprocess.run(
        self.cli_args(parallel_root, parallel_root / "mapping.json", workers=2),
        cwd=MODULE_PATH.parents[3],
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(serial.returncode, 0, serial.stderr)
    self.assertEqual(parallel.returncode, 0, parallel.stderr)
    self.assertEqual(
        self.artifact_bytes(serial_root),
        self.artifact_bytes(parallel_root),
    )
    self.assertIn("status=complete", parallel.stderr)
    self.assertIn("completed=6", parallel.stderr)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python -m unittest \
  tests.test_build_text_control_tags.BuildTextControlTagsCliTest.test_parser_defaults_to_eighty_workers_and_one_thousand_scenarios \
  tests.test_build_text_control_tags.BuildTextControlTagsCliTest.test_invalid_parallel_settings_fail_before_reading_input \
  tests.test_build_text_control_tags.BuildTextControlTagsCliTest.test_serial_conversion_reports_exact_progress \
  tests.test_build_text_control_tags.BuildTextControlTagsCliTest.test_parallel_and_serial_cli_outputs_are_identical \
  -v
```

Expected: FAIL because the parser and conversion API do not accept the new options.

- [ ] **Step 3: Implement progress reporting**

Import `FIRST_COMPLETED`, `ProcessPoolExecutor`, `wait`, `sys`, and `time`. Add:

```python
def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _ProgressReporter:
    def __init__(self, *, split, workers, every, input_count, stream):
        self.split = split
        self.workers = workers
        self.every = every
        self.input_count = input_count
        self.stream = stream
        self.started_at = time.monotonic()
        self.submitted_count = 0
        self.completed_count = 0
        self.row_count = 0
        self.tag_count = 0

    def start(self):
        self._emit("start")

    def submitted(self):
        self.submitted_count += 1

    def completed(self, row_count, tag_count):
        self.completed_count += 1
        self.row_count += row_count
        self.tag_count += tag_count
        if self.completed_count % self.every == 0:
            self._emit("running")

    def finish(self):
        self._emit("complete")

    def _emit(self, status):
        elapsed = max(time.monotonic() - self.started_at, 0.0)
        rate = self.completed_count / elapsed if elapsed else 0.0
        pending = self.submitted_count - self.completed_count
        print(
            f"[text-tags {self.split}] status={status} workers={self.workers} "
            f"inputs={self.input_count} completed={self.completed_count} "
            f"submitted={self.submitted_count} rows={self.row_count} "
            f"tags={self.tag_count} pending={pending} "
            f"rate={rate:.1f} scenes/s elapsed={_format_elapsed(elapsed)}",
            file=self.stream,
            flush=True,
        )
```

Add a top-level, pickleable worker:

```python
def _write_scene_tags(
    rows: Sequence[Mapping[str, str]],
    tag_path: str | Path,
    stop_speed_mps: float,
    acceleration_threshold: float,
) -> Tuple[int, int]:
    tags = build_intervals(
        rows,
        stop_speed_mps=stop_speed_mps,
        accel_mps2=acceleration_threshold,
    )
    _write_json(Path(tag_path), tags)
    return len(rows), len(tags)
```

- [ ] **Step 4: Implement serial and bounded parallel scheduling**

Extend `convert_action_rows` with the three parameters from the interface. Validate `workers >= 1` and `progress_every >= 1` before opening input. Use `sys.stderr` when no progress stream is supplied.

Keep scene ownership and mapping validation in the main process. For one worker call `_write_scene_tags` directly. For more workers, maintain a future set and cap it:

```python
max_pending = workers * 2
future = executor.submit(
    _write_scene_tags,
    rows,
    tag_path,
    stop_speed_mps,
    acceleration_threshold,
)
pending.add(future)
reporter.submitted()
if len(pending) >= max_pending:
    done, pending = wait(pending, return_when=FIRST_COMPLETED)
    for completed_future in done:
        row_count, tag_count = completed_future.result()
        reporter.completed(row_count, tag_count)
```

Drain remaining futures after input exhaustion. On error, cancel pending futures and call `shutdown(wait=True, cancel_futures=True)` before re-raising. Only after all tasks succeed, write the sorted mapping and call `reporter.finish()`.

Add and forward:

```python
parser.add_argument("--workers", type=int, default=80)
parser.add_argument("--progress-every", type=int, default=1000)
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command again.

Expected: all four focused tests PASS and serial/parallel artifacts are byte-identical.

- [ ] **Step 6: Run the complete converter test module**

Run:

```bash
python -m unittest tests.test_build_text_control_tags -v
```

Expected: all tests PASS without traceback or resource warnings.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/smart/datasets/build_text_control_tags.py tests/test_build_text_control_tags.py
git commit -m "feat: parallelize text tag conversion"
```

---

### Task 2: Shell controls, operator documentation, and final verification

**Files:**
- Modify: `tests/test_build_text_control_tags.py`
- Modify: `scripts/build_text_control_tags.sh`
- Modify: `docs/text_control_pre_bc.md`

**Interfaces:**
- Consumes: Task 1 CLI flags.
- Produces: `TAG_WORKERS` defaulting to 80 and `TAG_PROGRESS_EVERY` defaulting to 1000.

- [ ] **Step 1: Write the failing wrapper test**

```python
def test_shell_wrapper_forwards_parallel_defaults(self):
    script_path = MODULE_PATH.parents[3] / "scripts" / "build_text_control_tags.sh"
    script = script_path.read_text(encoding="utf-8")
    self.assertIn('TAG_WORKERS="${TAG_WORKERS:-80}"', script)
    self.assertIn(
        'TAG_PROGRESS_EVERY="${TAG_PROGRESS_EVERY:-1000}"',
        script,
    )
    self.assertIn('--workers "${TAG_WORKERS}"', script)
    self.assertIn('--progress-every "${TAG_PROGRESS_EVERY}"', script)
```

- [ ] **Step 2: Run the wrapper test and verify RED**

```bash
python -m unittest \
  tests.test_build_text_control_tags.BuildTextControlTagsCliTest.test_shell_wrapper_forwards_parallel_defaults \
  -v
```

Expected: FAIL because the wrapper does not define or forward the variables.

- [ ] **Step 3: Update the wrapper**

Add:

```bash
TAG_WORKERS="${TAG_WORKERS:-80}"
TAG_PROGRESS_EVERY="${TAG_PROGRESS_EVERY:-1000}"
```

Forward:

```bash
  --workers "${TAG_WORKERS}" \
  --progress-every "${TAG_PROGRESS_EVERY}"
```

- [ ] **Step 4: Update operator documentation**

After exporting `TEXT_PROMPT_ROOT` in `docs/text_control_pre_bc.md`, add:

```bash
# 默认 80 个进程；若 PFS 小文件写入拥塞，可降低到 16 或 32
export TAG_WORKERS=80
export TAG_PROGRESS_EVERY=1000
```

Explain that gzip reading remains sequential, workers parallelize scenario computation and atomic JSON writes, progress goes to stderr, and `TAG_WORKERS=1` is diagnostic serial mode.

- [ ] **Step 5: Verify wrapper, help, and tests**

```bash
bash -n scripts/build_text_control_tags.sh
python src/smart/datasets/build_text_control_tags.py --help
python -m unittest tests.test_build_text_control_tags -v
```

Expected: shell syntax and help exit zero; help lists both new options; all tests PASS.

- [ ] **Step 6: Verify scope and commit Task 2**

```bash
git diff --check -- \
  src/smart/datasets/build_text_control_tags.py \
  tests/test_build_text_control_tags.py \
  scripts/build_text_control_tags.sh \
  docs/text_control_pre_bc.md
git diff --stat -- \
  src/smart/datasets/build_text_control_tags.py \
  tests/test_build_text_control_tags.py \
  scripts/build_text_control_tags.sh \
  docs/text_control_pre_bc.md
git add tests/test_build_text_control_tags.py scripts/build_text_control_tags.sh docs/text_control_pre_bc.md
git commit -m "docs: expose text tag conversion controls"
```
