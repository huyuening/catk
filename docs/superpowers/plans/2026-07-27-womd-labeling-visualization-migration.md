# WOMD Labeling and Visualization Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the current WOMD road, agent-size, and per-frame action labeling algorithms inside CatK and provide resumable full-dataset annotation plus scenario and aggregate visualization commands.

**Architecture:** Preserve the source algorithms in an isolated `src.womd_labeling` package, reuse CatK's bundled WOMD protobuf, and expose each pipeline stage as a Python module. A top-level runner resolves training/validation/testing splits and invokes annotation, statistics, scenario rendering, and aggregate plotting with split-isolated outputs.

**Tech Stack:** Python 3.11, NumPy, SciPy, Shapely, Matplotlib, protobuf, tqdm, pytest.

## Global Constraints

- The remote runtime must not depend on a checkout of `WOMD-Traffic-Signal-Data-Improvement`.
- Existing CatK training, inference, preprocessing, and tokenization behavior must remain unchanged.
- Label definitions and default thresholds must match the current source working tree.
- Raw TFRecords must be streamed without TensorFlow.
- Completed shard files must be written atomically and resume validation must reject corrupt or incompatible outputs.
- Generated WOMD protobuf code must be reused from `src.smart.tokens.womd_proto`, not duplicated.
- Migrated code must carry source provenance and license text.

---

### Task 1: Isolated compatibility package and labeling core

**Files:**
- Create: `src/womd_labeling/__init__.py`
- Create: `src/womd_labeling/_compat/__init__.py`
- Create: `src/womd_labeling/_compat/generic.py`
- Create: `src/womd_labeling/_compat/geometry.py`
- Create: `src/womd_labeling/_compat/waymo.py`
- Create: `src/womd_labeling/_compat/waymonizer.py`
- Create: `src/womd_labeling/map_annotation.py`
- Create: `src/womd_labeling/agent_size_classification.py`
- Create: `src/womd_labeling/agent_action_classification.py`
- Create: `src/womd_labeling/road_type_statistics.py`
- Create: `src/womd_labeling/LICENSE.WOMD_TRAFFIC_SIGNAL_DATA_IMPROVEMENT.txt`
- Test: `tests/womd_labeling/test_map_annotation.py`
- Test: `tests/womd_labeling/test_agent_size_classification.py`
- Test: `tests/womd_labeling/test_agent_action_classification.py`
- Test: `tests/womd_labeling/test_road_type_statistics.py`
- Test: `tests/womd_labeling/test_package_isolation.py`

**Interfaces:**
- Consumes: `src.smart.tokens.womd_proto.scenario_pb2.Scenario`
- Produces:
  - `annotate_scenario(scenario, config=None, *, scenario_index=None, source_file=None, frame_indices=None) -> ScenarioMapAnnotation`
  - `extract_agent_size_records(scenario, frame_index, config=None) -> tuple[list[dict], dict]`
  - `label_scenario_actions(scenario, config=None) -> tuple[list[dict], dict]`
  - `classify_ego_frame(frame, driveway_index=None) -> RoadTypeLabel`

- [ ] **Step 1: Add namespace-adapted source behavior tests**

Copy the six current source tests into `tests/womd_labeling`, changing imports
from `src.<module>` to `src.womd_labeling.<module>` and protobuf imports to:

```python
from src.smart.tokens.womd_proto import scenario_pb2
```

Add an isolation assertion:

```python
def test_labeling_imports_do_not_reference_source_checkout():
    import src.womd_labeling.map_annotation as module

    assert "WOMD-Traffic-Signal-Data-Improvement" not in str(module.__file__)
```

- [ ] **Step 2: Run tests to verify the package is missing**

Run:

```bash
python -m pytest -q tests/womd_labeling/test_map_annotation.py \
  tests/womd_labeling/test_agent_size_classification.py \
  tests/womd_labeling/test_agent_action_classification.py \
  tests/womd_labeling/test_road_type_statistics.py \
  tests/womd_labeling/test_package_isolation.py
```

Expected: collection fails with `ModuleNotFoundError: src.womd_labeling`.

- [ ] **Step 3: Migrate the minimal compatibility layer and core algorithms**

Mechanically copy the current source implementations, then make only these
namespace changes:

```python
# generic.py
from src.smart.tokens.womd_proto import scenario_pb2

# geometry.py
from .generic import Direction, Pt, UnionFind

# waymo.py
from .generic import Pt, TLS

# waymonizer.py
from .geometry import (
    distance_between_points,
    find_polyline_nearest_point,
    polyline_length,
    real_neighbor_type,
    two_lines_parallel,
)
from .generic import UnionFind
from .waymo import Boundary, LaneCenter, LaneType, Neighbor, Pt, WaymonicTLS

# map_annotation.py
from ._compat.waymonizer import Waymonizer

class ScenarioProcessor(Waymonizer):
    """Map-topology-only compatibility adapter used by EgoMapAnnotator."""

    def __init__(self, scenario, load_boundaries: bool = False) -> None:
        super().__init__(scenario, load_boundaries=load_boundaries)
```

Change cross-module imports in action and road modules to package-relative
imports. Do not migrate traffic-signal generators or source protobuf files.

- [ ] **Step 4: Run migrated core tests**

Run the Task 1 test command. Expected: all tests pass.

- [ ] **Step 5: Run the original source tests as a parity reference**

Run:

```bash
python -m pytest -q \
  /Users/huyuening/PycharmProjects/WOMD-Traffic-Signal-Data-Improvement/tests/test_map_annotation.py \
  /Users/huyuening/PycharmProjects/WOMD-Traffic-Signal-Data-Improvement/tests/test_agent_size_classification.py \
  /Users/huyuening/PycharmProjects/WOMD-Traffic-Signal-Data-Improvement/tests/test_agent_action_classification.py \
  /Users/huyuening/PycharmProjects/WOMD-Traffic-Signal-Data-Improvement/tests/test_road_type_statistics.py
```

Expected: both source and CatK suites pass the same behavioral cases.

- [ ] **Step 6: Commit**

```bash
git add src/womd_labeling tests/womd_labeling
git commit -m "feat: migrate WOMD labeling core"
```

### Task 2: Shared streaming TFRecord and resumable annotation pipeline

**Files:**
- Create: `src/womd_labeling/tfrecord_io.py`
- Create: `src/womd_labeling/annotate.py`
- Test: `tests/womd_labeling/test_tfrecord_io.py`
- Test: `tests/womd_labeling/test_annotate_cli.py`

**Interfaces:**
- Consumes: Task 1 `annotate_scenario`
- Produces:
  - `resolve_tfrecord_paths(entries: Iterable[str | Path]) -> list[Path]`
  - `iter_tfrecord(path: Path) -> Iterator[tuple[int, bytes]]`
  - `count_tfrecord_records(path: Path) -> int`
  - `validate_completed_annotation(path: Path, *, source_file: str, expected_records: int) -> dict`
  - `annotate_paths(args: argparse.Namespace) -> dict`

- [ ] **Step 1: Write failing streaming and resume tests**

Use a test helper that writes valid TFRecord framing around serialized synthetic
scenarios. Assert:

```python
assert [path.name for path in resolve_tfrecord_paths([directory])] == [
    "training.tfrecord-00000-of-00002",
    "training.tfrecord-00001-of-00002",
]
assert count_tfrecord_records(first_path) == 2
assert list(iter_tfrecord(first_path))[0][0] == 0
```

Write one annotation shard, validate it, truncate its gzip payload, and assert
`validate_completed_annotation` raises `ValueError`. Assert `--resume` skips a
valid completed shard and `--overwrite` regenerates it.

- [ ] **Step 2: Run tests to verify missing APIs**

Run:

```bash
python -m pytest -q tests/womd_labeling/test_tfrecord_io.py \
  tests/womd_labeling/test_annotate_cli.py
```

Expected: import failure for `src.womd_labeling.tfrecord_io`.

- [ ] **Step 3: Implement shared TFRecord streaming**

Implement deterministic path resolution, raw TFRecord framing reads, record
counting, gzip/plain JSONL readers, and atomic `.partial` output helpers.
Reject duplicate absolute paths and empty path matches.

- [ ] **Step 4: Migrate and adapt the annotation CLI**

Port the current `annotate_ego_map.py` implementation into
`src.womd_labeling.annotate`. Replace local TFRecord helpers with Task 2 shared
helpers and add:

```text
--resume / --no-resume   default: resume
--overwrite              takes precedence over resume
```

Each successful JSONL record retains schema version, source file, scenario
index, scenario ID, junctions, ego frames, and statistics. Preserve ordered
multi-process writes.

- [ ] **Step 5: Run Task 2 tests**

Run the Task 2 test command. Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/womd_labeling/tfrecord_io.py src/womd_labeling/annotate.py \
  tests/womd_labeling/test_tfrecord_io.py tests/womd_labeling/test_annotate_cli.py
git commit -m "feat: add resumable WOMD annotation pipeline"
```

### Task 3: Statistics pipeline for roads, sizes, and all-frame actions

**Files:**
- Create: `src/womd_labeling/statistics.py`
- Test: `tests/womd_labeling/test_statistics_cli.py`

**Interfaces:**
- Consumes: Task 1 labelers and Task 2 TFRecord utilities
- Produces:
  - `process_scenario(task: tuple, frame_index: int, map_config, size_config, action_config) -> dict`
  - `run_statistics(args: argparse.Namespace) -> dict`
  - gzip detail CSVs, count CSVs, `errors.jsonl`, and `summary.json`

- [ ] **Step 1: Add a failing one-scenario statistics test**

Build a synthetic scenario with one SDC vehicle, one pedestrian, timestamps,
and a simple lane. Invoke `run_statistics` and assert these files exist and are
internally consistent:

```python
assert summary["aggregate"]["scenarios"] == 1
assert summary["aggregate"]["errors"] == 0
assert sum(summary["aggregate"]["road_type_counts"].values()) == 1
assert summary["aggregate"]["action_diagnostics"]["valid_state_frames"] > 0
```

- [ ] **Step 2: Run the test to verify the CLI is missing**

Run:

```bash
python -m pytest -q tests/womd_labeling/test_statistics_cli.py
```

Expected: import failure for `src.womd_labeling.statistics`.

- [ ] **Step 3: Migrate the statistics CLI**

Port `summarize_road_types.py`, use package-relative imports and Task 2
TFRecord helpers, and retain:

```text
current_frame_road_types.csv.gz
current_frame_agent_sizes.csv.gz
agent_actions_by_frame.csv.gz
current_frame_road_type_counts.csv
current_frame_agent_size_counts.csv
agent_action_counts.csv
agent_action_counts_by_frame.csv
errors.jsonl
summary.json
```

Write large detail files through `.partial` paths and atomically rename the
complete output set. Keep `--overwrite` mandatory when replacing a complete
statistics directory.

- [ ] **Step 4: Run Task 3 tests**

Run the Task 3 test command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/womd_labeling/statistics.py \
  tests/womd_labeling/test_statistics_cli.py
git commit -m "feat: add WOMD label statistics pipeline"
```

### Task 4: Scenario-level annotation visualization

**Files:**
- Create: `src/womd_labeling/map_annotation_visualization.py`
- Create: `src/womd_labeling/visualize.py`
- Test: `tests/womd_labeling/test_map_annotation_visualization.py`
- Test: `tests/womd_labeling/test_visualize_cli.py`

**Interfaces:**
- Consumes: raw TFRecords and Task 2 annotation JSONL files
- Produces:
  - `render_initial_frame_map(...) -> RenderedMapAnnotation`
  - `visualize_paths(args: argparse.Namespace) -> dict`
  - PNG files, `manifest.csv`, and visualization `summary.json`

- [ ] **Step 1: Add namespace-adapted source visualization tests**

Copy `test_map_annotation_visualization.py` and add an integration test that
renders one synthetic scenario with one valid annotation to a temporary PNG.
Assert the PNG is nonempty and its manifest status is `written`.

- [ ] **Step 2: Run tests to verify visualization modules are missing**

Run:

```bash
python -m pytest -q \
  tests/womd_labeling/test_map_annotation_visualization.py \
  tests/womd_labeling/test_visualize_cli.py
```

Expected: import failure for the visualization module.

- [ ] **Step 3: Migrate renderer and CLI**

Port `map_annotation_visualization.py` and
`visualize_ego_map_annotations.py`. Use Matplotlib's `Agg` backend, shared
TFRecord/annotation readers, package-relative imports, and output-relative
Matplotlib cache paths. Preserve worker support, existing-image skipping,
scenario-ID filtering, and bilingual annotation text.

- [ ] **Step 4: Run Task 4 tests**

Run the Task 4 test command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/womd_labeling/map_annotation_visualization.py \
  src/womd_labeling/visualize.py \
  tests/womd_labeling/test_map_annotation_visualization.py \
  tests/womd_labeling/test_visualize_cli.py
git commit -m "feat: add WOMD annotation visualizer"
```

### Task 5: Multi-shard aggregate visualization

**Files:**
- Create: `src/womd_labeling/plot_statistics.py`
- Test: `tests/womd_labeling/test_plot_statistics.py`

**Interfaces:**
- Consumes: a statistics directory and one or more annotation files/directories
- Produces: PNG, PDF, SVG, plotted-count CSV, and optional HTML SVG fragment

- [ ] **Step 1: Add source plot tests and a failing multi-file aggregation test**

Adapt `test_plot_statistics_summary.py` to the package namespace. Add two
annotation gzip files with distinct road labels and assert:

```python
result = recount_road_hierarchy([first, second], frame_index=10)
assert result["rows"] == 2
assert sum(result["top_counts"].values()) == 2
```

- [ ] **Step 2: Run the test to verify the plot module is missing**

Run:

```bash
python -m pytest -q tests/womd_labeling/test_plot_statistics.py
```

Expected: import failure for `src.womd_labeling.plot_statistics`.

- [ ] **Step 3: Migrate and generalize aggregate plotting**

Port `plot_statistics_summary.py`, then change:

```python
def recount_road_hierarchy(
    paths: Iterable[Path],
    frame_index: int = 10,
) -> dict:
    ...
```

Resolve direct files, directories, and glob expressions with Task 2 helpers.
Retain source hierarchy validation, labels, colors, compact count formatting,
and all output formats.

- [ ] **Step 4: Run Task 5 tests**

Run the Task 5 test command. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/womd_labeling/plot_statistics.py \
  tests/womd_labeling/test_plot_statistics.py
git commit -m "feat: add aggregate WOMD label visualization"
```

### Task 6: Full-dataset orchestrator, dependencies, and user documentation

**Files:**
- Create: `src/womd_labeling/run_dataset.py`
- Create: `scripts/label_womd_dataset.sh`
- Create: `docs/womd-labeling.md`
- Modify: `install/requirements.txt`
- Modify: `README.md`
- Test: `tests/womd_labeling/test_run_dataset.py`

**Interfaces:**
- Consumes: Tasks 2–5 stage APIs
- Produces: split-isolated output tree and top-level `run_summary.json`

- [ ] **Step 1: Write a failing runner layout test**

Create temporary `training`, `validation`, and `testing` directories and patch
stage callables with lightweight recorders. Assert the runner sends each split
to:

```text
annotations/<split>
statistics/<split>
visualizations/scenarios/<split>
visualizations/aggregate/<split>
```

Assert `--visualize-max-scenarios 0` is translated to no scenario limit and
that a missing requested split raises `FileNotFoundError`.

- [ ] **Step 2: Run the test to verify the runner is missing**

Run:

```bash
python -m pytest -q tests/womd_labeling/test_run_dataset.py
```

Expected: import failure for `src.womd_labeling.run_dataset`.

- [ ] **Step 3: Implement the runner and shell wrapper**

Expose:

```text
--input-root
--output-root
--splits training validation testing
--workers
--stages annotations statistics scenario-visualizations aggregate-visualization
--visualize-max-scenarios
--overwrite
--resume / --no-resume
```

Call stage functions directly rather than spawning nested shell processes. Write
`run_summary.json` atomically after every completed split so progress survives
an interruption. The shell wrapper forwards:

```bash
WOMD_ROOT
LABEL_OUTPUT_ROOT
NUM_WORKERS
SPLITS
VISUALIZE_MAX_SCENARIOS
```

- [ ] **Step 4: Declare dependencies and document commands**

Add Matplotlib, Shapely, and tqdm to `install/requirements.txt`. Document:

- full training/validation/testing command;
- annotation-only resume command;
- single-scenario visualization;
- aggregate plot regeneration;
- output schema and disk/runtime expectations.

- [ ] **Step 5: Run runner tests**

Run the Task 6 test command. Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/womd_labeling/run_dataset.py scripts/label_womd_dataset.sh \
  docs/womd-labeling.md install/requirements.txt README.md \
  tests/womd_labeling/test_run_dataset.py
git commit -m "feat: add full-dataset WOMD labeling workflow"
```

### Task 7: End-to-end validation and final integration

**Files:**
- Modify only if verification exposes a tested defect.

**Interfaces:**
- Consumes: all previous tasks
- Produces: verification evidence and a merge-ready branch

- [ ] **Step 1: Run all migrated tests**

```bash
python -m pytest -q tests/womd_labeling
```

Expected: all pass.

- [ ] **Step 2: Run the complete CatK test suite**

```bash
python -m pytest -q
```

Expected: existing 186 tests plus migrated tests pass; existing dependency-gated
skips remain skips.

- [ ] **Step 3: Run compile and import checks**

```bash
python -m compileall -q src/womd_labeling
python -m src.womd_labeling.annotate --help
python -m src.womd_labeling.statistics --help
python -m src.womd_labeling.visualize --help
python -m src.womd_labeling.plot_statistics --help
python -m src.womd_labeling.run_dataset --help
```

Expected: all exit successfully.

- [ ] **Step 4: Run one-real-scenario end-to-end smoke test**

Use:

```text
/Users/huyuening/PycharmProjects/WOMD-Traffic-Signal-Data-Improvement/
dataset/training.tfrecord-00000-of-01000
```

Run annotation and statistics with `--max-scenarios 1 --workers 1`, render one
scenario, then build aggregate PNG/PDF/SVG/CSV. Validate every output exists,
is nonempty, and each summary reports one scenario with zero errors.

- [ ] **Step 5: Compare one scenario with the source implementation**

Serialize both CatK and source `annotate_scenario(...).to_dict()` results to
canonical JSON and assert equality. Compare size-label records and action-label
records the same way.

- [ ] **Step 6: Inspect diff and repository state**

```bash
git diff --check main...HEAD
git status --short --branch
git log --oneline --decorate main..HEAD
```

Expected: no whitespace errors and only intentional migration files.

- [ ] **Step 7: Finish the branch**

Use `superpowers:verification-before-completion`, then
`superpowers:finishing-a-development-branch` to merge the verified feature into
local `main` while preserving the user's unrelated working-tree changes.
