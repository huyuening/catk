# WOMD Labeling and Visualization Migration Design

## Goal

Migrate the current labeling and visualization implementation from
`WOMD-Traffic-Signal-Data-Improvement` into CatK so that a remote development
machine can label all raw WOMD training, validation, and testing shards and
produce both scenario-level and aggregate visualizations without depending on
the source repository at runtime.

## Scope

The migration preserves the source repository's current label definitions and
default thresholds:

- ego-map labels for every frame:
  - road segment or intersection;
  - signalized, stop-controlled, roundabout, or geometry-only junction;
  - freeway mainline/ramp, urban street, or parking-lot environment;
  - matched lane, lane count, junction arms, distances, confidence, and reasons;
- current-frame agent-size proxy labels:
  - large/small vehicle and motorcycle proxy;
  - bicycle/e-bike proxy;
  - adult/child pedestrian proxy;
- all-valid-frame agent action labels:
  - stop, U-turn, left/right turn, left/right lane change;
  - deceleration, keep, and acceleration;
- scenario map visualization with annotations and agent boxes;
- aggregate road, agent-size, and action visualizations in PNG, PDF, SVG, and
  machine-readable CSV form.

Trajectory reconstruction, traffic-signal imputation, and TFRecord rewriting
remain outside this migration because CatK already contains the required
trajectory reconstruction code and the requested feature is labeling plus
visualization.

## Architecture

All migrated code lives under `src/womd_labeling` to avoid collisions with
CatK's existing `src.utils`, training code, and token reconstruction modules.
The package reuses CatK's bundled WOMD protobuf implementation from
`src.smart.tokens.womd_proto`; it does not duplicate generated protobuf files.

The package is divided into:

- pure labeling modules for map, road hierarchy, agent size, and agent action;
- a minimal private Waymonizer compatibility layer used by map labeling;
- shared streaming TFRecord utilities;
- CLI modules for annotation, statistics, scenario rendering, aggregate
  plotting, and full-dataset orchestration.

The source algorithms are copied from the current working tree of
`WOMD-Traffic-Signal-Data-Improvement`, with imports and project paths adapted
to the isolated CatK package. The map annotator uses only the Waymonizer
topology functionality; traffic-signal generation classes are not copied.

## Data Flow

1. `python -m src.womd_labeling.run_dataset` resolves raw WOMD shards from the
   requested split directories.
2. The annotation stage streams TFRecord records and writes one ordered
   `*.map-annotations.jsonl.gz` file per input shard.
3. Each shard is first written to a `.partial` file and atomically renamed after
   completion. Completed shards are skipped in resume mode only after their
   metadata and record count pass validation.
4. The statistics stage streams the same raw records and produces:
   - current-frame road labels;
   - current-frame size labels;
   - all-frame action labels;
   - count tables, diagnostics, errors, and `summary.json`.
5. The visualization stage can render selected or all scenarios from raw
   records plus annotation JSONL files.
6. The aggregate plotter consumes every annotation shard in an annotation
   directory together with the statistics output and writes PNG/PDF/SVG/CSV.

## Full-Dataset Interface

The main entry point is:

```bash
python -m src.womd_labeling.run_dataset \
  --input-root /path/to/uncompressed/scenario \
  --output-root /path/to/catk_womd_labels \
  --splits training validation testing \
  --workers 24
```

Expected raw layout:

```text
<input-root>/
  training/
  validation/
  testing/
```

The runner exposes independent `annotations`, `statistics`,
`scenario-visualizations`, and `aggregate-visualization` stages, allowing an
expensive full-dataset run to be resumed or a visualization to be regenerated
without repeating labeling. Scenario rendering defaults to a bounded sample;
explicit `--visualize-max-scenarios 0` renders every labeled scenario.

For compatibility and debugging, every stage is also directly callable through
its own `python -m src.womd_labeling.<command>` entry point.

## Output Layout

```text
<output-root>/
  annotations/
    training/
    validation/
    testing/
  statistics/
    training/
    validation/
    testing/
  visualizations/
    scenarios/
      training/
      validation/
      testing/
    aggregate/
      training.*
      validation.*
      testing.*
  run_summary.json
```

Per-split directories prevent shard-name collisions and make restart,
inspection, and deletion boundaries explicit.

## Reliability and Errors

- Input resolution is deterministic and sorted.
- Raw TFRecords are streamed without requiring TensorFlow.
- Worker results are buffered only until they can be written in source-record
  order.
- Individual scenario failures are emitted as structured error records and do
  not abort an otherwise valid shard.
- Completed shard outputs are immutable unless `--overwrite` is supplied.
- Resume mode validates gzip readability, schema version, source shard name,
  and record count before skipping a shard.
- Summaries report inputs, outputs, counts, errors, configuration, and skipped
  work.
- A nonzero stage error count is reflected in the top-level run summary.

## Compatibility

- Python 3.11 and CatK's current protobuf files remain the baseline.
- SciPy and NumPy are already CatK dependencies.
- Matplotlib, Shapely, and tqdm are declared explicitly for the migrated
  visualization and geometry behavior.
- Labels remain JSON/CSV rather than being inserted into CatK training cache,
  so existing training and preprocessing behavior is unchanged.
- The package does not import or require
  `/root/workspace/WOMD-Traffic-Signal-Data-Improvement`.

## Testing

Tests are migrated and namespace-adapted for:

- synthetic junction construction and map matching;
- freeway, urban-road, parking, and roundabout classification;
- agent size thresholds and action decisions;
- annotation text and coordinate transforms;
- aggregate hierarchy/count validation.

Additional CatK integration tests cover:

- direct imports without the source repository on `PYTHONPATH`;
- deterministic multi-shard input discovery;
- atomic per-shard annotation output;
- valid resume and rejection of corrupt/incompatible completed shards;
- aggregate plotting across multiple annotation files;
- full-dataset command construction and split output isolation.

An end-to-end smoke test runs one real scenario from the locally available WOMD
TFRecord, writes annotations and statistics, renders one scenario image, and
produces aggregate figures.

## Provenance

The migrated implementation retains an attribution notice and a copy of the
source repository license alongside `src/womd_labeling`. No runtime link to the
source checkout is retained.
