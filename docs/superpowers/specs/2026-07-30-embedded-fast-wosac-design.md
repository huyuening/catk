# Embedded Fast WOSAC Design

## Status

Approved approach: embed the Fast WOSAC backend directly in CatK and make
CatK's validation preprocessing produce the required ground-truth artifacts.

## Goal

CatK must run Fast WOSAC 2024 and 2025 validation without a TrajTok checkout,
`TRAJTOK_ROOT`, `sys.path` mutation, network access, or runtime source
discovery. Existing TrajTok-generated `validation_gt` files must remain
compatible.

## Scope

This change includes:

- the Fast WOSAC PyTorch metric backend used by CatK;
- the 2024 and 2025 WOSAC textproto configurations;
- conversion of a Waymo `Scenario` proto into the preprocessed Fast WOSAC
  ground-truth dictionary;
- automatic `validation_gt/<scenario_id>.pkl` generation during CatK
  validation preprocessing;
- removal of CatK's runtime dependency on a sibling TrajTok checkout;
- source provenance, compatibility documentation, and regression tests.

This change does not include TrajTok training, tokenization, visualization, or
its standalone offline evaluation CLI. CatK's existing `FastWOSACMetrics`
adapter remains the supported evaluation interface.

## Source Provenance

The embedded backend is copied from:

- repository: `https://github.com/Thinklab-SJTU/TrajTok`;
- source commit: `5920c89e26b62e8337512c253ab59efee995a496`;
- source directory: `wosac_fast_eval_tool`;
- source authors: the TrajTok authors identified by that repository.

The user explicitly authorized embedding this source in CatK. A provenance
notice will accompany the embedded files. The numeric backend files will be
kept byte-for-byte equivalent to the source wherever package placement does
not require an import-path adjustment.

## Architecture

### Embedded backend

Create a private CatK package:

```text
src/smart/metrics/fast_wosac_backend/
├── __init__.py
├── NOTICE.md
├── scenario_gt_converter.py
└── fast_sim_agents_metrics/
    ├── __init__.py
    ├── challenge_2024_config.textproto
    ├── challenge_2025_sim_agents_config.textproto
    ├── estimators.py
    ├── interaction_features.py
    ├── map_metric_features.py
    ├── metric_features.py
    ├── metrics.py
    ├── traffic_light_features.py
    └── trajectory_features.py
```

The package is private to CatK's metrics layer. Existing relative imports
inside `fast_sim_agents_metrics` remain unchanged.

### CatK adapter

`src/smart/metrics/fast_wosac_metrics.py` will import the internal backend and
scenario converter directly. It will no longer:

- inspect a TrajTok filesystem path;
- append any external directory to `sys.path`;
- raise an error asking for `/root/workspace/TrajTok`;
- retain `trajtok_root` as runtime state.

`FastWOSACMetrics` keeps its existing public behavior:

- supported versions are exactly `"2024"` and `"2025"`;
- metric output names and tensor types do not change;
- strict preprocessed-GT behavior does not change;
- raw validation TFRecord fallback remains available when strict mode is
  disabled;
- TensorFlow remains restricted to CPU for TFRecord fallback.

### Model and configuration

`SMART` will construct `FastWOSACMetrics` without passing `trajtok_root`.

The `trajtok_root` configuration entry will remain temporarily accepted as a
deprecated, ignored compatibility field. Its default will no longer reference
an external directory or read `TRAJTOK_ROOT`. This allows previously recorded
Hydra configurations and old command-line overrides to load without restoring
TrajTok, while making the field operationally irrelevant.

New commands will not mention `TRAJTOK_ROOT` or
`model.model_config.trajtok_root`.

## Validation Ground-Truth Generation

When `python -m src.data_preprocess` runs with `--split validation`, it will:

1. parse each Waymo `Scenario` exactly once;
2. continue writing the existing CatK scenario pickle;
3. continue writing the existing per-scenario validation TFRecord;
4. call CatK's embedded `extract_gt_scenario`;
5. write the returned dictionary to
   `<output_dir>/validation_gt/<scenario_id>.pkl`.

Training and testing preprocessing will not create `validation_gt`.

The generated dictionary must match the existing TrajTok representation,
including the WOSAC 2025 fields:

- `scenario_id`;
- `tracks`;
- `track_masks`;
- `object_ids`;
- `object_types`;
- `road_edges`;
- `predict_index`;
- `sim_agent_ids`;
- `lane_ids`;
- `lane_polylines`;
- `traffic_signals`.

Existing TrajTok-generated files remain valid and require no conversion.

Ground-truth files will use the existing pickle format because the current
loader and stored validation set already rely on it. The migration will not
introduce a second artifact format.

## Data Flow

```text
Waymo validation TFRecord
    ├── CatK scenario cache
    ├── split per-scenario TFRecord
    └── embedded scenario_gt_converter
            └── validation_gt/<scenario_id>.pkl

CatK validation rollout
    └── FastWOSACMetrics
            ├── PreprocessedScenarioGT
            ├── embedded 2024/2025 textproto
            └── embedded PyTorch Fast WOSAC backend
                    └── existing val_closed/wosac* metrics
```

## Error Handling

- Unsupported WOSAC versions continue to raise `ValueError`.
- Missing strict `validation_gt` directories or scenario files continue to
  raise before silently falling back.
- WOSAC 2025 ground truth missing required fields continues to raise `KeyError`,
  but the message will refer to CatK's current preprocessing command rather
  than TrajTok.
- A missing embedded textproto raises `FileNotFoundError` naming the internal
  resource.
- External TrajTok paths are never consulted, even if `TRAJTOK_ROOT` is set.

## Compatibility

- Existing CatK checkpoints are unaffected because the backend has no learned
  state.
- Existing TrajTok `validation_gt` artifacts remain readable.
- Metric keys, aggregation logic, rollout shapes, and 2024/2025 selection stay
  unchanged.
- Existing commands that set `model.model_config.trajtok_root` remain accepted
  during the compatibility period, but the value is ignored.
- A machine containing only CatK, the existing Python environment, the CatK
  cache, and `validation_gt` must be able to initialize and run Fast WOSAC.

## Testing

Tests will cover:

1. importing and constructing the embedded backend with no TrajTok directory
   or `sys.path` mutation;
2. parsing both embedded WOSAC textproto configurations;
3. preserving the set of 2024 and 2025 metric names;
4. converting a synthetic Waymo scenario into all required GT fields;
5. validation preprocessing writing a compatible
   `validation_gt/<scenario_id>.pkl`;
6. training and testing preprocessing not writing validation GT;
7. strict missing/malformed GT errors referring to CatK rather than TrajTok;
8. Hydra configuration no longer resolving `TRAJTOK_ROOT`;
9. byte-level or hash-level provenance checks for copied backend sources;
10. the complete existing CatK test suite.

Where local dependencies permit it, verification will also compare the
embedded backend files against the authorized TrajTok source and run a
same-input parity check. Such a comparison is a development verification
step, not a runtime or CI dependency.

## Documentation

Update the README to:

- state that Fast WOSAC is built into CatK;
- remove the sibling TrajTok checkout requirement;
- remove `TRAJTOK_ROOT` from training and inference examples;
- show that CatK validation preprocessing creates `validation_gt`;
- retain the disclaimer that the fast evaluator is unofficial and official
  submissions must be judged by the official WOSAC server;
- credit the TrajTok Fast WOSAC implementation and link its source commit.

## Acceptance Criteria

The migration is complete when:

- deleting `/root/workspace/TrajTok` cannot cause CatK model initialization or
  Fast WOSAC validation to fail;
- `pre_bc` and `clsft` retain their default 10% Fast WOSAC validation behavior;
- CatK can generate required validation GT from raw validation TFRecords;
- existing validation GT and checkpoints continue to work;
- Fast WOSAC 2024 and 2025 remain selectable;
- no production code imports `wosac_fast_eval_tool` from an external checkout;
- relevant focused tests and the complete test suite pass.
