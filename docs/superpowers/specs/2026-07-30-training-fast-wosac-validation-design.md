# Training-Time Fast WOSAC Validation Design

## Goal

Make every `pre_bc` and `clsft` training epoch evaluate a deterministic 10%
prefix of the WOMD validation split with TrajTok's Fast WOSAC 2025 evaluator
and preprocessed `validation_gt`. Preserve the existing open-loop validation
loss and accuracy, and fail immediately rather than falling back to raw
TFRecords when preprocessed ground truth is unavailable.

## Scope

This change applies to the base `pre_bc` and `clsft` experiment configurations.
Experiments that inherit either base configuration, including history-dynamics,
future-token-dynamics, reconstructed-vocabulary, hybrid, and TrajTok-original
loss variants, inherit the same validation protocol unless explicitly
overridden at invocation time.

Standalone inference, local-validation, submission-generation, and model-wide
defaults retain their current behavior. The change does not alter training
losses, optimizer settings, data preprocessing, vocabularies, checkpoints, or
rollout generation outside validation.

## Validation Protocol

Both base training experiments use the following settings:

- validate after every epoch with `check_val_every_n_epoch: 1`;
- process `trainer.limit_val_batches: 0.1`;
- keep both open-loop and closed-loop validation enabled;
- evaluate every processed validation batch with
  `n_batch_wosac_metric: -1`;
- use the `fast` WOSAC backend and metric version `2025`;
- generate 32 closed-loop rollouts per scenario;
- use `topk_prob` validation sampling with `num_k: 48` and `temp: 1.0`;
- keep training-time visualization settings unchanged.

Because the validation loader is not shuffled, `0.1` denotes a stable prefix
of the validation loader rather than a new random sample each epoch. With the
current 44,104-scenario validation split, this is approximately 4,400
scenarios.

Open-loop `val_open/loss` and `val_open/acc` continue to be computed on the
same 10% prefix. Fast WOSAC adds the existing `val_closed/wosac/*` and
`val_closed/wosac_likelihood/*` metrics to terminal output and the configured
logger at validation epoch end.

## Ground-Truth Path

Training-time Fast WOSAC uses a validation-GT path independent of the CatK
training cache. This is necessary because history-dynamics caches can live at
paths such as
`/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact`,
while TrajTok ground truth remains under the original preprocessing root.

The default path is:

```text
/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario/validation_gt
```

Users can override it with:

```bash
export FAST_WOSAC_GT_DIR=/path/to/validation_gt
```

The resolved path is exposed through the shared paths configuration and passed
to the Fast WOSAC metric from the `pre_bc` and `clsft` experiment
configurations. It must not be derived from a history-dynamics `CACHE_ROOT`.

## Strict Ground-Truth Semantics

Strict preprocessed-GT mode is opt-in at the model level and enabled by both
training experiments. It has the following behavior:

1. If the configured `validation_gt` directory is absent or not a directory,
   model initialization raises `FileNotFoundError`.
2. For every evaluated scenario, the file
   `<validation_gt>/<scenario_id>.pkl` must exist. A missing file raises
   `FileNotFoundError`.
3. The loaded artifact must be a dictionary and its `scenario_id` must match
   the validation batch.
4. WOSAC 2025 required fields must all be present; the existing key validation
   remains active.
5. Raw per-scenario TFRecord extraction is never used in strict mode.

Fast WOSAC's existing fallback behavior remains available to other call sites
whose strict option is false. This avoids changing standalone inference or
other model configurations outside the requested training experiments.

## Configuration Structure

The model configuration gains a boolean Fast WOSAC strict-GT option whose
default is false. `SMART` forwards this option to `FastWOSACMetrics`.

The shared paths configuration gains `validation_gt_dir`, resolved from
`FAST_WOSAC_GT_DIR` with the deployment default above.

`pre_bc.yaml` and `clsft.yaml` explicitly enable:

- Fast WOSAC 2025;
- strict preprocessed GT;
- the shared validation-GT path;
- all-batch WOSAC accumulation;
- the 32-rollout, `K=48` validation protocol;
- 10% validation every epoch.

Keeping these values in the two base experiment files limits the behavior
change to training while allowing all inheriting variants to compose normally.

## Error Handling

Failures identify the missing directory or scenario-specific `.pkl` path and
state that strict Fast WOSAC validation does not permit TFRecord fallback.
Invalid artifact type, scenario-ID mismatch, and missing 2025 fields retain
specific errors so the user can distinguish preprocessing corruption from a
path configuration mistake.

The directory check occurs during model construction, before a training epoch
starts. Per-scenario checks occur when that scenario reaches validation,
because eagerly indexing hundreds of thousands of files would add unnecessary
startup cost.

## Testing

Configuration-composition tests verify that both base experiments resolve to:

- `limit_val_batches == 0.1`;
- `check_val_every_n_epoch == 1`;
- open-loop and closed-loop validation enabled;
- Fast WOSAC version 2025;
- strict GT enabled;
- `n_batch_wosac_metric == -1`;
- 32 rollouts and `K=48`, temperature `1.0`;
- the expected default and environment-overridden GT paths.

Composition tests also verify that representative inheriting configurations
such as `pre_bc_history_dynamics`, `pre_bc_trajtok_original`, and
`clsft_history_dynamics` retain the protocol.

Fast-metric unit tests cover:

- strict mode rejecting a missing GT directory;
- strict mode rejecting a missing scenario artifact without invoking the raw
  TFRecord extractor;
- strict mode accepting and loading a valid scenario artifact;
- non-strict mode retaining the existing fallback behavior.

The focused tests run before the repository test suite. YAML resolution is
also checked through Hydra so interpolation and defaults ordering are tested,
not merely the raw dictionary text.

## Documentation

The training documentation will state that `pre_bc` and `clsft` now run Fast
WOSAC 2025 on 10% of validation after every epoch, describe the expected
runtime cost, document `FAST_WOSAC_GT_DIR`, and explain the strict failure
behavior.

It will also show the existing Hydra overrides for users who intentionally
want a cheaper experiment, such as reducing the validation fraction or
disabling closed-loop validation.

