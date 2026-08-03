# Online Raw History Dynamics Design

## Context

CatK currently supports an optional history-dynamics branch containing signed
longitudinal acceleration, angular speed, and signed lateral acceleration. The
existing preprocessing path reconstructs the observable 11-frame trajectory
before calculating these values and stores the result in every scenario cache.

This experiment must retain the same three model inputs while removing the
additional history reconstruction. The quantities will instead be calculated
online from the ordinary CatK agent tensors during training, validation, and
test submission. Ordinary CatK preprocessing remains unchanged, including its
legacy interpolation between the first and last valid observations. The new
mode performs no further gap filling, smoothing, outlier filtering, polynomial
fitting, or heading reconstruction.

## Goals

- Preserve the three history-dynamics input channels and their existing model
  embedding.
- Calculate two dynamics vectors online at CatK history-token endpoints 5 and
  10.
- Use direct causal backward finite differences over cached position and
  heading.
- Apply identical behavior during training, validation, and test submission.
- Run from ordinary CatK caches without cached `history_dynamics` fields.
- Keep original CatK and reconstructed-history experiments unchanged.

## Non-goals

- Reading WOMD TFRecords in the training data loader.
- Recovering the pre-interpolation WOMD validity mask from an existing cache.
- Reconstructing, smoothing, or fitting any trajectory.
- Calculating dynamics for future generated tokens.
- Changing the agent vocabulary, loss function, rollout reconstruction, or
  WOSAC submission format.

## Configuration

The existing `model.model_config.history_dynamics` block gains a `mode` field:

```yaml
history_dynamics:
  is_active: false
  mode: cached_reconstructed
  normalization_scale: [5.0, 1.0, 5.0]
  initial_gate: 1.0
```

Supported modes are:

- `cached_reconstructed`: current behavior. Read `history_dynamics` and
  `history_dynamics_valid` from the cache and fail strictly when either field
  is absent.
- `online_raw`: ignore any cached dynamics fields and calculate the values
  from `position`, `heading`, and `valid_mask` in the current batch.

`cached_reconstructed` remains the default for backward compatibility. A new
experiment, `pre_bc_history_dynamics_online_raw`, inherits `pre_bc` and enables
the history branch with `mode: online_raw`.

## Online Calculation

Let the cache sampling interval be `dt = 0.1` seconds. For token endpoint
`t` in `{5, 10}`, use positions at `t-2`, `t-1`, and `t`:

```text
v_previous = (p[t-1] - p[t-2]) / dt
v_current  = (p[t]   - p[t-1]) / dt
a_world    = (v_current - v_previous) / dt
```

Calculate the wrapped heading difference without unwrapping or filtering the
trajectory:

```text
delta_heading = atan2(sin(theta[t] - theta[t-1]),
                      cos(theta[t] - theta[t-1]))
angular_speed = delta_heading / dt
```

Project world-frame acceleration into the body frame at the endpoint heading:

```text
a_longitudinal =  a_x * cos(theta[t]) + a_y * sin(theta[t])
a_lateral      = -a_x * sin(theta[t]) + a_y * cos(theta[t])
```

The output order remains:

```text
[a_longitudinal, angular_speed, a_lateral]
```

Values are clipped to the existing absolute limits of `[15, 3, 15]`. This
clipping is a feature-range safeguard and does not alter the trajectory.

## Validity and Numerical Handling

A token dynamics vector is valid only when:

- cached frames `t-2`, `t-1`, and `t` are all marked valid;
- all three XY positions are finite; and
- headings at `t-1` and `t` are finite.

An invalid vector is stored as zero and its validity mask is false. The model's
existing validity gate therefore prevents the zero placeholder from acting as
an observed dynamics value. Unknown agent types require no special handling
because this calculation is purely geometric.

The online function must reject malformed tensor shapes, non-positive `dt`,
and history lengths that do not contain both configured endpoints.

## Data Flow

1. The data module loads ordinary CatK scenario caches.
2. `TokenProcessor.tokenize_agent` forms the existing tokenized agent fields.
3. When history dynamics are active:
   - `cached_reconstructed` copies the two cached fields as before;
   - `online_raw` calls a tensor-only dynamics function using the batch's
     cached position, heading, and validity tensors.
4. The decoder normalizes and embeds the resulting `[agent, 2, 3]` tensor with
   the unchanged history-dynamics embedding and gate.
5. Only the two observed history tokens receive these embeddings. Future token
   inputs remain unchanged.

No CPU round trip, cache write, or WOMD record access occurs in the online
path. The calculation is vectorized on the same device as the token processor.

## Compatibility

- `history_dynamics.is_active: false` retains original CatK behavior.
- Existing `pre_bc_history_dynamics`, `clsft_history_dynamics`, and
  `inference_history_dynamics` configurations retain cached reconstructed
  behavior.
- Existing checkpoints retain the same history embedding parameter shapes.
  Nevertheless, this experiment starts pre-BC from scratch because changing
  the feature definition is the intended ablation.
- A cache may contain reconstructed dynamics fields while using `online_raw`;
  those fields must be ignored. An ordinary cache without those fields must
  also work.
- The loss setting is independent. The comparison command will explicitly use
  hard-label cross entropy so the history calculation is the only intended
  model-input change relative to the prior hard-label experiment.

## Testing

Tests will cover:

1. Constant velocity and heading produce zero acceleration and angular speed.
2. Known quadratic XY motion produces the expected body-frame acceleration at
   endpoints 5 and 10.
3. Heading wraparound across `pi` produces a small signed angular speed rather
   than a near-`2*pi/dt` spike.
4. Missing or non-finite support produces a zero vector with a false validity
   mask.
5. Values exceeding the configured limits are clipped.
6. `online_raw` works without cached dynamics fields and ignores conflicting
   cached values when they are present.
7. `cached_reconstructed` keeps its existing strict missing-field error.
8. Hydra composition enables the new experiment without changing the defaults
   of existing experiments.

## Expected Training Interface

After implementation, the experiment will be started with the existing
history-dynamics cache or an ordinary CatK cache; only standard agent tensors
are read:

```bash
MY_EXPERIMENT=pre_bc_history_dynamics_online_raw \
MY_TASK_NAME=pre_bc_history_dynamics_online_raw_hard_ce_b200 \
CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact \
WANDB_OFFLINE=false \
WANDB_ENTITY=huyuening911-beijing-jiaotong-university \
bash scripts/train.sh \
  ckpt_path=null \
  model.model_config.training_loss.spatial_aware_smoothing=false \
  model.model_config.training_loss.label_smoothing=0.0 \
  model.model_config.val_closed_loop=false
```

Using the existing exact-cache directory here is intentional: `online_raw`
uses only its normal CatK position, heading, and validity fields and ignores
the precomputed reconstructed dynamics fields. This holds scenario selection
and all unrelated inputs constant for a fair ablation.
