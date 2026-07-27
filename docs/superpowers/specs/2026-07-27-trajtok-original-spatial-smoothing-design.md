# TrajTok Original Spatial-Smoothing Compatibility Design

## Goal

Add an explicit CatK loss mode that reproduces the original TrajTok
spatial-aware label-smoothing formula, including its second division by the
same spatial-weight sum. This mode is intended for a controlled experiment;
it must not replace or silently alter CatK's current raw-GT normalized loss.

The reference implementation is the initial TrajTok implementation in:

- `/Users/huyuening/PycharmProjects/TrajTok/src/smart/metrics/utils.py`
- TrajTok commit `5920c89`

The later TrajTok compatibility switch in commit `b5b4b97` confirms that the
original behavior is the branch with
`spatial_aware_normalize_non_gt_mass=false`.

## Scope

- Add a selectable `trajtok_original` spatial-smoothing mode to CatK pre-BC.
- Preserve the original TrajTok probability construction exactly, including
  the non-normalized target mass.
- Use CatK's already-computed teacher-forced `gt_idx` for the 16 supervised
  future tokens.
- Add a plain pre-BC experiment entry for the compatibility mode.
- Allow the same mode to be selected by one Hydra override for existing
  history-dynamics and future-token-dynamics pre-BC experiments.
- Keep model parameters, vocabulary files, checkpoint keys, validity masks,
  and endpoint-only supervision unchanged.

This change does not modify CLSFT, inference rollout behavior, token matching,
WOSAC metrics, history dynamics, future token dynamics, or the model
architecture.

## Existing CatK Modes

CatK currently has two effective target-generation paths:

1. With `spatial_aware_smoothing=false`, CatK creates a one-hot target and
   delegates uniform label smoothing to PyTorch.
2. With `spatial_aware_smoothing=true`, CatK constructs a normalized spatial
   target centered on the raw ground-truth endpoint. The selected token
   receives `1 - epsilon`, and all non-selected tokens together receive
   exactly `epsilon`.

The second path intentionally corrected and adapted the original TrajTok
formula. It must remain the default behavior of existing pre-BC experiment
files so prior CatK experiments remain reproducible.

## Original TrajTok Target

For each agent and future token step, let \(j\) be the teacher-forced token
index `gt_idx`. Let \(C_j\) be the final four-corner contour of that token and
\(C_i\) the corresponding contour of candidate token \(i\).

TrajTok computes:

\[
d_i = \frac{1}{4}\sum_{c=1}^{4}
      \left\lVert C_{j,c} - C_{i,c}\right\rVert_2
\]

\[
w_i =
\begin{cases}
0, & i=j,\\
(d_i + 10^{-4})^{-2}, & i\ne j,
\end{cases}
\qquad
S = \sum_i w_i.
\]

The original implementation first normalizes `proj = w / S`, then divides
that normalized value by the same original `S` again:

\[
q_j = 1-\epsilon,
\qquad
q_i = \epsilon\frac{w_i}{S^2}, \quad i\ne j.
\]

Consequently:

\[
\sum_i q_i = 1-\epsilon+\frac{\epsilon}{S},
\]

which is generally not one. The compatibility mode must retain this behavior.
It must not clamp `S`, renormalize the result, redistribute a fixed
non-ground-truth mass, or move the spatial center to raw ground truth.

With the standard 2048-token vocabulary, `S` has non-ground-truth terms. A
single-token or otherwise degenerate vocabulary can reproduce TrajTok's
division-by-zero behavior; the compatibility path will not add a mathematical
fallback that changes the reference formula.

## CatK Integration

### Configuration

Retain the existing boolean gate:

```yaml
training_loss:
  label_smoothing: 0.1
  spatial_aware_smoothing: false
  spatial_aware_smoothing_mode: raw_gt_normalized
```

`spatial_aware_smoothing_mode` accepts:

- `raw_gt_normalized`: the current CatK spatial target;
- `trajtok_original`: the original TrajTok target described above.

When `spatial_aware_smoothing=false`, the mode is ignored and the existing
uniform PyTorch label-smoothing path is used. An unknown mode raises a clear
configuration error during loss construction.

The base default is `raw_gt_normalized`. Existing `pre_bc` continues enabling
spatial smoothing and therefore retains its current behavior.

Add:

```text
configs/experiment/pre_bc_trajtok_original.yaml
```

This file inherits `pre_bc` and changes only
`training_loss.spatial_aware_smoothing_mode` to `trajtok_original`.

Existing composed experiments can enable the mode without creating a
combinatorial set of experiment files:

```text
model.model_config.training_loss.spatial_aware_smoothing_mode=trajtok_original
```

### Data Flow

CatK's token processor already returns:

```text
tokenized_agent["gt_idx"]  # [n_agent, 18]
```

The decoder predicts the 16 transitions after the two historical tokens.
`SMART.training_step` and open-loop `validation_step` will therefore pass:

```text
tokenized_agent["gt_idx"][:, 2:]  # [n_agent, 16]
```

to `CrossEntropy`.

The original mode uses this index directly. The current
`raw_gt_normalized` mode continues using CatK's rollout-relative Euclidean
target and does not change its label construction.

### Loss Helper

Add a separate helper for the original TrajTok formula instead of adding
branches inside CatK's current normalized helper. The separate helper makes
the compatibility behavior auditable and reduces the risk of changing the
existing experiment.

The helper will:

1. gather the final token contour at every `gt_idx`;
2. compute mean four-corner distances to all candidate tokens;
3. mask the selected token;
4. compute inverse-square weights with the original `1e-4` offset;
5. divide by `proj_sum`;
6. divide by the same `proj_sum` again while multiplying by
   `label_smoothing`;
7. assign `1 - label_smoothing` to the selected token.

The resulting soft target is passed to `torch.nn.functional.cross_entropy`
with PyTorch's built-in `label_smoothing=0`, matching TrajTok and avoiding a
second independent uniform smoothing operation.

### Architectural Difference

TrajTok uses type-split prediction heads and type-specific token tensors.
CatK uses one shared 2048-class head, while `token_traj` is already populated
per agent with the appropriate vehicle, pedestrian, or cyclist vocabulary
geometry.

The compatibility mode therefore reproduces TrajTok's target index, contour
distance, probability formula, and cross-entropy handling. It does not replace
CatK's classifier with TrajTok's type-split architecture; doing so would be a
different model experiment and would break checkpoint compatibility.

## Backward Compatibility

- Existing `pre_bc` behavior remains `raw_gt_normalized`.
- Existing history/future-dynamics pre-BC experiments retain their current
  behavior unless explicitly overridden.
- `clsft` continues setting `spatial_aware_smoothing=false`.
- Model parameter shapes and state-dict keys do not change.
- Existing checkpoints remain loadable.
- The current normalized raw-GT helper remains unchanged.
- Uniform label smoothing remains unchanged when the spatial gate is off.

## Testing

Add focused tests that verify:

1. The CatK compatibility helper is numerically equal to a local reference
   implementation copied from TrajTok commit `5920c89`.
2. The helper centers distances on the contour selected by `gt_idx`, not the
   raw ground-truth endpoint.
3. The second `proj_sum` division is retained.
4. The resulting target mass matches
   `1 - epsilon + epsilon / S` and is not forced to one.
5. The selected token remains exactly `1 - epsilon`.
6. `CrossEntropy` selects the original helper only for
   `trajtok_original` and uses built-in label smoothing equal to zero.
7. Missing `gt_idx` in original mode raises a clear error.
8. Existing raw-GT normalized and uniform-smoothing tests still pass.
9. The new experiment composes to `trajtok_original`, while existing
   `pre_bc` and `clsft` modes remain unchanged.

Run the focused loss tests and the adjacent history/future-dynamics tests to
detect unintended integration regressions.

## Experimental Interpretation

Because the original soft-target mass varies with \(S\), its reported
cross-entropy is not directly comparable in scale with the normalized CatK
loss. It also changes per-sample gradient magnitude according to vocabulary
geometry.

The comparison should therefore prioritize:

- open-loop token accuracy;
- validation trajectory metrics;
- closed-loop WOSAC metrics;
- training stability and non-finite-loss checks.

For a controlled comparison, use the same vocabulary, cached data, model
inputs, random seed, batch size, optimizer, and validation settings, changing
only `spatial_aware_smoothing_mode`.

## Success Criteria

- The compatibility helper matches the original TrajTok formula numerically.
- A plain pre-BC run can select the mode through
  `experiment=pre_bc_trajtok_original`.
- Any existing pre-BC family can select it through one Hydra override.
- Existing CatK experiments preserve their prior resolved configuration.
- No model parameters or checkpoint keys change.
- Focused and adjacent regression tests pass.
