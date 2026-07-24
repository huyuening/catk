# Raw-GT Spatial-Aware Label Smoothing Design

## Goal

Replace uniform label smoothing in CatK behavior-cloning pre-training with a
paper-correct, spatial-aware target distribution centered on the raw
ground-truth endpoint. Keep supervision endpoint-only and preserve CatK's
existing rollout-relative target construction.

## Scope

- Apply the new loss to `pre_bc` and experiments that inherit from it,
  including `pre_bc_history_dynamics`.
- Keep the default model configuration backward compatible.
- Keep `clsft`, inference, model architecture, vocabulary files, token
  matching, and checkpoint structure unchanged.
- Continue supervising only the final contour of each 0.5-second token.
- Use a normalized non-ground-truth probability mass, as required by the
  TrajTok paper.

Full six-frame token supervision, focal loss, class-frequency reweighting,
separate prediction heads, and future-dynamics auxiliary losses are outside
this change.

## Current Behavior

CatK first expresses the next ground-truth endpoint in the local frame of the
current predicted or tokenized state. It converts that endpoint and heading
into a four-corner contour, selects the closest token endpoint, and builds a
one-hot target. PyTorch then applies uniform label smoothing, assigning equal
probability to every non-ground-truth token.

This target construction must remain intact because closed-loop CatK targets
are relative to the current rollout state rather than a fixed logged-history
token index.

## Proposed Target Distribution

For each valid agent and prediction step:

1. Use CatK's existing `get_euclidean_targets` result to construct the raw
   ground-truth endpoint contour \(C^*\).
2. Compute the endpoint contour distance to every type-appropriate token:

   \[
   d_i = \frac{1}{4}\sum_{c=1}^{4}\lVert C^*_c - C_{i,c}\rVert_2.
   \]

3. Select the hard target \(j=\arg\min_i d_i\).
4. Assign \(1-\epsilon\) to token \(j\).
5. Distribute the complete remaining mass \(\epsilon\) across all
   non-ground-truth tokens using inverse-square distance:

   \[
   q_i =
   \begin{cases}
   1-\epsilon, & i=j,\\
   \epsilon
   \dfrac{(d_i+\delta)^{-2}}
   {\sum_{m\ne j}(d_m+\delta)^{-2}}, & i\ne j.
   \end{cases}
   \]

The numerical stabilizer is fixed at \(\delta=10^{-4}\), matching the TrajTok
implementation. The resulting distribution must sum to one within floating
point tolerance. With `label_smoothing=0.1`, the target token receives `0.9`
and all other tokens together receive exactly `0.1`.

Unlike TrajTok, the non-ground-truth weights are centered directly on the raw
ground-truth contour rather than on the already-quantized target token. This
retains CatK's endpoint quantization residual: if the raw endpoint lies between
two tokens, the second-nearest token receives probability according to its
actual distance from the raw endpoint.

## Loss Integration

`CrossEntropy` receives a new boolean configuration option,
`spatial_aware_smoothing`, defaulting to `false`.

- When disabled, the current one-hot target plus PyTorch uniform
  `label_smoothing` path remains unchanged.
- When enabled, the raw-GT spatial target is constructed first and passed to
  cross entropy with PyTorch's built-in `label_smoothing=0`. This prevents
  applying smoothing twice.
- Existing validity masks, `train_mask`, `use_gt_raw`,
  `gt_thresh_scale_length`, and `rollout_as_gt` behavior remain unchanged.
- `label_smoothing` must be in the half-open interval `[0, 1)`. A value of
  zero produces an exact one-hot distribution.

The base model configuration explicitly keeps spatial smoothing disabled.
`pre_bc.yaml` enables it. Because `pre_bc_history_dynamics.yaml` inherits from
`pre_bc`, history-dynamics pre-training receives the new loss automatically.
`clsft.yaml` continues using its existing zero-smoothing objective.

## Differences from TrajTok

- TrajTok constructs its spatial neighborhood around a pre-tokenized
  ground-truth token; this CatK variant constructs it around the raw
  ground-truth endpoint.
- TrajTok consumes type-split logits from separate vehicle, pedestrian, and
  cyclist heads. CatK keeps one shared 2048-class head and supplies
  type-appropriate token geometry per agent.
- TrajTok consumes a fixed `gt_idx`. CatK retains a target computed relative
  to the current predicted or tokenized state.
- Both implementations compare only the final four-corner token contour.
- This design always normalizes the non-ground-truth mass. It does not expose
  the legacy TrajTok compatibility branch that divides by the normalization
  term twice.

## Numerical and Error Handling

- Clamp the non-ground-truth weight denominator to a small positive value
  before division.
- Mask the selected hard target before inverse-distance normalization so its
  zero distance cannot dominate the non-ground-truth mass.
- Preserve the input tensor device and floating-point dtype.
- Reject smoothing values outside `[0, 1)`.
- If a vocabulary contains only one token, return a one-hot target because no
  non-ground-truth class exists.

## Testing

Add focused unit tests that verify:

1. Spatial targets sum to one and assign exactly `1-epsilon` to the nearest
   token.
2. A token closer to the raw ground-truth contour receives more of the
   non-ground-truth mass than a farther token.
3. The distribution is centered on raw ground truth rather than the selected
   target token, using an asymmetric synthetic vocabulary.
4. `epsilon=0` produces the same one-hot target as the legacy helper.
5. Invalid smoothing values raise `ValueError`.
6. Spatial smoothing is enabled by `pre_bc` and inherited by
   `pre_bc_history_dynamics`, while the base model and `clsft` retain legacy
   behavior.
7. Existing repository tests continue to pass.

## Success Criteria

- No model parameter shapes or checkpoint keys change.
- Legacy loss output is unchanged when spatial smoothing is disabled.
- Pre-training uses a normalized raw-GT spatial target with endpoint-only
  supervision.
- History-dynamics pre-training uses the same revised pre-training loss.
- CLSFT remains unchanged.
- The focused tests and existing test suite pass.
