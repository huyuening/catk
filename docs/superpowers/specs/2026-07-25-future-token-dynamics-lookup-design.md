# Future-Token Dynamics Lookup Design

## Goal

Condition each autoregressive prediction after the first future token on three
body-frame dynamics values derived directly from the selected token's six-frame
vocabulary trajectory:

\[
[a_{\mathrm{lon}},\ \omega,\ a_{\mathrm{lat}}].
\]

Provide two matched experiment families:

- the existing unreconstructed vocabulary,
  `agent_vocab_555_s2.pkl`;
- the offline reconstructed vocabulary,
  `agent_vocab_reconstructed.pkl`.

The two families must share one implementation and differ only in
`agent_token_file`. Existing raw-history dynamics remain unchanged and continue
to supply the two observed history-token features.

## Scope

- Add an optional future-token dynamics branch, disabled in the base model.
- Build a lookup table once from whichever agent vocabulary is loaded.
- Train the branch causally in pre-BC teacher-forced decoding.
- Use the selected rollout token to condition only the following prediction in
  CLSFT and inference.
- Provide pre-BC, CLSFT, and inference configurations for both vocabulary
  variants.
- Preserve the existing raw 11-frame history reconstruction, cache format, map
  inputs, traffic-light inputs, token matching, sampling, and prediction head.
- Keep the original CatK and history-only experiment configurations backward
  compatible.

Generating `agent_vocab_reconstructed.pkl` is outside this change. The existing
offline vocabulary-reconstruction command remains responsible for creating it.
The reconstructed configuration consumes that file from
`src/smart/tokens/`.

Angular acceleration, cross-token seam features, a dynamics auxiliary loss,
and any modification of the selected token trajectory are separate future
ablations.

## Chosen Dynamics

The lookup keeps angular speed rather than replacing it with angular
acceleration.

- Angular speed directly carries turn direction and turn intensity.
- Angular acceleration is ambiguous without angular speed: a positive value
  can mean entering a left turn or leaving a right turn.
- Six 10 Hz samples support a substantially more stable first derivative than
  a second derivative at the token endpoint.
- Lateral acceleration and angular speed remain complementary: lateral
  acceleration depends on speed, whereas angular speed directly describes
  heading evolution.

A read-only analysis of `agent_vocab_555_s2.pkl` also found substantially
greater endpoint sensitivity for angular acceleration. At the 95th percentile,
the absolute difference between the endpoint estimate and the mean over the
last three frames was:

| Agent type | Angular speed | Angular acceleration |
| --- | ---: | ---: |
| Vehicle | 0.0695 rad/s | 0.777 rad/s² |
| Pedestrian | 6.44 rad/s | 63.48 rad/s² |
| Cyclist | 0.592 rad/s | 6.11 rad/s² |

The units differ, so these values are not a direct error ratio. They
nevertheless confirm the expected boundary sensitivity of the second
derivative. Angular acceleration may later be evaluated as a separately gated
fourth feature computed from an 11-point, two-token window.

## Lookup-Table Construction

Each class-specific vocabulary contains a tensor with shape
`[n_token, 6, 4, 2]`. For each token:

1. Compute the six center positions as the mean of the four contour corners.
2. Compute heading from the existing CatK contour convention, using the vector
   from corner 3 to corner 0.
3. Unwrap heading along the six-frame time dimension.
4. Use the same 0.1-second, second-order finite-difference convention as the
   existing history-dynamics calculation:
   - differentiate center position to obtain velocity;
   - differentiate velocity to obtain global acceleration;
   - differentiate unwrapped heading to obtain angular speed.
5. Project global acceleration into the token's endpoint body frame:

   \[
   a_{\mathrm{lon}}=a_x\cos\theta+a_y\sin\theta,
   \]

   \[
   a_{\mathrm{lat}}=-a_x\sin\theta+a_y\cos\theta.
   \]

6. Retain the values at frame 5, producing one `[3]` vector per token.
7. Clip the values to the same physical limits used by history dynamics:
   `[15 m/s², 3 rad/s, 15 m/s²]`.

This is direct differentiation of the vocabulary trajectory. It must not call
the trajectory filter/reconstructor. A reconstructed vocabulary has already
received its smoothing offline; reconstructing its six points again would be
redundant and could alter token semantics.

The table is constructed once during `TokenProcessor` initialization and
registered as three non-persistent buffers:

- `agent_token_dynamics_veh`;
- `agent_token_dynamics_ped`;
- `agent_token_dynamics_cyc`.

The buffers have shape `[n_token, 3]` and follow the same token ordering as the
corresponding vocabulary trajectory buffers. They are exposed to the decoder
through the tokenized-agent dictionary. Initialization rejects an invalid
trajectory shape or any non-finite derived value with an error that identifies
the vocabulary file and agent class.

Constructing the table from the loaded trajectory, rather than storing new
fields in the pickle, preserves compatibility with both vocabulary formats and
guarantees that token indices and lookup rows cannot drift apart.

## Decoder Branch

Add a separate optional `future_token_dynamics` configuration:

```yaml
future_token_dynamics:
  is_active: false
  normalization_scale: [5.0, 1.0, 5.0]
  initial_gate: 1.0
```

When active, the agent decoder owns:

- a three-input MLP embedding dedicated to token-derived future dynamics;
- a learnable scalar gate;
- a non-persistent normalization-scale buffer.

The new embedding is separate from the existing raw-history dynamics
embedding. The two signals have different provenance and uncertainty:
history dynamics come from reconstructed observations, while future dynamics
are deterministic properties of a selected vocabulary token.

The lookup vector is normalized, embedded, gated, and added to the already
fused feature for that token. It does not replace the token embedding or the
existing displacement-and-angle motion feature.

## Causal Data Flow

CatK has two observed history tokens, ending at frames 5 and 10. Their features
continue to receive only the cached raw-history dynamics.

### Pre-BC Open-Loop Training

Pre-BC processes the matched token sequence with teacher forcing:

1. Positions 0 and 1 are observed history tokens. Future-token lookup is
   masked to zero for these positions.
2. Position 1 predicts the first future token at position 2 without access to
   that token's dynamics.
3. Starting at position 2, gather the lookup vector for the teacher-forced
   `sampled_idx`.
4. The feature at position \(t\) uses \(D(k_t)\) to predict token
   \(k_{t+1}\).

Thus the branch is trained during pre-BC without leaking the target token into
its own prediction.

### CLSFT and Inference Rollout

At rollout step \(t\):

1. Predict logits for \(k_t\) from the current autoregressive state.
2. Select \(k_t\) using the unchanged CAT-K sampling procedure.
3. Gather the class-appropriate lookup row \(D(k_t)\).
4. Add its embedded value to `feat_a_next`.
5. Append that feature to the temporal sequence.
6. Use it only while predicting \(k_{t+1}\).

The first future-token logits are therefore identical in data availability to
the history-only model. Dynamics from a candidate or selected token can never
affect the logits that selected that same token.

## Matched Experiment Families

The implementation is shared. Dedicated configurations make run provenance
explicit.

### Unreconstructed vocabulary

- `pre_bc_history_future_token_dynamics`
- `clsft_history_future_token_dynamics`
- `inference_history_future_token_dynamics`

These inherit their existing `*_history_dynamics` counterparts, enable
`future_token_dynamics`, and retain:

```yaml
agent_token_file: agent_vocab_555_s2.pkl
```

### Reconstructed vocabulary

- `pre_bc_history_future_token_dynamics_reconstructed`
- `clsft_history_future_token_dynamics_reconstructed`
- `inference_history_future_token_dynamics_reconstructed`

These inherit the corresponding unreconstructed future-dynamics experiment
and override only:

```yaml
agent_token_file: agent_vocab_reconstructed.pkl
```

All other architecture, optimizer, loss, sampling, history-dynamics, and
validation settings remain identical. Each CLSFT run must load the pre-BC
checkpoint trained with the same vocabulary, and inference must use the
matching configuration. A checkpoint trained against one vocabulary must not
be evaluated with the other vocabulary merely because both contain 2048
tokens per class.

## Backward Compatibility and Errors

- `future_token_dynamics.is_active` defaults to `false`, so existing model and
  experiment configurations retain their current parameter set and behavior.
- The new configurations require caches containing the existing raw-history
  dynamics fields, just like the current history-dynamics experiments.
- Enabling future-token dynamics changes checkpoint keys because it creates a
  new embedding and gate. Pre-BC, CLSFT, and inference must all enable the
  branch for those checkpoints.
- A missing reconstructed vocabulary fails at model initialization with its
  expected path. The implementation does not silently fall back to the
  unreconstructed vocabulary.
- Vocabulary classes must have the same token count because the current shared
  prediction head assumes one common class count.

## Testing

Add focused tests for:

1. A straight, constant-acceleration six-frame token produces the expected
   longitudinal acceleration and zero angular/lateral motion.
2. A constant-radius token produces the expected angular speed and lateral
   acceleration.
3. Heading that crosses \(-\pi/\pi\) is unwrapped before differentiation.
4. Rigid rotation and translation leave body-frame lookup dynamics unchanged.
5. Lookup construction preserves `[n_token, 3]`, dtype, device movement, token
   order, and class separation.
6. Invalid shapes and non-finite values are rejected.
7. Open-loop positions 0 and 1 receive no token-derived future dynamics, while
   each position from 2 onward receives the lookup value for its own
   teacher-forced token and uses that feature only to predict the next token.
8. In rollout, a selected token's lookup is added to `feat_a_next` only after
   its logits have been computed.
9. Disabling the branch reproduces the existing feature path.
10. The three reconstructed experiment configs inherit their unreconstructed
    counterparts and differ only by `agent_token_file`.
11. The six new configs retain raw-history dynamics and enable future-token
    dynamics.
12. Existing repository tests continue to pass.

The feature tests must follow a red-green cycle. Full model verification should
include the focused tests, the complete repository test suite, and Hydra
configuration composition for all six experiment names.

## Success Criteria

- Both vocabulary variants train and evaluate through the same lookup code.
- The only experimental difference between matched variants is
  `agent_token_file`.
- Existing raw-history dynamics remain active and unchanged.
- Future dynamics have the exact order
  `[a_lon, angular_speed, a_lat]`.
- The first future token cannot consume its own or any later token's dynamics.
- Every later prediction can consume the dynamics of the token selected at the
  preceding rollout step.
- No online trajectory reconstruction is introduced.
- Legacy configurations remain behaviorally and checkpoint compatible when
  the feature is disabled.
- Focused tests, configuration-composition tests, and the existing test suite
  pass.
