# Training-Trajectory Token-Transition Dynamics Design

## Status and Relationship to the Previous Design

This design supersedes only the lookup-source portion of
`2026-07-25-future-token-dynamics-lookup-design.md`.

The existing optional decoder branch, three-value order, normalization, gate,
causal placement, matched experiment families, and disabled-by-default behavior
remain. The lookup values will no longer be calculated independently from each
six-frame vocabulary trajectory. They will instead be estimated offline from
complete 91-frame trajectories in the training split.

## Goal

Build a fixed, class-specific token-transition lookup from complete CatK
training trajectories:

\[
D(c, k_{t-1}, k_t)
  = [a_{\mathrm{lon}},\ \omega,\ a_{\mathrm{lat}}],
\]

where \(c\) is the agent class and \(k_{t-1}, k_t\) are the previous and current
token IDs.

The lookup must:

- use only the training split;
- compute dynamics continuously over each available 91-frame trajectory,
  rather than independently inside each six-frame token;
- preserve cross-token motion at token endpoints;
- support both the CatK raw-cache vocabulary and the full-trajectory
  reconstructed vocabulary;
- be fixed before model training;
- use only already matched or selected token IDs during model execution;
- never read validation/test trajectory dynamics or future ground truth.

## Why the Lookup Is Keyed by a Token Pair

A single token describes a local six-frame displacement, but its endpoint
velocity and acceleration also depend on how the trajectory entered that
token. The same token can therefore have different boundary dynamics in
different contexts.

Three approaches were considered:

1. **Current-token marginal lookup.** Small and fast, but averages away the
   cross-token context that motivates using complete trajectories.
2. **Previous/current token-pair lookup.** Retains the incoming boundary
   context, is available causally after the current token is selected, and
   directly serves the next-token prediction.
3. **Per-occurrence training features.** Retains all context, but cannot be
   reproduced for generated validation/test rollouts and would create a
   training/inference mismatch.

The second approach is selected. Per-token marginal estimates remain only as
the fallback for unobserved or rare pairs.

## Data Boundary and Leakage Guarantee

The offline builder accepts one explicit `--training-dir`. It does not accept a
validation or test directory and does not instantiate a validation/test
dataloader.

Using complete trajectories, centered finite differences, or a non-causal
full-trajectory reconstruction is permitted during table construction because
the result is aggregated across the training corpus into a fixed function of
token IDs. No occurrence-specific dynamics leave the builder.

At model runtime:

- open-loop pre-BC may use teacher-forced previous/current token IDs, as it
  already does for token embeddings;
- closed-loop validation, CLSFT, and inference use selected rollout token IDs;
- no runtime path reads a scene's cached future dynamics;
- the dynamics for \(k_t\) are added only after \(k_t\) has been matched or
  selected, and affect only prediction of \(k_{t+1}\).

This prevents both cross-split contamination and target-token leakage.

## Supported Training-Trajectory Sources

The builder supports two explicit source modes.

### CatK raw-cache mode

- Read the `training` cache used by the original CatK vocabulary.
- Use the cached 91-frame `position`, `heading`, and `valid_mask`.
- Apply the same heading cleanup and sequential token matching used by CatK.
- Do not invoke the full-trajectory reconstructor.

The existing CatK cache linearly fills internal gaps between the first and last
valid observations. “Raw” therefore means the legacy CatK cache trajectory,
not untouched Scenario proto arrays.

### Full-trajectory reconstructed mode

- Read the reconstructed `training` cache produced for vocabulary clustering.
- Require its `trajectory_reconstructed` provenance marker.
- Use its reconstructed 91-frame position and heading directly.
- Do not reconstruct the trajectory a second time.
- Match against `agent_vocab_reconstructed.pkl`.

The source cache and vocabulary must belong to the same raw/reconstructed
family.

## Offline Builder

Add a module runnable as:

```bash
python -m src.smart.tokens.build_transition_dynamics \
  --training-dir /path/to/cache/training \
  --agent-token-file /path/to/agent_vocab.pkl \
  --source raw \
  --output /path/to/agent_transition_dynamics.pt
```

The reconstructed form uses `--source reconstructed` and the reconstructed
training cache and vocabulary.

The builder processes files in a deterministic sorted order. Configurable
batch size and dataloader worker count affect throughput only. An optional
scenario limit supports smoke tests.

### Per-agent processing

For each eligible agent:

1. Read all 91 cached frames.
2. Clean or validate heading according to the selected source mode.
3. Unwrap heading over each contiguous valid run.
4. Differentiate global \(x,y\) twice at 10 Hz to obtain acceleration.
5. Differentiate unwrapped heading once to obtain angular speed.
6. Project acceleration into the reconstructed or cached body heading:

   \[
   a_{\mathrm{lon}}=a_x\cos\theta+a_y\sin\theta,
   \]

   \[
   a_{\mathrm{lat}}=-a_x\sin\theta+a_y\cos\theta.
   \]

7. Clip values to `[15 m/s², 3 rad/s, 15 m/s²]`.
8. Match the complete trajectory sequentially to the selected vocabulary at
   endpoints `[5, 10, ..., 90]`, using CatK's deterministic `num_k=1`
   matching rule.
9. For every adjacent valid token pair, accumulate the current endpoint's
   three values under `(agent_type, previous_token, current_token)`.
10. Also accumulate a marginal estimate under `(agent_type, current_token)`.

An occurrence is skipped when either token is invalid, the endpoint dynamics
are unavailable, or any derived value is non-finite. Skipping an occurrence
does not discard the rest of the agent trajectory.

### Aggregation and sparse-pair fallback

Per-occurrence values are clipped before accumulation. The builder maintains
float64 sums and integer counts, then computes:

- a current-token marginal mean;
- a token-pair mean;
- an empirical shrinkage estimate

  \[
  \hat D_{\mathrm{pair}}
  =
  \frac{n D_{\mathrm{pair}}+\alpha D_{\mathrm{current}}}
       {n+\alpha},
  \]

  with configurable \(\alpha=8\) by default.

This uses rare pair observations without allowing a single noisy occurrence to
dominate. An unobserved pair falls back exactly to the current-token marginal.
If a vocabulary token is not observed at all, its marginal falls back to the
existing isolated-token geometry calculation. That final fallback guarantees
a defined value for every valid token ID without consulting another split.

## Artifact Format

The builder writes a tensor-only PyTorch artifact that is compatible with
`torch.load(..., weights_only=True)`.

Required fields:

- `format_version`;
- `feature_order = ["a_lon", "angular_speed", "a_lat"]`;
- `values`, shaped `[3, n_token, n_token, 3]`;
- `vocabulary_sha256`;
- `vocabulary_size`;
- `source` (`raw` or `reconstructed`);
- `dt`;
- `clipping_limits`;
- `shrinkage_count`;
- aggregate occurrence and coverage statistics.

`values` is stored as float16 after aggregation. For 2,048 tokens and three
agent classes it occupies approximately 72 MiB. The runtime performs direct
constant-time indexing and casts the gathered three-value vector to the model
feature dtype. Sums and dense counts are builder-only and are not loaded on
each GPU.

The output is written to a temporary sibling and atomically renamed only after
all validation succeeds. An interrupted build cannot masquerade as a complete
artifact.

The builder also writes a small JSON summary containing class counts, pair
coverage, marginal coverage, skipped occurrences, vocabulary hash, and source
mode.

## Runtime Loading and Validation

Extend `future_token_dynamics` configuration with:

```yaml
future_token_dynamics:
  is_active: false
  lookup_file: null
  source: raw
  normalization_scale: [5.0, 1.0, 5.0]
  initial_gate: 1.0
```

When active:

1. `lookup_file` is required.
2. `TokenProcessor` loads the artifact once with `weights_only=True`.
3. It calculates the SHA-256 digest of the configured agent vocabulary.
4. Loading fails before training if the digest, token count, source family,
   tensor shape, feature order, or finite-value check does not match.
5. The transition tensor is registered as a non-persistent buffer so it moves
   with the model but is not duplicated inside every checkpoint.

The large artifact is generated on the development machine and is not
committed to Git.

## Causal Decoder Data Flow

### Open-loop pre-BC

CatK's matched sequence is
`[k_0, k_1, ..., k_17]`.

- Positions 0 and 1 retain the existing actual history-dynamics inputs.
- Future-transition dynamics are zero at positions 0 and 1.
- For every position \(t \ge 2\), gather
  `D(type, sampled_idx[t-1], sampled_idx[t])`.
- Add the result to the feature at position \(t\).
- That feature predicts only \(k_{t+1}\).

No value indexed by \(k_{t+1}\) is available while predicting \(k_{t+1}\).

### Closed-loop CLSFT and inference

After the model selects a new token \(k_t\):

1. retain the previously selected token \(k_{t-1}\);
2. gather `D(type, k[t-1], k[t])`;
3. add it to the appended autoregressive feature;
4. use that feature only to predict \(k_{t+1}\).

The transition lookup never uses `gt_idx`, `gt_pos_raw`, or cached future
dynamics in rollout.

## Experiment Configurations

Keep the six experiment names introduced by the previous design:

- `pre_bc_history_future_token_dynamics`;
- `clsft_history_future_token_dynamics`;
- `inference_history_future_token_dynamics`;
- and their three `_reconstructed` variants.

Update them to require the matching transition artifact. The unreconstructed
family uses the raw-cache vocabulary and raw table. The reconstructed family
uses the reconstructed vocabulary and reconstructed table.

The lookup path can be supplied through a Hydra override, so generated
artifacts may remain on high-capacity shared storage:

```bash
model.model_config.future_token_dynamics.lookup_file=/path/to/table.pt
```

The base CatK, history-only, and all configurations with
`future_token_dynamics.is_active=false` remain checkpoint compatible.

## Error Handling

Fail early with an actionable message when:

- the training directory or vocabulary does not exist;
- reconstructed mode lacks reconstructed-cache provenance;
- raw mode is paired with a reconstructed cache;
- an agent type or vocabulary shape is unsupported;
- fewer than three usable frames exist in a derivative run;
- the final artifact contains a non-finite value;
- a runtime artifact has an unsupported version;
- a vocabulary digest or token count differs;
- an active experiment has no lookup path.

Individual malformed occurrences are counted and skipped. Structural
configuration or provenance errors abort the build.

## Testing

Use test-driven implementation with focused tests for:

1. A complete constant-acceleration trajectory produces the expected
   longitudinal acceleration at all token endpoints.
2. A constant-radius trajectory produces the expected angular speed and
   lateral acceleration.
3. A value at a token endpoint uses the continuous 91-frame derivative rather
   than a separately differentiated six-frame slice.
4. Heading wraparound is handled over the complete valid run.
5. Sequential matching produces the same token IDs as CatK's deterministic
   matcher.
6. Pair and marginal accumulators remain separated by agent class.
7. Shrinkage, unseen-pair fallback, and unseen-token isolated fallback are
   numerically correct.
8. Raw/reconstructed provenance mismatches are rejected.
9. Artifact vocabulary-hash and shape mismatches are rejected.
10. Open-loop position \(t\) gathers `(t-1, t)`, masks positions 0 and 1, and
    cannot consume token \(t+1\).
11. Rollout gathers only the previous and newly selected token IDs.
12. Disabling the branch preserves legacy behavior.
13. All six experiment configurations compose with an explicit table path.
14. Existing CatK tests continue to pass.

## Success Criteria

- Only training-cache files are read while creating the table.
- The raw and reconstructed variants use their matching full 91-frame
  trajectories and vocabulary hashes.
- Runtime validation and inference do not read occurrence-level future
  dynamics.
- Cross-token dynamics are indexed by previous/current token pairs.
- All token pairs have a deterministic value through shrinkage and fallback.
- The first future-token prediction remains free of future-token lookup
  information.
- A selected token can affect only the following prediction.
- Existing experiments remain unchanged when the branch is disabled.
- Focused tests, Hydra composition checks, and the repository test suite pass.
