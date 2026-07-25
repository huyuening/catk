# Original-Vocabulary Reconstructed Transition Dynamics Design

## Status

Approved for implementation on 2026-07-25.

This design extends the training-derived transition-dynamics lookup described
in `2026-07-25-training-trajectory-transition-dynamics-design.md`. It does not
change the decoder branch, feature order, causal placement, normalization,
gating, or disabled-by-default behavior.

## Goal

Build a fixed CatK transition-dynamics lookup that deliberately combines two
training-cache views of the same WOMD scenarios:

- the original CatK trajectory determines the original vocabulary token IDs;
- the full-trajectory reconstructed trajectory supplies the three dynamics
  values.

For every accepted adjacent original-token pair, accumulate

\[
D(c, k_{t-1}^{raw}, k_t^{raw})
  = [a_{\mathrm{lon}},\ \omega,\ a_{\mathrm{lat}}],
\]

where the values on the right are evaluated at the endpoint of the current
segment in the corresponding reconstructed trajectory.

The resulting artifact remains bound to
`src/smart/tokens/agent_vocab_555_s2.pkl`. The reconstructed vocabulary is not
read or used by this workflow.

## Non-Goals

- Do not change how the original CatK vocabulary was clustered.
- Do not rematch reconstructed trajectories to either vocabulary.
- Do not write reconstructed positions, headings, or dynamics into the normal
  model-training cache.
- Do not expose occurrence-specific future ground truth at model runtime.
- Do not scan validation or testing data.
- Do not alter legacy single-cache transition-dynamics builds.

## Selected Approach

Extend the existing transition-dynamics builder with an explicit paired-cache
mode and provide a small shell wrapper for the common command.

Two alternatives were rejected:

1. **Occurrence sidecar followed by a second aggregation pass.** This makes
   individual samples easy to inspect but doubles full-corpus I/O and creates
   another large intermediate artifact.
2. **Writing reconstructed dynamics back into CatK caches.** This increases
   storage, mixes fixed training-corpus statistics with runtime inputs, and
   makes accidental future-information use easier.

The selected paired mode performs matching, dynamics extraction, and
aggregation in one deterministic pass while preserving the current runtime
lookup interface.

## Inputs

The paired builder accepts:

- an assignment training directory, normally
  `datasets/original/training`;
- a dynamics training directory, normally
  `datasets/reconstructed/training`;
- the original CatK agent vocabulary;
- an output artifact path;
- the existing batch-size, worker-count, scenario-limit, and shrinkage
  controls.

The existing `--training-dir` single-cache interface remains supported. A
caller must select exactly one input form:

- `--training-dir`; or
- both `--assignment-training-dir` and `--dynamics-training-dir`.

Supplying an incomplete or mixed form is an error.

## Scenario and Agent Alignment

Pairing must not depend only on DataLoader position or agent array order.

### Scenario pairing

1. Enumerate canonical cache files in both directories in sorted order.
2. Require identical filename sets.
3. Load paired files and require equal `scenario_id`.
4. Require equal `current_time_index` when the field exists.

### Agent pairing

1. Require a one-dimensional, unique `agent.id` array in each cache.
2. Require identical agent-ID sets.
3. Reorder the reconstructed agent fields into original-cache agent order.
4. Require the aligned `agent.type` values to match.
5. Require compatible trajectory frame counts and tensor shapes.

`role` and `shape` are not consumed by the lookup calculation, but paired
caches produced by CatK's reconstruction comparison pipeline already validate
that they remain unchanged. The paired builder records identity/type checks
that are necessary for safe lookup generation and does not silently drop
unmatched agents.

The assignment cache must be raw according to
`trajectory_reconstructed`; the dynamics cache must mark every agent as
reconstructed. Violations stop the build.

## Token Assignment and Dynamics Extraction

For each aligned scenario:

1. Pass only the original-cache batch to CatK's existing
   `TokenProcessor.tokenize_agent`.
2. Match against `agent_vocab_555_s2.pkl` with deterministic `num_k=1`
   behavior.
3. Preserve the resulting original token indices and validity at endpoints
   `[5, 10, ..., 90]`.
4. Read only reconstructed `position`, `heading`, and `valid_mask` for dynamics
   extraction.
5. Compute dynamics continuously over each reconstructed 91-frame trajectory
   using the existing `extract_full_trajectory_dynamics` implementation.
6. At each current-token endpoint, collect
   `[a_lon, angular_speed, a_lat]` from the reconstructed result.
7. Accumulate that value under
   `(agent_type, previous_original_token, current_original_token)`.

An occurrence is accepted only when:

- both adjacent original tokens are valid;
- the reconstructed endpoint dynamics are valid;
- the agent type and both token IDs are in range;
- all three values are finite.

The builder skips only the invalid occurrence, not the remaining valid
segments of the agent.

## Aggregation

Aggregation remains identical to the existing lookup:

- float64 pair sums and integer pair counts;
- a current-token marginal estimated from reconstructed dynamics assigned to
  that original current token;
- empirical shrinkage toward the current-token marginal:

  \[
  \hat D_{\mathrm{pair}}
  =
  \frac{nD_{\mathrm{pair}}+\alpha D_{\mathrm{current}}}{n+\alpha},
  \]

  with \(\alpha=8\) by default;
- an unobserved pair falls back to the current-token marginal;
- an entirely unobserved original token falls back to dynamics derived from
  its original vocabulary geometry.

The final tensor shape remains `[3, n_token, n_token, 3]` and its dtype remains
`float16`.

## Artifact Provenance

The hybrid artifact uses the explicit source identifier:

```text
raw_tokens_reconstructed_dynamics
```

This identifier is added to the artifact and runtime validation allowlists. It
prevents a hybrid table from being silently loaded as either a pure raw table
or a reconstructed-vocabulary table.

The artifact remains bound to the exact original vocabulary bytes through its
SHA-256 digest. Its statistics and neighboring summary record:

- `assignment_source: raw`;
- `dynamics_source: reconstructed`;
- scenario and aligned-agent counts;
- candidate, accepted, and skipped occurrences;
- class-specific occurrence counts;
- pair/marginal coverage and fallback counts;
- original vocabulary digest and size.

The artifact layout does not otherwise change, so its format version need not
change.

## Runtime Configuration

The model continues to receive only token IDs and the fixed lookup. No
reconstructed scene trajectory is loaded during training, validation, or
inference.

Add original-vocabulary hybrid experiment variants for pre-BC, CLSFT, and
inference. They inherit the existing original-vocabulary future-dynamics
experiments and override only:

```yaml
model:
  model_config:
    future_token_dynamics:
      source: raw_tokens_reconstructed_dynamics
```

The base CatK configuration remains unchanged and keeps this branch disabled.

## Shell Entry Point

Add:

```text
scripts/build_original_vocab_reconstructed_dynamics.sh
```

The common invocation is:

```bash
RECON_OUTPUT=/mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1 \
bash scripts/build_original_vocab_reconstructed_dynamics.sh
```

Defaults:

- assignment directory:
  `$RECON_OUTPUT/datasets/original/training`;
- dynamics directory:
  `$RECON_OUTPUT/datasets/reconstructed/training`;
- vocabulary:
  `$CATK_ROOT/src/smart/tokens/agent_vocab_555_s2.pkl`;
- output:
  `$RECON_OUTPUT/agent_transition_dynamics_original_vocab_reconstructed.pt`.

`CATK_ROOT`, `VOCAB_FILE`, `LOOKUP_FILE`, `BATCH_SIZE`, `NUM_WORKERS`,
`SHRINKAGE_COUNT`, and an optional `MAX_SCENARIOS` remain overridable through
environment variables. The wrapper changes into `CATK_ROOT` before invoking
the Python module so imports and relative defaults are stable.

## Failure Handling

The builder fails before writing the final artifact when it encounters:

- a missing or empty input directory;
- different scenario-file sets;
- mismatched scenario IDs;
- duplicate, missing, or extra agent IDs;
- agent-type disagreement;
- malformed or non-91-frame trajectory tensors;
- raw/reconstructed provenance disagreement;
- an invalid original vocabulary;
- invalid numeric or builder configuration.

Artifact saving remains atomic. A failed build may leave logs but must not
leave a partially valid final `.pt` file.

## Testing Strategy

Implementation follows test-driven development.

1. Parser tests cover the exclusive single-cache and paired-cache input forms.
2. Alignment tests deliberately swap reconstructed agent order and verify
   pairing by `agent.id`.
3. A synthetic case gives original and reconstructed trajectories different
   accelerations, then verifies:
   - token IDs come from the original batch;
   - accumulated values come from the reconstructed batch.
4. Negative tests cover file-set mismatch, duplicate IDs, missing IDs, type
   mismatch, and invalid provenance.
5. Existing single-cache builder tests remain unchanged and passing.
6. Artifact tests accept the new source only when requested and still reject
   source or vocabulary mismatches.
7. Hydra composition tests verify the three hybrid experiment variants retain
   the original vocabulary and select the hybrid source.
8. A shell syntax check and help/smoke invocation cover the wrapper.

## Expected End-to-End Command

After the reconstruction comparison output exists:

```bash
export CATK_ROOT=/root/workspace/catk
export RECON_OUTPUT=/mnt/pfs/waymo_motion_1_3_0/catk_batch_vocab_v1

bash scripts/build_original_vocab_reconstructed_dynamics.sh
```

This command creates an original-vocabulary lookup whose token membership is
unchanged from CatK and whose aggregated three-value dynamics come only from
the aligned reconstructed training trajectories.
