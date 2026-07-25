# Training-Trajectory Token-Transition Dynamics

This optional CatK branch conditions each future token on three body-frame
dynamics features associated with the selected transition
`(previous_token, current_token)`:

- signed longitudinal acceleration `a_lon`;
- angular speed `omega`;
- signed lateral acceleration `a_lat`.

The lookup is built once from complete 91-frame trajectories in the **training
split only**. Validation and test trajectories are never scanned by the
builder. At training, validation, and inference time, CatK reads only the fixed
lookup and the already-known or already-selected token IDs.

The base `smart` configuration keeps the branch disabled. Use one of the
dedicated experiments below to enable it.

## 1. Build the fixed lookup

Run the builder from the CatK repository root. For the original CatK
vocabulary and original CatK training cache:

```bash
RAW_CACHE=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact
RAW_LOOKUP="$RAW_CACHE/agent_transition_dynamics.pt"

python -m src.smart.tokens.build_transition_dynamics \
  --training-dir "$RAW_CACHE/training" \
  --agent-token-file src/smart/tokens/agent_vocab_555_s2.pkl \
  --source raw \
  --output "$RAW_LOOKUP"
```

For the full-trajectory-reconstructed training cache and matching reconstructed
vocabulary:

```bash
RECON_CACHE=/path/to/reconstruction_output/datasets/reconstructed
RECON_LOOKUP="$RECON_CACHE/agent_transition_dynamics_reconstructed.pt"

python -m src.smart.tokens.build_transition_dynamics \
  --training-dir "$RECON_CACHE/training" \
  --agent-token-file src/smart/tokens/agent_vocab_reconstructed.pkl \
  --source reconstructed \
  --output "$RECON_LOOKUP"
```

The reconstructed cache must contain
`agent.trajectory_reconstructed=true` for every agent. The raw builder rejects
that marker, while the reconstructed builder requires it; this prevents the
two table families from being mixed accidentally.

Each command writes:

- the float16 lookup artifact specified by `--output`;
- a neighboring `*.summary.json` file containing scenario, occurrence, and
  lookup-coverage statistics.

The artifact records the exact vocabulary SHA-256 digest. CatK refuses to load
the table if its vocabulary file, token count, feature order, or source does
not match.

Useful optional builder arguments are:

```text
--batch-size 8
--num-workers 8
--shrinkage-count 8
--max-scenarios N
```

Omit `--max-scenarios` for the complete training split. It is intended only
for a smoke test. No validation or testing directory is accepted by the
builder.

## 2. Pre-BC training

Raw vocabulary/table:

```bash
export CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact
export LOOKUP_FILE="$CACHE_ROOT/agent_transition_dynamics.pt"
export MY_EXPERIMENT=pre_bc_history_future_token_dynamics
export MY_TASK_NAME=pre_bc_history_future_token_dynamics_b200

bash scripts/train.sh \
  ckpt_path=null \
  model.model_config.future_token_dynamics.lookup_file="$LOOKUP_FILE"
```

Reconstructed vocabulary/table:

```bash
export CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact
export LOOKUP_FILE=/path/to/agent_transition_dynamics_reconstructed.pt
export MY_EXPERIMENT=pre_bc_history_future_token_dynamics_reconstructed
export MY_TASK_NAME=pre_bc_history_future_token_dynamics_reconstructed_b200

bash scripts/train.sh \
  ckpt_path=null \
  model.model_config.future_token_dynamics.lookup_file="$LOOKUP_FILE"
```

`CACHE_ROOT` above remains the normal CatK runtime cache containing
`history_dynamics` for training and validation. The reconstructed,
agent-only training cache is used only by the offline lookup builder.

## 3. CLSFT

Use the experiment family and lookup that match the pre-BC checkpoint:

```bash
PRE_BC_CKPT=/path/to/pre_bc/checkpoints/last.ckpt
LOOKUP_FILE=/path/to/matching/agent_transition_dynamics.pt

MY_EXPERIMENT=clsft_history_future_token_dynamics \
MY_TASK_NAME=clsft_history_future_token_dynamics_b200 \
bash scripts/train.sh \
  ckpt_path="$PRE_BC_CKPT" \
  model.model_config.future_token_dynamics.lookup_file="$LOOKUP_FILE"
```

For the reconstructed family, replace the experiment with:

```text
clsft_history_future_token_dynamics_reconstructed
```

and provide the reconstructed lookup.

## 4. Full validation

Raw family:

```bash
CATK_CKPT=/path/to/checkpoints/last.ckpt \
CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact \
python run.py \
  experiment=inference_history_future_token_dynamics \
  model.model_config.future_token_dynamics.lookup_file=/path/to/agent_transition_dynamics.pt \
  trainer.limit_val_batches=1.0 \
  task_name=history_future_token_dynamics_full
```

Reconstructed family:

```bash
CATK_CKPT=/path/to/checkpoints/last.ckpt \
CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact \
python run.py \
  experiment=inference_history_future_token_dynamics_reconstructed \
  model.model_config.future_token_dynamics.lookup_file=/path/to/agent_transition_dynamics_reconstructed.pt \
  trainer.limit_val_batches=1.0 \
  task_name=history_future_token_dynamics_reconstructed_full
```

Use a fractional `trainer.limit_val_batches`, such as `0.1`, for a deterministic
prefix of the validation loader rather than the full set.

## Compatibility rules

- A raw checkpoint, raw vocabulary, and raw lookup form one compatible family.
- A reconstructed checkpoint, reconstructed vocabulary, and reconstructed
  lookup form the other compatible family.
- The same fixed lookup must be used for pre-BC, CLSFT, and evaluation of a
  given run family.
- The lookup contains no trainable parameters and is registered as a
  non-persistent model buffer, so it is not duplicated inside checkpoints.
- Legacy CatK experiments and checkpoints remain unchanged because
  `future_token_dynamics.is_active` defaults to `false`.
