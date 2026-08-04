# Hard-CE PRE_BC Text-Control Default Design

**Date:** 2026-08-04

## Objective

Replace the canonical frozen-PRE_BC text-control training workflow so that it
warm-starts from the user's selected hard-label cross-entropy checkpoint:

```text
/root/workspace/catk/logs/
pre_bc_history_dynamics_hard_ce_b200/
runs/2026-07-30_21-15-08/checkpoints/last.ckpt
```

The existing `text_control_pre_bc` experiment, launcher, checkpoint auditor,
tests, and operator guide will all describe this one default. The previous
`trajtok_original` text-training default will not be retained as a second
experiment. Text-control checkpoints already produced by the previous code
remain loadable for inference because inference reconstructs the model from
the configuration saved inside each checkpoint.

## Fixed Base Contract

The selected PRE_BC checkpoint is treated as:

```yaml
history_dynamics:
  is_active: true
  mode: cached_reconstructed

training_loss:
  spatial_aware_smoothing: false
  label_smoothing: 0.0
```

The agent vocabulary remains `agent_vocab_555_s2.pkl`. A real-checkpoint
preflight must fail if the checkpoint reports disabled history dynamics,
another history mode, enabled spatial smoothing, or nonzero label smoothing.
State-dict loading remains stricter than the metadata check: missing keys are
allowed only below `encoder.agent_encoder.text_control_adapter.`, unexpected
keys are forbidden, shapes and dtype classes must match, and base tensors must
be finite.

## Training Boundary

`action=finetune` continues to mean a new weights-only warm start. It must not
restore the PRE_BC optimizer, scheduler, AMP scaler, callbacks, epoch, global
step, checkpoint directory, or W&B run identity.

Every existing CatK parameter remains frozen, including map encoding,
motion-token embedding, temporal/map/agent attention, history-dynamics
embedding, token prediction head, and any future-token dynamics branch. The
only trainable parameters remain:

- rank-16 LoRA matrices in the last six DistilBERT attention layers;
- the DistilBERT-to-256 text projection;
- the six identity-initialized FiLM adapters.

The hard-CE loss is also used during text-adapter training. It is not merely
checkpoint provenance: the active text experiment explicitly sets
`spatial_aware_smoothing=false` and `label_smoothing=0.0` so Hydra inheritance
cannot silently restore the former TrajTok smoothing path.

Closed-loop adapter training keeps the existing initial protocol: 32 sampled
rollouts, temperature `1e-5`, learning rate `5e-5`, per-GPU batch size 4,
10 epochs, 10% deterministic validation slice, Fast WOSAC 2025, and a fresh
W&B run. These values are replication defaults rather than claims of optimal
hyperparameters.

## Text Data Contract

The source of text supervision does not change. Training prompts are generated
from WOMD `training` future trajectories through the repository's all-frame
action-labeling pipeline, converted to ECoSim-compatible action intervals, and
rendered with deterministic English templates. Validation-future prompts are
permitted only for oracle validation. Custom inference accepts user text and
must not load action tags, mappings, validation GT, test GT, or any other
future-derived artifact.

No external ECoSim natural-language corpus is required. ECoSim contributes the
tag organization and conditioning architecture, while the sample-level action
supervision comes from the user's WOMD training split.

## Configuration and Naming

The canonical files retain their existing names:

- `configs/experiment/text_control_pre_bc.yaml`;
- `scripts/train_text_control_pre_bc.sh`;
- `docs/text_control_pre_bc.md`;
- `src/smart/inference/audit_text_control.py`.

The launcher default checkpoint becomes the selected hard-CE path, and the
default task name becomes:

```text
text_control_pre_bc_history_dynamics_hard_ce
```

The experiment comments, README link target, operator guide, audit output, and
tests must no longer claim that `trajtok_original` is the active loss.

## Checkpoint Audit

The audit command remains:

```bash
python -m src.smart.inference.audit_text_control "$PRE_BC_CKPT"
```

It constructs the text-enabled runtime, performs the strict state-dict warm
start, freezes the base, and prints:

- checkpoint epoch and global step as informational metadata only;
- every allowed missing text-adapter key and zero unexpected keys;
- total, frozen, and trainable parameter counts and names;
- vocabulary path and SHA-256;
- `history_dynamics_mode: cached_reconstructed`;
- `loss_mode: spatial=False, label_smoothing=0.0`;
- `CFG disabled`.

The audit must stop before distributed training on any contract mismatch.

## Inference and Control Semantics

Single-agent counterfactual inference is unchanged. It retains only the first
11 agent frames, erases hidden future state, resolves one real target agent ID,
encodes one prompt once, and performs only conditional `topk_prob` rollouts.
There is no CFG branch.

Text conditioning remains soft guidance. Instructions such as mandatory
collision, forced lane change, or red-light running are not guaranteed by this
adapter-only stage. AUTO agents can react to the controlled agent through the
ordinary 60 m agent-to-agent attention graph.

## Verification

The change is accepted only when automated tests demonstrate that:

1. the canonical experiment resolves to cached reconstructed history dynamics
   plus hard CE;
2. the launcher contains the exact selected checkpoint and hard-CE task name;
3. the auditor accepts hard CE and rejects spatial smoothing or nonzero label
   smoothing;
4. no-prompt logits and the complete 8-second deterministic rollout remain
   bitwise identical to the frozen base;
5. only text LoRA, projection, and FiLM parameters receive gradients or change;
6. the full repository test suite, source compilation, shell syntax checks,
   and Git whitespace checks pass.

The local repository cannot validate the server-only checkpoint bytes or
DistilBERT cache. Before GPU training, the operator must resolve the Hydra
configuration and run the real checkpoint audit on `/root/workspace/catk`.
