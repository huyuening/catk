# Hard-CE PRE_BC Text-Control Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the canonical frozen-PRE_BC text-control workflow warm-start from the selected cached-history hard-CE checkpoint and fail closed on every incompatible loss configuration.

**Architecture:** Keep the existing ECoSim `tag_only`, frozen CatK, DistilBERT LoRA, projection, six-FiLM, no-CFG architecture unchanged. Replace the canonical experiment and launcher defaults with hard CE, extract a pure checkpoint-metadata validator for unit testing, and update the operator guide so server-side configuration resolution and real-checkpoint audit happen before distributed training.

**Tech Stack:** Python 3.9/3.11, PyTorch, Lightning, Hydra/OmegaConf, YAML, Bash, `unittest`.

## Global Constraints

- Use `/root/workspace/catk/logs/pre_bc_history_dynamics_hard_ce_b200/runs/2026-07-30_21-15-08/checkpoints/last.ckpt` as the canonical default.
- Require `history_dynamics.is_active=true` and effective mode `cached_reconstructed`.
- Require `training_loss.spatial_aware_smoothing=false` and `training_loss.label_smoothing=0.0` in both checkpoint metadata and the active text experiment.
- Keep `agent_vocab_555_s2.pkl`, reconstructed historical dynamics, model architecture, sampling implementation, and Fast WOSAC 2025 unchanged.
- Treat the PRE_BC checkpoint as weights only; never restore optimizer, scheduler, AMP scaler, callbacks, epoch, global step, W&B identity, or checkpoint directory.
- Freeze every existing CatK parameter; train only text LoRA, the text projection, and six FiLM adapters.
- Keep custom inference history-only, conditional-only, and free of CFG or future-derived artifacts.
- Do not add a second TrajTok-default text experiment; the canonical workflow is replaced in place.
- Do not alter the WOMD-to-ECoSim prompt-generation pipeline.
- Preserve unrelated user-owned changes in the main worktree.

---

### Task 1: Switch the canonical experiment and launcher to hard CE

**Files:**
- Modify: `tests/test_text_control_configs.py`
- Modify: `tests/test_text_control_integration.py`
- Modify: `configs/experiment/text_control_pre_bc.yaml`
- Modify: `scripts/train_text_control_pre_bc.sh`

**Interfaces:**
- Consumes: Hydra experiment name `text_control_pre_bc` and environment variable `PRE_BC_CKPT`.
- Produces: one hard-CE text-training configuration and a launcher whose defaults select the exact checkpoint and task name `text_control_pre_bc_history_dynamics_hard_ce`.

- [ ] **Step 1: Change configuration tests to require hard CE and the exact launcher defaults**

In `tests/test_text_control_configs.py`, replace the TrajTok assertions in both the raw-YAML and resolved-Hydra tests with:

```python
self.assertFalse(model["training_loss"]["spatial_aware_smoothing"])
self.assertEqual(model["training_loss"]["label_smoothing"], 0.0)
```

and:

```python
self.assertFalse(model.training_loss.spatial_aware_smoothing)
self.assertEqual(model.training_loss.label_smoothing, 0.0)
```

Extend `test_launch_script_is_syntax_valid_and_never_resumes_wandb` with:

```python
self.assertIn(
    "/root/workspace/catk/logs/pre_bc_history_dynamics_hard_ce_b200/"
    "runs/2026-07-30_21-15-08/checkpoints/last.ckpt",
    text,
)
self.assertIn("text_control_pre_bc_history_dynamics_hard_ce", text)
self.assertNotIn("trajtok_original", text)
```

In `tests/test_text_control_integration.py`, require the experiment loss block to contain `spatial_aware_smoothing=false` and `label_smoothing=0.0`.

- [ ] **Step 2: Run the two test modules and verify RED**

Run:

```bash
python -m unittest \
  tests.test_text_control_configs \
  tests.test_text_control_integration -v
```

Expected: the hard-CE assertions fail because the experiment still enables `trajtok_original`, and the launcher-path assertion fails on the former checkpoint.

- [ ] **Step 3: Make the experiment's loss override explicitly hard CE**

In `configs/experiment/text_control_pre_bc.yaml`, keep the `pre_bc_history_dynamics` base but replace the loss override with:

```yaml
training_loss:
  spatial_aware_smoothing: false
  label_smoothing: 0.0
```

Update the file header to identify the frozen base as cached reconstructed history dynamics plus hard-label cross entropy. Do not change LoRA, FiLM, sampling, WOSAC, batch-size, epoch, data, or W&B settings.

- [ ] **Step 4: Replace the launcher checkpoint and task-name defaults**

In `scripts/train_text_control_pre_bc.sh`, set:

```bash
export PRE_BC_CKPT="${PRE_BC_CKPT:-/root/workspace/catk/logs/pre_bc_history_dynamics_hard_ce_b200/runs/2026-07-30_21-15-08/checkpoints/last.ckpt}"
export MY_TASK_NAME="${MY_TASK_NAME:-text_control_pre_bc_history_dynamics_hard_ce}"
```

Keep `PRE_BC_CKPT` overridable, retain every existing path check, and retain the W&B `unset`, `id=null`, and `resume=never` safeguards.

- [ ] **Step 5: Run configuration and integration tests and verify GREEN**

Run:

```bash
python -m unittest \
  tests.test_text_control_configs \
  tests.test_text_control_integration -v
bash -n scripts/train_text_control_pre_bc.sh
```

Expected: all tests pass except the existing Hydra skip in an environment without Hydra, and Bash syntax exits zero.

- [ ] **Step 6: Resolve the experiment in the Hydra-enabled environment**

Run:

```bash
TEXT_PROMPT_ROOT=/tmp/text-control-tags \
TEXT_MODEL_PATH=/tmp/distilbert \
PRE_BC_CKPT=/tmp/pre-bc.ckpt \
/Users/huyuening/opt/anaconda3/envs/convert_dataset/bin/python \
  -m unittest tests.test_text_control_configs -v
```

Expected: all five configuration tests pass, including the real Hydra composition.

- [ ] **Step 7: Commit the canonical configuration switch**

```bash
git add \
  configs/experiment/text_control_pre_bc.yaml \
  scripts/train_text_control_pre_bc.sh \
  tests/test_text_control_configs.py \
  tests/test_text_control_integration.py
git commit -m "config: use hard-CE PRE-BC for text control"
```

---

### Task 2: Make the checkpoint auditor enforce the hard-CE contract

**Files:**
- Create: `tests/test_text_control_audit.py`
- Modify: `src/smart/inference/audit_text_control.py`

**Interfaces:**
- Consumes: the `hyper_parameters.model_config` object saved in a PRE_BC Lightning checkpoint.
- Produces: `validate_hard_ce_contract(model_config: Any) -> tuple[Any, Any]`, returning the validated history and loss configuration objects or raising `RuntimeError` before model construction.

- [ ] **Step 1: Write pure contract-validator tests**

Create `tests/test_text_control_audit.py`:

```python
import unittest

from src.smart.inference.audit_text_control import validate_hard_ce_contract


def config(*, active=True, mode="cached_reconstructed", spatial=False, smoothing=0.0):
    return {
        "history_dynamics": {"is_active": active, "mode": mode},
        "training_loss": {
            "spatial_aware_smoothing": spatial,
            "label_smoothing": smoothing,
        },
    }


class HardCECheckpointContractTest(unittest.TestCase):
    def test_accepts_cached_history_hard_ce(self):
        history, loss = validate_hard_ce_contract(config())
        self.assertEqual(history["mode"], "cached_reconstructed")
        self.assertEqual(loss["label_smoothing"], 0.0)

    def test_rejects_disabled_or_online_history(self):
        with self.assertRaisesRegex(RuntimeError, "history dynamics disabled"):
            validate_hard_ce_contract(config(active=False))
        with self.assertRaisesRegex(RuntimeError, "cached_reconstructed"):
            validate_hard_ce_contract(config(mode="online_raw"))

    def test_rejects_spatial_smoothing(self):
        with self.assertRaisesRegex(RuntimeError, "spatial smoothing"):
            validate_hard_ce_contract(config(spatial=True))

    def test_rejects_nonzero_label_smoothing(self):
        with self.assertRaisesRegex(RuntimeError, "label_smoothing=0.0"):
            validate_hard_ce_contract(config(smoothing=0.1))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the audit test and verify RED**

Run:

```bash
python -m unittest tests.test_text_control_audit -v
```

Expected: import fails because `validate_hard_ce_contract` does not exist.

- [ ] **Step 3: Implement the pure fail-closed validator**

Add this function to `src/smart/inference/audit_text_control.py` after `_sha256`:

```python
def validate_hard_ce_contract(model_config: Any) -> tuple[Any, Any]:
    history = _get(model_config, "history_dynamics", {})
    loss = _get(model_config, "training_loss", {})
    if not bool(_get(history, "is_active", False)):
        raise RuntimeError("selected PRE_BC checkpoint has history dynamics disabled")
    history_mode = str(_get(history, "mode", "cached_reconstructed"))
    if history_mode != "cached_reconstructed":
        raise RuntimeError(
            "selected PRE_BC checkpoint must use "
            f"history mode cached_reconstructed, got {history_mode!r}"
        )
    if bool(_get(loss, "spatial_aware_smoothing", False)):
        raise RuntimeError(
            "selected PRE_BC checkpoint must disable spatial smoothing"
        )
    try:
        label_smoothing = float(_get(loss, "label_smoothing", 0.0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "selected PRE_BC checkpoint must use label_smoothing=0.0"
        ) from exc
    if label_smoothing != 0.0:
        raise RuntimeError(
            "selected PRE_BC checkpoint must use label_smoothing=0.0, "
            f"got {label_smoothing}"
        )
    return history, loss
```

Replace the existing TrajTok checks in `audit_pre_bc_for_text_control` with:

```python
history, loss = validate_hard_ce_contract(model_config)
```

Change the printed loss summary to:

```python
print(
    "loss_mode: "
    f"spatial={_get(loss, 'spatial_aware_smoothing')}, "
    f"label_smoothing={float(_get(loss, 'label_smoothing', 0.0))}"
)
```

- [ ] **Step 4: Run audit and existing warm-start tests and verify GREEN**

Run:

```bash
python -m unittest \
  tests.test_text_control_audit \
  tests.test_checkpoint_warm_start \
  tests.test_text_control_training -v
python -m src.smart.inference.audit_text_control --help
```

Expected: all tests pass and the CLI help exits zero without constructing a model.

- [ ] **Step 5: Commit the audited hard-CE contract**

```bash
git add src/smart/inference/audit_text_control.py tests/test_text_control_audit.py
git commit -m "fix: audit hard-CE text-control warm starts"
```

---

### Task 3: Update the operator guide and perform final verification

**Files:**
- Modify: `docs/text_control_pre_bc.md`

**Interfaces:**
- Consumes: the canonical hard-CE experiment, launcher, action-tag artifacts, and server paths.
- Produces: a complete operator sequence from text-tag preparation through audit, training, validation monitoring, checkpoint selection, and counterfactual inference.

- [ ] **Step 1: Replace the former base description and checkpoint path**

In `docs/text_control_pre_bc.md`, replace every canonical PRE_BC path with:

```text
/root/workspace/catk/logs/pre_bc_history_dynamics_hard_ce_b200/
runs/2026-07-30_21-15-08/checkpoints/last.ckpt
```

Describe the base as reconstructed historical longitudinal acceleration,
angular speed, and lateral acceleration plus hard-label cross entropy. State
the exact active loss:

```text
spatial_aware_smoothing=false
label_smoothing=0.0
```

- [ ] **Step 2: Update the audit acceptance criteria and training sequence**

Require the audit output to include:

```text
history_dynamics_mode: cached_reconstructed
loss_mode: spatial=False, label_smoothing=0.0
unexpected_keys: 0
CFG disabled
```

Document this exact server sequence:

```bash
cd /root/workspace/catk
source /root/anaconda3/etc/profile.d/conda.sh
conda activate trajtok

git pull --ff-only origin main

export PRE_BC_CKPT=/root/workspace/catk/logs/pre_bc_history_dynamics_hard_ce_b200/runs/2026-07-30_21-15-08/checkpoints/last.ckpt
export CACHE_ROOT=/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact
export TEXT_PROMPT_ROOT=/mnt/pfs/waymo_motion_1_3_0/text_control_tags

python src/run.py experiment=text_control_pre_bc --cfg job --resolve
python -m src.smart.inference.audit_text_control "$PRE_BC_CKPT"
bash scripts/train_text_control_pre_bc.sh
```

State that tag generation may be skipped only when both mapping JSON files and
their referenced train/validation tag JSON files already exist.

- [ ] **Step 3: Scan canonical files for stale default claims**

Run:

```bash
rg -n "pre_bc_history_dynamics_trajtok_original_b200|text_control_pre_bc_history_dynamics_trajtok_original|loss mode.*trajtok_original" \
  configs/experiment/text_control_pre_bc.yaml \
  scripts/train_text_control_pre_bc.sh \
  src/smart/inference/audit_text_control.py \
  docs/text_control_pre_bc.md \
  tests/test_text_control_configs.py \
  tests/test_text_control_integration.py \
  tests/test_text_control_audit.py
```

Expected: no matches.

- [ ] **Step 4: Run the focused text-control suite**

Run:

```bash
python -m unittest \
  tests.test_text_prompts \
  tests.test_build_text_control_tags \
  tests.test_text_prompt_dataset \
  tests.test_text_control_modules \
  tests.test_text_control_decoder \
  tests.test_text_control_training \
  tests.test_checkpoint_warm_start \
  tests.test_text_control_configs \
  tests.test_text_control_audit \
  tests.test_text_control_inference \
  tests.test_text_control_integration -v
```

Expected: all available tests pass; only environment-dependent Hydra coverage may skip in the default Python environment.

- [ ] **Step 5: Run complete repository and static verification**

Run:

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
python -m compileall -q src
bash -n \
  scripts/build_text_control_tags.sh \
  scripts/train_text_control_pre_bc.sh \
  scripts/infer_text_control.sh
python -m src.smart.inference.text_control --help
python -m src.smart.inference.audit_text_control --help
git diff --check
```

Expected: the full suite has zero failures, compilation and shell syntax exit zero, both CLIs display help, and Git reports no whitespace errors.

- [ ] **Step 6: Commit the guide and verification handoff**

```bash
git add docs/text_control_pre_bc.md
git commit -m "docs: describe hard-CE text-control fine-tuning"
```

- [ ] **Step 7: Perform the server-only preflight before allocating eight GPUs**

On `/root/workspace/catk`, run the documented configuration resolution and
audit against the real checkpoint. Do not start `scripts/train_text_control_pre_bc.sh`
unless the audit prints zero unexpected keys, only allowed missing adapter
keys, cached reconstructed history dynamics, hard CE, and `CFG disabled`.

The local implementation is complete without this server-only check, but GPU
training is not authorized by the workflow until that check succeeds.
