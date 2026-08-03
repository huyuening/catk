#!/usr/bin/env bash
set -euo pipefail

cd /root/workspace/catk

source /root/anaconda3/etc/profile.d/conda.sh
conda activate trajtok

export PRE_BC_CKPT="${PRE_BC_CKPT:-/root/workspace/catk/logs/pre_bc_history_dynamics_trajtok_original_b200/runs/2026-07-27_19-49-13/checkpoints/last.ckpt}"
export CACHE_ROOT="${CACHE_ROOT:-/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario_history_dynamics_exact}"
export TEXT_PROMPT_ROOT="${TEXT_PROMPT_ROOT:-/mnt/pfs/waymo_motion_1_3_0/text_control_tags}"
export TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-distilbert-base-uncased}"
export FAST_WOSAC_GT_DIR="${FAST_WOSAC_GT_DIR:-/mnt/pfs/waymo_motion_1_3_0/preprocessed_scenario/validation_gt}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MY_EXPERIMENT=text_control_pre_bc
export MY_TASK_NAME="${MY_TASK_NAME:-text_control_pre_bc_history_dynamics_trajtok_original}"
export WANDB_OFFLINE="${WANDB_OFFLINE:-false}"
export WANDB_ENTITY="${WANDB_ENTITY:-huyuening911-beijing-jiaotong-university}"

# The PRE-BC checkpoint contributes model tensors only. Start a distinct run.
unset WANDB_RUN_ID WANDB_RESUME

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
}

require_directory() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Required directory not found: ${path}" >&2
    exit 1
  fi
}

require_file "${PRE_BC_CKPT}"
require_directory "${CACHE_ROOT}/training"
require_directory "${CACHE_ROOT}/validation"
require_file "${TEXT_PROMPT_ROOT}/train_scenario_mapping.json"
require_file "${TEXT_PROMPT_ROOT}/val_scenario_mapping.json"

bash scripts/train.sh \
  ckpt_path="${PRE_BC_CKPT}" \
  logger.wandb.id=null \
  logger.wandb.resume=never \
  "$@"
