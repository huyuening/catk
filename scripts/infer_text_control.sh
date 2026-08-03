#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 || $# > 7 )); then
  echo "Usage: $0 CHECKPOINT SCENARIO_PICKLE TARGET_AGENT_ID PROMPT OUTPUT_DIR [N_ROLLOUTS] [SEED]" >&2
  exit 2
fi

CHECKPOINT="$1"
SCENARIO_PICKLE="$2"
TARGET_AGENT_ID="$3"
PROMPT="$4"
OUTPUT_DIR="$5"
N_ROLLOUTS="${6:-32}"
SEED="${7:-0}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${SCENARIO_PICKLE}" ]]; then
  echo "Scenario pickle not found: ${SCENARIO_PICKLE}" >&2
  exit 1
fi
if [[ -z "${PROMPT//[[:space:]]/}" ]]; then
  echo "Prompt must be non-empty." >&2
  exit 1
fi

cd /root/workspace/catk
source /root/anaconda3/etc/profile.d/conda.sh
conda activate trajtok

python -m src.smart.inference.text_control \
  "${CHECKPOINT}" \
  "${SCENARIO_PICKLE}" \
  "${TARGET_AGENT_ID}" \
  "${PROMPT}" \
  "${OUTPUT_DIR}" \
  "${N_ROLLOUTS}" \
  "${SEED}"
