#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATK_ROOT="${CATK_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
: "${RECON_OUTPUT:?Set RECON_OUTPUT to the batch reconstruction output directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
VOCAB_FILE="${VOCAB_FILE:-${CATK_ROOT}/src/smart/tokens/agent_vocab_555_s2.pkl}"
LOOKUP_FILE="${LOOKUP_FILE:-${RECON_OUTPUT}/agent_transition_dynamics_original_vocab_reconstructed.pt}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SHRINKAGE_COUNT="${SHRINKAGE_COUNT:-8.0}"

BUILD_COMMAND=(
  "${PYTHON_BIN}"
  -m src.smart.tokens.build_transition_dynamics
  --assignment-training-dir "${RECON_OUTPUT}/datasets/original/training"
  --dynamics-training-dir "${RECON_OUTPUT}/datasets/reconstructed/training"
  --agent-token-file "${VOCAB_FILE}"
  --output "${LOOKUP_FILE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --shrinkage-count "${SHRINKAGE_COUNT}"
)
if [[ -n "${MAX_SCENARIOS:-}" ]]; then
  BUILD_COMMAND+=(--max-scenarios "${MAX_SCENARIOS}")
fi

cd "${CATK_ROOT}"
"${BUILD_COMMAND[@]}"
