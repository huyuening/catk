#!/usr/bin/env bash
set -euo pipefail

: "${WOMD_ROOT:?Set WOMD_ROOT to the raw WOMD scenario root}"

LABEL_OUTPUT_ROOT="${LABEL_OUTPUT_ROOT:-${WOMD_ROOT%/}/catk_labels}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SPLITS="${SPLITS:-training validation testing}"
STAGES="${STAGES:-annotations statistics scenario-visualizations aggregate-visualization}"
VISUALIZE_MAX_SCENARIOS="${VISUALIZE_MAX_SCENARIOS:-100}"
PYTHON_BIN="${PYTHON_BIN:-python}"

read -r -a split_args <<< "$SPLITS"
read -r -a stage_args <<< "$STAGES"

extra_args=()
if [[ "${OVERWRITE:-false}" == "true" ]]; then
  extra_args+=(--overwrite)
fi
if [[ "${RESUME:-true}" == "false" ]]; then
  extra_args+=(--no-resume)
fi

"$PYTHON_BIN" -m src.womd_labeling.run_dataset \
  --input-root "$WOMD_ROOT" \
  --output-root "$LABEL_OUTPUT_ROOT" \
  --splits "${split_args[@]}" \
  --stages "${stage_args[@]}" \
  --workers "$NUM_WORKERS" \
  --visualize-max-scenarios "$VISUALIZE_MAX_SCENARIOS" \
  "${extra_args[@]}"
