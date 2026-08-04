#!/usr/bin/env bash
set -euo pipefail

ACTION_ROWS="${ACTION_ROWS:?Set ACTION_ROWS to a comma-separated list of action CSV.GZ files}"
TEXT_PROMPT_ROOT="${TEXT_PROMPT_ROOT:?Set TEXT_PROMPT_ROOT}"
TEXT_SPLIT="${TEXT_SPLIT:-train}"
TEXT_MAPPING_PATH="${TEXT_MAPPING_PATH:?Set TEXT_MAPPING_PATH}"
TAG_WORKERS="${TAG_WORKERS:-80}"
TAG_PROGRESS_EVERY="${TAG_PROGRESS_EVERY:-1000}"

IFS=',' read -r -a ACTION_INPUTS <<< "${ACTION_ROWS}"
python -m src.smart.datasets.build_text_control_tags \
  --input "${ACTION_INPUTS[@]}" \
  --output-root "${TEXT_PROMPT_ROOT}" \
  --split "${TEXT_SPLIT}" \
  --mapping-output "${TEXT_MAPPING_PATH}" \
  --workers "${TAG_WORKERS}" \
  --progress-every "${TAG_PROGRESS_EVERY}"
