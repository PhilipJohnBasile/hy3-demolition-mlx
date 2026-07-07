#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/00_env.sh"

MODEL="${1:-$HY3_MODEL_DIR}"
PROMPT="${2:-Return exactly: ready}"
MAX_TOKENS="${MAX_TOKENS:-8}"
NUM_DRAFT_TOKENS="${NUM_DRAFT_TOKENS:-0}"

"$HY3_PYTHON" -m mlx_lm generate \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --max-tokens "$MAX_TOKENS" \
  --temp 0 \
  --top-p 1 \
  --top-k 0 \
  --num-draft-tokens "$NUM_DRAFT_TOKENS" \
  --chat-template-config '{"reasoning_effort":"no_think"}'
