#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$ROOT/scripts/00_env.sh"

MODEL="${1:-$HY3_MLX_BASE}"
TARGET="${2:-$HY3_MODEL_DIR}"

mkdir -p "$TARGET"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-4}"

if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL" --local-dir "$TARGET"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$MODEL" --local-dir "$TARGET"
else
  echo "Install huggingface_hub or run: pip install huggingface_hub" >&2
  exit 1
fi

echo "Downloaded $MODEL to $TARGET"
