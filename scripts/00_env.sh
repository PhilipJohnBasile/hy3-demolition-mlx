#!/usr/bin/env bash
set -euo pipefail

# BASH_SOURCE is bash-only; zsh sets $0 to the sourced file instead.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

export HY3_PROJECT_ROOT="${HY3_PROJECT_ROOT:-$ROOT}"
export HY3_MLX_BASE="${HY3_MLX_BASE:-ox-ox/Hy3-295B-Instruct-w2q3exp-AProjQ8-SExpQ8-OutQ8-MTP-mlx}"
export HY3_MODEL_DIR="${HY3_MODEL_DIR:-$ROOT/models/hy3-mlx-base}"
export HY3_LITE_FUSED="${HY3_LITE_FUSED:-$ROOT/dist/hy3-demolition-mlx-lite-v1-fused}"

if [ -d "$ROOT/.venv/bin" ]; then
  export PATH="$ROOT/.venv/bin:$PATH"
  export HY3_PYTHON="${HY3_PYTHON:-$ROOT/.venv/bin/python}"
else
  export HY3_PYTHON="${HY3_PYTHON:-python3}"
fi

# Build-time-only teachers/gates. All three are private internal repos (not
# published), so these default paths will not exist outside the author's
# machine. They are not needed to run the fused model.
export AGENT_TOOLKIT_PATH="${AGENT_TOOLKIT_PATH:-/Users/pjb/git/agent-toolkit}"
export AGENT_BRAIN_BLUEPRINT_PATH="${AGENT_BRAIN_BLUEPRINT_PATH:-/Users/pjb/git/agent-brain-blueprint}"
export TINYGPT_SOULS_PATH="${TINYGPT_SOULS_PATH:-/Users/pjb/git/tinygpt-souls}"

export PYTHONPATH="$ROOT/src:$AGENT_TOOLKIT_PATH/verify:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

echo "HY3_PROJECT_ROOT=$HY3_PROJECT_ROOT"
echo "HY3_MLX_BASE=$HY3_MLX_BASE"
echo "HY3_MODEL_DIR=$HY3_MODEL_DIR"
echo "HY3_LITE_FUSED=$HY3_LITE_FUSED"
echo "HY3_PYTHON=$HY3_PYTHON"
echo "AGENT_TOOLKIT_PATH=$AGENT_TOOLKIT_PATH"
echo "AGENT_BRAIN_BLUEPRINT_PATH=$AGENT_BRAIN_BLUEPRINT_PATH"
echo "TINYGPT_SOULS_PATH=$TINYGPT_SOULS_PATH"
