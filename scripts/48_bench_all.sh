#!/bin/bash
# Master benchmark orchestrator across our models.
# Sequential (one GPU job at a time). Idempotent — skips any receipt that
# already exists. Order: discriminating benches first (AIME/GPQA), then the
# saturated ones (GSM8K), then slow reap25 last.
set -u
cd /Users/pjb/git/Hy3
PY=.venv/bin/python
SIB=dist/hy3-family-mini-qwen35b-v1
QWEN=models/qwen35b-a3b-base
REAP=dist/hy3-demolition-mlx-reap25-v1-fused

run() {  # receipt_stem  script  model  thinking  [limit_args...]
  local stem=$1 script=$2 model=$3 think=$4; shift 4
  if [ -f "eval/receipts/${stem}.json" ]; then echo "[bench] SKIP ${stem} (done)"; return; fi
  echo "[bench] RUN ${stem}"
  $PY "scripts/${script}" "$model" --thinking "$think" "$@"
}

while pgrep -f "4[5679]_.*\.py|46_gsm8k|50_gpqa" | grep -qv $$ 2>/dev/null; do sleep 5; done

# --- fast models: DISCRIMINATING first (AIME, GPQA), then saturated (GSM8K) ---
run aime_hy3-family-mini-qwen35b-v1    49_aime.py   $SIB  sibling
run aime_qwen35b-a3b-base              49_aime.py   $QWEN sibling
run aime_think_hy3-family-mini-qwen35b-v1  49_aime.py  $SIB  sibling --reason
run aime_think_qwen35b-a3b-base            49_aime.py  $QWEN sibling --reason
run gpqa_hy3-family-mini-qwen35b-v1    50_gpqa.py   $SIB  sibling
run gpqa_qwen35b-a3b-base              50_gpqa.py   $QWEN sibling
run gsm8k_hy3-family-mini-qwen35b-v1   46_gsm8k.py  $SIB  sibling
run gsm8k_qwen35b-a3b-base             46_gsm8k.py  $QWEN sibling

# --- reap25 (Hy3, slow): full small benches, subset the big ones ---
run humaneval_hy3-demolition-mlx-reap25-v1-fused  45_humaneval.py  $REAP hy3
run aime_hy3-demolition-mlx-reap25-v1-fused        49_aime.py       $REAP hy3
run mbpp_hy3-demolition-mlx-reap25-v1-fused        47_mbpp.py       $REAP hy3 --limit 150
run gpqa_hy3-demolition-mlx-reap25-v1-fused        50_gpqa.py       $REAP hy3 --limit 80
run gsm8k_hy3-demolition-mlx-reap25-v1-fused       46_gsm8k.py      $REAP hy3 --limit 150

echo "[bench] ALL DONE"
