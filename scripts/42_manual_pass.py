#!/usr/bin/env python3
"""Manual quality pass (#16), run by Claude across the daily-use models.

Six real daily-driver / daily-agent tasks — not pass/fail unit tests but "is
this pleasant and capable to actually use." Runs one model at a time (never two
resident — the OOM lesson), no_think production mode, saves outputs for a
qualitative verdict.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS = [
    ("code_cli",
     "Write a small Python CLI (argparse) that reads a CSV file and prints, for "
     "each numeric column, its count / mean / min / max. Handle non-numeric "
     "columns gracefully."),
    ("review_bug",
     "Review this for correctness and style, and fix anything wrong:\n"
     "def average(nums):\n    total = 0\n    for n in nums:\n        total += n\n"
     "    return total / len(nums)"),
    ("agent_plan",
     "I have a Flask JSON API and need to add per-IP rate limiting (100 req/min) "
     "without external services. Give me a concrete plan and the core code."),
    ("reasoning",
     "Two trains: one leaves A at 60 mph, another leaves B (180 miles away) at "
     "40 mph, heading toward each other. When and where do they meet? Show the "
     "work."),
    ("explain",
     "Explain Python async/await to someone who already understands threads, in "
     "one tight paragraph. No fluff."),
    ("tool_json",
     "Emit ONLY a JSON tool call (no prose) to search users named 'Kim' created "
     "after 2020-01-01, for a function search_users(name: str, after_date: str)."),
]


def run(tag: str, path: str, thinking_kwarg: dict) -> None:
    from mlx_lm import load, generate
    print(f"[mp {time.strftime('%H:%M:%S')}] loading {tag}", flush=True)
    model, tok = load(path)
    out = []
    for pid, prompt in TASKS:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       **thinking_kwarg)
        t0 = time.time()
        resp = generate(model, tok, prompt=text, max_tokens=600, verbose=False)
        out.append({"id": pid, "output": resp, "secs": round(time.time() - t0, 1)})
        print(f"[mp {time.strftime('%H:%M:%S')}] {tag}/{pid} ({len(resp)} chars)", flush=True)
    (REPO / f"eval/receipts/manual_pass_{tag}.json").write_text(json.dumps(out, indent=1))
    del model, tok
    import mlx.core as mx
    mx.clear_cache()


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "reap25":
        run("reap25", str(REPO / "dist/hy3-demolition-mlx-reap25-v1-fused"),
            {"reasoning_effort": "no_think"})
    elif which == "sibling":
        run("sibling", str(REPO / "dist/hy3-family-mini-qwen35b-v1"),
            {"enable_thinking": False})
