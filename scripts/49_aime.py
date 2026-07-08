#!/usr/bin/env python3
"""AIME 2024 (30 problems) — competition math, integer answers 0-999.

A *discriminating* benchmark (unlike saturated GSM8K): frontier models score
high-with-tools, small local models score low. Extract the final integer and
match gold. accuracy = correct / 30.

Usage: 49_aime.py <model_dir> [--thinking sibling|hy3] [--limit N]
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROBLEMS = json.load(open("/tmp/aime.json"))


def extract(resp: str) -> int | None:
    m = re.search(r"\\boxed\{\s*(-?\d+)", resp)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", resp)
    return int(nums[-1]) if nums else None


def main() -> int:
    model_dir = sys.argv[1]
    thinking = sys.argv[sys.argv.index("--thinking") + 1] if "--thinking" in sys.argv else "hy3"
    reason = "--reason" in sys.argv  # thinking ON (fair vs frontier AIME)
    if reason:
        tkw = {"enable_thinking": True} if thinking == "sibling" else {"reasoning_effort": "high"}
        max_tok, stem = 8192, "aime_think"
    else:
        tkw = {"enable_thinking": False} if thinking == "sibling" else {"reasoning_effort": "no_think"}
        max_tok, stem = 1024, "aime"
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(PROBLEMS)
    tag = Path(model_dir).name

    from mlx_lm import load, generate
    print(f"[aime] loading {tag} (reason={reason})", flush=True)
    model, tok = load(model_dir)
    probs = PROBLEMS[:limit]
    correct = 0
    t0 = time.time()
    for i, p in enumerate(probs):
        instr = (f"{p['Problem']}\n\nThink step by step and give the final "
                 "integer answer as \\boxed{N}.")
        text = tok.apply_chat_template([{"role": "user", "content": instr}],
                                       tokenize=False, add_generation_prompt=True, **tkw)
        resp = generate(model, tok, prompt=text, max_tokens=max_tok, verbose=False)
        got = extract(resp)
        ok = got is not None and got == int(p["Answer"])
        correct += ok
        print(f"[aime] {i+1}/{len(probs)} | {correct}/{i+1} correct", flush=True)
    out = {"model": tag, "n": len(probs), "correct": correct,
           "accuracy_pct": round(100 * correct / len(probs), 1),
           "reasoning": reason, "minutes": round((time.time() - t0) / 60, 1)}
    (REPO / f"eval/receipts/{stem}_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"[aime] {tag} (reason={reason}): accuracy = {out['accuracy_pct']}% "
          f"({correct}/{len(probs)}) in {out['minutes']} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
