#!/usr/bin/env python3
"""GPQA Diamond (198) — graduate-level science, the open-ended \\boxed variant.

Discriminating (frontier ~90%; near-impossible for small models). Extract the
model's \\boxed{} answer and match the gold boxed answer (normalized, lenient).
NOTE: this is the open-ended mirror, not the original 4-way multiple choice, so
grading is approximate (units/format vary). accuracy = matched / N.

Usage: 50_gpqa.py <model_dir> [--thinking sibling|hy3] [--limit N]
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROBLEMS = json.load(open("/tmp/gpqa.json"))


def boxed(s: str) -> str | None:
    m = re.search(r"\\boxed\{(.+?)\}", s, re.S)
    return m.group(1) if m else None


def norm(s: str) -> str:
    s = s.lower().replace(" ", "").replace("$", "").replace("\\", "")
    s = re.sub(r"[{}()\[\]]", "", s)
    return s.replace("times10^", "e").replace("*10^", "e").replace("^", "")


def main() -> int:
    model_dir = sys.argv[1]
    thinking = sys.argv[sys.argv.index("--thinking") + 1] if "--thinking" in sys.argv else "hy3"
    tkw = {"enable_thinking": False} if thinking == "sibling" else {"reasoning_effort": "no_think"}
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(PROBLEMS)
    tag = Path(model_dir).name

    from mlx_lm import load, generate
    print(f"[gpqa] loading {tag}", flush=True)
    model, tok = load(model_dir)
    probs = PROBLEMS[:limit]
    correct = 0
    t0 = time.time()
    for i, p in enumerate(probs):
        instr = (f"{p['problem']}\n\nReason step by step and give the final "
                 "answer as \\boxed{...}.")
        text = tok.apply_chat_template([{"role": "user", "content": instr}],
                                       tokenize=False, add_generation_prompt=True, **tkw)
        resp = generate(model, tok, prompt=text, max_tokens=1024, verbose=False)
        got, gold = boxed(resp), boxed(p["solution"])
        ok = bool(got and gold and norm(got) == norm(gold))
        correct += ok
        if (i + 1) % 20 == 0 or i + 1 == len(probs):
            print(f"[gpqa] {i+1}/{len(probs)} | {correct}/{i+1} = "
                  f"{100*correct/(i+1):.1f}%", flush=True)
    out = {"model": tag, "n": len(probs), "correct": correct,
           "accuracy_pct": round(100 * correct / len(probs), 1),
           "minutes": round((time.time() - t0) / 60, 1),
           "note": "open-ended boxed variant, lenient match — approximate"}
    (REPO / f"eval/receipts/gpqa_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"[gpqa] {tag}: accuracy = {out['accuracy_pct']}% "
          f"({correct}/{len(probs)}) in {out['minutes']} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
