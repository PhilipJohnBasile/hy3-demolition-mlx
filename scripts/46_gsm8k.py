#!/usr/bin/env python3
"""GSM8K — grade-school math word problems, the standard math benchmark.

Chain-of-thought prompt; extract the final number from the model's answer and
compare to gold (the number after '####'). accuracy = correct / N.

Usage: 46_gsm8k.py <model_dir> [--thinking sibling|hy3] [--limit N]
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROBLEMS = json.load(open("/tmp/gsm8k.json"))


def gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def extract(resp: str) -> str | None:
    m = re.findall(r"-?\$?\d[\d,]*\.?\d*", resp.replace(",", ""))
    return m[-1].replace("$", "").rstrip(".") if m else None


def main() -> int:
    model_dir = sys.argv[1]
    thinking = sys.argv[sys.argv.index("--thinking") + 1] if "--thinking" in sys.argv else "hy3"
    tkw = {"enable_thinking": False} if thinking == "sibling" else {"reasoning_effort": "no_think"}
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(PROBLEMS)
    tag = Path(model_dir).name

    from mlx_lm import load, generate
    print(f"[gsm8k] loading {tag}", flush=True)
    model, tok = load(model_dir)
    probs = PROBLEMS[:limit]
    correct = 0
    t0 = time.time()
    for i, p in enumerate(probs):
        instr = (f"{p['question']}\n\nSolve step by step, then end with a line "
                 "exactly like 'The answer is <number>.'")
        text = tok.apply_chat_template([{"role": "user", "content": instr}],
                                       tokenize=False, add_generation_prompt=True, **tkw)
        resp = generate(model, tok, prompt=text, max_tokens=512, verbose=False)
        g = gold(p["answer"])
        got = extract(resp)
        try:
            ok = got is not None and abs(float(got) - float(g)) < 1e-4
        except ValueError:
            ok = (got == g)
        correct += ok
        if (i + 1) % 25 == 0 or i + 1 == len(probs):
            print(f"[gsm8k] {i+1}/{len(probs)} | acc {correct}/{i+1} = "
                  f"{100*correct/(i+1):.1f}%", flush=True)
    out = {"model": tag, "n": len(probs), "correct": correct,
           "accuracy_pct": round(100 * correct / len(probs), 1),
           "minutes": round((time.time() - t0) / 60, 1)}
    (REPO / f"eval/receipts/gsm8k_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"[gsm8k] {tag}: accuracy = {out['accuracy_pct']}% "
          f"({correct}/{len(probs)}) in {out['minutes']} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
