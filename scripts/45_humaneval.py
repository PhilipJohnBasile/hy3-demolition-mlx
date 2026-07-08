#!/usr/bin/env python3
"""HumanEval pass@1 — the standard code benchmark AI sites report.

For each of the 164 problems: prompt the model to complete the function,
extract the code, run it against the problem's unit test in a sandboxed
subprocess (timeout), and count passes. pass@1 = passes / 164.

Usage: 45_humaneval.py <model_dir> [--thinking sibling|hy3] [--limit N]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROBLEMS = json.load(open("/tmp/humaneval.json"))


def extract_code(resp: str, prompt: str, entry: str) -> str:
    """Get a runnable module defining `entry` from the model output."""
    blocks = re.findall(r"```(?:python)?\n(.*?)```", resp, re.S)
    for b in blocks:
        if f"def {entry}" in b:
            return b
    if blocks:
        return blocks[0]
    if f"def {entry}" in resp:  # raw, no fence
        return resp
    return prompt + resp  # model may have only emitted the body


def run_one(code: str, test: str, entry: str) -> bool:
    program = f"{code}\n\n{test}\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=10, text=True)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> int:
    model_dir = sys.argv[1]
    thinking = "sibling" if "--thinking" in sys.argv and sys.argv[sys.argv.index("--thinking") + 1] == "sibling" else "hy3"
    tkw = {"enable_thinking": False} if thinking == "sibling" else {"reasoning_effort": "no_think"}
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(PROBLEMS)
    tag = Path(model_dir).name

    from mlx_lm import load, generate
    print(f"[humaneval] loading {tag}", flush=True)
    model, tok = load(model_dir)
    probs = PROBLEMS[:limit]
    passed, results = 0, []
    t0 = time.time()
    for i, p in enumerate(probs):
        instr = ("Complete this Python function. Reply with ONLY the complete "
                 "function in a ```python code block, no explanation:\n\n```python\n"
                 f"{p['prompt']}```")
        text = tok.apply_chat_template([{"role": "user", "content": instr}],
                                       tokenize=False, add_generation_prompt=True, **tkw)
        resp = generate(model, tok, prompt=text, max_tokens=640, verbose=False)
        code = extract_code(resp, p["prompt"], p["entry_point"])
        ok = run_one(code, p["test"], p["entry_point"])
        passed += ok
        results.append({"task_id": p["task_id"], "passed": ok})
        if (i + 1) % 10 == 0 or i + 1 == len(probs):
            print(f"[humaneval] {i+1}/{len(probs)} | pass@1 so far {passed}/{i+1} "
                  f"= {100*passed/(i+1):.1f}%", flush=True)
    dt = time.time() - t0
    out = {"model": tag, "n": len(probs), "passed": passed,
           "pass_at_1_pct": round(100 * passed / len(probs), 1),
           "minutes": round(dt / 60, 1)}
    (REPO / f"eval/receipts/humaneval_{tag}.json").write_text(
        json.dumps({**out, "per_task": results}, indent=1))
    print(f"[humaneval] {tag}: pass@1 = {out['pass_at_1_pct']}% "
          f"({passed}/{len(probs)}) in {out['minutes']} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
