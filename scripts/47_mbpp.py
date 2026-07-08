#!/usr/bin/env python3
"""MBPP (sanitized, 257) — Mostly Basic Python Problems, standard code benchmark.

Prompt with the task text + the first assertion (signature hint), generate a
function, run it against all test_list assertions in a sandboxed subprocess.
pass@1 = passed / N.

Usage: 47_mbpp.py <model_dir> [--thinking sibling|hy3] [--limit N]
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
PROBLEMS = json.load(open("/tmp/mbpp.json"))


def extract_code(resp: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", resp, re.S)
    if blocks:
        return max(blocks, key=len)
    return resp


def run_one(code: str, imports: list, tests: list) -> bool:
    program = "\n".join(imports) + "\n" + code + "\n" + "\n".join(tests) + "\n"
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
    thinking = sys.argv[sys.argv.index("--thinking") + 1] if "--thinking" in sys.argv else "hy3"
    tkw = {"enable_thinking": False} if thinking == "sibling" else {"reasoning_effort": "no_think"}
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(PROBLEMS)
    tag = Path(model_dir).name

    from mlx_lm import load, generate
    print(f"[mbpp] loading {tag}", flush=True)
    model, tok = load(model_dir)
    probs = PROBLEMS[:limit]
    passed = 0
    t0 = time.time()
    for i, p in enumerate(probs):
        instr = (f"{p['prompt']}\n\nYour function must satisfy this test:\n"
                 f"{p['test_list'][0]}\n\nReply with ONLY the function in a "
                 "```python code block.")
        text = tok.apply_chat_template([{"role": "user", "content": instr}],
                                       tokenize=False, add_generation_prompt=True, **tkw)
        resp = generate(model, tok, prompt=text, max_tokens=512, verbose=False)
        code = extract_code(resp)
        ok = run_one(code, p.get("test_imports") or [], p["test_list"])
        passed += ok
        if (i + 1) % 25 == 0 or i + 1 == len(probs):
            print(f"[mbpp] {i+1}/{len(probs)} | pass@1 {passed}/{i+1} = "
                  f"{100*passed/(i+1):.1f}%", flush=True)
    out = {"model": tag, "n": len(probs), "passed": passed,
           "pass_at_1_pct": round(100 * passed / len(probs), 1),
           "minutes": round((time.time() - t0) / 60, 1)}
    (REPO / f"eval/receipts/mbpp_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"[mbpp] {tag}: pass@1 = {out['pass_at_1_pct']}% "
          f"({passed}/{len(probs)}) in {out['minutes']} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
