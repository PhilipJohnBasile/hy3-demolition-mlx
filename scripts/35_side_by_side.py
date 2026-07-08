#!/usr/bin/env python3
"""reap25 vs lite-v1 on the same real prompts, for PJB's #16 manual-pass call.

ONE model per invocation (ARCHITECTURE #3: never two resident models — the
first attempt jetsam-died loading the second in-process). Persists each
model's outputs immediately so a crash never loses completed work.

Usage:
  35_side_by_side.py --model-tag reap25   # runs one model, writes its JSON
  35_side_by_side.py --model-tag lite-v1
  35_side_by_side.py --merge              # merges both JSONs into the .md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATHS = {
    "reap25": str(REPO / "dist" / "hy3-demolition-mlx-reap25-v1-fused"),
    "lite-v1": str(REPO / "dist" / "hy3-demolition-mlx-lite-v1-fused"),
}
PROMPTS = [
    ("code_ratelimiter",
     "Implement a thread-safe sliding-window rate limiter class in Python: "
     "allow(key) returns True if fewer than N calls happened for that key in "
     "the last window_seconds, else False. No external deps."),
    ("planning_oauth",
     "I have a working Flask app with session login. Give me a concrete, "
     "ordered plan to add 'Sign in with Google' (OAuth2) alongside it — steps, "
     "the libraries, and the two biggest correctness pitfalls."),
    ("repair_offbyone",
     "This returns the wrong result for the last window. Fix it and say what "
     "was wrong:\n\ndef rolling_max(xs, k):\n    out = []\n    for i in range(len(xs)-k):\n"
     "        out.append(max(xs[i:i+k]))\n    return out"),
    ("soul_music",
     "In 4 sentences, explain why a V7 chord creates tension that resolves to I, "
     "using the tritone. Be precise, not flowery."),
]
MAXTOK = 512


def run_one(tag: str) -> None:
    from mlx_lm import generate, load
    print(f"[sbs {time.strftime('%H:%M:%S')}] loading {tag}", flush=True)
    model, tok = load(PATHS[tag])
    out = []
    dst = REPO / "dist" / f"sbs_{tag}.json"
    for pid, prompt in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        resp = generate(model, tok, prompt=text, max_tokens=MAXTOK, verbose=False)
        out.append({"id": pid, "output": resp, "secs": round(time.time() - t0, 1),
                    "stopped_clean": bool(resp.strip()) and not resp.rstrip().endswith("!")})
        dst.write_text(json.dumps(out, indent=2))  # persist after EACH prompt
        print(f"[sbs {time.strftime('%H:%M:%S')}] {tag}/{pid} done ({len(resp)} chars)", flush=True)
    print(f"[sbs] wrote {dst}", flush=True)


def merge() -> None:
    res = {t: json.loads((REPO / "dist" / f"sbs_{t}.json").read_text()) for t in PATHS}
    md = ["# reap25 vs lite-v1 — manual-pass material (#16)\n",
          "Same prompts, both models, one process each. Judge quality yourself. "
          f"Generated {time.strftime('%Y-%m-%d %H:%M')}.\n"]
    for i, (pid, prompt) in enumerate(PROMPTS):
        md.append(f"\n## {pid}\n\n**Prompt:** {prompt}\n")
        for tag in ("reap25", "lite-v1"):
            r = res[tag][i]
            md.append(f"\n### {tag} ({r['secs']}s, clean_stop={r['stopped_clean']})\n\n"
                      f"```\n{r['output'].strip()}\n```\n")
    (REPO / "dist" / "reap25_vs_lite_side_by_side.md").write_text("\n".join(md))
    print("[sbs] wrote dist/reap25_vs_lite_side_by_side.md", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", choices=list(PATHS))
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    elif a.model_tag:
        run_one(a.model_tag)
    else:
        ap.error("pass --model-tag or --merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
