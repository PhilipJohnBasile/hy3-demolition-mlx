#!/usr/bin/env python3
"""Generate reap25 vs lite-v1 outputs on the same real prompts, side by side,
so PJB can make the #16 manual-pass judgment on concrete material.

Loads one model, runs all prompts, unloads; then the other (never both
resident). Writes a readable markdown diff. Promotes nothing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load

REPO = Path(__file__).resolve().parents[1]
MODELS = {
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


def run_model(tag: str, path: str) -> list[dict]:
    print(f"[sbs {time.strftime('%H:%M:%S')}] loading {tag}", flush=True)
    model, tok = load(path)
    out = []
    for pid, prompt in PROMPTS:
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t0 = time.time()
        resp = generate(model, tok, prompt=text, max_tokens=MAXTOK, verbose=False)
        out.append({"id": pid, "output": resp, "secs": round(time.time() - t0, 1),
                    "stopped_clean": not resp.rstrip().endswith("!")})
        print(f"[sbs {time.strftime('%H:%M:%S')}] {tag}/{pid} done ({len(resp)} chars)", flush=True)
    del model, tok
    mx.clear_cache()
    return out


def main() -> int:
    results = {tag: run_model(tag, path) for tag, path in MODELS.items()}
    # readable side-by-side markdown
    md = ["# reap25 vs lite-v1 — manual-pass material\n",
          "Same prompts, both models. Judge quality yourself (#16). "
          f"Generated {time.strftime('%Y-%m-%d %H:%M')}.\n"]
    for i, (pid, prompt) in enumerate(PROMPTS):
        md.append(f"\n## {pid}\n\n**Prompt:** {prompt}\n")
        for tag in ("reap25", "lite-v1"):
            r = results[tag][i]
            md.append(f"\n### {tag}  ({r['secs']}s, clean_stop={r['stopped_clean']})\n\n"
                      f"```\n{r['output'].strip()}\n```\n")
    (REPO / "dist" / "reap25_vs_lite_side_by_side.md").write_text("\n".join(md))
    (REPO / "dist" / "reap25_vs_lite_side_by_side.json").write_text(
        json.dumps(results, indent=2))
    print("[sbs] wrote dist/reap25_vs_lite_side_by_side.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
