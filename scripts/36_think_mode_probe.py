#!/usr/bin/env python3
"""Prove the 'think leak' is a serving-mode choice, not a model defect.

Loads reap25 once, runs the same prompts under reasoning_effort='high'
(default) vs 'no_think', and also shows a strip-through-</think> post-process
on the 'high' output. Writes a readable receipt.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = str(REPO / "dist" / "hy3-demolition-mlx-reap25-v1-fused")
THINK_END = "</think:opensource>"
PROMPTS = [
    ("soul_music", "In 4 sentences, explain why a V7 chord creates tension that "
     "resolves to I, using the tritone. Be precise, not flowery."),
    ("code_double", "Write a Python function that returns the two largest distinct "
     "values in a list, or None if there are fewer than two distinct values."),
]


def strip_think(text: str) -> str:
    # show only what follows the think close tag (the actual answer)
    if THINK_END in text:
        return text.split(THINK_END, 1)[1].lstrip()
    return text  # never closed -> pure reasoning, nothing to show


def main() -> int:
    from mlx_lm import generate, load
    print(f"[probe {time.strftime('%H:%M:%S')}] loading reap25", flush=True)
    model, tok = load(MODEL)
    out = {}
    for pid, prompt in PROMPTS:
        out[pid] = {"prompt": prompt}
        for effort in ("high", "no_think"):
            text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, reasoning_effort=effort)
            t0 = time.time()
            resp = generate(model, tok, prompt=text, max_tokens=400, verbose=False)
            out[pid][effort] = {
                "raw": resp,
                "answer_after_strip": strip_think(resp),
                "closed_think": THINK_END in resp,
                "secs": round(time.time() - t0, 1),
            }
            print(f"[probe {time.strftime('%H:%M:%S')}] {pid}/{effort} done "
                  f"(closed_think={THINK_END in resp}, {len(resp)} chars)", flush=True)
    (REPO / "eval/receipts/hy3_think_mode_probe.json").write_text(json.dumps(out, indent=2))
    # readable markdown
    md = ["# Think-mode probe — reap25\n",
          "The 'think leak' is a serving choice: `high` reasons first (and can run "
          "out of tokens mid-thought on long tasks); `no_think` answers directly. "
          f"Generated {time.strftime('%Y-%m-%d %H:%M')}.\n"]
    for pid, prompt in PROMPTS:
        md.append(f"\n## {pid}\n\n**Prompt:** {prompt}\n")
        for effort in ("high", "no_think"):
            d = out[pid][effort]
            shown = d["answer_after_strip"] if d["closed_think"] else d["raw"]
            tag = "answer (after </think> strip)" if d["closed_think"] else "RAW — think never closed"
            md.append(f"\n### reasoning_effort={effort} ({d['secs']}s, closed_think={d['closed_think']})\n\n"
                      f"_{tag}:_\n```\n{shown.strip()[:1200]}\n```\n")
    (REPO / "eval/receipts/hy3_think_mode_probe.md").write_text("\n".join(md))
    print("[probe] wrote eval/receipts/hy3_think_mode_probe.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
