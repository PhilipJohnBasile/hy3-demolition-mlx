#!/usr/bin/env python3
"""Eval: local base vs local+prompt vs local+distilled-LoRA on held-out soul
topics. Also records token counts per condition for the cost comparison
(cloud D is scored separately from the already-generated Sonnet answers).

Usage: 58_soul_distill_eval.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = str(REPO / "dist/hy3-family-mini-qwen35b-v1")
ADAPTER = str(REPO / "dist/soul-distill-lora")

EXPERT_PROMPT = (
    "You are a deeply knowledgeable domain expert across software engineering, "
    "security, math, science, design, game development, music theory, visual "
    "art, and perfumery. Give thorough, accurate, well-structured technical "
    "answers with concrete examples."
)


def gen(model, tok, system, user, max_tokens=400):
    from mlx_lm import generate
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    n_prompt_tokens = len(tok.encode(text))
    out = generate(model, tok, prompt=text, max_tokens=max_tokens, verbose=False)
    return out, n_prompt_tokens


def main() -> int:
    from mlx_lm import load
    prompts = json.loads((REPO / "data/soul_distill_prompts.json").read_text())["eval"]

    print("[eval] loading base (conditions A, B)", flush=True)
    model, tok = load(BASE)
    rows = {p["id"]: {"facet": p["facet"], "prompt": p["prompt"]} for p in prompts}
    for p in prompts:
        a_out, a_pt = gen(model, tok, "", p["prompt"])
        b_out, b_pt = gen(model, tok, EXPERT_PROMPT, p["prompt"])
        rows[p["id"]]["A_base"] = a_out
        rows[p["id"]]["A_prompt_tokens"] = a_pt
        rows[p["id"]]["B_prompted"] = b_out
        rows[p["id"]]["B_prompt_tokens"] = b_pt
        print(f"[eval] {p['id']} A/B done", flush=True)
    del model

    print("[eval] loading base + soul-distill LoRA (condition C)", flush=True)
    model, tok = load(BASE, adapter_path=ADAPTER)
    for p in prompts:
        c_out, c_pt = gen(model, tok, "", p["prompt"])
        rows[p["id"]]["C_distilled"] = c_out
        rows[p["id"]]["C_prompt_tokens"] = c_pt
        print(f"[eval] {p['id']} C done", flush=True)

    out = [{"id": k, **v} for k, v in rows.items()]
    (REPO / "eval/receipts/soul_distill_eval.json").write_text(json.dumps(out, indent=1))
    print("[eval] wrote eval/receipts/soul_distill_eval.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
