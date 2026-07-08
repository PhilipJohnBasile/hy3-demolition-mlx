#!/usr/bin/env python3
"""Decisive test: does the <|soul:facet|> token actually DO anything?

Same task, WITH vs WITHOUT the soul token, on one model. Measures:
  - first-token KL divergence between the two next-token distributions
    (mechanistic: how much the token moves the model)
  - whether the greedy generations differ (behavioral)

Run on sibling (healed) AND base (control). If the heal wired the token to
something, the sibling's KL should exceed the base's. If both ~0, the token is
inert prompt text.

Usage: 53_soul_ablation.py <model_dir>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FACETS = ["coding", "perfumery", "music", "gamedev"]
PROMPTS = {json.loads(l)["facet"]: json.loads(l)
           for l in open(REPO / "eval/souls/prompts.jsonl")}


def strip_soul(text: str) -> str:
    return re.sub(r"<\|soul:[a-z]+\|>\s*", "", text).strip()


def main() -> int:
    import mlx.core as mx
    from mlx_lm import load, generate
    model_dir = sys.argv[1]
    tag = Path(model_dir).name
    print(f"[ablation] loading {tag}", flush=True)
    model, tok = load(model_dir)

    def first_logits(user_content: str):
        text = tok.apply_chat_template([{"role": "user", "content": user_content}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = mx.array([tok.encode(text)])
        return model(ids)[0, -1]

    def gen(user_content: str) -> str:
        text = tok.apply_chat_template([{"role": "user", "content": user_content}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        return generate(model, tok, prompt=text, max_tokens=120, verbose=False)

    rows = []
    for f in FACETS:
        with_tok = PROMPTS[f]["prompt"]           # has <|soul:f|>
        without = strip_soul(with_tok)             # token removed
        lw, lo = first_logits(with_tok), first_logits(without)
        pw = mx.softmax(lw.astype(mx.float32)); po = mx.softmax(lo.astype(mx.float32))
        eps = 1e-9
        kl = float(mx.sum(pw * (mx.log(pw + eps) - mx.log(po + eps))).item())
        gw, go = gen(with_tok), gen(without)
        identical = gw.strip() == go.strip()
        # crude divergence: first char position where they differ
        div_at = next((i for i in range(min(len(gw), len(go))) if gw[i] != go[i]), min(len(gw), len(go)))
        rows.append({"facet": f, "first_token_kl": round(kl, 4),
                     "gen_identical": identical, "diverge_at_char": div_at,
                     "with_head": gw[:120], "without_head": go[:120]})
        print(f"[ablation] {f:10} KL={kl:.4f}  identical={identical}  diverge@char={div_at}", flush=True)

    out = {"model": tag, "facets": rows,
           "mean_kl": round(sum(r["first_token_kl"] for r in rows) / len(rows), 4)}
    (REPO / f"eval/receipts/soul_ablation_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"[ablation] {tag}: mean first-token KL = {out['mean_kl']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
