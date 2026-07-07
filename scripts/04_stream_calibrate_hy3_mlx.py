#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load


def load_prompts(path: str) -> list[str]:
    prompts: list[str] = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(obj.get("prompt") or obj.get("content") or line.strip())
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/hy3-mlx-base")
    parser.add_argument("--prompts", default="eval/coding/prompts.jsonl")
    parser.add_argument("--out", default="dist/hy3-reap-saliency.json")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--reasoning-effort", default="no_think")
    args = parser.parse_args()

    model, tokenizer = load(args.model)

    layers = {}
    for i, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        router = getattr(mlp, "router", None)
        if router is not None:
            setattr(router, "_hy3_reap_layer", i)
            layers[i] = {
                "counts": [0 for _ in range(router.expert_bias.shape[0])],
                "score_sum": [0.0 for _ in range(router.expert_bias.shape[0])],
            }

    if not layers:
        raise RuntimeError("no Hy3 MoE routers found")

    gate_cls = next(
        getattr(getattr(layer, "mlp", None), "router").__class__
        for layer in model.layers
        if getattr(getattr(layer, "mlp", None), "router", None) is not None
    )
    original_call = gate_cls.__call__

    def recording_call(self, x):
        inds, scores = original_call(self, x)
        layer_idx = getattr(self, "_hy3_reap_layer", None)
        if layer_idx is not None:
            mx.eval(inds, scores)
            idx_rows = inds.reshape(-1).tolist()
            score_rows = scores.reshape(-1).tolist()
            bucket = layers[layer_idx]
            for idx, score in zip(idx_rows, score_rows):
                bucket["counts"][idx] += 1
                bucket["score_sum"][idx] += float(score)
        return inds, scores

    gate_cls.__call__ = recording_call
    try:
        for prompt in load_prompts(args.prompts):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                chat_template_kwargs={"reasoning_effort": args.reasoning_effort},
            )
            generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=args.max_tokens,
                temp=0.0,
            )
    finally:
        gate_cls.__call__ = original_call

    payload = {
        "model": args.model,
        "prompts": args.prompts,
        "created_at": time.time(),
        "layers": {str(k): v for k, v in sorted(layers.items())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

