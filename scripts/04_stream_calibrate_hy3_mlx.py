#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

mlx_generate = importlib.import_module("mlx_lm.generate")


@contextlib.contextmanager
def no_wired_limit(*_args, **_kwargs):
    yield


@dataclass(frozen=True)
class PromptCase:
    prompt: str
    facet: str = "default"
    source: str = "calibration"


def load_prompts(path: str) -> list[PromptCase]:
    prompts: list[PromptCase] = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(
                PromptCase(
                    prompt=obj.get("prompt") or obj.get("content") or line.strip(),
                    facet=obj.get("facet") or obj.get("soul") or "default",
                    source=str(path),
                )
            )
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/hy3-mlx-base")
    parser.add_argument("--prompts", default="eval/coding/prompts.jsonl")
    parser.add_argument("--soul-prompts", default="eval/souls/protected_prompts.jsonl")
    parser.add_argument("--out", default="dist/hy3-reap-saliency.json")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--reasoning-effort", default="no_think")
    parser.add_argument("--wired-limit", action="store_true")
    args = parser.parse_args()

    if not args.wired_limit:
        mlx_generate.wired_limit = no_wired_limit

    model, tokenizer = load(args.model, lazy=True)
    model.num_nextn_predict_layers = 0
    if hasattr(model, "mtp"):
        delattr(model, "mtp")

    layers = {}
    for i, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        router = getattr(mlp, "router", None)
        if router is not None:
            setattr(router, "_hy3_reap_layer", i)
            layers[i] = {
                "counts": [0 for _ in range(router.expert_bias.shape[0])],
                "score_sum": [0.0 for _ in range(router.expert_bias.shape[0])],
                "facets": {},
            }

    if not layers:
        raise RuntimeError("no Hy3 MoE routers found")

    gate_cls = next(
        getattr(getattr(layer, "mlp", None), "router").__class__
        for layer in model.layers
        if getattr(getattr(layer, "mlp", None), "router", None) is not None
    )
    original_call = gate_cls.__call__
    active_facet = "default"

    def recording_call(self, x):
        inds, scores = original_call(self, x)
        layer_idx = getattr(self, "_hy3_reap_layer", None)
        if layer_idx is not None:
            mx.eval(inds, scores)
            idx_rows = inds.reshape(-1).tolist()
            score_rows = scores.reshape(-1).tolist()
            bucket = layers[layer_idx]
            facet_bucket = bucket["facets"].setdefault(
                active_facet,
                {
                    "counts": [0 for _ in bucket["counts"]],
                    "score_sum": [0.0 for _ in bucket["score_sum"]],
                },
            )
            for idx, score in zip(idx_rows, score_rows):
                bucket["counts"][idx] += 1
                bucket["score_sum"][idx] += float(score)
                facet_bucket["counts"][idx] += 1
                facet_bucket["score_sum"][idx] += float(score)
        return inds, scores

    gate_cls.__call__ = recording_call
    try:
        cases = load_prompts(args.prompts)
        soul_path = Path(args.soul_prompts)
        if soul_path.exists():
            cases.extend(load_prompts(str(soul_path)))
        for case in cases:
            active_facet = case.facet
            messages = [{"role": "user", "content": case.prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=args.reasoning_effort,
            )
            prompt_tokens = tokenizer.encode(formatted, add_special_tokens=False)
            mlx_generate.generate(
                model,
                tokenizer,
                prompt=prompt_tokens,
                max_tokens=args.max_tokens,
                verbose=False,
            )
    finally:
        gate_cls.__call__ = original_call

    payload = {
        "model": args.model,
        "prompts": args.prompts,
        "soul_prompts": args.soul_prompts,
        "created_at": time.time(),
        "layers": {str(k): v for k, v in sorted(layers.items())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
