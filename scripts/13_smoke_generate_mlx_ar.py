#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import time
from pathlib import Path
from typing import Iterator

import mlx.core as mx
from mlx_lm.utils import load

mlx_generate = importlib.import_module("mlx_lm.generate")


@contextlib.contextmanager
def no_wired_limit(*_args, **_kwargs) -> Iterator[None]:
    yield


def build_prompt(tokenizer, user_prompt: str, reasoning_effort: str) -> list[int]:
    if tokenizer.has_chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.encode(user_prompt)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test Hy3 with plain AR MLX decoding and no wired-limit pin."
    )
    parser.add_argument("--model", default="models/hy3-mlx-base-ar")
    parser.add_argument("--prompt", default="Return exactly: ready")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--prefill-step-size", type=int, default=128)
    parser.add_argument("--reasoning-effort", default="no_think")
    parser.add_argument("--eager-load", action="store_true")
    parser.add_argument("--receipt", default="eval/receipts/hy3_ar_smoke.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist; run scripts/13_prepare_hy3_ar_view.py first"
        )

    mlx_generate.wired_limit = no_wired_limit

    started = time.perf_counter()
    model, tokenizer, config = load(
        str(model_path),
        lazy=not args.eager_load,
        tokenizer_config={"fix_mistral_regex": True},
        return_config=True,
    )
    model.num_nextn_predict_layers = 0
    if hasattr(model, "mtp"):
        delattr(model, "mtp")
    mx.clear_cache()

    prompt_tokens = build_prompt(tokenizer, args.prompt, args.reasoning_effort)
    text = mlx_generate.generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=args.max_tokens,
        prefill_step_size=args.prefill_step_size,
        verbose=True,
    )
    elapsed = time.perf_counter() - started

    receipt = {
        "model": str(model_path),
        "model_type": config.get("model_type"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
        "prompt_tokens": len(prompt_tokens),
        "max_tokens": args.max_tokens,
        "prefill_step_size": args.prefill_step_size,
        "reasoning_effort": args.reasoning_effort,
        "elapsed_s": elapsed,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "metal": mx.metal.is_available(),
        "device": mx.device_info() if mx.metal.is_available() else None,
        "output": text,
    }
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"receipt: {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
