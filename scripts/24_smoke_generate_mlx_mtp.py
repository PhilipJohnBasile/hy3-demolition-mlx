#!/usr/bin/env python3
"""Smoke the Hy3 MTP self-speculative path and compare tok/s against AR.

Loads the MTP view (num_nextn_predict_layers=1 + sidecar), applies the same
no-wired-limit patch that made AR generation survive the Metal watchdog, and
generates via stream_generate so the fork's mtp_generate_step activates. The
receipt records decode tok/s for the MTP run and, with --ar-model, an AR
baseline on the identical prompt, plus whether the outputs match (the MTP
path verifies drafts greedily, so temp-0 output should be identical).
"""
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
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )
    return tokenizer.encode(text, add_special_tokens=False)


def run_once(model_path: str, prompt: str, max_tokens: int, reasoning_effort: str) -> dict:
    mx.reset_peak_memory()
    started = time.perf_counter()
    model, tokenizer, config = load(
        model_path,
        lazy=True,
        tokenizer_config={"fix_mistral_regex": True},
        return_config=True,
    )
    load_s = time.perf_counter() - started

    prompt_tokens = build_prompt(tokenizer, prompt, reasoning_effort)
    text = ""
    gen_started = time.perf_counter()
    last = None
    for response in mlx_generate.stream_generate(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=max_tokens,
        prefill_step_size=128,
    ):
        text += response.text
        last = response
    gen_s = time.perf_counter() - gen_started

    result = {
        "model": model_path,
        "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
        "mtp_active": getattr(model, "num_nextn_predict_layers", 0) > 0,
        "load_s": round(load_s, 2),
        "generation_s": round(gen_s, 2),
        "generation_tokens": last.generation_tokens if last else 0,
        "generation_tps": round(last.generation_tps, 3) if last else None,
        "prompt_tokens": len(prompt_tokens),
        "prompt_tps": round(last.prompt_tps, 3) if last else None,
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
        "output": text,
    }
    del model
    mx.clear_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/hy3-mlx-base-mtp")
    parser.add_argument("--ar-model", default="",
                        help="optional AR view to benchmark the same prompt for comparison")
    parser.add_argument("--prompt", default="Explain in three sentences why tests must run before a fix is called done.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--reasoning-effort", default="no_think")
    parser.add_argument("--receipt", default="eval/receipts/hy3_mtp_smoke.json")
    args = parser.parse_args()

    mlx_generate.wired_limit = no_wired_limit

    receipt: dict = {
        "date": time.strftime("%Y-%m-%d"),
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "metal": mx.metal.is_available(),
        "device": mx.device_info() if mx.metal.is_available() else None,
        "mtp": run_once(args.model, args.prompt, args.max_tokens, args.reasoning_effort),
    }
    if args.ar_model:
        receipt["ar"] = run_once(args.ar_model, args.prompt, args.max_tokens, args.reasoning_effort)
        mtp_tps = receipt["mtp"]["generation_tps"] or 0
        ar_tps = receipt["ar"]["generation_tps"] or 0
        receipt["speedup"] = round(mtp_tps / ar_tps, 3) if ar_tps else None
        receipt["outputs_match"] = receipt["mtp"]["output"] == receipt["ar"]["output"]

    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("device",)}, indent=2, default=str))
    print(f"receipt: {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
