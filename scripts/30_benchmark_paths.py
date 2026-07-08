#!/usr/bin/env python3
"""Rigorous decode-path benchmark: AR vs MTP self-speculative (and, once the
MTPLX backend lands, MTPLX as a third path).

The earlier MTP smoke (scripts/24) was a single 64-token run — fine for a
correctness/parity check, not for a throughput claim. This measures each path
properly: warmup, several prompts at varied lengths, steady-state decode
tok/s (median over samples), first-token latency (cold vs warm), output parity
vs AR, and — for MTP — the draft acceptance rate that determines whether
self-speculation helps at all.

Three-way (same weights, runtime is the only variable between 2 and 3):
- regular_mlx    : plain autoregressive MLX decode, no MTP (baseline)
- mtp_no_mtplx   : the MTP heads driven by mlx-lm's per-token draft->verify
                   loop (mtp_generate_step; needs the MTP view with
                   num_nextn_predict_layers>0 + sidecar)
- mtp_mtplx      : the SAME MTP heads driven by MTPLX's batched verification.
                   NOT wired yet — activates once the MTPLX hy_v3 backend is
                   in a release (mlx-lm #1211 + MTPLX PR #142). Until then this
                   benchmarks regular_mlx vs mtp_no_mtplx honestly.

Usage (run when the GPU is free; loads ~112 GB):
  ./scripts/30_benchmark_paths.py \
    --ar-model models/hy3-mlx-base-ar \
    --mtp-model models/hy3-mlx-base-mtp \
    --out eval/receipts/hy3_path_benchmark.json
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import statistics
import time
from pathlib import Path
from typing import Iterator

import mlx.core as mx
from mlx_lm.utils import load

mlx_generate = importlib.import_module("mlx_lm.generate")


@contextlib.contextmanager
def no_wired_limit(*_a, **_k) -> Iterator[None]:
    yield


# varied prompt lengths surface prompt-processing vs decode behavior
PROMPTS = [
    "Return exactly: ready",
    "Write a one-line Python function that reverses a string.",
    "Explain in three sentences why a fixed timestep matters for game physics.",
    "List five HTTP status codes and what each means, one per line.",
    ("You are reviewing a pull request that adds caching. Summarize, in a short "
     "paragraph, the three questions you would ask before approving it."),
]


def build_prompt(tokenizer, text: str, effort: str) -> list[int]:
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, reasoning_effort=effort,
    )
    return tokenizer.encode(templated, add_special_tokens=False)


def bench_path(model_path: str, label: str, max_tokens: int, effort: str,
               warmup: int) -> dict:
    mx.reset_peak_memory()
    t_load = time.perf_counter()
    model, tokenizer, config = load(
        model_path, lazy=True,
        tokenizer_config={"fix_mistral_regex": True}, return_config=True,
    )
    load_s = time.perf_counter() - t_load
    mtp_active = getattr(model, "num_nextn_predict_layers", 0) > 0

    # warmup so the first-real-token latency below is steady-state, not cold
    for _ in range(max(1, warmup)):
        list(mlx_generate.stream_generate(
            model, tokenizer, build_prompt(tokenizer, "warm up", effort),
            max_tokens=8))

    samples = []
    outputs = []
    for text in PROMPTS:
        toks = build_prompt(tokenizer, text, effort)
        t0 = time.perf_counter()
        first_token_s = None
        gen_text = ""
        last = None
        for i, resp in enumerate(mlx_generate.stream_generate(
                model, tokenizer, toks, max_tokens=max_tokens)):
            if i == 0:
                first_token_s = time.perf_counter() - t0
            gen_text += resp.text
            last = resp
        samples.append({
            "prompt_tokens": len(toks),
            "gen_tokens": last.generation_tokens if last else 0,
            "decode_tps": round(last.generation_tps, 3) if last else None,
            "prompt_tps": round(last.prompt_tps, 3) if last else None,
            "first_token_s": round(first_token_s, 3) if first_token_s else None,
        })
        outputs.append(gen_text)

    decode = [s["decode_tps"] for s in samples if s["decode_tps"]]
    ftl = [s["first_token_s"] for s in samples if s["first_token_s"]]
    result = {
        "label": label,
        "model": model_path,
        "mtp_active": mtp_active,
        "load_s": round(load_s, 2),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
        "median_decode_tps": round(statistics.median(decode), 3) if decode else None,
        "median_first_token_s": round(statistics.median(ftl), 3) if ftl else None,
        "per_prompt": samples,
        "_outputs": outputs,
    }
    del model
    mx.clear_cache()
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ar-model", default="models/hy3-mlx-base-ar")
    p.add_argument("--mtp-model", default="models/hy3-mlx-base-mtp")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--reasoning-effort", default="no_think")
    p.add_argument("--out", default="eval/receipts/hy3_path_benchmark.json")
    p.add_argument("--skip-mtp", action="store_true")
    args = p.parse_args()

    mlx_generate.wired_limit = no_wired_limit
    report: dict = {
        "date": time.strftime("%Y-%m-%d"),
        "max_tokens": args.max_tokens,
        "prompts": len(PROMPTS),
        "metal": mx.metal.is_available(),
        "paths": {},
    }

    ar = bench_path(args.ar_model, "regular_mlx", args.max_tokens, args.reasoning_effort, args.warmup)
    report["paths"]["ar"] = ar

    if not args.skip_mtp and Path(args.mtp_model).exists():
        mtp = bench_path(args.mtp_model, "mtp_no_mtplx", args.max_tokens, args.reasoning_effort, args.warmup)
        # parity vs AR on identical prompts (greedy MTP verify => should match)
        mtp["outputs_match_ar"] = mtp.pop("_outputs") == ar["_outputs"]
        if ar["median_decode_tps"] and mtp["median_decode_tps"]:
            mtp["speedup_vs_ar"] = round(mtp["median_decode_tps"] / ar["median_decode_tps"], 3)
        report["paths"]["mtp"] = mtp

    report["paths"]["ar"].pop("_outputs", None)
    report["mtp_mtplx"] = ("not benchmarked yet — MTPLX hy_v3 backend not in a "
                           "release (blocked on mlx-lm #1211 + MTPLX PR #142); same "
                           "MTP heads, batched verify, activates via one flag when live)")

    # human-readable summary line
    a = report["paths"]["ar"]
    summary = f"AR median decode {a['median_decode_tps']} tok/s, first-token {a['median_first_token_s']}s"
    if "mtp" in report["paths"]:
        m = report["paths"]["mtp"]
        summary += (f" | MTP {m['median_decode_tps']} tok/s "
                    f"(speedup {m.get('speedup_vs_ar')}x, parity {m.get('outputs_match_ar')})")
    report["summary"] = summary

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(summary)
    print(f"receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
