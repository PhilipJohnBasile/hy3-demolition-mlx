#!/usr/bin/env python3
"""Three-model shootout on this Mac: sibling vs reap25 vs lite-v1.

Same prompts, production (no-think) mode, measured warm decode tok/s +
prompt-processing tok/s + peak memory. One model per process (memory fully
releases between them — fair + no OOM). Run via the orchestrator at the bottom
of this file's __main__, or `--model <tag>` for one.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

MODELS = {
    "sibling": (str(REPO / "dist/hy3-family-mini-qwen35b-v1"), {"enable_thinking": False}),
    "reap25": (str(REPO / "dist/hy3-demolition-mlx-reap25-v1-fused"), {"reasoning_effort": "no_think"}),
    "lite-v1": (str(REPO / "dist/hy3-demolition-mlx-lite-v1-fused"), {"reasoning_effort": "no_think"}),
}

PROMPTS = [
    "Write a Python function that checks if a string is a valid IPv4 address.",
    "Explain the difference between a process and a thread in three sentences.",
    "Compute 17 * 23 and show the steps.",
]


def run_one(tag: str) -> None:
    from mlx_lm import load, stream_generate
    import mlx.core as mx
    path, tkw = MODELS[tag]
    t0 = time.time()
    model, tok = load(path)
    load_s = time.time() - t0
    # warm-up (excluded from timing) so decode tok/s is steady-state
    warm = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                   tokenize=False, add_generation_prompt=True, **tkw)
    for _ in stream_generate(model, tok, warm, max_tokens=8):
        pass

    dec, pre, peak = [], [], 0.0
    for p in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True, **tkw)
        last = None
        for last in stream_generate(model, tok, text, max_tokens=200):
            pass
        dec.append(last.generation_tps)
        pre.append(last.prompt_tps)
        peak = max(peak, last.peak_memory)
    out = {
        "model": tag,
        "load_s": round(load_s, 1),
        "decode_tps_median": round(statistics.median(dec), 1),
        "decode_tps_each": [round(x, 1) for x in dec],
        "prompt_tps_median": round(statistics.median(pre), 1),
        "peak_gb": round(peak, 1),
    }
    (REPO / f"eval/receipts/shootout_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"[shootout] {tag}: load {out['load_s']}s | decode {out['decode_tps_median']} tok/s "
          f"| prefill {out['prompt_tps_median']} tok/s | peak {out['peak_gb']} GB", flush=True)
    del model, tok
    mx.clear_cache()


def aggregate() -> None:
    rows = []
    for tag in MODELS:
        f = REPO / f"eval/receipts/shootout_{tag}.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
    print("\n=== SHOOTOUT (M5 Max 128 GB, no-think, 200-tok gens) ===")
    print(f"{'model':10} {'decode tok/s':>13} {'prefill tok/s':>14} {'peak GB':>9} {'load s':>8}")
    for r in rows:
        print(f"{r['model']:10} {r['decode_tps_median']:>13} {r['prompt_tps_median']:>14} "
              f"{r['peak_gb']:>9} {r['load_s']:>8}")


if __name__ == "__main__":
    if "--aggregate" in sys.argv:
        aggregate()
    elif "--model" in sys.argv:
        run_one(sys.argv[sys.argv.index("--model") + 1])
