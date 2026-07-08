#!/usr/bin/env python3
"""M6b: streaming model loader for hy_v3 (BACKLOG #37).

Build the model, quantize every module EXCEPT switch_mlp, swap each MoE's
switch_mlp for a disk-backed StreamingSwitchGLU, and load ONLY the non-expert
weights. Then measure resident memory — the 60.7 GB expert stack should stay on
disk, so resident ≈ attention/router/shared/embeddings (~10 GB) + a small LRU.

This proves (or disproves) the 64 GB claim on the memory side. Speed (tok/s) is
the follow-on once a forward is confirmed correct.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import hy_v3

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hy3_streaming import MultiShardExpertSource, StreamingSwitchGLU  # noqa: E402

import os
MODEL = REPO / "dist" / "hy3-demolition-mlx-reap25-v1-fused"
CACHE = int(os.environ.get("STREAM_CACHE", "24"))  # experts/proj kept resident
MEM_LIMIT_GB = float(os.environ.get("STREAM_MEM_LIMIT_GB", "0"))  # 0 = none


def log(m):
    print(f"[load {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    if MEM_LIMIT_GB > 0:
        mx.set_memory_limit(int(MEM_LIMIT_GB * 1e9))
        log(f"HARD memory limit = {MEM_LIMIT_GB} GB (simulating a smaller Mac; "
            f"allocations past this fail)")
    config = json.loads((MODEL / "config.json").read_text())
    args = hy_v3.ModelArgs.from_dict(config)
    log(f"building hy_v3 model: {args.num_hidden_layers} layers, "
        f"{args.num_experts} experts")
    model = hy_v3.Model(args)

    # load ONLY non-expert weights (lazy/mmap — experts never materialized)
    log("loading non-expert weights (skipping switch_mlp experts)")
    weights = {}
    for shard in sorted(glob.glob(str(MODEL / "*.safetensors"))):
        d = mx.load(shard)
        for k, v in d.items():
            if "switch_mlp" not in k:
                weights[k] = v
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # quantize everything the checkpoint quantized, EXCEPT switch_mlp
    def class_predicate(p, m):
        if "switch_mlp" in p:
            return False
        if p in config["quantization"]:
            return config["quantization"][p]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{p}.scales" in weights

    q = config["quantization"]
    nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                mode=q.get("mode", "affine"), class_predicate=class_predicate)

    # swap each MoE's switch_mlp for a disk-backed streaming block
    log("swapping switch_mlp -> StreamingSwitchGLU (disk-backed)")
    src = MultiShardExpertSource(str(MODEL))
    n_swapped = 0
    for i, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp"):
            pre = f"model.layers.{i}.mlp.switch_mlp."
            mk = lambda proj, bits: dict(source=src, wkey=pre + proj + ".weight",
                                         group_size=64, bits=bits, num_experts=args.num_experts)
            # gate/up 2-bit, down 3-bit (from config per-module quant)
            gb = config["quantization"].get(pre + "gate_proj", {}).get("bits", 2)
            db = config["quantization"].get(pre + "down_proj", {}).get("bits", 3)
            mlp.switch_mlp = StreamingSwitchGLU(mk("gate_proj", gb), mk("up_proj", gb),
                                                mk("down_proj", db), cache_size=CACHE)
            n_swapped += 1
    log(f"swapped {n_swapped} MoE layers")

    mx.clear_cache()
    mx.reset_peak_memory()
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    del weights
    mx.clear_cache()

    active = mx.get_active_memory() / 1e9
    log(f"RESIDENT after streaming load: {active:.1f} GB "
        f"(vs ~87 GB fully-resident reap25)")
    disk = REPO / "dist" / "hy3-demolition-mlx-reap25-v1-fused"
    total = sum(f.stat().st_size for f in disk.glob("*.safetensors")) / 1e9
    log(f"model on disk = {total:.0f} GB; experts stay on disk, "
        f"resident is non-expert weights + a {CACHE}-expert/proj LRU")
    if active < 40:
        log("MEMORY PROOF: streaming load fits well under a 64 GB budget")
    else:
        log("CAUTION: resident higher than expected — investigate what wired in")

    # M6c: real generation — correctness + decode tok/s
    log("M6c: generating from the streaming model")
    from mlx_lm.utils import load_tokenizer
    from mlx_lm import stream_generate
    tok = load_tokenizer(MODEL)

    # logit probe: is a single forward sane (not NaN, plausible argmax)?
    ids = tok.encode("The capital of France is")
    logits = model(mx.array([ids]))
    mx.eval(logits)
    last = logits[0, -1]
    nan = bool(mx.any(mx.isnan(last)))
    top = int(mx.argmax(last))
    log(f"PROBE: logits nan={nan}, argmax token={top} -> "
        f"{tok.decode([top])!r} (sane if a real word/token)")
    prompt = "The capital of France is Paris, which is famous for"
    t0 = time.time()
    text, n = "", 0
    for resp in stream_generate(model, tok, prompt, max_tokens=48):
        text += resp.text
        n = resp.generation_tokens
    dt = time.time() - t0
    peak = mx.get_peak_memory() / 1e9
    log(f"OUTPUT: {text.strip()[:160]!r}")
    log(f"decode: {n} tokens in {dt:.1f}s = {n/dt:.2f} tok/s | "
        f"peak resident during gen = {peak:.1f} GB")
    log("M6c: streaming generation WORKS end-to-end" if text.strip()
        else "M6c: empty output — investigate")
    src.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
