#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hy3_weight_store import copy_non_weight_files, iter_weight_shards, load_config


def quant_params_for_key(cfg: dict, weight_key: str) -> dict:
    quant = cfg.get("quantization_config") or cfg.get("quantization") or {}
    module_key = weight_key[:-7] if weight_key.endswith(".weight") else weight_key
    if isinstance(quant, dict):
        exact = quant.get(module_key)
        if isinstance(exact, dict):
            return exact
        return {
            "group_size": quant.get("group_size", 64),
            "bits": quant.get("bits", 4),
            "mode": quant.get("mode", "affine"),
        }
    return {"group_size": 64, "bits": 4, "mode": "affine"}


def bits_for_key(key: str, expert_bits: int, high_bits: int) -> int:
    high_markers = (
        "embed_tokens",
        "lm_head",
        "self_attn",
        "shared_mlp",
        "router.gate",
        "expert_bias",
        "norm",
    )
    return high_bits if any(marker in key for marker in high_markers) else expert_bits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="dist/hy3-requant")
    parser.add_argument("--expert-bits", type=int, default=3)
    parser.add_argument("--high-bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", default="affine")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "source": args.model,
        "expert_bits": args.expert_bits,
        "high_bits": args.high_bits,
        "group_size": args.group_size,
        "mode": args.mode,
    }
    (out / "requant_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"wrote plan: {out / 'requant_plan.json'}")
    if not args.write:
        print("dry run only; pass --write to stream and write requantized safetensors")
        return

    copy_non_weight_files(args.model, out)
    cfg = load_config(args.model)
    old_cfg = dict(cfg)
    cfg["hy3_mixed_quant"] = plan

    weight_map = {}
    total_size = 0
    quant_config = {}
    for shard in iter_weight_shards(args.model):
        tensors = mx.load(str(shard))
        emitted = {}
        skip = set()
        for key, tensor in list(tensors.items()):
            if key in skip or not key.endswith(".weight"):
                if key not in skip and not (key.endswith(".scales") or key.endswith(".biases")):
                    emitted[key] = tensor
                continue
            if tensor.shape[-1] % args.group_size != 0:
                emitted[key] = tensor
                continue
            base = key[:-7]
            scale_key = base + ".scales"
            bias_key = base + ".biases"
            if scale_key in tensors:
                old_params = quant_params_for_key(old_cfg, key)
                old_biases = tensors.get(bias_key)
                tensor = mx.dequantize(
                    tensor,
                    tensors[scale_key],
                    old_biases,
                    old_params.get("group_size", 64),
                    old_params.get("bits", 4),
                    old_params.get("mode", "affine"),
                )
            bits = bits_for_key(key, args.expert_bits, args.high_bits)
            q_weight, scales, *biases = mx.quantize(
                tensor.astype(mx.float16),
                group_size=args.group_size,
                bits=bits,
                mode=args.mode,
            )
            emitted[key] = q_weight
            emitted[scale_key] = scales
            if biases:
                emitted[bias_key] = biases[0]
            quant_config[base] = {
                "group_size": args.group_size,
                "bits": bits,
                "mode": args.mode,
            }
            skip.add(scale_key)
            skip.add(bias_key)
        out_name = shard.name
        mx.save_safetensors(str(out / out_name), emitted, metadata={"format": "mlx"})
        for key, tensor in emitted.items():
            weight_map[key] = out_name
            total_size += tensor.nbytes
        print(f"wrote {out / out_name}")

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {k: weight_map[k] for k in sorted(weight_map)},
    }
    cfg["quantization"] = {k: quant_config[k] for k in sorted(quant_config)}
    cfg["quantization_config"] = cfg["quantization"]
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
