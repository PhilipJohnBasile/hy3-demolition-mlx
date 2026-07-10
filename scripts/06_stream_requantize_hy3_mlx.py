#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hy3_weight_store import copy_non_weight_files, iter_weight_shards, load_config


def quant_params_for_key(cfg: dict, weight_key: str, weight, scales) -> dict:
    """Resolve the ORIGINAL quant params for a stored tensor.

    The config's per-module entries use runtime module names
    (mlp.switch_mlp.*), while checkpoint tensors use storage names
    (mlp.experts.*) — a raw lookup misses every expert tensor and the global
    fallback (bits=2) silently corrupts the 3-bit down_proj. So: translate
    storage->runtime names for the lookup, then cross-check against the bits
    implied by the packed shapes and trust the shapes on any mismatch.
    """
    quant = cfg.get("quantization_config") or cfg.get("quantization") or {}
    module_key = weight_key[:-7] if weight_key.endswith(".weight") else weight_key
    runtime_key = module_key.replace(".mlp.experts.", ".mlp.switch_mlp.")
    entry = quant.get(runtime_key) if isinstance(quant, dict) else None
    if not isinstance(entry, dict):
        entry = {
            "group_size": quant.get("group_size", 64) if isinstance(quant, dict) else 64,
            "bits": quant.get("bits", 4) if isinstance(quant, dict) else 4,
        }
    group_size = int(entry.get("group_size", 64))
    mode = entry.get("mode", "affine")
    # packed uint32 last dim holds in_dim*bits/32; scales last dim holds
    # in_dim/group_size — solve for bits and verify.
    in_dim = scales.shape[-1] * group_size
    inferred_bits = weight.shape[-1] * 32 // in_dim
    bits = int(entry.get("bits", inferred_bits))
    if bits != inferred_bits:
        print(
            f"WARN {weight_key}: config says {bits}-bit but packed shape implies "
            f"{inferred_bits}-bit; trusting the shape"
        )
        bits = inferred_bits
    return {"group_size": group_size, "bits": bits, "mode": mode}


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
            base = key[:-7]
            scale_key = base + ".scales"
            bias_key = base + ".biases"
            if scale_key not in tensors:
                # Not quantized in the source (norms, unquantized embeds):
                # pass through untouched. Quantizing e.g. an RMSNorm vector
                # produces weights its module class cannot consume.
                emitted[key] = tensor
                continue
            old_params = quant_params_for_key(old_cfg, key, tensor, tensors[scale_key])
            old_biases = tensors.get(bias_key)
            tensor = mx.dequantize(
                tensor,
                tensors[scale_key],
                old_biases,
                old_params["group_size"],
                old_params["bits"],
                old_params["mode"],
            )
            if ".mlp.router.gate." in key:
                # Routing is a discrete decision path: keep the gate in bf16
                # (~58 MB total) so quantization noise cannot flip expert
                # selection. No quant_config entry -> loads as a plain Linear.
                emitted[key] = tensor.astype(mx.bfloat16)
                mx.eval(emitted[key])
                skip.add(scale_key)
                skip.add(bias_key)
                continue
            if tensor.shape[-1] % args.group_size != 0:
                raise ValueError(
                    f"{key}: dequantized in_dim {tensor.shape[-1]} not divisible "
                    f"by target group size {args.group_size}"
                )
            bits = bits_for_key(key, args.expert_bits, args.high_bits)
            q_weight, scales, *biases = mx.quantize(
                tensor.astype(mx.float16),
                group_size=args.group_size,
                bits=bits,
                mode=args.mode,
            )
            mx.eval(q_weight, scales, *biases)
            emitted[key] = q_weight
            emitted[scale_key] = scales
            if biases:
                emitted[bias_key] = biases[0]
            # Write the runtime module name so the loader's config lookup hits.
            runtime_base = base.replace(".mlp.experts.", ".mlp.switch_mlp.")
            quant_config[runtime_base] = {
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
        del tensors, emitted
        mx.clear_cache()
        print(f"wrote {out / out_name}")

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {k: weight_map[k] for k in sorted(weight_map)},
    }
    # mlx_lm's loader requires a top-level {bits, group_size} default in this dict (used as the
    # global fallback nn.quantize() call itself needs) IN ADDITION to the per-module overrides
    # below — matching the native checkpoint's own config.json convention. Without it, loading
    # this checkpoint raises `KeyError: 'group_size'` in mlx_lm.utils._quantize(). Every quantized
    # tensor already gets an explicit per-module override, so this default is a required-but-inert
    # fallback for us; set it to the expert bit-width since that's the majority of the model.
    cfg["quantization"] = {
        "bits": args.expert_bits,
        "group_size": args.group_size,
        **{k: quant_config[k] for k in sorted(quant_config)},
    }
    cfg["quantization_config"] = cfg["quantization"]
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
