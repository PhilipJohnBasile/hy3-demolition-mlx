#!/usr/bin/env python3
"""Fuse a LoRA adapter into the base model without materializing all weights.

`mlx_lm fuse` loads the whole model eagerly (lazy=False), which for the 105 GB
Hy3 checkpoint exhausts unified memory and either dies or thrashes swap. This
script does the same fuse with lazy=True: weights stay memory-mapped, each
LoRA fuse is a lazy graph, and save_model() evaluates and frees one shard at a
time. Peak memory is roughly one shard plus fuse buffers instead of the whole
model.

Fuse against the stock AR view (not the -train view) so the released model
keeps the standard chat template.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten
from mlx_lm.utils import load, save


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/hy3-mlx-base-ar")
    parser.add_argument("--adapter-path", default="dist/adapters-hy3-lite-v1")
    parser.add_argument("--save-path", default="dist/hy3-demolition-mlx-lite-v1-fused")
    parser.add_argument("--receipt", default="eval/receipts/hy3_lite_v1_fuse.json")
    parser.add_argument(
        "--card",
        default="cards/hy3-demolition-mlx-lite-v1.md",
        help="model card copied to <save-path>/README.md after fusing "
        "(mlx_lm save writes a frontmatter-only stub otherwise); '' skips",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    print("Loading model lazily")
    model, tokenizer, config = load(
        args.model,
        adapter_path=args.adapter_path,
        lazy=True,
        return_config=True,
    )

    fused_linears = [
        (n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")
    ]
    if not fused_linears:
        raise RuntimeError("no fusable LoRA modules found; wrong adapter path?")
    print(f"Fusing {len(fused_linears)} LoRA modules")
    model.update_modules(tree_unflatten(fused_linears))

    print("Saving shard by shard")
    save(Path(args.save_path), args.model, model, tokenizer, config)
    elapsed = time.perf_counter() - started

    card = Path(args.card) if args.card else None
    if card is not None:
        if not card.exists():
            raise FileNotFoundError(f"model card {card} not found; pass --card '' to skip")
        (Path(args.save_path) / "README.md").write_text(
            card.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(f"model card: {card} -> {args.save_path}/README.md")

    receipt = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "save_path": args.save_path,
        "fused_modules": len(fused_linears),
        "elapsed_s": elapsed,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "lazy": True,
    }
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"receipt: {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
