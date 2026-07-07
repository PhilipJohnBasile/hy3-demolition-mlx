#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hy3_reap import build_plan, is_expert_axis_tensor, layer_from_key, load_saliency, plan_for_layer
from hy3_weight_store import copy_non_weight_files, iter_weight_shards, load_config


DEFAULT_PROTECTED_FACETS = [
    "coding",
    "math",
    "science",
    "security",
    "design",
    "fullstack",
    "gamedev",
    "legacy",
    "music",
    "art",
    "perfumery",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/hy3-mlx-base-ar")
    parser.add_argument("--saliency", required=True)
    parser.add_argument("--out", default="dist/hy3-reap-pruned")
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--protected-facet", action="append", dest="protected_facets")
    parser.add_argument("--min-keep-per-protected-facet", type=int, default=8)
    parser.add_argument("--no-soul-protection", action="store_true")
    parser.add_argument("--allow-missing-soul-saliency", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.model)
    if int(cfg.get("num_nextn_predict_layers") or 0) > 0:
        raise SystemExit(
            f"{args.model}: num_nextn_predict_layers="
            f"{cfg['num_nextn_predict_layers']} — pruning a source with an MTP "
            "sidecar would emit unpruned mtp.* experts against a pruned config "
            "and break loading. Prune the AR view (models/hy3-mlx-base-ar) or "
            "resolve the sidecar strategy (backlog #27) first."
        )
    old_num_experts = int(cfg["num_experts"])
    protected_facets = [] if args.no_soul_protection else (
        args.protected_facets or DEFAULT_PROTECTED_FACETS
    )
    plan = build_plan(
        load_saliency(args.saliency),
        source=args.model,
        ratio=args.ratio,
        num_experts=old_num_experts,
        protected_facets=protected_facets,
        min_keep_per_protected_facet=args.min_keep_per_protected_facet,
        require_protected=bool(protected_facets) and not args.allow_missing_soul_saliency,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reap_plan.json").write_text(json.dumps(plan.to_json(), indent=2) + "\n")
    print(f"wrote plan: {out / 'reap_plan.json'}")

    if not args.write:
        print("dry run only; pass --write to stream and write pruned safetensors")
        return

    copy_non_weight_files(args.model, out)
    new_cfg = cfg.copy()
    new_cfg["num_experts"] = plan.new_num_experts
    new_cfg["hy3_reap"] = plan.to_json()
    (out / "config.json").write_text(json.dumps(new_cfg, indent=2) + "\n")

    weight_map = {}
    total_size = 0
    for shard in iter_weight_shards(args.model):
        tensors = mx.load(str(shard))
        changed = {}
        for key, tensor in tensors.items():
            layer_idx = layer_from_key(key)
            layer_plan = plan_for_layer(plan, layer_idx) if layer_idx is not None else None
            if layer_plan and is_expert_axis_tensor(key, tuple(tensor.shape), old_num_experts):
                keep = mx.array(layer_plan.keep, dtype=mx.int32)
                tensor = mx.take(tensor, keep, axis=0)
            changed[key] = tensor
            total_size += tensor.nbytes
        out_name = shard.name
        mx.save_safetensors(str(out / out_name), changed, metadata={"format": "mlx"})
        for key in changed:
            weight_map[key] = out_name
        print(f"wrote {out / out_name}")

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {k: weight_map[k] for k in sorted(weight_map)},
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    for file_name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"):
        src = Path(args.model) / file_name
        dst = out / file_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
