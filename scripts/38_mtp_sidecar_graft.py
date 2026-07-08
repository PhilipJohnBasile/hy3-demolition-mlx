#!/usr/bin/env python3
"""Sidecar-graft the base MTP head onto a fused AR artifact → an MTP-equipped
variant, without re-pruning the trunk or re-healing.

The MTP head consumes the trunk's final hidden state (hidden_size, unchanged by
pruning) and has its own MoE that uses the GLOBAL num_experts. So:
  - lite-v1 (num_experts=192): graft mtp.* as-is.
  - reap25 (num_experts=144): slice the mtp.* experts 192->144 first (same
    packing-safe whole-expert axis slice as the trunk prune; MTP is a verified
    draft head, so a norm-based pick only affects acceptance, never correctness).

CPU only. Symlinks the trunk shards, writes one mtp sidecar shard, merges the
weight index, and flips num_nextn_predict_layers to 1.

Usage: 38_mtp_sidecar_graft.py --source <fused_dir> --out <variant_dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx

REPO = Path(__file__).resolve().parents[1]
MTP_BASE = REPO / "models" / "hy3-mlx-base-mtp"


def load_mtp_tensors() -> dict:
    out = {}
    for shard in sorted(glob.glob(str(MTP_BASE / "*.safetensors"))):
        d = mx.load(shard)
        for k, v in d.items():
            if k.startswith("mtp."):
                out[k] = v
    return out


def expert_scores(mtp: dict) -> mx.array:
    """Per-expert importance for the MTP MoE, dequant-free: the quantization
    scales encode per-group weight magnitude, so the L2 of each expert's
    gate/up/down scales is a sound importance proxy. Keeps the experts whose
    weights carry the most signal. (MTP is a verified draft head, so this only
    affects draft acceptance, never correctness.)
    """
    total = None
    for proj in ("gate_proj", "up_proj", "down_proj"):
        s = mtp[f"mtp.layer.mlp.experts.{proj}.scales"].astype(mx.float32)  # [E, ...]
        e = mx.sum(s ** 2, axis=tuple(range(1, s.ndim)))  # [E]
        total = e if total is None else total + e
    return mx.sqrt(total)


def slice_mtp_experts(mtp: dict, keep: mx.array) -> dict:
    """Slice every per-expert MTP tensor to the kept indices (axis 0)."""
    keep_list = [int(i) for i in keep.tolist()]
    out = {}
    for k, v in mtp.items():
        # per-expert tensors live under experts.* (E on axis0), router.gate is
        # [E, hidden] families, expert_bias is [E]
        if ".experts." in k or k.endswith("expert_bias") or ".router.gate." in k:
            out[k] = mx.take(v, mx.array(keep_list), axis=0)
        else:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = Path(a.source)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    n_exp = cfg["num_experts"]
    print(f"source num_experts={n_exp}")

    mtp = load_mtp_tensors()
    base_exp = mtp["mtp.layer.mlp.experts.down_proj.weight"].shape[0]
    print(f"base MTP experts={base_exp}")

    if n_exp < base_exp:
        scores = expert_scores(mtp)
        keep = mx.argsort(scores)[::-1][:n_exp]  # top-n by L2 norm
        keep = mx.sort(keep)
        mtp = slice_mtp_experts(mtp, keep)
        print(f"sliced MTP experts {base_exp} -> {n_exp} (top-L2 down_proj)")
        after = mtp["mtp.layer.mlp.experts.down_proj.weight"].shape[0]
        assert after == n_exp, f"slice failed: {after}"

    # write the mtp sidecar shard
    sidecar = "model-mtp-sidecar.safetensors"
    mx.save_safetensors(str(out / sidecar), mtp)
    print(f"wrote {sidecar}: {len(mtp)} tensors")

    # symlink the trunk shards + merge the index
    src_index = json.loads((src / "model.safetensors.index.json").read_text())
    wmap = dict(src_index["weight_map"])
    for shard in sorted(glob.glob(str(src / "*.safetensors"))):
        name = os.path.basename(shard)
        link = out / name
        if not link.exists():
            os.symlink(os.path.abspath(shard), link)
    for k in mtp:
        wmap[k] = sidecar
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": src_index.get("metadata", {}), "weight_map": wmap}, indent=2))

    # config: enable the MTP head
    cfg["num_nextn_predict_layers"] = 1
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # aux files (tokenizer, template, model code)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "hy_v3.py", "special_tokens_map.json"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    # a short note card
    (out / "README.md").write_text(
        f"# {out.name}\n\nMTP-equipped variant: the fused AR trunk of "
        f"`{src.name}` with the Hy3 NextN sidecar grafted back on "
        f"(num_nextn_predict_layers=1). For MTPLX / MTP speculative decoding "
        f"once the hy_v3 backend lands. num_experts={n_exp}.\n")
    print(f"GRAFT DONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
