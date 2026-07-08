#!/usr/bin/env python3
"""Run the 10-prompt stress battery on lite-v1 via the STREAMING pager, so we
can score it for the shootout without a 112GB resident load (which OOMs).
Reuses the M6 streaming loader machinery.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import hy_v3
from mlx_lm import stream_generate
from mlx_lm.utils import load_tokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hy3_streaming import MultiShardExpertSource, StreamingSwitchGLU  # noqa: E402

MODEL = REPO / "dist" / "hy3-demolition-mlx-lite-v1-fused"
CACHE = 24

spec = importlib.util.spec_from_file_location("s37", REPO / "scripts/37_reap25_stress.py")
s37 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s37)
PROMPTS = s37.PROMPTS


def build_streaming(model_dir: Path):
    cfg = json.loads((model_dir / "config.json").read_text())
    args = hy_v3.ModelArgs.from_dict(cfg)
    model = hy_v3.Model(args)
    weights = {}
    for shard in sorted(glob.glob(str(model_dir / "*.safetensors"))):
        for k, v in mx.load(shard).items():
            if "switch_mlp" not in k:
                weights[k] = v
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    def pred(p, m):
        if "switch_mlp" in p:
            return False
        if p in cfg["quantization"]:
            return cfg["quantization"][p]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{p}.scales" in weights
    q = cfg["quantization"]
    nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                mode=q.get("mode", "affine"), class_predicate=pred)
    src = MultiShardExpertSource(str(model_dir))
    for i, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp"):
            pre = f"model.layers.{i}.mlp.switch_mlp."
            mk = lambda proj, b: dict(source=src, wkey=pre + proj + ".weight",
                                      group_size=64, bits=b, num_experts=args.num_experts)
            gb = cfg["quantization"].get(pre + "gate_proj", {}).get("bits", 2)
            db = cfg["quantization"].get(pre + "down_proj", {}).get("bits", 3)
            mlp.switch_mlp = StreamingSwitchGLU(mk("gate_proj", gb), mk("up_proj", gb),
                                                mk("down_proj", db), cache_size=CACHE)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


def main() -> int:
    print("[stream-stress] building streaming lite-v1", flush=True)
    model = build_streaming(MODEL)
    tok = load_tokenizer(MODEL)
    print(f"[stream-stress] resident {mx.get_active_memory()/1e9:.1f} GB", flush=True)
    out = []
    for pid, prompt in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       reasoning_effort="no_think")
        resp = ""
        for r in stream_generate(model, tok, text, max_tokens=512):
            resp += r.text
        out.append({"id": pid, "output": resp})
        print(f"[stream-stress] {pid} done ({len(resp)} chars)", flush=True)
    (REPO / "eval/receipts/lite_v1_stream_stress.json").write_text(json.dumps(out, indent=1))
    print("[stream-stress] wrote eval/receipts/lite_v1_stream_stress.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
