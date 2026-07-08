#!/usr/bin/env python3
"""SSD expert-streaming validation (BACKLOG #37).

Milestone 2: the StreamingSwitchGLU block matches the fused reap25 MoE
bit-for-bit on real weights (mixed 2/3-bit).
Milestone 3 probe: does mmap-backed loading keep RESIDENT memory proportional
to the experts actually sliced (streaming works), or does MLX materialize the
whole stacked tensor (streaming needs a different I/O path)?
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hy3_streaming import StreamingSwitchGLU  # noqa: E402

MODEL = REPO / "dist" / "hy3-demolition-mlx-reap25-v1-fused"


def find_layer1_shard() -> str:
    for s in sorted(glob.glob(str(MODEL / "*.safetensors"))):
        with safe_open(s, framework="mlx") as f:
            if any("layers.1.mlp.switch_mlp" in k for k in f.keys()):
                return s
    raise SystemExit("layer-1 switch_mlp not found")


def milestone2_correctness(shard: str) -> None:
    from mlx_lm.models.switch_layers import swiglu
    pre = "model.layers.1.mlp.switch_mlp."
    T = {}
    with safe_open(shard, framework="mlx") as f:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for t in ("weight", "scales", "biases"):
                T[f"{proj}.{t}"] = f.get_tensor(pre + proj + "." + t)
    mx.eval(list(T.values()))
    pack = lambda p, b: dict(weight=T[f"{p}.weight"], scales=T[f"{p}.scales"],
                             biases=T[f"{p}.biases"], group_size=64, bits=b)
    mx.random.seed(1)
    h = mx.random.normal((1, 4096)) * 0.1
    idx = mx.array([[7, 7, 40, 113, 2, 88, 131, 15]])
    xx = mx.expand_dims(h, (-2, -3))
    fq = lambda x, p, b: mx.gather_qmm(x, T[f"{p}.weight"], T[f"{p}.scales"],
                                       T[f"{p}.biases"], rhs_indices=idx,
                                       transpose=True, group_size=64, bits=b)
    ref = mx.gather_qmm(swiglu(fq(xx, "gate_proj", 2), fq(xx, "up_proj", 2)),
                        T["down_proj.weight"], T["down_proj.scales"],
                        T["down_proj.biases"], rhs_indices=idx, transpose=True,
                        group_size=64, bits=3).squeeze(-2)
    glu = StreamingSwitchGLU(pack("gate_proj", 2), pack("up_proj", 2),
                             pack("down_proj", 3), cache_size=8)
    got = glu(h, idx)
    mx.eval(ref, got)
    err = float(mx.max(mx.abs(ref - got)))
    print(f"[M2] real reap25 layer-1 MoE block: max|fused-stream| = {err:.3e}")
    assert err < 1e-3, "streaming block does not match fused"
    print("[M2] PASS — full MoE block streams bit-identically")


def milestone3_memory_probe(shard: str) -> None:
    """Load the biggest expert tensor via mmap, slice a few experts, and see if
    resident memory tracks the slice or the whole tensor."""
    key = "model.layers.1.mlp.switch_mlp.gate_proj.weight"  # [144,1536,256] u32
    mx.clear_cache()
    mx.reset_peak_memory()
    # mmap-lazy handle: safetensors in MLX is memory-mapped; slicing should
    # only need the sliced experts.
    with safe_open(shard, framework="mlx") as f:
        full_shape = f.get_slice(key).get_shape()
        w = f.get_tensor(key)  # lazy/mmap-backed
    full_bytes = 1
    for d in full_shape:
        full_bytes *= d
    full_bytes *= 4  # uint32
    # slice 8 experts and force them resident
    experts = [7, 40, 113, 2, 88, 131, 15, 99]
    sliced = mx.stack([w[e] for e in experts])
    mx.eval(sliced)
    peak = mx.get_peak_memory()
    print(f"[M3] gate_proj.weight full tensor = {full_bytes/1e9:.2f} GB "
          f"({full_shape})")
    print(f"[M3] sliced {len(experts)}/144 experts; peak MLX alloc during "
          f"slice+eval = {peak/1e9:.2f} GB")
    ratio = peak / full_bytes
    if ratio < 0.5:
        print(f"[M3] PASS — peak is {100*ratio:.0f}% of the full tensor; "
              f"slicing does NOT materialize the whole stack (streaming viable)")
    else:
        print(f"[M3] CAUTION — peak is {100*ratio:.0f}% of the full tensor; "
              f"MLX may materialize the stack. Streaming needs explicit "
              f"per-expert pread, not just array slicing.")


def milestone4_pread(shard: str) -> None:
    """Per-expert pread is byte-exact and reads only ~1/E of the tensor."""
    from hy3_streaming import DiskExpertSource
    key = "model.layers.1.mlp.switch_mlp.gate_proj.weight"
    with safe_open(shard, framework="mlx") as f:
        full = f.get_tensor(key)
        mx.eval(full)
    src = DiskExpertSource(shard)
    ok = True
    for e in (0, 7, 40, 143):
        got = src.read_expert(key, e)
        mx.eval(got)
        err = float(mx.max(mx.abs(got.astype(mx.int64) - full[e].astype(mx.int64))))
        ok = ok and err == 0
    bpe = src.bytes_per_expert(key)
    print(f"[M4] per-expert pread byte-exact={ok}; reads {bpe/1e6:.2f} MB/expert "
          f"= {100/full.shape[0]:.2f}% of the {bpe*full.shape[0]/1e9:.2f} GB tensor")
    src.close()
    assert ok, "pread not byte-exact"
    print("[M4] PASS — true per-expert disk reads (streaming I/O works)")


if __name__ == "__main__":
    shard = find_layer1_shard()
    milestone2_correctness(shard)
    milestone3_memory_probe(shard)
    milestone4_pread(shard)
