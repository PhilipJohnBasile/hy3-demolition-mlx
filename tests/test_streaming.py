"""Regression tests for the SSD expert-streaming pager (src/hy3_streaming.py).

The streaming path is the load-bearing 64GB claim; these lock in that the
streaming gathered-qmm stays bit-identical to the fused path and that the
per-expert byte math is exact. No model download needed — synthetic quantized
MoE.
"""
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hy3_streaming import StreamingSwitchLinear, _fused  # noqa: E402


def test_streaming_matches_fused_bit_identical():
    mx.random.seed(0)
    E, OUT, IN, bits, gs = 64, 256, 128, 4, 64
    x = mx.random.normal((1, 1, IN))
    w, s, b = mx.quantize(mx.random.normal((E, OUT, IN)), group_size=gs, bits=bits)
    idx = mx.array([[3, 3, 17, 40, 0, 5, 17, 20]])
    ref = _fused(x, w, s, b, idx, gs, bits)
    got = StreamingSwitchLinear(w, s, b, group_size=gs, bits=bits, cache_size=4)(x, idx)
    mx.eval(ref, got)
    assert float(mx.max(mx.abs(ref - got))) == 0.0


def test_streaming_only_touches_routed_experts():
    mx.random.seed(1)
    E = 64
    w, s, b = mx.quantize(mx.random.normal((E, 128, 64)), group_size=64, bits=4)
    lin = StreamingSwitchLinear(w, s, b, group_size=64, bits=4, cache_size=64)
    idx = mx.array([[1, 1, 2, 3]])  # 3 unique of 64
    lin(mx.random.normal((1, 1, 64)), idx)
    assert lin.reads == 3  # only the routed experts were faulted in


def test_lru_cache_bounds_reads():
    mx.random.seed(2)
    E = 64
    w, s, b = mx.quantize(mx.random.normal((E, 128, 64)), group_size=64, bits=4)
    lin = StreamingSwitchLinear(w, s, b, group_size=64, bits=4, cache_size=8)
    x = mx.random.normal((1, 1, 64))
    for _ in range(5):
        lin(x, mx.array([[0, 1, 2, 3]]))  # same 4 experts repeatedly
    assert lin.reads == 4 and lin.hits > 0  # 4 cold reads, rest cache hits
