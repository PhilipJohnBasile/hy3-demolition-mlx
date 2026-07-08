"""Milestone 1 of the SSD expert-streaming pager (BACKLOG #37).

Core component: a streaming replacement for mlx-lm's SwitchLinear that keeps
the stacked per-expert quantized weights DISK-BACKED (memory-mapped) and, per
forward, gathers only the indices actually routed to — with an LRU cache of hot
experts. This is the piece that decouples resident RAM from the 60.7 GB expert
floor so Hy3 can run on a 64 GB Mac (bit-identical, ~2-5 tok/s).

This file proves the *mechanism* is correct against the fused SwitchLinear on a
controlled quantized MoE. Wiring it into hy_v3's MoE + real mmap I/O is the
next milestone.
"""
from __future__ import annotations

from collections import OrderedDict

import mlx.core as mx


class StreamingSwitchLinear:
    """Disk-backed gathered quantized matmul.

    Holds the quantized expert stack (weight/scales/biases, each [E, ...]) as
    a source that is sliced per-expert on demand. Only the experts in `indices`
    are gathered into a small dense stack for the matmul, so at most
    `len(unique(indices))` experts are touched per call — the rest never leave
    disk. An LRU cache keeps recently-used experts materialized to avoid
    re-reading hot experts every token.
    """

    def __init__(self, weight, scales, biases, group_size: int, bits: int,
                 cache_size: int = 16):
        # In the real pager these are mmap handles; here they are the source
        # arrays and we slice lazily. Slicing an mmap'd mx.array only faults in
        # the requested experts' pages.
        self._w, self._s, self._b = weight, scales, biases
        self.group_size, self.bits = group_size, bits
        self.num_experts = weight.shape[0]
        self._cache: OrderedDict[int, tuple] = OrderedDict()
        self.cache_size = cache_size
        self.reads = 0  # experts faulted from "disk" (cache misses)
        self.hits = 0

    def _expert(self, e: int):
        if e in self._cache:
            self.hits += 1
            self._cache.move_to_end(e)
            return self._cache[e]
        self.reads += 1
        tup = (self._w[e], self._s[e], self._b[e])  # <- per-expert page-in
        mx.eval(tup)
        self._cache[e] = tup
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return tup

    def __call__(self, x, indices):
        # x: [..., 1, in]  indices: [..., n_selected]
        flat_idx = [int(i) for i in indices.reshape(-1).tolist()]
        needed = sorted(set(flat_idx))
        w = mx.stack([self._expert(e)[0] for e in needed])
        s = mx.stack([self._expert(e)[1] for e in needed])
        b = mx.stack([self._expert(e)[2] for e in needed])
        remap = {e: j for j, e in enumerate(needed)}
        local = mx.array([remap[i] for i in flat_idx]).reshape(indices.shape)
        return mx.gather_qmm(x, w, s, b, rhs_indices=local, transpose=True,
                             group_size=self.group_size, bits=self.bits)


def _fused(x, weight, scales, biases, indices, group_size, bits):
    """Reference: the fused all-resident gather (what SwitchLinear does)."""
    return mx.gather_qmm(x, weight, scales, biases, rhs_indices=indices,
                         transpose=True, group_size=group_size, bits=bits)


def _selfcheck():
    mx.random.seed(0)
    E, OUT, IN, bits, gs = 144, 512, 256, 4, 64
    x = mx.random.normal((1, 1, IN))
    dense = mx.random.normal((E, OUT, IN))
    w, s, b = mx.quantize(dense, group_size=gs, bits=bits)
    idx = mx.array([[3, 3, 17, 140, 0, 88, 17, 200 % E]])  # top-8, with dups

    ref = _fused(x, w, s, b, idx, gs, bits)
    stream = StreamingSwitchLinear(w, s, b, gs, bits, cache_size=8)
    got = stream(x, idx)
    err = float(mx.max(mx.abs(ref - got)))
    uniq = len({int(i) for i in idx.reshape(-1).tolist()})
    print(f"max|ref-stream| = {err:.2e}  (should be ~0)")
    print(f"experts touched: {stream.reads} unique (of {E}); "
          f"cache hits: {stream.hits}")
    print(f"resident-if-streamed: {uniq}/{E} experts = "
          f"{100*uniq/E:.1f}% of the expert stack for this token")
    assert err < 1e-3, "streaming matmul does not match fused reference"
    print("PASS: streaming gathered-qmm is bit-close to the fused path")


if __name__ == "__main__":
    _selfcheck()
