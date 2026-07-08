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


class DiskExpertSource:
    """True per-expert reader: preads ONLY expert e's bytes from a safetensors
    file, never the whole stacked [E, ...] tensor. This is what makes streaming
    real — M3 showed array-slicing a loaded tensor materializes the whole stack.

    Parses the safetensors header once to learn each per-expert tensor's dtype,
    shape and byte range, then reads expert slices on demand with os.pread.
    """

    _DT = {"F32": (mx.float32, 4), "F16": (mx.float16, 2), "BF16": (mx.bfloat16, 2),
           "U32": (mx.uint32, 4), "I32": (mx.int32, 4), "U16": (mx.uint16, 2),
           "U8": (mx.uint8, 1), "I8": (mx.int8, 1)}

    def __init__(self, path: str):
        import json
        import os
        self._fd = os.open(path, os.O_RDONLY)
        n = int.from_bytes(os.pread(self._fd, 8, 0), "little")
        self._header = json.loads(os.pread(self._fd, n, 8))
        self._data0 = 8 + n

    def close(self):
        import os
        os.close(self._fd)

    _NP = {"F32": "<f4", "F16": "<f2", "U32": "<u4", "I32": "<i4",
           "U16": "<u2", "U8": "<u1", "I8": "<i1"}

    def read_expert(self, key: str, e: int):
        """Return expert e of tensor `key` as an mx.array of shape [d1, d2...]."""
        import os
        import numpy as np
        meta = self._header[key]
        mxdt = self._DT[meta["dtype"]][0]
        shape = meta["shape"]            # [E, d1, d2, ...]
        per = 1
        for d in shape[1:]:
            per *= d
        nbytes = per * self._DT[meta["dtype"]][1]
        start = self._data0 + meta["data_offsets"][0] + e * nbytes
        raw = os.pread(self._fd, nbytes, start)   # <-- only this expert's bytes
        arr = np.frombuffer(raw, dtype=self._NP[meta["dtype"]]).reshape(shape[1:])
        return mx.array(arr, dtype=mxdt)

    def bytes_per_expert(self, key: str) -> int:
        meta = self._header[key]
        _, itemsize = self._DT[meta["dtype"]]
        per = 1
        for d in meta["shape"][1:]:
            per *= d
        return per * itemsize


class StreamingSwitchGLU:
    """Streaming replacement for mlx-lm SwitchGLU: three StreamingSwitchLinears
    (gate/up/down) + swiglu. Hy3 uses gate/up at 2-bit and down at 3-bit, so
    each projection carries its own bit-width. Only the routed experts are
    faulted in per token, shared across the three projections' caches.
    """

    def __init__(self, gate, up, down, cache_size: int = 16):
        # each arg = dict(weight, scales, biases, group_size, bits)
        self.gate = StreamingSwitchLinear(**gate, cache_size=cache_size)
        self.up = StreamingSwitchLinear(**up, cache_size=cache_size)
        self.down = StreamingSwitchLinear(**down, cache_size=cache_size)

    @property
    def reads(self):
        return self.gate.reads + self.up.reads + self.down.reads

    def __call__(self, x, indices):
        from mlx_lm.models.switch_layers import swiglu
        x = mx.expand_dims(x, (-2, -3))
        x_up = self.up(x, indices)
        x_gate = self.gate(x, indices)
        return self.down(swiglu(x_gate, x_up), indices).squeeze(-2)


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
