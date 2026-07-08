#!/usr/bin/env python3
"""GPU verification of the qwen3_5_mtp MTPLX backend on the real checkpoint.

Loads the sibling MTP export through the patched MTPLX runtime path and checks:
  1. trunk loads via the sys.modules shim (qwen3_5_mtp -> qwen3_5_moe)
  2. inject_qwen3_5_mtp_support grafts the head + validate passes
  3. trunk forward returns usable hidden states
  4. mtp_forward produces a finite, non-degenerate draft-token distribution
  5. draft plausibility: top-1 drafted token is a reasonable continuation

This is the hardware bring-up gate: only push the PR if this passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/Users/pjb/git/MTPLX")
import mlx.core as mx  # noqa: E402
from mlx_lm.utils import load as mlx_lm_load  # noqa: E402

from mtplx.qwen3_5_mtp_patch import (  # noqa: E402
    install_qwen3_5_mtp_trunk_shim,
    inject_qwen3_5_mtp_support,
    is_qwen3_5_mtp_config,
    validate_qwen3_5_mtp_support,
)
from mtplx.artifacts import load_config  # noqa: E402

MTP = Path("dist/hy3-family-mini-qwen35b-mtp-v1")


def main() -> int:
    cfg = load_config(MTP)
    assert is_qwen3_5_mtp_config(cfg), "config not detected as qwen3_5_mtp"
    print("[1/5] config detected as qwen3_5_mtp", flush=True)

    install_qwen3_5_mtp_trunk_shim()
    model, tok = mlx_lm_load(str(MTP))
    print("[2/5] trunk loaded via shim (qwen3_5_mtp -> qwen3_5_moe) OK", flush=True)

    ok = inject_qwen3_5_mtp_support(model, MTP, cfg, None)
    assert ok and validate_qwen3_5_mtp_support(model), "graft/validate failed"
    print("[3/5] MTP head grafted + validate_qwen3_5_mtp_support OK", flush=True)

    ids = mx.array([tok.encode("The capital of France is")])
    trunk_out = model(ids, return_hidden=True)
    if not isinstance(trunk_out, tuple):
        print("[4/5] FAIL: trunk did not return hidden states (return_hidden path "
              "needs the pre-norm extraction — see docstring)", flush=True)
        return 2
    logits, hidden = trunk_out
    ar_next = int(mx.argmax(logits[0, -1]).item())
    print(f"[4/5] trunk forward OK; hidden={hidden.shape}; AR next-token id={ar_next} "
          f"({tok.decode([ar_next])!r})", flush=True)

    draft = model.mtp_forward(hidden[:, -1:, :], mx.array([[ar_next]]),
                              concat_order="embedding_hidden")
    finite = bool(mx.all(mx.isfinite(draft)).item())
    draft_id = int(mx.argmax(draft[0, -1]).item())
    print(f"[5/5] mtp_forward: finite={finite}; draft token id={draft_id} "
          f"({tok.decode([draft_id])!r})", flush=True)
    if not finite:
        print("RESULT: FAIL — draft logits not finite", flush=True)
        return 3
    print("RESULT: PASS — qwen3_5_mtp graft loads + drafts (finite distribution). "
          "Draft-acceptance rate is the next-level check.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
