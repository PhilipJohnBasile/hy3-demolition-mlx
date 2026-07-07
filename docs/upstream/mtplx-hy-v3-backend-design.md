# MTPLX hy_v3 backend — design + status

> Status: PROTOTYPE built and committed on branch `hy-v3-mtp-backend` in the
> MTPLX scratchpad clone; patch exported here
> (`0001-Add-hy_v3-Tencent-Hy3-native-MTP-backend.patch`). Existing MTPLX
> artifact tests pass with the change. Do not open the upstream PR until the
> dependency chain below is satisfied and PJB gives the go-ahead.

## What the prototype contains

- `mtplx/backends/hy_v3_mtp.py` — facade following the glm_mtp pattern
  (contract-gated load through `mtplx.runtime`, health metadata, drafting
  wired through the shared `generation.py` sampler).
- `mtplx/hy_v3_mtp_patch.py` — `is_hy_v3_mtp_config` (model_type/architecture
  + `num_nextn_predict_layers > 0`) and `inject_hy_v3_mtp_support`. Because
  the mlx-lm hy_v3 MTP revision exposes the head natively
  (`predict_next_tokens`, `return_hidden_states=True`), injection *binds* the
  existing surface (`mtplx_draft_next`, depth 1) rather than grafting weights.
  It raises a clear error if config promises MTP but the checkpoint is an
  AR-only export.
- `mtplx/runtime.py` — dispatch line before the deepseek fallback.
- `mtplx/commands/public.py` — added to the runtime import smoke list.
- `mtplx/backends/registry.py` — hy-v3-mtp upgraded from
  `recognized-backend-pending` to `experimental-native-contract-gated`
  (`can_run_verified=True`), with architecture notes and the mlx-lm reference.
- `tests/test_artifacts.py` — hy_v3 removed from the pending-markers test
  (that's the point of the change); remaining artifact tests pass.

## Architecture facts the backend encodes

- One appended NextN layer (`num_nextn_predict_layers=1`), with its **own**
  192-expert MoE MLP (sigmoid top-8 routing + expert bias) — structurally
  independent of the trunk experts.
- Input: `eh_proj(concat[enorm(next-token embedding), hnorm(trunk hidden)])`
  where the trunk hidden is **pre-final-norm**; output through a final RMSNorm
  and the shared lm_head/embeddings.
- MLX sidecar naming: `mtp.*` in `model-mtp.safetensors` (1.4 GB, quantized to
  match trunk).

## Dependency chain before the PR can be real

1. mlx-lm PR #1211 (hy_v3 base) merges upstream.
2. Our MTP follow-up PR merges (adds `predict_next_tokens` — the surface this
   backend binds). Prepared: `0001-Use-the-Hy3-MTP-layer-...patch`.
3. MTPLX bumps its pinned mlx-lm to a release containing both.
4. Live verification on the M5 Max: `mtplx inspect` classifies our artifact as
   hy-v3-mtp; `mtplx bench` measures the runtime contract (exactness baseline
   + speedup) that promotion past `experimental` requires.

## Open questions — RESOLVED (generation.py/runtime.py read, 2026-07-07)

- **Call shape:** the sampler calls `rt.draft_mtp(hidden, mx.array([[token]]),
  mtp_cache=rt.make_mtp_cache())` and `rt.forward_ar(tokens, cache=...,
  return_hidden=True)` — a **runtime-level** surface, not model attributes.
  Signatures map 1:1 onto hy_v3's `predict_next_tokens(hidden, token_ids,
  cache)` and `model(inputs, cache, return_hidden_states=True)`. The injector
  should therefore install/parameterize the runtime adapter (like
  glm_mtp_patch's `make_mtp_cache`) rather than the model-attribute binding in
  the current prototype patch — revise `inject_hy_v3_mtp_support` at live
  wiring.
- **Draft cache:** MTPLX passes a FRESH `make_mtp_cache()` per draft call and
  discards it — no trim-on-rejection bookkeeping needed (simpler than
  mlx-lm's mtp_generate_step, which trims by 1).
- **Contract parameters:** `runtime.draft_mtp` resolves `hidden_variant` and
  `concat_order` from the RuntimeContract — exactly Hy3's two sensitivities.
  The hy_v3 contract must pin: hidden = trunk **pre-final-norm**, concat =
  **[enorm(embedding), hnorm(hidden)]** (embedding first; "order matters" per
  the mlx-lm implementation). Fixtures get measured by `mtplx bench` on the
  M5 Max once the model loads.
