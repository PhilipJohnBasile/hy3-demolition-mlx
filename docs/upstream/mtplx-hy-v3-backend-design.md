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

## Open questions for the live-wiring pass

- Whether MTPLX's shared sampler calls `mtplx_draft_next` with the same
  (hidden, token_ids, cache) shapes the mlx-lm method expects, or needs a thin
  adapter in the patch (check `mtplx/generation.py` draft call sites).
- KV-cache handling for the draft layer across accepted/rejected tokens
  (mlx-lm's mtp_generate_step trims the mtp cache by 1 on rejection; MTPLX's
  cache_bank may want to own that).
- Contract fixtures: which prompts + depths their `bench` records.
