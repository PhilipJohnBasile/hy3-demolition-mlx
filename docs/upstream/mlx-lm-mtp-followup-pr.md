# Draft follow-up PR: Hy3 MTP self-speculative decoding

> Status: DRAFT — branch is prepared locally; do not push/open without PJB's
> go-ahead. Depends on #1211 merging first.
>
> Branch: `hy-v3-mtp` in the scratchpad clone (based on `pr-1211`), commit
> "Use the Hy3 MTP layer for self-speculative decoding".
> To recreate: fetch PR 1211, apply the fork delta from
> `eauchs/mlx-lm@a7cc3054` + the sanitize n_mtp==0 drop + the MTP unit test
> (all in the branch commit).

## Title
Use the Hy3 MTP layer for self-speculative decoding

## Body

Follow-up to #1211. Hy3 ships a Multi-Token-Prediction layer
(`num_nextn_predict_layers=1`); #1211 strips it. This PR keeps it and uses it
for self-speculative decoding — no external draft model needed:

- `models/hy_v3.py`: `MTPBlock` (enorm/hnorm → eh_proj → one decoder layer →
  final norm), `Model.predict_next_tokens(hidden, tokens, cache)`, and
  `return_hidden_states` on the forward pass. `sanitize` remaps checkpoint MTP
  weights (either `model.layers.{n}.*` or `mtp.*` naming) onto the `mtp.*`
  submodule when enabled, drops them under either naming when disabled, and
  loads already-stacked MLX-converted expert tensors.
- `generate.py`: `mtp_generate_step` — drafts one token with the MTP head,
  verifies it in the next trunk forward (up to 2 tokens per target pass),
  greedy verification so temp-0 output is identical to AR.
  `stream_generate` uses it automatically when the model exposes
  `num_nextn_predict_layers > 0` and no `draft_model` is passed.
- `tests/test_models.py`: `test_hy_v3_mtp` covering the MTP forward/draft
  shapes and both sanitize behaviors.

Measured on Hy3 295B (mixed-quant MLX, M5 Max 128 GB, warm cache,
64-token generations): outputs are **exactly identical** to AR (greedy
verification working as designed), but throughput is 0.54 tok/s vs
7.40 tok/s AR — the per-token draft→verify loop with per-step cache trims
dominates. **Frame this PR as a correctness reference for the Hy3 MTP
layer** (weights load, head drafts, parity holds), explicitly not a speed
win yet; batched verification is the follow-up that could make it one.
Consider marking the mtp path opt-in (flag) rather than automatic given
the measured regression.

Credit: MTP implementation originally by @eauchs
(eauchs/mlx-lm@hy_v3-mtp); base model support by @kernelpool (#1211).

## Checklist before opening
- [ ] #1211 merged (rebase this branch onto main once it lands)
- [ ] MTP smoke receipt from scripts/24 (fills the measured-numbers gap)
- [ ] Confirm @eauchs is credited appropriately / consider co-authoring
- [ ] pre-commit run --files on the three changed files
- [ ] PJB go-ahead to fork + push + open

## BLOCKERS before opening this PR (found by ultracode audit 2026-07-07)

The fork's generate.py mtp_generate_step has 3 criticals that must be fixed
before this becomes a real PR (they don't affect the AR-only artifacts we
ship, only the MTP self-speculative path this PR would upstream):

1. **Silently ignores sampler/logits_processors** — stream_generate
   auto-activates mtp_generate_step for any num_nextn_predict_layers>0 model
   with no draft; that path is greedy-argmax only. Every temperature>0 request
   silently becomes temp-0. Fix: gate behind an explicit opt-in flag AND/OR
   fall back to generate_step when a non-greedy sampler is requested.
2. **prompt_cache split is wrong** — mtp_generate_step does
   prompt_cache[:-1] (trunk) / [-1] (mtp), but make_prompt_cache(model)
   returns exactly num_hidden_layers entries (no +1 for mtp), so cache reuse
   indexes past the end / misassigns. Fix: build/derive the mtp cache slot
   explicitly, don't assume caller layout.
3. **First token yields logprobs=None** — unlike generate_step, breaking
   mlx_lm.server (indexes gen.logprobs[gen.token]) with TypeError. Fix: yield
   a real logprobs vector for the first token.

Also: mtp_generate_step ignores max_kv_size/kv_bits/prompt_progress_callback
(accepts but no-ops), the draft cache is never prefilled over the prompt, and
test_hy_v3_mtp exercises none of mtp_generate_step. Do NOT open the PR until
1-3 are fixed and tested against the real path.
