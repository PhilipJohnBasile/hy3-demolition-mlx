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

Measured on Hy3 295B (mixed-quant MLX, M5 Max 128 GB): [FILL IN after
scripts/24 smoke: AR tok/s vs MTP tok/s, speedup, outputs_match receipt].

Credit: MTP implementation originally by @eauchs
(eauchs/mlx-lm@hy_v3-mtp); base model support by @kernelpool (#1211).

## Checklist before opening
- [ ] #1211 merged (rebase this branch onto main once it lands)
- [ ] MTP smoke receipt from scripts/24 (fills the measured-numbers gap)
- [ ] Confirm @eauchs is credited appropriately / consider co-authoring
- [ ] pre-commit run --files on the three changed files
- [ ] PJB go-ahead to fork + push + open
