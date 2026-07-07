# Comment for ml-explore/mlx-lm#1211 (Add Hy3)

> Status: POSTED 2026-07-07 with PJB's go-ahead —
> https://github.com/ml-explore/mlx-lm/pull/1211#issuecomment-4907074517
> (final text scoped the eval claim to the verified 19 cases; souls half of
> the suite was still running at post time)

---

Validation report from running this PR's architecture on real hardware, plus
one small robustness suggestion.

**Setup:** Hy3 295B MoE (mixed-quant MLX build,
`ox-ox/Hy3-295B-Instruct-w2q3exp-AProjQ8-SExpQ8-OutQ8-MTP-mlx`, ~105 GB) on an
M5 Max / 128 GB, macOS. Model code from this PR (via the `eauchs/mlx-lm`
`hy_v3-mtp` branch, which builds on it).

**What worked:**
- Load + generate, AR-only view (MTP sidecar omitted,
  `num_nextn_predict_layers=0`): coherent output, clean EOS stops, peak
  memory 112.3 GB, ~1.4 tok/s cold decode.
- `mlx_lm.server` OpenAI-compatible endpoint: 30-case agent eval (coding,
  tool-calls, strict JSON, repair, planning) all passing at temp 0.
- LoRA fine-tune (rank 8, 8 layers via `mlx_lm lora`) + fuse: works; one
  template note below.

**Two field notes for anyone using this with chat fine-tuning / converted checkpoints:**
1. The chat template only appends `eos_token` to the final assistant turn when
   `is_training` is set. `ChatDataset` in the tuner applies the template with
   no kwargs, so SFT on chat data silently teaches the model to never emit
   EOS (we saw token-0 spam after answers until training against a template
   view that defaults `is_training=true`).
2. MLX-converted checkpoints can carry the MTP weights as a separate
   `model-mtp.safetensors` whose tensors are named `mtp.*` (not
   `model.layers.80.*`). The current `sanitize` only drops the
   `model.layers.{n+i}.*` naming, so such checkpoints fail strict loading.
   One-line fix: also drop keys starting with `"mtp."`.

Happy to share receipts (JSON) for any of the numbers above.

---

Notes to self before posting:
- Verify the eauchs fork's relationship to this PR is fairly described.
- Attach or link specific receipts if asked: hy3_ar_smoke_16.json,
  hy3_server_smoke.json, hy3_lite_v1_train.json, hy3_environment.json.
