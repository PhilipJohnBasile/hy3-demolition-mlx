# REAP Decision Rubric

## D0. Research grounding (2026-07-07)

Validated against the REAP paper (arXiv:2510.13999, Cerebras; 20B–1T MoEs)
and community Hy3 work:

- **Criterion**: true REAP saliency is `mean over routed tokens of
  g_j(x)·||f_j(x)||₂` — the mean decouples frequency from impact, so
  rare-but-strong experts (our soul concern) score correctly. Calibration
  (script 04) records `reap_sum` + `counts`; planning ranks by the mean.
  Gate-only `score_sum` is kept for comparison, not ranking.
- **Expectations**: one-shot 25% prune ≈ −2.8% on codegen (no healing);
  50% ≈ −8% and clearly degraded. Our 25%→40% ladder with healing sits on
  the safe side of both numbers. Late layers are most sensitive — read the
  analyzer's per-layer output with that prior.
- **Calibration data**: domain-matched calibration is critical (C4-style
  generic calibration collapsed their code models to 0%). Our pack is
  domain-matched by construction; volume (~1k prompts) matches their
  small-model floor, well short of their 12k×16k large-model recipe —
  acceptable trade for 0.5 tok/s hardware, noted honestly.
- **Routing numerics** (avlp12, mlx-lm PR #1211 thread): the reference
  implementation computes the router matmul in fp32; bf16 flips top-8
  selections on tight margins. Applied to our installed fork (see RESTORE);
  calibration must run with this patch or saliency is measured on wrong
  routing. Same source: keep `router.gate` unquantized in requant
  (~58 MB) — implemented in script 06.

Pre-committed judgment for the decisions between here and reap40, written
2026-07-07 with full session context. Change it deliberately, not casually —
each rule exists because of something measured or a failure already hit.

## D1. Prune plan acceptance (#19)

Run `scripts/25_analyze_reap_plan.py plan.json --saliency saliency.json`.

- **REJECT** from the analyzer is final — do not override structure or
  protected-coverage failures by hand-editing a plan.
- **REVIEW** flags get a written note in the plan receipt before `--write`:
  what the flag was, why proceeding is safe. Two specific interpretations:
  - `protection_pressure > 60%` in a few layers is expected in late layers
    (souls concentrate); in MANY layers it means min_keep=8 × 11 facets is
    eating the budget — reduce overlap by checking `soul_concentration`
    first, not by lowering min_keep.
  - `saliency_mass < 70%` in isolated layers: acceptable if those layers'
    routing is flat (check counts); in >10 layers it means the calibration
    pack is too narrow — recalibrate before pruning, never prune anyway.

## D2. The heal EOS landmine (#22) — DO NOT SKIP

Training the heal LoRA directly on `dist/hy3-reap25-requant` re-creates the
EOS bug that broke the first lite adapter: the stock chat template only
appends EOS when `is_training` is set. Before ANY heal training:

```bash
./scripts/17_prepare_hy3_train_view.py --source dist/hy3-reap25-requant \
  --out dist/hy3-reap25-requant-train
```

Train against the `-train` view; fuse against the stock-template dir.
Smoke MUST include a stop-behavior check (generate 64 tokens, expect
finish before cap with no trailing repetition).

**Second heal landmine, from the GLM record (misdiagnosed FOUR times
there):** `--mask-prompt` divides loss by completion-token count, and
mlx_lm's own prompt/completion boundary can yield ZERO completion tokens
for some rows → 0/0 → "Val loss nan" at iter 1. Our script 07 passes
--mask-prompt (fine on the current pack — lite-v1 trained clean), but if
any heal shows iter-1 NaN: switch to full-sequence loss (drop
--mask-prompt) FIRST. Do not blame quantization — that misdiagnosis cost
the GLM project its biggest time-sink and an over-correction spiral.

## D2b. Requant is OPTIONAL for reap25 — default is skip (#21)

Arithmetic, 2026-07-07: experts are ~101 of 105 GB and already sit at
gate/up 2-bit, down 3-bit. The inherited "experts 3-bit" requant policy
would GROW gate/up by 50%, canceling the prune: 0.75 × 9/7 ≈ 0.96 of
original size. Therefore:
- **Default reap25 path: prune only.** Pruned tensors keep their original
  quantization; skip script 06 entirely (saves hours, avoids a
  dequant→requant noise cycle). Expected ~80 GB disk / ~85-90 GB peak.
- Requant exists for pushing SMALLER (e.g. down 3→2-bit toward ~70 GB) and
  only runs as a separate candidate compared against the prune-only
  artifact — never bundled into the first reap25 build.
- If requant does run: router.gate stays unquantized bf16 (script 06 does
  this now), and remember the stacked-loss caveat in D3.
- **Bit-width choice is a benchmark, not a default** (PJB, 2026-07-07):
  Metal kernel paths favor power-of-2 widths — 4/8-bit pack uint32 cleanly,
  3-bit needs cross-word extraction. But decode may be bandwidth-bound
  (fewer bits = fewer bytes/token), and 4-bit experts don't fit until after
  a deep prune (~175 GB unpruned, ~110 GB at reap25, viable ~88 GB at
  reap40). Any requant candidate benchmarks 3-bit vs 4-bit (incl. mxfp4
  mode, which has first-class Metal kernels) on tok/s + evals before
  choosing. The native 2/2/3 checkpoint remains the measured reference.
- **PRIOR EVIDENCE — the GLM demolition record (checked 2026-07-07, repo
  PhilipJohnBasile/glm52-demolition + HF q4a4-soul card):**
  - v1 (3-bit requant + **77% prune** + generic calibration) "broke —
    hallucinates, sentence-loops". v2/v3 (4-bit, 77% prune, code/soul
    calibration) shipped. The shipped card's verdict: *"3-bit was just
    below the quality cliff; 4-bit is just above it and MLX's
    best-optimized kernel."* → **never requant to 3-bit** stands.
  - Nuance the table hides: v1 changed three variables at once (bits,
    calibration, and a 77% prune — far past the REAP paper's 50% cliff).
    Hy3's 25% prune-only default avoids that entire regime; and Hy3's
    NATIVE 2/3-bit works because the original builder quantized once from
    full precision — the rule is about OUR dequant→requant, not about odd
    bit-widths per se.
  - GLM's decode microbench: 3-bit matmul 158µs vs 4-bit 220µs — decode is
    bandwidth-bound, smaller was FASTER. 4-bit won on quality + kernels,
    not speed. Keep this when benchmarking any requant candidate.

## D3. reap25 promotion (#23)

Mechanical gate first: `scripts/20_compare_receipts.py baseline candidate`
must say PROMOTE (no hard-domain regressions, pass rate >= lite-v1).
Then the judgment layer:

- **The BRUTAL tier is the primary sensitive instrument** (added
  2026-07-07 evening; `eval/brutal/prompts.jsonl`). The 8-case hard tier
  scored lite-v1 8/8, so it has no discriminating power — a pruned model
  that quietly lost capability could still score 8/8 there. The brutal tier
  (multi-constraint code, two-bug repair, cross-field JSON, tool-trap,
  safety-refusal planning, named-theory souls) is designed to register
  partial capability loss as partial failure. GATE: establish lite-v1's
  brutal-tier baseline FIRST (run when GPU frees, before reap25); then
  reap25 losing 1 brutal case = REVIEW (read outputs), losing 2+ = do not
  promote (heal more or keep lite-v1). The old hard tier stays as a
  secondary check but is not decisive on its own.
- **Soul flips**: a single soul flip either way is noise (keyword
  heuristics); 2+ soul regressions = treat as real damage, check whether
  the flipped facets match low-`saliency_mass` layers from the plan report
  — if yes, the prune ate those souls and healing may not recover them.
- **"Materially better for daily use" means**: peak memory <= ~90 GB
  (leaves ~35 GB for real work on the 128 GB machine) AND no gate failures.
  tok/s is NOT expected to improve (top-8 × 21B active is unchanged) — do
  not hold the promotion hostage to speed.
- Manual quality pass (PJB) stays mandatory: one code, one planning, one
  tool-use, one creative/soul prompt, judged side-by-side vs lite-v1.

## D4. reap40 go/no-go (#24)

Go only if ALL of:
1. reap25 promoted cleanly (no REVIEW overrides needed at D3).
2. reap25's hard-tier score matched lite-v1 exactly (a model that already
   lost headroom at 25% will fall off a cliff at 40%).
3. The 25% plan's `saliency_mass` averaged >= 85% — meaning routing was
   concentrated and there was genuinely dead weight to cut.
Stop conditions (from the endgame plan, restated as hard rules): never 70%+,
never aggregate-only saliency, never promote with tool-call/code regressions.

## D5. MTP strategy — RESOLVED by the smoke (2026-07-07 17:17)

Measured (eval/receipts/hy3_mtp_smoke.json): the fork's self-speculative
path is **correct** (exact output parity with AR) but **13.7× slower** —
0.54 tok/s vs 7.40 tok/s AR on identical warm-cache prompts. The per-token
draft→verify loop with cache trims is the bottleneck, not the heads.

Decision: **all fused artifacts ship AR-only** (no sidecar, nextn=0). The
sidecar stays preserved in the base checkpoint; the MTP speed play moves
entirely to the MTPLX backend (#28), whose batched-verify runtime is
engineered for these heads. The mlx-lm follow-up PR gets reframed
honestly: a correctness reference for the MTP layer, with the measured
numbers stated, not a speed claim.

Bonus fact from the same receipt: **warm AR decode is 7.4 tok/s** — the
oft-quoted 1.4 tok/s was cold-load page-cache behavior. Daily-driver and
eval planning should use the warm number after first generation.

## D6. Heal data priorities (measured, not guessed)

From the lite-v1 baseline failures: design/dashboard vocabulary, long-form
scope discipline (fullstack truncation), soul-tag echo at answer start.
Canon batch 3 (data/hy3_canon_sft/rows_b3.jsonl) targets exactly these.
If reap25 evals surface new failures, add rows for those before extending
training iterations — targeted rows beat more epochs on repair-heavy data.
