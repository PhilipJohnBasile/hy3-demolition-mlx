# Hy3-Demolition-MLX Backlog

Status board for the endgame plan. Updated as work lands; receipts under
`eval/receipts/` are the source of truth for anything marked done.

Goal: two private MLX-only releases — **lite-v1** (done) and **reap25-v1**
(prune 25% of experts, requant, heal, promote only if it matches lite-v1's
eval pass rate). 40% only if 25% wins. Hard stop: no 70%+ pruning, no
aggregate-only saliency, no promoting a model that loses tool-call/code
reliability.

## Done

- [x] **lite-v1 promoted** (2026-07-07, tag `lite-v1`, commit `509fe41`)
  - Length-normalized SFT pack (305/16/16, ≤2048 tokens, quarantine deduped by hash)
  - LoRA 200 iters, 8 layers (16 OOMs on 128 GB), val loss 1.154 → 0.722
  - EOS fix: train against `models/hy3-mlx-base-ar-train` (`is_training` template view, script 17) — plain view teaches the model to never stop
  - Streamed lazy fuse (script 18; stock `mlx_lm fuse` eager-loads 105 GB and dies)
  - 5/5 agent eval with adapter and fused via `/v1/chat/completions`; 112.3 GB peak
  - Private HF `philipjohnbasile/hy3-demolition-mlx-lite-v1`, revision tag `lite-v1`, real model card (`cards/`)
- [x] Normalizer idempotency + proof receipt
- [x] Artifact model card auto-copied by fuse script
- [x] Receipt set made intentional (EOS-evidence pair committed)
- [x] Expanded eval suite: 30 cases — coding, direct strict-JSON (schema-checked),
      direct tool-call payloads (shape/args/no-prose), repair with failing tests,
      planning (no-fake-execution verifier), 11 protected souls
- [x] Local verifiers (`src/hy3_local_verifiers.py`), unit-tested; two tightening
      passes: whitespace-aware degeneration check, truncation = fail
      (`finish_reason` captured; planning/souls budget raised to 768)

## In flight

- [ ] **#11 Baseline suite vs lite-v1 fused** — part 1 (15 short-form cases): 15/15
      passed. Part 2 (4 planning + 11 souls) rerunning under the tightened
      harness, ETA ~3 h at 0.49 tok/s. Then: merge → summary (pass rate, peak
      memory, tok/s) → commit as the REAP comparison baseline.
      **REAP does not start until this is committed.**

## Phase 2 wrap-up (cheap, no GPU)

- [ ] **#12 Mechanical promotion gate** — `scripts/20_compare_receipts.py`:
      per-case diff of two receipt files (regressions/improvements/flips,
      pass-rate and tok/s deltas). Every REAP promotion decision runs through it.
- [ ] **#13 Pin mlx-lm commit** — RESTORE installs from a moving branch
      (`eauchs/mlx-lm@hy_v3-mtp`); record + pin exact commit, add environment
      receipt. EOS/fuse behavior is version-specific.
- [ ] **#14 Eval tiers** — document fast tier (15 short-form cases, ~20 min warm)
      for iteration vs full 30-case suite for promotion gates.

## Phase 2 wrap-up (GPU / human)

- [ ] **#15 Base-model baseline** — run the 30-case suite against plain
      `models/hy3-mlx-base-ar` (~4 h) to prove whether the LoRA added measurable
      value over base. Decide after #11 lands.
- [ ] **#16 Manual quality pass (PJB)** — drive
      `mlx_lm.chat --model dist/hy3-demolition-mlx-lite-v1-fused` on one real
      code task, planning task, tool-use task, creative/soul task. Human gate
      from the endgame plan; can't be automated.

## Phase 2.6: MTP sidecar (the speed lever)

Requirement: the released artifacts should run with the MTP sidecar
(multi-token prediction) — still a pure model directory, loaded by the
`hy_v3-mtp` mlx-lm fork with no wrapper. The sidecar exists in the base
checkpoint and was untouched by the LoRA; the AR-only view exists because the
MTP path previously tripped the Metal watchdog with the stock wired-limit pin.

- [ ] **#25 MTP smoke on base** — does MTP run with the no-wire patch? tok/s
      MTP vs AR + output-quality parity receipt. Blocked by #11 (GPU busy).
- [ ] **#26 MTP-enabled lite-v1** — attach `model-mtp.safetensors` + restore
      `num_nextn_predict_layers`; fast-tier eval must match AR verdicts;
      update HF artifact + card. Blocked by #25.
- [ ] **#27 MTP strategy for REAP** — carry the sidecar through
      prune/requant/heal or document reap artifacts as AR-only. Decision gates
      #20. Blocked by #25.

### MTPLX compatibility (hard requirement: artifacts must run on MTPLX)

Studied MTPLX v2.0.1 source (github.com/youssofal/MTPLX, Apache-2.0, native
macOS MLX server + CLI, MTP speculative decoding with exact rejection
sampling, OpenAI/Anthropic APIs). Findings:

- MTPLX **recognizes our architecture**: `HyV3ForCausalLM` / `hy_v3` maps to
  arch id `hy-v3-mtp` in its registry — but at support level
  `recognized-backend-pending`, tier "architecture-compatible-but-unverified",
  **can_run: False**. The hy_v3_mtp backend module does not exist yet.
- MTPLX pins **stock mlx-lm 0.31**, which has `hunyuan` (v1) but **no hy_v3**
  — it cannot load our trunk at all today. Our `eauchs/mlx-lm@hy_v3-mtp` fork
  is the working MLX implementation.
- Backend facades are ~50 lines; the real work is the family weight-injector
  patch + a runtime contract with hardware-measured exactness fixtures. vLLM's
  `hy_v3_mtp.py` is the named reference spec.

- [ ] **#28 Contribute hy_v3_mtp backend to MTPLX** — facade + injector patch
      + runtime contract fixtures; PR upstream. Blocked by #25, #29.
- [ ] **#29 hy_v3 trunk into stock mlx-lm** — upstream from the fork (or land
      the model class inside the MTPLX contribution). Prerequisite for #28.

## Phase 2.5: data before demolition

- [ ] **#17 Balanced facet import** — current pack is ~92% repair (music 4,
      perfumery 8, security 8 rows); wrong mix for healing pruned souls. Import
      from `agent-brain-blueprint`, `tinygpt-souls`, `agent-toolkit` per
      `data/build_time_sources.json`; verifier-gate, facet-tag,
      length-normalize. Blocks the REAP heal (#22).

## Phase 3: REAP 25% (each step gated on the previous)

- [ ] **#18 Calibration** — streamed saliency with soul buckets for all 11
      protected facets → `dist/hy3-reap-saliency-v1.json`. Blocked by #11.
- [ ] **#19 Dry-run prune plan** — ratio 0.25, min keep 8 experts per protected
      facet per layer; reject if any soul bucket missing. Commit plan receipt
      **before** writing weights.
- [ ] **#20 Apply prune + smoke** — `dist/hy3-reap25-pruned`, keep
      `num_experts_per_tok=8`, direct MLX smokes with receipts.
- [ ] **#21 Mixed requant + smoke** — experts 3-bit; attention/router/shared/
      norm/head 8-bit; group 64 → `dist/hy3-reap25-requant`.
- [ ] **#22 Heal + fuse** — LoRA on the balanced pack (+ REAP eval failures),
      trained via the `-train` template view, 8 layers, 200→500 iters only if
      healthy; streamed fuse → `dist/hy3-demolition-mlx-reap25-v1-fused`.
      Blocked by #21 and #17.
- [ ] **#23 Evaluate + promote** — full suite vs reap25; `compare_receipts`
      against the lite-v1 baseline (match-or-beat pass rate; memory, load time,
      tok/s recorded); manual soul review. Promote (tag `reap25-v1`, private HF)
      only if materially better for daily use. Blocked by #22 and #12.

## Phase 4: conditional

- [ ] **#24 REAP 40%** — only if reap25-v1 promotes. Same protections, same
      pipeline, same gates.

## Expected payoff (why we're doing this)

| Artifact | Size | Peak memory | Fits alongside real work? |
|---|---|---|---|
| lite-v1 (now) | 104 GB | 112.3 GB | Barely — owns the machine |
| reap25 (target) | ~80 GB | ~85–90 GB | Yes, with headroom |
| reap40 (stretch) | ~65 GB | ~70–75 GB | Comfortable |

Decode speed stays ~top-8 × 21B active params regardless of pruning — REAP
buys memory headroom, not tok/s. The tok/s levers are the MTP sidecar
(#25/#26, potentially 1.5–2× decode), prompt caching (on), and memory
pressure relief.
