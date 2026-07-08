# BUILD NOTES — Hy3-Demolition-MLX

Running dialog of what we did right and wrong, in the spirit of
glm52-demolition's BUILD_NOTES.md (which saved this project real time twice
on 2026-07-07 alone — see incidents 12 and 13). Append, don't rewrite;
misdiagnoses stay in the record with their corrections.

## Tool lineage — who built what

- **OpenCode** (early July 2026): built `agent-brain-blueprint` — the
  operating-stance curriculum that seeds the SFT behavior rows and (via
  CLAUDE.md) governs how the agents on this project work.
- **Codex** (2026-07-06 23:16 → 07-07 ~08:40, session on record): project
  charter and foundation. Set the rules (MLX-native, no GGUF/CUDA; "in the
  end we just want to run the model and nothing else"; MTPLX compatibility
  was a day-one ask, parked with "forget about mtplx and lets just focus on
  mlx"). Fought the Metal watchdog (below), invented the no-wire AR-only
  view, initialized private git+HF, imported the GLM data, took PJB's
  mid-session directive "when pruning be sure to look at our souls and dont
  prune one of our souls" into the soul-preserving REAP design, wrote the
  endgame plan, then handed off: "im going to do this plan in claude and
  you stay on as the reviewer".
- **Claude (Fable 5)** (2026-07-07, this file's first author): executed and
  extended the plan — lite-v1 promoted, eval suite ×6, REAP corrected to
  the real criterion, ecosystem PRs prepped, decision rubric written.
- **PJB**: the directives that shaped everything ("pure model", souls,
  MTPLX + LM Studio musts, "use all the chips but leave breathing room"),
  the GLM scar tissue, and the go/no-go calls.

## Incident log

1. **Metal watchdog kills on load (Codex).**
   `kIOGPUCommandBufferCallbackErrorTimeout` twice while loading the 105 GB
   checkpoint, even with memory freed. *Cause:* MLX's wired-limit pin
   while wiring ~110 GB. *Fix:* patch out `wired_limit` (no-wire) + an
   AR-only symlink view that omits the 1.3 GB MTP sidecar and sets
   `num_nextn_predict_layers=0`, avoiding the self-speculative path.
   *Lesson:* on 128 GB, loading is a negotiated settlement, not an event.

2. **MTPLX 2.0.0 tried empirically (Codex).** Patched MTPLX's bundled
   runtime with `hy_v3.py` as a reversible experiment. Verdict:
   `recognized-backend-pending, can_run: false` — MTPLX knows the
   architecture, ships no backend. Claude's later source-read of 2.0.1
   independently confirmed, then prototyped the missing backend
   (docs/upstream/). *Lesson:* the experiment and the source-read agreed;
   do both when cheap.

3. **zsh ate the env script (Claude).** `00_env.sh` used `BASH_SOURCE`,
   empty under zsh → wrong Python → first training launch died on a
   missing module. *Fix:* `${BASH_SOURCE[0]:-$0}`. *Lesson:* sourced
   scripts get the user's shell, not the shebang.

4. **16-layer LoRA OOM (Claude).** Plan said `--num-layers 16`; Metal OOM
   at ~iter 70. 8 layers fits (~113 GB peak) and healed fine (val
   1.154→0.722). *Lesson:* training profiles are hypotheses until they
   survive iteration 100.

5. **THE EOS BUG (Claude — the big one).** First 200-iter adapter answered
   correctly then spammed token-0 `!` forever. *Cause:* Hy3's chat template
   appends `eos` to the final assistant turn only when `is_training` is
   set; mlx_lm's ChatDataset passes no kwargs → 200 iterations of
   "assistant turns never end". *Fix:* `models/hy3-mlx-base-ar-train`
   (script 17), a view whose template defaults `is_training=true`. Train
   against the train view, fuse against the stock view. *Lesson:* an SFT
   pipeline that silently drops EOS trains an un-stoppable model; check
   tokenized sequence ENDINGS before training, not after.

6. **Stock `mlx_lm fuse` cannot fuse this model (Claude).** Eager-loads all
   105 GB; first attempt SIGKILLed, retry crawled at 0 GB free through
   swap. *Fix:* script 18 — lazy load, fuse as graphs, `save_model`
   evaluates shard-by-shard: 22 s, 44 GB peak. *Lesson:* for
   bigger-than-RAM artifacts, streaming isn't an optimization, it's the
   difference between works and dies.

7. **Monitor deleted the receipts inode (Claude, self-inflicted).** A
   results monitor `rm`/`touch`ed the receipts file the eval had already
   opened → 14 receipts written to a deleted inode, unrecoverable. *Fix:*
   monitors poll line-counts; never recreate files another process owns.
   *Lesson:* observation must not mutate the observed.

8. **Server readiness lies (Claude).** `mlx_lm.server` binds its port
   before loading; `/v1/models` answers instantly, the first request then
   stalls ~14 min. An eval client with a 600 s socket timeout died mid
   long-form case. *Fix:* 1800 s+ timeouts; treat first-request stall as
   load. *Lesson:* "port open" and "model loaded" are different facts.

9. **Verifier false positives, three flavors (Claude).** (a) Degeneration
   regex flagged repeated `'  '` indentation in good code — whitespace
   spans now exempt; (b) `soul_design` keyword list was graphic-design
   vocabulary while the prompt asked dashboard IA — the model's good answer
   failed the instrument, recalibrated against the recorded output; (c)
   `plan_db_migration` hit its token cap mid-answer and PASSED — now
   `finish_reason` is recorded and truncation fails unless the case allows
   it. *Lesson:* baselines measure the instrument as much as the model;
   calibrate cases before freezing, and record enough (finish_reason,
   tokens, elapsed) to re-adjudicate later.

10. **Requant script would have corrupted the model (Claude, caught in
    audit BEFORE any run).** Two bugs: it quantized unquantized RMSNorm
    vectors (unloadable), and its quant-param lookup used storage names
    (`mlp.experts.*`) against a config keyed by runtime names
    (`mlp.switch_mlp.*`) — every expert missed, fell back to global
    bits=2, silently dequantizing the 3-bit down_proj as garbage. *Fix:*
    requant only what has scales; translate names AND cross-check bits
    against packed uint32 shapes. Verified on real tensors (0.9% roundtrip
    err). *Lesson:* audit never-run pipeline code against real checkpoint
    headers before spending GPU-days; storage names ≠ runtime names.

11. **We weren't actually doing REAP (Claude, research pass).** Calibration
    summed router gate scores; the real criterion (arXiv:2510.13999) is the
    MEAN of gate × ‖expert output‖ per routed token — sums conflate
    frequency with impact and mis-rank exactly the rare soul experts this
    project protects. *Fix:* hook the MoE forward, record `reap_sum` +
    `counts`, rank by the mean (hook verified bit-identical to the
    unhooked forward). Same pass imported two community findings on our
    exact model (mlx-lm PR #1211 thread): the reference computes the
    router matmul in **fp32** (bf16 flips top-8 selections — patched into
    our fork; calibration must not measure wrong routing), and
    `router.gate` should stay unquantized (~58 MB). *Lesson:* name-check
    your methods against their papers before running them.

12. **Requant would have cancelled the prune (arithmetic).** Experts are
    ~101/105 GB at native 2/2/3-bit; "requant experts to 3-bit" grows
    gate/up 50% → 0.75 × 9/7 ≈ 0.96 of original size. *Fix:* D2b — reap25
    default is prune-only, native quant kept (~80 GB expected). *Lesson:*
    multiply before you build.

13. **PJB's memory said "3-bit was a disaster on GLM" — the record says
    more.** GLM v1 broke (hallucinating, sentence-looping) with THREE
    variables at once: 3-bit requant + 77% prune + generic calibration;
    the shipped card still concluded "3-bit just below the quality cliff,
    4-bit just above it and MLX's best kernel." And GLM's infamous
    iter-1 "Val loss nan" was misdiagnosed as quantization FOUR times —
    real cause: `--mask-prompt` with rows whose mlx_lm-computed completion
    span is zero tokens → 0/0. *Carried into Hy3 (D2/D2b):* never requant
    to 3-bit; if a heal NaNs at iter 1, drop `--mask-prompt` first; GLM's
    microbench (3-bit 158 µs < 4-bit 220 µs, bandwidth-bound) means any
    future bit-width choice is a quality call, not a speed call.
    *Lesson:* human memory carries the warning, the written record carries
    the mechanism — keep both, trust the record.

## What worked (keep doing these)

- **Receipts-first.** Every claim has a JSON receipt; the EOS bug, the
  truncation bug, and the verifier miscalibrations were all caught by
  receipts, none by vibes.
- **Audit before GPU.** Incidents 10–12 cost about an hour of reading and
  saved days of compute and at least one corrupt artifact.
- **Dry-run gates.** Plan-before-write (05), analyzer verdicts (25),
  promotion gate (20) — decisions are mechanical, judgment is
  pre-committed in DECISIONS.md.
- **One GPU lane, saturated CPU lanes.** The entire eval/verifier/data/
  ecosystem buildout happened while the GPU ran evals.
- **Cross-tool relay.** Codex planned and reviewed, Claude executed, PJB
  arbitrated; the GLM repo (prior project) served as institutional memory.
  This file extends that memory for whoever (or whatever) works next.

## State as of 2026-07-07 evening

- **lite-v1**: promoted, tagged, on private HF. 104 GB / 112.3 GB peak /
  ~1.4 tok/s. Baseline: 15/15 short-form, planning 4/4, souls passing
  under recalibrated verifiers (design/fullstack amended + rerun); hard
  tier + MTP smoke + base-model comparison running in the automated chain.
- **REAP**: calibration pack + true-REAP hook + plan analyzer +
  decision rubric ready; ~80 GB prune-only target next.
- **Ecosystem**: mlx-lm hy_v3 base PR (#1211) active upstream with our
  validation posted; MTP follow-up patch and MTPLX backend prototype
  staged in docs/upstream/, gated on external merges.

14. **MTP self-speculation: correct but 13.7× SLOWER (chain smoke,
    2026-07-07 17:17).** The fork's mtp_generate_step drafts and verifies
    with exact output parity (lossless as designed) — at 0.54 tok/s vs
    7.40 tok/s plain AR. The per-token draft→verify loop with per-step
    cache trims eats far more than the drafted token saves. *Consequences
    (D5):* all fused artifacts ship AR-only; the MTP speed play moves to
    MTPLX's batched-verify backend (#28); the upstream PR reframed as a
    correctness reference with honest numbers. *Bonus finding:* warm AR
    decode is **7.4 tok/s** — the 1.4 tok/s number everyone planned around
    was cold-load page-cache behavior. *Lesson:* "has the feature" and
    "the feature helps" are separate receipts; and always re-measure
    baselines warm.

15. **Base server jetsam-killed by memory competition (2026-07-07 ~20:00,
    self-inflicted).** While the 112 GB base model was resident and serving
    the #15 baseline eval, an ultracode audit ran 34 parallel agents + pytest
    suites in scratch clones. Memory pressure spiked and the OS SIGKILL'd the
    base server (no error logged — classic jetsam). The chain's stage 5 then
    ignored the eval's nonzero rc (audit finding, same run) and committed a
    meaningless 24-vs-30 comparison, pushed publicly. Marked the receipt
    INVALID with reason; #15 needs a clean re-run (optional, non-blocking —
    calibration is independent and proceeded fine). *Lessons:* (1) do NOT run
    memory-heavy build/audit work alongside a resident bigger-than-half-RAM
    model — they compete and the OS picks a loser; pause one or the other.
    (2) An orchestration step that ignores return codes will faithfully
    publish garbage; the audit called this exact bug and it fired within
    hours — fix rc-handling in 27 before reuse.

## What worked (addendum)

- **Tiny-checkpoint end-to-end prune test (2026-07-07 night).** Before the
  real 295B prune, ran scripts/05 on a KB-sized quantized hy_v3 checkpoint
  built in the true on-disk format (stacked quantized experts, router.gate,
  expert_bias). Confirmed: experts pruned on axis 0 with quant packing
  intact, router/bias shrunk consistently, config rewritten, and the pruned
  model LOADS + FORWARD-RUNS. De-risked the one never-run weight-writing step
  in seconds instead of discovering a misalignment after GPU-hours. Reusable
  pattern: test weight-mutating scripts on a tiny faithful checkpoint, not
  just by reading them (the GLM "reproduce on the real failing path" lesson,
  applied preventively).

- **SFT data leakage audit + D2-on-REAP verification (2026-07-07 night, CPU).**
  scripts/31 audits the combined pack: ZERO train/valid/test leakage (exact or
  identical-user-prompt), no intra-train duplicates, no malformed rows — the
  eval numbers are not compromised. (Facet balance is 76% repair — known, from
  the repair-heavy GLM import; canon batches mitigate.) Separately verified
  script 17 builds a working is_training train-view from a *pruned* dir, so the
  REAP heal (#22) cannot recreate the EOS bug (incident 5) on the pruned path.

- **Eval-harness code-execution safety documented (2026-07-07, public-repo
  hygiene).** The verifier RUNS model-generated code (subprocess python +
  30s timeout in a temp cwd) — NOT a sandbox; it executes with the user's
  full privileges. Safe for our use (own model, own curated harnesses, own
  machine); documented in SECURITY.md with a sandbox-it warning for anyone
  running the harness on untrusted output. Runtime (the fused model) executes
  nothing — the risk is entirely in the build/eval path, and only when
  pointed at output you don't control.

## 16. The "think leak" was a serving-mode default, not a defect (2026-07-08)

The manual-pass side-by-side (script 35) showed both lite-v1 and reap25
"leaking" chain-of-thought into answers and, on long prompts, running out of
tokens mid-reasoning. Investigation (script 36 probe + reading
chat_template.jinja) found the cause: Hy3 is a **reasoning model** whose
template defaults `reasoning_effort='high'`. At high/low the generation prompt
ends with an OPEN `<think:opensource>`, so the model reasons first, emits
`</think:opensource>`, then the answer. `apply_chat_template` with no
`reasoning_effort` (as script 35 and stock `mlx_lm.server` both do) therefore
serves raw reasoning. The `</think:opensource>` "artifact" at answer start is
just the think-close token.

Probe result (eval/receipts/hy3_think_mode_probe.md): `reasoning_effort='no_think'`
pre-closes the think block (`<think:opensource></think:opensource>` in the
prompt) and yields clean, correct, direct answers in ~half the time. Same
correctness, no leak. NOT a prune artifact — lite-v1 and reap25 behave
identically.

Implication: this is a serving-config choice, not a retrain. For an interactive
daily driver, serve `no_think` (fast, clean) or `low`; keep `high` for hard
agent tasks where the eval showed reasoning helps. Whether to change the
SHIPPED template default (currently high) is a product call — high maximizes
eval correctness, no_think maximizes clean UX. Documented in the model card;
default unchanged pending PJB's decision.
