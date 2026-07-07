# REAP Decision Rubric

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

## D3. reap25 promotion (#23)

Mechanical gate first: `scripts/20_compare_receipts.py baseline candidate`
must say PROMOTE (no hard-domain regressions, pass rate >= lite-v1).
Then the judgment layer:

- **Hard tier is the sensitive instrument.** lite-v1's hard-tier score is
  the reference. reap25 losing 1 hard case = REVIEW (read the outputs);
  losing 2+ = do not promote, heal more or keep lite-v1.
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

## D5. MTP strategy for REAP artifacts (#27) — pre-decided, pending one fact

The sidecar's MTP layer owns a private 192-expert MoE, structurally
independent of trunk experts. Recommendation: **carry the sidecar through
REAP unpruned** (it is 1.4 GB — pruning it saves ~350 MB at 25%, not worth
a separate saliency pass), requantize it with the same mixed policy, and
verify draft-acceptance rate on the healed model. The one fact that can
overturn this: if the MTP smoke (#25) shows the draft path is broken or
acceptance is very low even on the base model, ship reap artifacts AR-only
and note it in the card.

## D6. Heal data priorities (measured, not guessed)

From the lite-v1 baseline failures: design/dashboard vocabulary, long-form
scope discipline (fullstack truncation), soul-tag echo at answer start.
Canon batch 3 (data/hy3_canon_sft/rows_b3.jsonl) targets exactly these.
If reap25 evals surface new failures, add rows for those before extending
training iterations — targeted rows beat more epochs on repair-heavy data.
