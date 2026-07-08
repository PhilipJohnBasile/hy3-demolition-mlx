# Tests

Pure-Python regression tests for the logic modules (no model load, no GPU).
Run: `.venv/bin/python -m pytest tests/ -q`

- `test_reap_core.py` — build_plan (ratio bounds, keep/drop partition, the
  protected-padding fix, REAP-mean ranking, expert-axis detection)
- `test_verifiers.py` — all four local verifiers incl. the whitespace-aware
  degeneration fix and the JSON cross-field extra_checks
- `test_compare_gate.py` — the promotion gate's PROMOTE/REVIEW/REJECT verdicts
  (incl. the D3 soul-flip-is-REVIEW-not-REJECT rule this suite surfaced)
- `test_behavioral_regressions.py` — frozen structural markers distilled from
  the blueprint judged experiments (docs/blueprint-ablation.md,
  docs/blueprint-verdict.md); reads saved receipts, no model load. Pattern
  from the Mimosa project (data-model-brain): judged finding once → cheap
  check forever.
- `test_toolcall_live_regression.py` — live-weights xfail test pinning the
  reap25 tool-call tag bug (data-model-brain
  findings/reap25-toolcall-regression.md). **Skipped by default**
  (loads an 87GB checkpoint); run with `RUN_LIVE_TESTS=1`. Matches the
  Mimosa lesson "deterministic-path green != model-path works" — this is
  the live-weights half no code-only check can substitute for.

Each test pins a real bug found/fixed on 2026-07-07 (see BUILD_NOTES) or a
real finding from a judged behavioral experiment (2026-07-08).
