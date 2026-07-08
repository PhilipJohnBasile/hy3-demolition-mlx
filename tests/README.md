# Tests

Pure-Python regression tests for the logic modules (no model load, no GPU).
Run: `.venv/bin/python -m pytest tests/ -q`

- `test_reap_core.py` — build_plan (ratio bounds, keep/drop partition, the
  protected-padding fix, REAP-mean ranking, expert-axis detection)
- `test_verifiers.py` — all four local verifiers incl. the whitespace-aware
  degeneration fix and the JSON cross-field extra_checks
- `test_compare_gate.py` — the promotion gate's PROMOTE/REVIEW/REJECT verdicts
  (incl. the D3 soul-flip-is-REVIEW-not-REJECT rule this suite surfaced)

Each test pins a real bug found/fixed on 2026-07-07 (see BUILD_NOTES).
