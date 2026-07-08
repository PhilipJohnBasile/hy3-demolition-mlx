"""Regression tests for local verifiers (src/hy3_local_verifiers.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hy3_local_verifiers import (
    verify_json_schema, verify_tool_call, verify_no_fake_execution,
    verify_soul, _is_degenerate,
)

OK = {"schema": {"type": "object", "required": ["ok", "reason"],
      "additionalProperties": False,
      "properties": {"ok": {"type": "boolean"}, "reason": {"type": "string"}}}}


def test_json_accepts_valid_rejects_invalid():
    assert verify_json_schema('{"ok": true, "reason": "x"}', OK)[0]
    assert verify_json_schema('```json\n{"ok": true, "reason": "y"}\n```', OK)[0]
    assert not verify_json_schema('{"ok": "yes", "reason": "x"}', OK)[0]   # wrong type
    assert not verify_json_schema('{"ok": true, "reason": "x", "z": 1}', OK)[0]  # extra key
    assert not verify_json_schema('not json', OK)[0]


def test_json_cross_field_extra_check():
    spec = {"schema": {"type": "object", "required": ["start", "end", "span"],
            "properties": {"start": {"type": "integer"}, "end": {"type": "integer"},
                           "span": {"type": "integer"}}},
            "extra_checks": "span_equals_end_minus_start"}
    assert verify_json_schema('{"start":3,"end":10,"span":7}', spec)[0]
    assert not verify_json_schema('{"start":3,"end":10,"span":99}', spec)[0]
    assert not verify_json_schema('{"start":10,"end":3,"span":-7}', spec)[0]  # start<end


def test_tool_call_shape_and_trap():
    spec = {"tools": ["db.query", "calc.eval"], "expected_tool": "db.query",
            "required_args": ["sql"], "arg_types": {"sql": "string"}, "forbid_prose": True}
    assert verify_tool_call('{"tool": "db.query", "args": {"sql": "SELECT 1"}}', spec)[0]
    assert not verify_tool_call('{"tool": "calc.eval", "args": {"sql": "x"}}', spec)[0]  # wrong tool
    assert not verify_tool_call('{"tool": "db.query", "args": {}}', spec)[0]  # missing arg
    assert not verify_tool_call('Sure!\n{"tool": "db.query", "args": {"sql": "x"}}', spec)[0]  # prose


def test_no_fake_execution():
    spec = {"min_steps": 3, "must_mention": ["retry"]}
    good = "1. Inspect the error.\n2. Retry with backoff.\n3. Report to the user."
    assert verify_no_fake_execution(good, spec)[0]
    assert not verify_no_fake_execution("1. I ran it and tests pass.\n2. Done.\n3. Retry.", spec)[0]
    assert not verify_no_fake_execution("just retry", spec)[0]  # too few steps


def test_degeneration_ignores_whitespace_flags_real_loops():
    # indented code must NOT be flagged
    assert not _is_degenerate("    if x:\n        y()\n" * 12)
    # token-0 spam and word loops MUST be flagged
    assert _is_degenerate("def f(): return 1" + "!" * 60)
    assert _is_degenerate("la " * 60)


def test_soul_keyword_and_degeneration():
    spec = {"keywords": ["chord", "tonic", "progression"], "min_keywords": 2, "min_chars": 100}
    good = ("The I-V-vi-IV progression resolves because the tonic chord anchors home "
            "and each chord both tenses and points back to it. " * 2)
    assert verify_soul(good, spec)[0]
    assert not verify_soul("A story about a dog at the park. " * 4, spec)[0]  # off-topic
    assert not verify_soul("chord " * 60, spec)[0]  # degenerate
