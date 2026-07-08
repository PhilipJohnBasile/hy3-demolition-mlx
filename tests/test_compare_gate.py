"""Regression tests for the promotion gate (scripts/20_compare_receipts.py)."""
import json, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "bin" / "python")


def _write(rows):
    p = Path(tempfile.mktemp(suffix=".jsonl"))
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _run(base, cand):
    r = subprocess.run([PY, str(REPO / "scripts" / "20_compare_receipts.py"),
                        str(base), str(cand)], capture_output=True, text=True)
    return r.returncode, json.loads(r.stdout)


def _case(cid, domain, passed):
    return {"id": cid, "domain": domain, "passed": passed, "stage": "ok",
            "diag": "", "elapsed_s": 1.0, "completion_tokens": 10}


def test_promote_when_candidate_matches():
    rows = [_case("a", "python", True), _case("b", "json", True)]
    rc, rep = _run(_write(rows), _write(rows))
    assert rc == 0 and rep["verdict"] == "PROMOTE"


def test_reject_on_hard_domain_regression():
    base = [_case("a", "python", True), _case("b", "json", True)]
    cand = [_case("a", "python", False), _case("b", "json", True)]  # regressed a
    rc, rep = _run(_write(base), _write(cand))
    assert rc == 1 and rep["verdict"] == "REJECT"
    assert rep["regressions"][0]["id"] == "a"


def test_soul_regression_not_gating_by_default():
    base = [_case("s", "soul", True), _case("a", "python", True)]
    cand = [_case("s", "soul", False), _case("a", "python", True)]  # soul flip only
    rc, rep = _run(_write(base), _write(cand))
    # D3: a lone soul flip is not decisive -> REVIEW (human reads), not REJECT,
    # and NOT dragged to REJECT by the overall pass-rate drop. rc 0 = not a hard fail.
    assert rep["verdict"] == "REVIEW" and rc == 0
    assert rep["soul_regressions"] == ["s"]
