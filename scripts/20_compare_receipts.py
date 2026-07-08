#!/usr/bin/env python3
"""Mechanical promotion gate: compare two eval receipt files per-case.

Every REAP/quant/heal candidate is judged against the lite-v1 baseline with
this script, not by eyeballing JSONL. Exit 0 = candidate matches or beats the
baseline pass rate with no hard-domain regressions; exit 1 otherwise.

Soul cases use heuristic verifiers, so a lone soul flip is reported but only
counts as a regression when --strict-souls is passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HARD_DOMAINS = {"python", "json", "tool_call", "planning"}


def load(path: str) -> dict[str, dict]:
    rows = {}
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def rate(rows: dict[str, dict]) -> float:
    return sum(r["passed"] for r in rows.values()) / len(rows) if rows else 0.0


def tok_s(rows: dict[str, dict]) -> float | None:
    t = sum(r.get("elapsed_s", 0) for r in rows.values())
    k = sum(r.get("completion_tokens", 0) for r in rows.values())
    return round(k / t, 3) if t and k else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", help="baseline receipts JSONL (e.g. lite-v1)")
    parser.add_argument("candidate", help="candidate receipts JSONL")
    parser.add_argument("--strict-souls", action="store_true",
                        help="count soul-case regressions against the gate")
    parser.add_argument("--out", help="optional JSON report path")
    args = parser.parse_args()

    base = load(args.baseline)
    cand = load(args.candidate)

    shared = sorted(set(base) & set(cand))
    only_base = sorted(set(base) - set(cand))
    only_cand = sorted(set(cand) - set(base))

    regressions, improvements = [], []
    for case_id in shared:
        b, c = base[case_id], cand[case_id]
        if b["passed"] and not c["passed"]:
            regressions.append({
                "id": case_id,
                "domain": c["domain"],
                "hard": c["domain"] in HARD_DOMAINS,
                "candidate_stage": c["stage"],
                "candidate_diag": c["diag"][:200],
            })
        elif not b["passed"] and c["passed"]:
            improvements.append({"id": case_id, "domain": c["domain"]})

    hard_regressions = [r for r in regressions if r["hard"]]
    soul_regressions = [r for r in regressions if not r["hard"]]

    # Rate that gates the decision is over HARD (gating) domains only, so a
    # tolerated soul flip (keyword-heuristic noise, DECISIONS.md D3) can't drag
    # the overall rate below baseline and force a spurious REJECT.
    def hard_rate(rows: dict[str, dict]) -> float:
        hard = [r for r in rows.values() if r["domain"] in HARD_DOMAINS]
        return sum(r["passed"] for r in hard) / len(hard) if hard else 1.0

    base_rate, cand_rate = rate(base), rate(cand)
    reject = bool(hard_regressions) or hard_rate(cand) < hard_rate(base) \
        or (args.strict_souls and bool(soul_regressions))
    # D3: a soul flip is not decisive on its own, but a human reads it.
    if reject:
        verdict = "REJECT"
    elif soul_regressions:
        verdict = "REVIEW"
    else:
        verdict = "PROMOTE"

    report = {
        "baseline": {"path": args.baseline, "cases": len(base),
                     "pass_rate": round(base_rate, 4),
                     "hard_pass_rate": round(hard_rate(base), 4), "tok_s": tok_s(base)},
        "candidate": {"path": args.candidate, "cases": len(cand),
                      "pass_rate": round(cand_rate, 4),
                      "hard_pass_rate": round(hard_rate(cand), 4), "tok_s": tok_s(cand)},
        "shared_cases": len(shared),
        "missing_in_candidate": only_base,
        "new_in_candidate": only_cand,
        "regressions": regressions,
        "soul_regressions": [r["id"] for r in soul_regressions],
        "improvements": improvements,
        "soul_regressions_gating": args.strict_souls,
        "verdict": verdict,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    # REJECT is the only hard fail; REVIEW passes the gate but flags for a human
    # read (soul flips), matching scripts/25's ACCEPT/REVIEW/REJECT convention.
    return 1 if verdict == "REJECT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
