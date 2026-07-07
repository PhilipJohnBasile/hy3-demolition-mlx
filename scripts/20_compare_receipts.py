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
    gating = hard_regressions + (soul_regressions if args.strict_souls else [])

    base_rate, cand_rate = rate(base), rate(cand)
    verdict = "PROMOTE" if not gating and cand_rate >= base_rate else "REJECT"

    report = {
        "baseline": {"path": args.baseline, "cases": len(base),
                     "pass_rate": round(base_rate, 4), "tok_s": tok_s(base)},
        "candidate": {"path": args.candidate, "cases": len(cand),
                      "pass_rate": round(cand_rate, 4), "tok_s": tok_s(cand)},
        "shared_cases": len(shared),
        "missing_in_candidate": only_base,
        "new_in_candidate": only_cand,
        "regressions": regressions,
        "improvements": improvements,
        "soul_regressions_gating": args.strict_souls,
        "verdict": verdict,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if verdict == "PROMOTE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
