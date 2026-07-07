#!/usr/bin/env python3
"""Turn a REAP prune plan into decision-ready facts.

The dry-run plan (reap_plan.json) is a wall of expert indices; this report
answers the question the plan gate actually asks — "is this safe to apply?" —
with explicit ACCEPT/REVIEW/REJECT verdicts per check.

Checks and thresholds (rationale in BACKLOG.md / docs):
1. structure: every layer plans keep == new_num_experts, drop == old - new,
   no index out of range, no overlap between keep and drop. Violation = REJECT.
2. protected coverage: every protected facet contributes experts in every
   MoE layer, and 100% of protected experts are in the keep set (build_plan
   guarantees this; verify anyway — trust but check). Violation = REJECT.
3. protection pressure: protected experts / keep budget per layer. If any
   layer spends > 60% of its keep budget on protection, aggregate saliency
   barely matters there — REVIEW (probably fine, but understand why).
4. soul concentration: if two facets' protected sets overlap > 75% on
   average, their calibration prompts may not be distinguishing them —
   REVIEW the calibration pack.
5. saliency mass retained: with the saliency file, the fraction of total
   routing score mass kept per layer. Any MoE layer keeping < 70% of its
   score mass at a 25% prune = REVIEW (routing is unusually flat or the drop
   set is unusually hot).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", help="reap_plan.json from scripts/05 (dry run)")
    parser.add_argument("--saliency", help="saliency json used to build the plan")
    parser.add_argument("--out", help="optional JSON report path")
    args = parser.parse_args()

    plan = load(args.plan)
    old_n = plan["old_num_experts"]
    new_n = plan["new_num_experts"]
    layers = plan["layers"]
    facets = plan["protected_facets"]
    verdicts: list[tuple[str, str, str]] = []  # (check, verdict, detail)

    # 1. structure
    structural = []
    for lp in layers:
        keep, drop = set(lp["keep"]), set(lp["drop"])
        if len(lp["keep"]) != new_n:
            structural.append(f"layer {lp['layer']}: keep={len(lp['keep'])} != {new_n}")
        if keep & drop:
            structural.append(f"layer {lp['layer']}: keep/drop overlap")
        if len(keep) + len(drop) != old_n:
            structural.append(f"layer {lp['layer']}: keep+drop != {old_n}")
        if any(i < 0 or i >= old_n for i in keep | drop):
            structural.append(f"layer {lp['layer']}: index out of range")
    verdicts.append(("structure", "REJECT" if structural else "ACCEPT",
                     "; ".join(structural[:5]) or f"{len(layers)} layers consistent"))

    # 2. protected coverage
    coverage = []
    for lp in layers:
        keep = set(lp["keep"])
        for facet in facets:
            chosen = lp["protected"].get(facet)
            if not chosen:
                coverage.append(f"layer {lp['layer']}: facet {facet} has no protected experts")
            elif not set(chosen) <= keep:
                coverage.append(f"layer {lp['layer']}: facet {facet} protected experts dropped")
    verdicts.append(("protected_coverage", "REJECT" if coverage else "ACCEPT",
                     "; ".join(coverage[:5]) or f"all {len(facets)} facets protected in all layers"))

    # 3. protection pressure
    pressures = []
    for lp in layers:
        protected_union = set()
        for chosen in lp["protected"].values():
            protected_union.update(chosen)
        pressures.append((lp["layer"], len(protected_union) / new_n))
    worst = sorted(pressures, key=lambda x: -x[1])[:5]
    max_pressure = worst[0][1] if worst else 0.0
    verdicts.append(("protection_pressure",
                     "REVIEW" if max_pressure > 0.60 else "ACCEPT",
                     f"max {max_pressure:.0%} of keep budget on protection; top layers: "
                     + ", ".join(f"L{l}={p:.0%}" for l, p in worst)))

    # 4. soul concentration (pairwise overlap of protected sets, averaged over layers)
    overlaps: dict[str, list[float]] = {}
    for lp in layers:
        for i, fa in enumerate(facets):
            for fb in facets[i + 1:]:
                a, b = set(lp["protected"].get(fa) or []), set(lp["protected"].get(fb) or [])
                if a and b:
                    overlaps.setdefault(f"{fa}~{fb}", []).append(len(a & b) / min(len(a), len(b)))
    high = {k: sum(v) / len(v) for k, v in overlaps.items() if sum(v) / len(v) > 0.75}
    verdicts.append(("soul_concentration",
                     "REVIEW" if high else "ACCEPT",
                     ("indistinct facet pairs: " + ", ".join(f"{k}={v:.0%}" for k, v in sorted(high.items())))
                     if high else "protected sets are facet-distinct (<=75% overlap)"))

    # 5. saliency mass retained
    if args.saliency:
        sal = load(args.saliency)["layers"]
        retained = []
        for lp in layers:
            data = sal.get(str(lp["layer"]), {})
            scores = data.get("reap_sum") or data.get("score_sum")
            if not scores:
                continue
            total = sum(scores)
            kept = sum(scores[i] for i in lp["keep"])
            if total > 0:
                retained.append((lp["layer"], kept / total))
        low = [(l, r) for l, r in retained if r < 0.70]
        avg = sum(r for _, r in retained) / len(retained) if retained else 0.0
        verdicts.append(("saliency_mass",
                         "REVIEW" if low else "ACCEPT",
                         f"avg {avg:.0%} score mass kept; "
                         + (f"{len(low)} layers < 70%: " + ", ".join(f"L{l}={r:.0%}" for l, r in low[:5])
                            if low else "no layer below 70%")))
    else:
        verdicts.append(("saliency_mass", "REVIEW", "no --saliency provided; rerun with it"))

    overall = ("REJECT" if any(v == "REJECT" for _, v, _ in verdicts)
               else "REVIEW" if any(v == "REVIEW" for _, v, _ in verdicts)
               else "ACCEPT")
    report = {
        "plan": args.plan,
        "ratio": plan["ratio"],
        "experts": f"{old_n} -> {new_n}",
        "layers": len(layers),
        "checks": [{"check": c, "verdict": v, "detail": d} for c, v, d in verdicts],
        "overall": overall,
        "policy": ("ACCEPT = apply with --write; REVIEW = a human reads the flagged detail "
                   "before applying; REJECT = do not write weights"),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if overall != "REJECT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
