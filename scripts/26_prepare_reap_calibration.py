"""Assemble the REAP calibration pack for streamed saliency (#18).

Merges three sources into one prompt file so calibration is a single command:
- protected soul prompts (eval/souls/protected_prompts.jsonl) — the whole
  point of soul-preserving REAP; these must carry their facet tag.
- eval prompts across all packs — the behavior we actually measure.
- a capped sample of the GLM imported calibration (facet "mixed") for general
  routing coverage.

Every row keeps a `facet` so the calibration hook buckets saliency per soul.
Soul prompts are NOT capped; the mixed pool is (it dominates otherwise and
washes out rare-soul routing signal — the exact failure mode REAP must avoid).
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

EVAL_PACKS = [
    "eval/coding/prompts.jsonl",
    "eval/tool_calls/prompts.jsonl",
    "eval/agent_repair/prompts.jsonl",
    "eval/json_schema/prompts.jsonl",
    "eval/planning/prompts.jsonl",
    "eval/souls/prompts.jsonl",
    "eval/hard/prompts.jsonl",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def mixed_prompt_text(row: dict) -> str:
    """GLM calib rows stash a python-repr dict in `prompt`; pull real text out."""
    raw = row.get("prompt", "")
    try:
        obj = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw
    if isinstance(obj, dict):
        if "instruction" in obj:
            return str(obj["instruction"])
        msgs = obj.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if m.get("role") == "user":
                    return str(m.get("content", ""))
    return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/hy3_reap_calibration/prompts.jsonl")
    parser.add_argument("--mixed-cap", type=int, default=400,
                        help="max GLM 'mixed' rows (0 disables); soul/eval never capped")
    parser.add_argument("--seed", type=int, default=52)
    args = parser.parse_args()

    rows: list[dict] = []

    for r in read_jsonl(REPO / "eval/souls/protected_prompts.jsonl"):
        rows.append({"prompt": r["prompt"], "facet": r.get("facet", "default"),
                     "source": "protected_prompts"})

    for pack in EVAL_PACKS:
        p = REPO / pack
        if not p.exists():
            continue
        for r in read_jsonl(p):
            rows.append({"prompt": r["prompt"], "facet": r.get("facet", "default"),
                         "source": pack})

    calib = REPO / "data/glm52_import/normalized/calib_prompts.jsonl"
    if args.mixed_cap and calib.exists():
        import random
        pool = read_jsonl(calib)
        rng = random.Random(args.seed)
        rng.shuffle(pool)
        for r in pool[:args.mixed_cap]:
            text = mixed_prompt_text(r)
            if text.strip():
                rows.append({"prompt": text, "facet": r.get("facet", "mixed"),
                             "source": "glm_calib"})

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    facets = Counter(r["facet"] for r in rows)
    sources = Counter(r["source"] for r in rows)
    receipt = {
        "out": args.out,
        "total": len(rows),
        "by_source": dict(sources),
        "by_facet": dict(sorted(facets.items())),
        "mixed_cap": args.mixed_cap,
        "note": "feed to scripts/04 via --prompts; soul prompts already tagged per facet",
    }
    (REPO / "eval/receipts/hy3_reap_calibration_pack.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
