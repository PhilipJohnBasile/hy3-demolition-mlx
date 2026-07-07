#!/usr/bin/env python3
"""Merge the canon SFT pack into data/hy3_lite_sft_combined with dedupe.

Deterministic: rows are deduped against every existing split (and the
over-length quarantine files) by sha256 of their messages, then assigned
seeded round-robin so roughly 1 in 10 lands in valid and 1 in 10 in test.
Re-run scripts/16_normalize_lite_sft_lengths.py --write afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def row_key(row: dict) -> str:
    content = row.get("messages", row.get("text", ""))
    return hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canon", default="data/hy3_canon_sft/rows.jsonl")
    parser.add_argument("--combined", default="data/hy3_lite_sft_combined")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--receipt", default="eval/receipts/hy3_canon_sft_merge.json")
    args = parser.parse_args()

    combined = REPO_ROOT / args.combined
    splits = {s: read_jsonl(combined / f"{s}.jsonl") for s in ("train", "valid", "test")}
    seen = {
        row_key(r)
        for rows in splits.values()
        for r in rows
    } | {
        row_key(r)
        for s in ("train", "valid", "test")
        for r in read_jsonl(combined / f"{s}.overlength.jsonl")
    }

    canon = read_jsonl(REPO_ROOT / args.canon)
    fresh = [r for r in canon if row_key(r) not in seen]
    rng = random.Random(args.seed)
    assigned = Counter()
    for r in fresh:
        roll = rng.random()
        split = "valid" if roll < 0.1 else "test" if roll < 0.2 else "train"
        splits[split].append(r)
        assigned[split] += 1

    for split, rows in splits.items():
        with (combined / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    facets = Counter(
        r.get("metadata", {}).get("facet", "unknown")
        for rows in splits.values()
        for r in rows
    )
    receipt = {
        "canon_rows": len(canon),
        "merged_new": len(fresh),
        "skipped_duplicates": len(canon) - len(fresh),
        "assigned": dict(assigned),
        "split_sizes": {s: len(rows) for s, rows in splits.items()},
        "combined_facets": dict(sorted(facets.items())),
        "seed": args.seed,
        "note": "run scripts/16_normalize_lite_sft_lengths.py --write after merging",
    }
    receipt_path = REPO_ROOT / args.receipt
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
