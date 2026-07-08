#!/usr/bin/env python3
"""Build mlx-lm training data from Sonnet gold answers (paste JSON below or
load from /tmp/sonnet_train.json + /tmp/sonnet_eval.json).

Usage: 57_soul_distill_build.py
Reads data/soul_distill_prompts.json (prompts) + /tmp/sonnet_train.json
(id -> answer) and writes data/soul_distill/{train,valid}.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "soul_distill"


def main() -> int:
    prompts = json.loads((REPO / "data/soul_distill_prompts.json").read_text())
    gold = json.loads(Path("/tmp/sonnet_train.json").read_text())
    rows = []
    for p in prompts["train"]:
        ans = gold.get(p["id"])
        if not ans:
            print(f"MISSING gold for {p['id']}"); continue
        rows.append({"messages": [{"role": "user", "content": p["prompt"]},
                                  {"role": "assistant", "content": ans.strip()}]})
    OUT.mkdir(parents=True, exist_ok=True)
    # small valid split from the tail
    valid, train = rows[-4:], rows[:-4]
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    (OUT / "valid.jsonl").write_text("\n".join(json.dumps(r) for r in valid) + "\n")
    print(f"wrote {len(train)} train / {len(valid)} valid -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
