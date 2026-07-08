#!/usr/bin/env python3
"""Data-quality + leakage audit for the SFT pack (CPU-only).

Compromised eval numbers usually trace to train/test leakage or malformed
rows. This checks exact + near-duplicate leakage across splits, intra-train
duplicates, malformed rows, and facet balance, writing a receipt. Run after
any data change (15/16/21/22 scripts).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(split: str, data_dir: Path) -> list[dict]:
    p = data_dir / f"{split}.jsonl"
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def _norm(row: dict) -> str:
    txt = " ".join(m.get("content", "") for m in row.get("messages", []))
    return re.sub(r"\s+", " ", txt.lower()).strip()


def _user_prompt(row: dict) -> str:
    for m in row.get("messages", []):
        if m.get("role") == "user":
            return re.sub(r"\s+", " ", m["content"].lower()).strip()
    return ""


def _malformed(row: dict) -> str | None:
    m = row.get("messages")
    if not m or len(m) < 2:
        return "too-few-messages"
    if not any(x.get("role") == "assistant" and x.get("content", "").strip() for x in m):
        return "no-assistant-content"
    return None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hy3_lite_sft_combined")
    ap.add_argument("--receipt", default="eval/receipts/hy3_sft_data_audit.json")
    args = ap.parse_args()
    data_dir = REPO / args.data

    train, valid, test = (_load(s, data_dir) for s in ("train", "valid", "test"))
    tkeys = {hashlib.sha256(_norm(r).encode()).hexdigest() for r in train}
    tprompts = {_user_prompt(r) for r in train}
    tk_list = [hashlib.sha256(_norm(r).encode()).hexdigest() for r in train]

    report = {
        "data": args.data,
        "sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
        "exact_leakage_into_train": {
            "valid": sum(1 for r in valid if hashlib.sha256(_norm(r).encode()).hexdigest() in tkeys),
            "test": sum(1 for r in test if hashlib.sha256(_norm(r).encode()).hexdigest() in tkeys),
        },
        "user_prompt_leakage": sum(1 for r in valid + test if _user_prompt(r) in tprompts),
        "intra_train_duplicate_rows": sum(c - 1 for c in Counter(tk_list).values() if c > 1),
        "malformed_rows": sum(
            1 for rows in (train, valid, test) for r in rows if _malformed(r)),
        "train_facet_balance": dict(sorted(
            Counter(r.get("metadata", {}).get("facet", "?") for r in train).items(),
            key=lambda x: -x[1])),
    }
    leak = (report["exact_leakage_into_train"]["valid"]
            + report["exact_leakage_into_train"]["test"]
            + report["user_prompt_leakage"])
    report["clean"] = leak == 0 and report["intra_train_duplicate_rows"] == 0 \
        and report["malformed_rows"] == 0

    (REPO / args.receipt).parent.mkdir(parents=True, exist_ok=True)
    (REPO / args.receipt).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("CLEAN" if report["clean"] else "ISSUES FOUND")
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
