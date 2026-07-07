#!/usr/bin/env python3
"""Quarantine SFT rows whose tokenized length exceeds the training cap.

mlx_lm.lora silently truncates sequences above --max-seq-length, which for
repair-style data means training on cut-off assistant answers. This pass moves
over-length rows into a quarantine file next to the split and writes a receipt,
so the training pack never depends on silent truncation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SPLITS = ("train", "valid", "test")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_length(tokenizer, row: dict) -> int:
    if "messages" in row:
        return len(tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False))
    return len(tokenizer.encode(row.get("text", "")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/hy3_lite_sft_combined")
    parser.add_argument("--model", default="models/hy3-mlx-base-ar")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--receipt", default="eval/receipts/hy3_lite_sft_length_normalization.json")
    parser.add_argument("--write", action="store_true", help="rewrite splits; default is dry run")
    args = parser.parse_args()

    from mlx_lm.utils import load_tokenizer

    data_dir = REPO_ROOT / args.data
    tokenizer = load_tokenizer(REPO_ROOT / args.model)

    receipt: dict = {
        "data": args.data,
        "model": args.model,
        "max_seq_length": args.max_seq_length,
        "write": args.write,
        "splits": {},
    }
    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        kept: list[dict] = []
        kept_lengths: list[int] = []
        quarantined: list[dict] = []
        max_tokens = 0
        for row in rows:
            n = token_length(tokenizer, row)
            max_tokens = max(max_tokens, n)
            if n <= args.max_seq_length:
                kept.append(row)
                kept_lengths.append(n)
            else:
                quarantined.append(row)
        facets = Counter(
            row.get("metadata", {}).get("facet", "unknown") for row in quarantined
        )
        receipt["splits"][split] = {
            "total": len(rows),
            "kept": len(kept),
            "quarantined": len(quarantined),
            "quarantined_facets": dict(facets),
            "max_tokens": max_tokens,
            "kept_max_tokens": max(kept_lengths, default=0),
        }
        print(
            f"{split}: total={len(rows)} kept={len(kept)} "
            f"quarantined={len(quarantined)} max_tokens={max_tokens}"
        )
        if args.write and quarantined:
            write_jsonl(path, kept)
            quarantine_path = data_dir / f"{split}.overlength.jsonl"
            existing = read_jsonl(quarantine_path) if quarantine_path.exists() else []
            write_jsonl(quarantine_path, existing + quarantined)

    receipt_path = REPO_ROOT / args.receipt
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"receipt: {args.receipt}")


if __name__ == "__main__":
    main()
