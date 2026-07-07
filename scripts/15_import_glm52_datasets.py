#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

SFT_FILES = {
    "glm52-demolition-data/heal/gold_agentic/tool_reliability.jsonl": ("agentic", 80),
    "glm52-demolition-data/heal/gold_agentic/longhorizon.jsonl": ("agentic", 80),
    "glm52-demolition-data/heal/gold2/code_mlx.jsonl": ("coding", 120),
    "glm52-demolition-data/heal/gold2/code_modelinfra.jsonl": ("coding", 120),
    "glm52-demolition-data/heal/gold_cyber/securecode.jsonl": ("security", 120),
    "glm52-demolition-data/heal/gold_sound/sound.jsonl": ("music", 80),
    "glm52-demolition-data/heal/gold_perfumery/perfumery.jsonl": ("perfumery", 80),
    "glm52-verified-fixes/recover_data.jsonl": ("repair", 200),
    "glm52-verified-fixes/recover_data_diverse.jsonl": ("repair", 500),
}

CALIB_FILES = {
    "glm52-demolition-data/calib/reap_mix.jsonl": ("mixed", 512),
    "glm52-demolition-data/calib/facet_mix.jsonl": ("mixed", 512),
    "glm52-demolition-data/calib/soul_calib_v3.jsonl": ("soul", 512),
}

ROLE_MAP = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}

FACET_ALIASES = {
    "code": "coding",
    "sound": "music",
    "cyber": "security",
    "securecode": "security",
}


def normalize_facet(value: Any, fallback: str) -> str:
    facet = str(value or fallback).strip().lower().replace("-", "_")
    return FACET_ALIASES.get(facet, facet)


def stable_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reservoir_jsonl(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample: list[str] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            if len(sample) < limit:
                sample.append(line)
                continue
            idx = rng.randint(0, seen - 1)
            if idx < limit:
                sample[idx] = line
    return [json.loads(line) for line in sample]


def text_from_tool_calls(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls:
        return ""
    normalized = []
    for call in calls:
        function = call.get("function") or {}
        normalized.append(
            {
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        )
    return "\n\n<tool_calls>\n" + json.dumps(normalized, ensure_ascii=False) + "\n</tool_calls>"


def normalize_messages(row: dict[str, Any], fallback_facet: str, source: str) -> dict[str, Any] | None:
    messages = row.get("messages")
    if messages is None and row.get("text"):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": str(row["text"]).strip(),
                },
                {
                    "role": "assistant",
                    "content": "Acknowledged.",
                },
            ],
            "metadata": {"facet": row.get("facet", fallback_facet), "source": source},
        }
    if not isinstance(messages, list):
        return None

    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        tool_text = text_from_tool_calls(message)
        if role in ROLE_MAP:
            if role == "assistant" and tool_text:
                content = (content + tool_text).strip()
            if content:
                normalized.append({"role": ROLE_MAP[role], "content": content})
        elif role == "tool":
            tool_id = message.get("tool_call_id") or "tool"
            if content:
                normalized.append(
                    {
                        "role": "user",
                        "content": f"<tool_result id=\"{tool_id}\">\n{content}\n</tool_result>",
                    }
                )

    if not any(m["role"] == "user" for m in normalized):
        return None
    if not any(m["role"] == "assistant" for m in normalized):
        return None

    return {
        "messages": normalized,
        "metadata": {
            "facet": normalize_facet(row.get("facet") or row.get("soul"), fallback_facet),
            "source": source,
            "imported_from": "glm52",
        },
    }


def prompt_from_row(row: dict[str, Any], fallback_facet: str, source: str) -> dict[str, Any] | None:
    if isinstance(row.get("prompt"), str):
        prompt = row["prompt"].strip()
    elif isinstance(row.get("content"), str):
        prompt = row["content"].strip()
    elif isinstance(row.get("text"), str):
        prompt = row["text"].strip()
    elif isinstance(row.get("messages"), list):
        parts = []
        for message in row["messages"]:
            if message.get("role") in {"system", "user"} and message.get("content"):
                parts.append(str(message["content"]).strip())
        prompt = "\n\n".join(parts).strip()
    else:
        prompt = ""
    if not prompt:
        return None
    return {
        "id": stable_key(source + prompt)[:16],
        "facet": normalize_facet(row.get("facet") or row.get("soul"), fallback_facet),
        "prompt": prompt,
        "source": source,
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    valid_n = max(1, round(n * 0.05)) if n >= 20 else max(1, n // 10)
    test_n = max(1, round(n * 0.05)) if n >= 20 else max(1, n // 10)
    valid = rows[:valid_n]
    test = rows[valid_n : valid_n + test_n]
    train = rows[valid_n + test_n :]
    return train, valid, test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/glm52_import/raw")
    parser.add_argument("--seed-dir", default="data/hy3_lite_sft")
    parser.add_argument("--out-dir", default="data/hy3_lite_sft_combined")
    parser.add_argument("--normalized-dir", default="data/glm52_import/normalized")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--sft-scale", type=float, default=1.0)
    parser.add_argument("--calib-scale", type=float, default=1.0)
    args = parser.parse_args()

    raw_dir = REPO_ROOT / args.raw_dir
    normalized_dir = REPO_ROOT / args.normalized_dir
    out_dir = REPO_ROOT / args.out_dir
    seed_dir = REPO_ROOT / args.seed_dir

    imported_sft: list[dict[str, Any]] = []
    imported_calib: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    facet_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for rel, (facet, cap) in SFT_FILES.items():
        path = raw_dir / rel
        if not path.exists():
            skipped[f"missing:{rel}"] += 1
            continue
        rows = reservoir_jsonl(path, max(1, round(cap * args.sft_scale)), args.seed)
        for row in rows:
            normalized = normalize_messages(row, facet, rel)
            if normalized is None:
                skipped[f"unusable:{rel}"] += 1
                continue
            imported_sft.append(normalized)
            source_counts[rel] += 1
            facet_counts[str(normalized["metadata"]["facet"])] += 1

    seen: set[str] = set()
    deduped_sft: list[dict[str, Any]] = []
    for row in imported_sft:
        key = stable_key(json.dumps(row["messages"], sort_keys=True, ensure_ascii=False))
        if key in seen:
            skipped["duplicate_sft"] += 1
            continue
        seen.add(key)
        deduped_sft.append(row)

    for rel, (facet, cap) in CALIB_FILES.items():
        path = raw_dir / rel
        if not path.exists():
            skipped[f"missing:{rel}"] += 1
            continue
        rows = reservoir_jsonl(path, max(1, round(cap * args.calib_scale)), args.seed)
        for row in rows:
            prompt = prompt_from_row(row, facet, rel)
            if prompt is None:
                skipped[f"unusable_calib:{rel}"] += 1
                continue
            imported_calib.append(prompt)

    write_jsonl(deduped_sft, normalized_dir / "sft_import.jsonl")
    write_jsonl(imported_calib, normalized_dir / "calib_prompts.jsonl")

    seed_rows: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        path = seed_dir / f"{split}.jsonl"
        if path.exists():
            seed_rows.extend(read_jsonl(path))

    combined = seed_rows + deduped_sft
    train, valid, test = split_rows(combined, args.seed)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(valid, out_dir / "valid.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    receipt = {
        "raw_dir": args.raw_dir,
        "seed_dir": args.seed_dir,
        "out_dir": args.out_dir,
        "normalized_dir": args.normalized_dir,
        "seed": args.seed,
        "sft_imported": len(deduped_sft),
        "calib_imported": len(imported_calib),
        "combined": {
            "train": len(train),
            "valid": len(valid),
            "test": len(test),
        },
        "source_counts": dict(sorted(source_counts.items())),
        "facet_counts": dict(sorted(facet_counts.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    receipt_path = normalized_dir / "import_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
