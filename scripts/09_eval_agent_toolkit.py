#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hy3_chat_template import MODEL_FIRST_SYSTEM
from hy3_eval_receipts import add_common_eval_args, load_cases, run_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "eval/coding/prompts.jsonl",
            "eval/tool_calls/prompts.jsonl",
            "eval/agent_repair/prompts.jsonl",
            "eval/json_schema/prompts.jsonl",
        ],
    )
    parser.add_argument("--out", default="eval/receipts/agent_toolkit_eval.jsonl")
    args = parser.parse_args()

    all_cases = []
    for path in args.cases:
        all_cases.extend(load_cases(path))
    receipts = run_cases(
        cases=all_cases,
        base_url=args.base_url,
        model=args.model,
        backend=args.backend,
        out=args.out,
        system_prompt=MODEL_FIRST_SYSTEM,
        max_tokens=args.max_tokens,
    )
    passed = sum(1 for r in receipts if r.passed)
    total = len(receipts)
    print(f"agent_toolkit_eval: {passed}/{total} passed; receipts={args.out}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
