#!/usr/bin/env python3
"""Second, different tool-call sample on reap25 (different tool schema/prompt
than the first) to check whether the malformed-tag pattern reproduces or was
a one-off. Uses the streaming pager for safe memory footprint.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s44", REPO / "scripts/44_stream_stress.py")
s44 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s44)


def main() -> int:
    from mlx_lm import stream_generate
    from mlx_lm.utils import load_tokenizer

    model_dir = REPO / "dist" / "hy3-demolition-mlx-reap25-v1-fused"
    print("[toolcall-reap25-2] building streaming reap25", flush=True)
    model = s44.build_streaming(model_dir)
    tok = load_tokenizer(model_dir)

    tools = [{
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for flights between two cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    }]
    text = tok.apply_chat_template(
        [{"role": "user", "content": "Find me a flight from New York to London on March 15th."}],
        tools=tools, tokenize=False, add_generation_prompt=True, reasoning_effort="no_think")

    out = ""
    for r in stream_generate(model, tok, text, max_tokens=200):
        out += r.text
    print("=== RAW REAP25 OUTPUT (sample 2, different tool/prompt) ===", flush=True)
    print(out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
