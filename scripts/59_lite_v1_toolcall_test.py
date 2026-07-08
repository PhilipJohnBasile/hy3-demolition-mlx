#!/usr/bin/env python3
"""Same tool-call test as reap25, run on lite-v1 (unpruned, still LoRA-healed)
via the streaming pager (safe at ~10GB resident vs 112GB direct load).
Isolates: is the malformed tag output shared with reap25 (-> our heal), or
specific to the pruned model (-> pruning-induced)?
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("s44", REPO / "scripts/44_stream_stress.py")
s44 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s44)


def main() -> int:
    from mlx_lm import stream_generate
    from mlx_lm.utils import load_tokenizer

    print("[toolcall-lite] building streaming lite-v1", flush=True)
    model = s44.build_streaming(s44.MODEL)
    tok = load_tokenizer(s44.MODEL)

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
        },
    }]
    text = tok.apply_chat_template(
        [{"role": "user", "content": "What is the weather in Tokyo? Use the tool."}],
        tools=tools, tokenize=False, add_generation_prompt=True, reasoning_effort="no_think")

    out = ""
    for r in stream_generate(model, tok, text, max_tokens=150):
        out += r.text
    print("=== RAW LITE-V1 OUTPUT ===", flush=True)
    print(out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
