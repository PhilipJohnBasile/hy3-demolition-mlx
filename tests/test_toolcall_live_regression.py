"""Live-weights regression test for the reap25 tool-call tag bug
(data-model-brain findings/reap25-toolcall-regression.md).

reap25 (25%-pruned) was found to unreliably omit the <tool_sep:opensource>
tag in native tool-call output — confirmed on 2 independent schemas, isolated
against lite-v1 (same heal, no prune, correct both times). This test locks in
whichever behavior the current checkpoint actually has, so a future re-heal
or re-prune that silently fixes OR re-breaks this gets noticed instead of
assumed.

Matches the Mimosa lesson (data-model-brain what-works #3 / meta-lesson):
"deterministic-path green != model-path works" — only a live-weights smoke
test catches a native-format regression like this; a code-only check can't.

SKIPPED BY DEFAULT (loads an 87GB checkpoint via the streaming pager, ~1-2
min). Run explicitly: RUN_LIVE_TESTS=1 pytest tests/test_toolcall_live_regression.py
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="live model test — set RUN_LIVE_TESTS=1 to run (loads 87GB checkpoint via streaming pager)",
)


def _build_streaming(model_dir):
    spec = importlib.util.spec_from_file_location("s44", REPO / "scripts/44_stream_stress.py")
    s44 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s44)
    return s44.build_streaming(model_dir)


def test_reap25_toolcall_format_documented_state():
    """As of 2026-07-08 this is EXPECTED TO FAIL (documents the known bug).
    If it starts passing, the regression was fixed — update the model card
    and data-model-brain to reflect that, don't just let this test go green
    silently."""
    from mlx_lm import stream_generate
    from mlx_lm.utils import load_tokenizer

    model_dir = REPO / "dist" / "hy3-demolition-mlx-reap25-v1-fused"
    model = _build_streaming(model_dir)
    tok = load_tokenizer(model_dir)

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                           "required": ["location"]},
        },
    }]
    text = tok.apply_chat_template(
        [{"role": "user", "content": "What is the weather in Tokyo? Use the tool."}],
        tools=tools, tokenize=False, add_generation_prompt=True, reasoning_effort="no_think")

    out = ""
    for r in stream_generate(model, tok, text, max_tokens=150):
        out += r.text

    has_tool_sep = "<tool_sep:opensource>" in out
    if not has_tool_sep:
        pytest.xfail(
            "known bug reproduced: <tool_sep:opensource> missing from reap25's "
            f"tool-call output (see data-model-brain findings/reap25-toolcall-regression.md). "
            f"Raw output: {out[:300]!r}"
        )
    # if we get here, the tag was present — the bug may be fixed; xfail above
    # won't fire and this assertion documents the now-correct behavior
    assert has_tool_sep
