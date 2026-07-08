#!/usr/bin/env python3
"""Soul-facet eval — the ON-DISTRIBUTION test of the verifier heal.

The academic benchmarks (HumanEval/AIME/GPQA) are off-distribution: we never
healed on competition math. This runs the 11 protected soul facets — exactly
what the heal hardened (coding/math/science/security/design/fullstack/gamedev/
legacy/music/art/perfumery, each cued by a <|soul:facet|> token) — and grades
with the project's own verify_soul. This is where sibling should beat base.

Usage: 51_souls.py <model_dir> [--thinking sibling|hy3]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hy3_local_verifiers import verify_soul  # noqa: E402

PROMPTS = [json.loads(l) for l in open(REPO / "eval/souls/prompts.jsonl")]


def main() -> int:
    model_dir = sys.argv[1]
    thinking = sys.argv[sys.argv.index("--thinking") + 1] if "--thinking" in sys.argv else "hy3"
    tkw = {"enable_thinking": False} if thinking == "sibling" else {"reasoning_effort": "no_think"}
    tag = Path(model_dir).name

    from mlx_lm import load, generate
    print(f"[souls] loading {tag}", flush=True)
    model, tok = load(model_dir)
    passed, rows = 0, []
    t0 = time.time()
    for p in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}],
                                       tokenize=False, add_generation_prompt=True, **tkw)
        out = generate(model, tok, prompt=text, max_tokens=p.get("max_tokens", 768), verbose=False)
        ok, stage, diag = verify_soul(out, p["spec"])
        passed += ok
        rows.append({"facet": p["facet"], "passed": bool(ok), "stage": stage,
                     "diag": diag, "chars": len(out), "output": out,
                     "prompt": p["prompt"], "spec": p["spec"]})
        print(f"[souls] {p['facet']:10} {'PASS' if ok else 'FAIL':4} ({stage}) {len(out)}ch", flush=True)
    out = {"model": tag, "n": len(PROMPTS), "passed": passed,
           "accuracy_pct": round(100 * passed / len(PROMPTS), 1),
           "minutes": round((time.time() - t0) / 60, 1), "per_facet": rows}
    (REPO / f"eval/receipts/souls_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"[souls] {tag}: {passed}/{len(PROMPTS)} = {out['accuracy_pct']}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
