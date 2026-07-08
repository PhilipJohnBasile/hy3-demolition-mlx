#!/usr/bin/env python3
"""Preflight: catch pipeline breakage without loading a model.

We edit the numbered scripts a lot; a single import or syntax break wastes a
multi-hour GPU run. This validates everything cheap and CPU-only: scripts
compile, src modules import, eval packs load into EvalCase, verifiers behave,
the calibration/data packs are well-formed, the fp32 router patch is present,
and the REAP plan analyzer runs on a synthetic plan. Exit 0 = safe to launch
GPU work; nonzero = fix first.

Run before any REAP launch: `./scripts/29_preflight_check.py`
"""
from __future__ import annotations

import json
import py_compile
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "ok " if ok else "FAIL"
    print(f"  [{mark}] {name}{': ' + detail if detail and not ok else ''}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    print("preflight: scripts compile")
    for script in sorted((REPO / "scripts").glob("*.py")):
        try:
            py_compile.compile(str(script), doraise=True)
            check(script.name, True)
        except py_compile.PyCompileError as e:
            check(script.name, False, str(e).splitlines()[-1][:80])

    print("preflight: src modules import")
    sys.path.insert(0, str(REPO / "src"))
    for mod in ("hy3_reap", "hy3_weight_store", "hy3_eval_receipts",
                "hy3_local_verifiers", "hy3_chat_template"):
        try:
            __import__(mod)
            check(mod, True)
        except Exception as e:  # noqa: BLE001
            check(mod, False, f"{type(e).__name__}: {e}")

    print("preflight: eval packs load into EvalCase")
    from hy3_eval_receipts import load_cases  # type: ignore
    packs = list((REPO / "eval").glob("*/prompts.jsonl"))
    total = 0
    for pack in sorted(packs):
        try:
            cases = load_cases(pack)
            total += len(cases)
            check(f"{pack.parent.name}/prompts.jsonl", True, f"{len(cases)} cases")
        except Exception as e:  # noqa: BLE001
            check(f"{pack.parent.name}/prompts.jsonl", False, f"{type(e).__name__}: {e}")
    check("eval packs non-empty", total > 0, f"{total} cases across {len(packs)} packs")

    print("preflight: local verifiers behave")
    from hy3_local_verifiers import LOCAL_VERIFIERS  # type: ignore
    vj = LOCAL_VERIFIERS["json_schema"]
    schema = {"schema": {"type": "object", "required": ["ok"],
                         "properties": {"ok": {"type": "boolean"}}}}
    check("json verifier accepts valid", vj('{"ok": true}', schema)[0])
    check("json verifier rejects invalid", not vj('{"ok": 1}', schema)[0])
    vt = LOCAL_VERIFIERS["tool_call"]
    tspec = {"tools": ["a"], "expected_tool": "a", "required_args": ["x"],
             "arg_types": {"x": "string"}}
    check("tool verifier accepts valid", vt('{"tool": "a", "args": {"x": "y"}}', tspec)[0])
    check("tool verifier rejects wrong tool", not vt('{"tool": "b", "args": {"x": "y"}}', tspec)[0])

    print("preflight: data packs well-formed")
    for split in ("train", "valid", "test"):
        p = REPO / "data" / "hy3_lite_sft_combined" / f"{split}.jsonl"
        try:
            rows = [json.loads(l) for l in p.open() if l.strip()]
            bad = [i for i, r in enumerate(rows) if "messages" not in r]
            check(f"lite_sft_combined/{split}", not bad,
                  f"{len(bad)} rows missing 'messages'" if bad else f"{len(rows)} rows")
        except Exception as e:  # noqa: BLE001
            check(f"lite_sft_combined/{split}", False, f"{type(e).__name__}: {e}")
    calib = REPO / "data" / "hy3_reap_calibration" / "prompts.jsonl"
    try:
        rows = [json.loads(l) for l in calib.open() if l.strip()]
        tagged = sum(1 for r in rows if r.get("facet"))
        check("reap calibration pack", tagged == len(rows), f"{len(rows)} rows, all facet-tagged")
    except Exception as e:  # noqa: BLE001
        check("reap calibration pack", False, f"{type(e).__name__}: {e}")

    print("preflight: fp32 router patch present (D0)")
    try:
        import importlib.util
        spec = importlib.util.find_spec("mlx_lm.models.hy_v3")
        src = Path(spec.origin).read_text() if spec and spec.origin else ""
        check("fp32 router in installed fork", "astype(mx.float32)" in src,
              "MoEGate must run gate matmul in fp32 (see RESTORE.md)")
    except Exception as e:  # noqa: BLE001
        check("fp32 router patch", False, f"mlx_lm not importable: {e}")

    print("preflight: REAP plan analyzer runs on a synthetic plan")
    try:
        import subprocess
        import tempfile
        plan = {"source": "x", "ratio": 0.25, "old_num_experts": 8, "new_num_experts": 6,
                "protected_facets": ["coding"], "min_keep_per_protected_facet": 2,
                "layers": [{"layer": 1, "keep": [0, 1, 2, 3, 4, 5], "drop": [6, 7],
                            "protected": {"coding": [0, 1]}}]}
        tf = Path(tempfile.mktemp(suffix=".json"))
        tf.write_text(json.dumps(plan))
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "25_analyze_reap_plan.py"),
                            str(tf)], capture_output=True, text=True)
        tf.unlink(missing_ok=True)
        check("analyzer runs + emits verdict", '"overall"' in r.stdout)
    except Exception as e:  # noqa: BLE001
        check("analyzer smoke", False, f"{type(e).__name__}: {e}")

    print()
    if FAILURES:
        print(f"PREFLIGHT FAILED ({len(FAILURES)}): fix before launching GPU work")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("PREFLIGHT PASSED — pipeline is launch-ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
