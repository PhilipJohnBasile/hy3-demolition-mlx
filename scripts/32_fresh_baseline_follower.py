#!/usr/bin/env python3
"""After the REAP calibration/plan chain finishes, run the FRESH lite-v1
baseline (suite + hard + brutal) so it's ready for the reap25 gate.

DECISIONS.md D3 requires a fresh full baseline on the CURRENT eval set before
comparing reap25 — the committed 30/30 predates the brutal tier and the soul
tightening. This also answers REFLECTIONS risk #1: does the brutal tier
discriminate on lite-v1? Runs only when the GPU is otherwise idle.

Writes NO weights. Serves lite-v1, evals the three tiers, commits receipts,
stops. Waits for 'REAP PLAN READY' in dist/reap_chain.log first, then for the
GPU to actually clear.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "bin" / "python")
FUSED = str(REPO / "dist" / "hy3-demolition-mlx-lite-v1-fused")
TOOLKIT = {"AGENT_TOOLKIT_PATH": "/Users/pjb/git/agent-toolkit"}


def log(m: str) -> None:
    print(f"[freshbase {time.strftime('%H:%M:%S')}] {m}", flush=True)


def pgrep(pat: str) -> bool:
    return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0


def free_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "Pages free" in line:
            return int(line.split()[-1].rstrip(".")) * 16384 / 1e9
    return 0.0


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def main() -> int:
    log("waiting for REAP plan chain to finish")
    chain_log = REPO / "dist" / "reap_chain.log"
    while True:
        txt = chain_log.read_text() if chain_log.exists() else ""
        if "REAP PLAN READY" in txt:
            break
        if "ABORT" in txt:
            log("REAP chain aborted; not running baseline")
            return 1
        time.sleep(60)
    log("plan ready; clearing GPU + waiting for it to free")
    for pat in ("04_stream_calibrate", "05_apply_reap", "14_serve_mlx", "08_serve_mlx"):
        subprocess.run(["pkill", "-f", pat])
    while free_gb() < 90:
        time.sleep(30)
    time.sleep(30)

    import os
    logf = (REPO / "dist" / "serve_freshbase.log").open("w")
    server = subprocess.Popen(
        [PY, "scripts/08_serve_mlx.py", "--model", FUSED, "--port", "8080",
         "--prompt-cache-size", "8"], cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    try:
        # wait for readiness (first request loads the model)
        import urllib.request
        deadline = time.time() + 3600
        while time.time() < deadline:
            try:
                if b"lite-v1-fused" in urllib.request.urlopen(
                        "http://127.0.0.1:8080/v1/models", timeout=3).read():
                    break
            except Exception:
                pass
            time.sleep(5)
        log("server up; running suite + hard + brutal")
        env = dict(os.environ); env.update(TOOLKIT)
        rc = subprocess.run(
            [PY, "scripts/09_eval_agent_toolkit.py",
             "--base-url", "http://127.0.0.1:8080/v1", "--model", FUSED,
             "--backend", "mlx_lm_server_fused", "--timeout", "2400",
             "--cases", "eval/coding/prompts.jsonl", "eval/tool_calls/prompts.jsonl",
             "eval/agent_repair/prompts.jsonl", "eval/json_schema/prompts.jsonl",
             "eval/planning/prompts.jsonl", "eval/souls/prompts.jsonl",
             "eval/hard/prompts.jsonl", "eval/brutal/prompts.jsonl",
             "--out", "eval/receipts/hy3_lite_v1_fresh_baseline.jsonl"],
            cwd=REPO, env=env).returncode
    finally:
        server.terminate()

    rows = read_jsonl(REPO / "eval/receipts/hy3_lite_v1_fresh_baseline.jsonl")
    from collections import Counter
    by_pack = Counter()
    fails = []
    for r in rows:
        dom = r["domain"]
        by_pack[dom] += 1
        if not r["passed"]:
            fails.append(r["id"])
    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "model": "dist/hy3-demolition-mlx-lite-v1-fused",
        "cases": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "failed_ids": fails,
        "by_domain": dict(by_pack),
        "brutal_discriminates": any(r["domain"] in ("python", "json", "tool_call",
                                    "planning", "soul") and not r["passed"]
                                    for r in rows if r["id"].startswith(("brutal_", "hard_"))),
        "note": "D3 reap25-gate baseline on the CURRENT eval set (suite+hard+brutal). "
                "If brutal fails nothing, the instrument does not discriminate — "
                "see REFLECTIONS risk #1.",
    }
    (REPO / "eval/receipts/hy3_lite_v1_fresh_baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    log(f"fresh baseline: {summary['passed']}/{summary['cases']}; fails: {fails}")
    subprocess.run(["git", "add",
                    "eval/receipts/hy3_lite_v1_fresh_baseline.jsonl",
                    "eval/receipts/hy3_lite_v1_fresh_baseline_summary.json"], cwd=REPO)
    subprocess.run(["git", "commit", "-m",
                    "Fresh lite-v1 baseline (suite+hard+brutal) for the reap25 gate\n\n"
                    "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n"
                    "Claude-Session: https://claude.ai/code/session_01LJenYJzwFH5NTvNc76tTgY"],
                   cwd=REPO)
    subprocess.run(["git", "push", "origin", "main"], cwd=REPO)
    log("FRESH BASELINE READY — brutal-tier discrimination now measurable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
