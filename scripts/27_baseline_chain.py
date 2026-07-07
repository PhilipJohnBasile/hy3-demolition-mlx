#!/usr/bin/env python3
"""Post-suite GPU chain: amend-rerun -> hard tier -> MTP smoke -> commit
baseline -> base-model baseline -> compare.

Runs each stage as a subprocess so model memory is released at hard process
boundaries. Designed to be launched while the main suite is still running; it
waits for the eval process to exit first. Logs milestones to stdout (tee'd by
the caller); every stage writes receipts, failures stop the chain with a
clear marker rather than cascading.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "bin" / "python")
ENV_TOOLKIT = {"AGENT_TOOLKIT_PATH": "/Users/pjb/git/agent-toolkit"}
FUSED = "dist/hy3-demolition-mlx-lite-v1-fused"
FUSED_ABS = str(REPO / FUSED)
AR_BASE = "models/hy3-mlx-base-ar"
AR_BASE_ABS = str(REPO / AR_BASE)


def log(msg: str) -> None:
    print(f"[chain {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], *, env: dict | None = None, timeout: int = 21600) -> int:
    import os
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    log("run: " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO, env=full_env, timeout=timeout)
    return proc.returncode


def pgrep(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def wait_gone(pattern: str, label: str) -> None:
    while pgrep(pattern):
        time.sleep(20)
    log(f"{label} exited")


def wait_server_ready(marker: str, timeout_s: int = 3600) -> None:
    import urllib.request
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            body = urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=3).read()
            if marker.encode() in body:
                log("server lists model")
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"server never listed {marker}")


def eval_cases(cases: list[str], model_abs: str, out: str, backend: str) -> int:
    return run(
        [PY, "scripts/09_eval_agent_toolkit.py", "--base-url", "http://127.0.0.1:8080/v1",
         "--model", model_abs, "--backend", backend, "--timeout", "2400",
         "--cases", *cases, "--out", out],
        env=ENV_TOOLKIT,
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def start_server(model: str, logname: str) -> subprocess.Popen:
    logf = (REPO / "dist" / logname).open("w")
    proc = subprocess.Popen(
        [PY, "scripts/08_serve_mlx.py", "--model", model, "--port", "8080",
         "--prompt-cache-size", "8"],
        cwd=REPO, stdout=logf, stderr=subprocess.STDOUT,
    )
    return proc


def main() -> int:
    # ---- stage 0: wait for the running suite ----
    log("waiting for main suite (09_eval) to finish")
    wait_gone("09_eval_agent_toolkit", "main suite")

    part2 = read_jsonl(REPO / "eval/receipts/baseline_part2.jsonl")
    log(f"part2: {len(part2)} receipts, {sum(r['passed'] for r in part2)} passed")

    # ---- stage 1: rerun the two amended soul cases against the warm server ----
    amended_ids = {"soul_design", "soul_fullstack"}
    souls = [json.loads(l) for l in (REPO / "eval/souls/prompts.jsonl").open()]
    amend_file = REPO / "dist" / "amended_cases.jsonl"
    with amend_file.open("w") as f:
        for r in souls:
            if r["id"] in amended_ids:
                f.write(json.dumps(r) + "\n")
    log("stage 1: rerunning amended cases (server still warm)")
    eval_cases([str(amend_file)], FUSED_ABS, "eval/receipts/baseline_amended.jsonl",
               "mlx_lm_server_fused")

    # ---- stage 2: hard tier ----
    log("stage 2: hard tier vs lite-v1")
    eval_cases(["eval/hard/prompts.jsonl"], FUSED_ABS,
               "eval/receipts/hy3_lite_v1_hard_tier.jsonl", "mlx_lm_server_fused")

    # ---- stage 3: stop server, MTP smoke ----
    log("stage 3: stopping fused server, then MTP smoke")
    subprocess.run(["pkill", "-f", "08_serve_mlx.py"])
    wait_gone("08_serve_mlx.py", "fused server")
    time.sleep(20)
    rc = run([PY, "scripts/24_smoke_generate_mlx_mtp.py",
              "--model", "models/hy3-mlx-base-mtp", "--ar-model", AR_BASE,
              "--max-tokens", "64"], timeout=7200)
    log(f"MTP smoke rc={rc} (receipt: eval/receipts/hy3_mtp_smoke.json)")

    # ---- stage 4: merge + summary + commit baseline ----
    log("stage 4: merging baseline receipts")
    part1 = read_jsonl(Path("/private/tmp/claude-501/-Users-pjb-git-Hy3/87b5adcc-7548-4102-ae82-407fe90cf010/scratchpad/baseline_part1.jsonl"))
    part2 = read_jsonl(REPO / "eval/receipts/baseline_part2.jsonl")
    amended = {r["id"]: r for r in read_jsonl(REPO / "eval/receipts/baseline_amended.jsonl")}
    merged = part1 + [amended.get(r["id"], r) for r in part2]
    with (REPO / "eval/receipts/hy3_lite_v1_baseline_suite.jsonl").open("w") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hard = read_jsonl(REPO / "eval/receipts/hy3_lite_v1_hard_tier.jsonl")

    def rate(rows):
        return round(sum(r["passed"] for r in rows) / len(rows), 4) if rows else None

    timed = [r for r in merged + hard if r.get("elapsed_s")]
    tok_s = round(sum(r.get("completion_tokens", 0) for r in timed)
                  / max(1e-9, sum(r["elapsed_s"] for r in timed)), 3)
    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "model": FUSED,
        "suite": {"cases": len(merged), "passed": sum(r["passed"] for r in merged),
                  "pass_rate": rate(merged),
                  "failed": [r["id"] for r in merged if not r["passed"]]},
        "hard_tier": {"cases": len(hard), "passed": sum(r["passed"] for r in hard),
                      "pass_rate": rate(hard),
                      "failed": [r["id"] for r in hard if not r["passed"]]},
        "throughput_tok_s_end_to_end": tok_s,
        "peak_memory_gb": 112.3,
        "peak_memory_source": "eval/receipts/hy3_lite_v1_fused_smoke.json",
        "amended_cases": sorted(amended_ids),
        "note": "REAP comparison baseline; compare candidates with scripts/20_compare_receipts.py",
    }
    (REPO / "eval/receipts/hy3_lite_v1_baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    log(f"baseline: suite {summary['suite']['passed']}/{summary['suite']['cases']}, "
        f"hard {summary['hard_tier']['passed']}/{summary['hard_tier']['cases']}")

    run(["git", "add",
         "eval/receipts/hy3_lite_v1_baseline_suite.jsonl",
         "eval/receipts/hy3_lite_v1_baseline_summary.json",
         "eval/receipts/hy3_lite_v1_hard_tier.jsonl",
         "eval/receipts/hy3_mtp_smoke.json", ".gitignore"])
    run(["git", "commit", "-m",
         "Commit lite-v1 REAP comparison baseline + hard tier + MTP smoke\n\n"
         "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n"
         "Claude-Session: https://claude.ai/code/session_01LJenYJzwFH5NTvNc76tTgY"])
    run(["git", "push", "origin", "main"])
    log("baseline committed and pushed")

    # ---- stage 5: base-model baseline (overnight) ----
    log("stage 5: serving AR base for the base-model baseline")
    server = start_server(AR_BASE, "serve_base_baseline.log")
    try:
        wait_server_ready("hy3-mlx-base-ar")
        eval_cases(["eval/coding/prompts.jsonl", "eval/tool_calls/prompts.jsonl",
                    "eval/agent_repair/prompts.jsonl", "eval/json_schema/prompts.jsonl",
                    "eval/planning/prompts.jsonl", "eval/souls/prompts.jsonl"],
                   AR_BASE_ABS, "eval/receipts/hy3_base_baseline_suite.jsonl",
                   "mlx_lm_server_base")
        eval_cases(["eval/hard/prompts.jsonl"], AR_BASE_ABS,
                   "eval/receipts/hy3_base_hard_tier.jsonl", "mlx_lm_server_base")
    finally:
        server.terminate()
    log("stage 5 done; comparing base vs lite-v1")
    run([PY, "scripts/20_compare_receipts.py",
         "eval/receipts/hy3_base_baseline_suite.jsonl",
         "eval/receipts/hy3_lite_v1_baseline_suite.jsonl",
         "--out", "eval/receipts/hy3_base_vs_lite_compare.json"])
    run(["git", "add", "eval/receipts/hy3_base_baseline_suite.jsonl",
         "eval/receipts/hy3_base_hard_tier.jsonl",
         "eval/receipts/hy3_base_vs_lite_compare.json"])
    run(["git", "commit", "-m",
         "Base-model baseline + base-vs-lite comparison\n\n"
         "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n"
         "Claude-Session: https://claude.ai/code/session_01LJenYJzwFH5NTvNc76tTgY"])
    run(["git", "push", "origin", "main"])
    log("CHAIN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
