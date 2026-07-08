#!/usr/bin/env python3
"""Build and evaluate reap40 (PJB: "keep going let's do it"; plan was ACCEPT).

The deeper 40% tier (D4). Reuses the existing saliency (no re-calibration) and
the committed fresh lite-v1 baseline as the comparison reference. Same
de-risked chain as reap25: prune --write -> smoke -> is_training train-view ->
heal LoRA -> streamed fuse -> stop-smoke -> full eval -> compare vs lite-v1.

Commits receipts. Does NOT promote/upload — that stays PJB's explicit call.
settle() gates on competing PROCESSES (the reap25 deadlock fix), not free
memory.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "bin" / "python")
TOOLKIT = {"AGENT_TOOLKIT_PATH": "/Users/pjb/git/agent-toolkit", "HY3_PYTHON": PY}
RATIO = "0.40"
PRUNED = "dist/hy3-reap40-pruned"
PRUNED_TRAIN = "dist/hy3-reap40-pruned-train"
ADAPTER = "dist/adapters-hy3-reap40-heal-v1"
FUSED = "dist/hy3-demolition-mlx-reap40-v1-fused"


def log(m: str) -> None:
    print(f"[reap40 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _competing_alive() -> bool:
    pats = ("08_serve_mlx", "14_serve_mlx", "09_eval_agent", "30_benchmark",
            "04_stream_calibrate", "33_reap25")
    return any(subprocess.run(["pgrep", "-f", p], capture_output=True).returncode == 0
               for p in pats)


def sh(cmd, env=None, timeout=57600):
    e = dict(os.environ)
    if env:
        e.update(env)
    log("run: " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO, env=e, timeout=timeout).returncode


def settle():
    for pat in ("08_serve_mlx", "14_serve_mlx", "09_eval", "30_benchmark",
                "04_stream_calibrate"):
        subprocess.run(["pkill", "-f", pat])
    while _competing_alive():
        time.sleep(10)
    time.sleep(25)  # let Metal release wired memory


def git(*args):
    subprocess.run(["git", *args], cwd=REPO)


def commit(msg):
    git("commit", "-m", msg + "\n\nCo-Authored-By: Claude Fable 5 "
        "<noreply@anthropic.com>\nClaude-Session: "
        "https://claude.ai/code/session_01LJenYJzwFH5NTvNc76tTgY")
    git("push", "origin", "main")


def read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def main() -> int:
    settle()
    plan_report = REPO / "eval/receipts/hy3_reap40_plan_report.json"
    if json.loads(plan_report.read_text())["overall"] == "REJECT":
        log("ABORT: reap40 plan is REJECT")
        return 1

    log("stage 1: prune --write (192 -> 115 experts, 40%)")
    if sh([PY, "scripts/05_apply_reap_prune_hy3_mlx.py", "--model", "models/hy3-mlx-base-ar",
           "--saliency", "dist/hy3-reap-saliency-v1.json", "--out", PRUNED,
           "--ratio", RATIO, "--write"]) != 0:
        log("ABORT: prune failed")
        return 1

    log("stage 2: pruned-model smoke")
    settle()
    sh([PY, "scripts/13_smoke_generate_mlx_ar.py", "--model", PRUNED,
        "--prompt", "Write a one-line Python function that doubles a number.",
        "--max-tokens", "48", "--receipt", "eval/receipts/hy3_reap40_pruned_smoke.json"])

    log("stage 3: is_training train-view of the pruned dir")
    if sh([PY, "scripts/17_prepare_hy3_train_view.py", "--source", PRUNED,
           "--out", PRUNED_TRAIN]) != 0:
        log("ABORT: train-view prep failed")
        return 1

    log("stage 4: heal LoRA on the balanced pack")
    settle()
    if sh([PY, "scripts/07_heal_lora_hy3_mlx.py", "--model", PRUNED_TRAIN,
           "--data", "data/hy3_lite_sft_combined", "--adapter-path", ADAPTER,
           "--iters", "200", "--num-layers", "8", "--max-seq-length", "2048",
           "--train"]) != 0:
        log("ABORT: heal failed (if iter-1 NaN, D2: drop --mask-prompt)")
        return 1

    log("stage 5: streamed fuse")
    settle()
    if sh([PY, "scripts/18_fuse_lora_streamed.py", "--model", PRUNED,
           "--adapter-path", ADAPTER, "--save-path", FUSED,
           "--card", "cards/hy3-demolition-mlx-lite-v1.md"]) != 0:
        log("ABORT: fuse failed")
        return 1

    log("stage 6: fused reap40 stop-behavior smoke")
    settle()
    sh([PY, "scripts/13_smoke_generate_mlx_ar.py", "--model", FUSED,
        "--prompt", "Write a one-line Python function that doubles a number.",
        "--max-tokens", "64", "--receipt", "eval/receipts/hy3_reap40_fused_smoke.json"])

    git("add", "-f", f"{PRUNED}/reap_plan.json", f"{PRUNED}/config.json")
    git("add", "eval/receipts/hy3_reap40_pruned_smoke.json",
        "eval/receipts/hy3_reap40_fused_smoke.json")
    commit("reap40 built: prune 40% -> heal -> fuse (smokes committed)")

    log("stage 7: full eval (suite+hard+brutal) on reap40")
    settle()
    logf = (REPO / "dist" / "serve_reap40.log").open("w")
    srv = subprocess.Popen([PY, "scripts/08_serve_mlx.py", "--model", str(REPO / FUSED),
                            "--port", "8080", "--prompt-cache-size", "8"],
                           cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 3600
        ready = False
        while time.time() < deadline:
            try:
                if b"reap40" in urllib.request.urlopen(
                        "http://127.0.0.1:8080/v1/models", timeout=3).read():
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(5)
        if ready:
            sh([PY, "scripts/09_eval_agent_toolkit.py", "--base-url",
                "http://127.0.0.1:8080/v1", "--model", str(REPO / FUSED),
                "--backend", "mlx_lm_server_reap40", "--timeout", "2400",
                "--cases", "eval/coding/prompts.jsonl", "eval/tool_calls/prompts.jsonl",
                "eval/agent_repair/prompts.jsonl", "eval/json_schema/prompts.jsonl",
                "eval/planning/prompts.jsonl", "eval/souls/prompts.jsonl",
                "eval/hard/prompts.jsonl", "eval/brutal/prompts.jsonl",
                "--out", "eval/receipts/hy3_reap40_eval.jsonl"], env=TOOLKIT)
        else:
            log("server never came up; skipping eval")
    finally:
        srv.terminate()

    log("stage 8: compare vs fresh lite-v1 baseline")
    rc = sh([PY, "scripts/20_compare_receipts.py",
             "eval/receipts/hy3_lite_v1_fresh_baseline.jsonl",
             "eval/receipts/hy3_reap40_eval.jsonl",
             "--out", "eval/receipts/hy3_reap40_vs_lite_compare.json"])
    git("add", "eval/receipts/hy3_reap40_eval.jsonl",
        "eval/receipts/hy3_reap40_vs_lite_compare.json")
    verdict = "?"
    cmp_path = REPO / "eval/receipts/hy3_reap40_vs_lite_compare.json"
    if cmp_path.exists():
        verdict = json.loads(cmp_path.read_text()).get("verdict", "?")
    rows = read_jsonl(REPO / "eval/receipts/hy3_reap40_eval.jsonl")
    commit(f"reap40 eval + compare vs lite-v1: {verdict} ({sum(r['passed'] for r in rows)}/{len(rows)})")
    log(f"REAP40 TESTED — verdict: {verdict}, {sum(r['passed'] for r in rows)}/{len(rows)} (rc={rc})")
    log("NOT promoted — HF upload / tag is PJB's explicit call")
    log("REAP40 COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
