#!/usr/bin/env python3
"""Overnight GPU handoff: when the baseline chain completes, run REAP
calibration, then the dry-run prune plan + analyzer, and commit receipts.

Writes NO model weights — the prune --write step stays human-gated per
DECISIONS.md D1. Assumes the fp32-router patch is applied to the installed
fork (verified by RESTORE.md note; calibration must not measure bf16
routing).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "bin" / "python")


def log(msg: str) -> None:
    print(f"[overnight {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], timeout: int = 57600) -> int:
    log("run: " + " ".join(cmd))
    try:
        return subprocess.run(cmd, cwd=REPO, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT after {timeout}s: {' '.join(cmd)}")
        return 124


def pgrep(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


# every process pattern that can hold model memory; 14_serve is the
# GRANDCHILD the chain's server.terminate() does not reach (audit finding)
GPU_PATTERNS = (
    "14_serve_mlx_ar_nowire",
    "mlx_lm server",
    "mlx_lm.server",
    "08_serve_mlx",
    "09_eval_agent_toolkit",
)


def main() -> int:
    # wait for the baseline chain to fully finish
    log("waiting for baseline chain to complete")
    chain_log = REPO / "dist" / "chain.log"
    while True:
        if "CHAIN COMPLETE" in chain_log.read_text():
            break
        if not pgrep("27_baseline_chain"):
            # TOCTOU: the chain may have completed and exited between the log
            # read and the pgrep — re-read before declaring failure
            if "CHAIN COMPLETE" in chain_log.read_text():
                break
            log("chain process gone without CHAIN COMPLETE — check dist/chain.log; aborting")
            return 1
        time.sleep(60)
    log("chain complete; clearing GPU processes (incl. orphaned grandchild server)")
    deadline = time.time() + 600
    while any(pgrep(p) for p in GPU_PATTERNS):
        if time.time() > deadline:
            for p in GPU_PATTERNS:
                subprocess.run(["pkill", "-9", "-f", p])
            log("escalated to SIGKILL on lingering GPU processes")
            deadline = time.time() + 600
        else:
            for p in GPU_PATTERNS:
                subprocess.run(["pkill", "-f", p])
        time.sleep(20)
    log("GPU clear; settling 60s")
    time.sleep(60)

    # Memory precheck: calibration loads ~112 GB. If free memory is low, some
    # OTHER process (e.g. PJB using the GPU) owns it — do NOT load a second
    # giant model and trigger jetsam. Wait it out rather than fight.
    def free_gb() -> float:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "Pages free" in line:
                return int(line.split()[-1].rstrip(".")) * 16384 / 1e9
        return 0.0

    waited = 0
    while free_gb() < 90:
        if waited == 0:
            log(f"only {free_gb():.0f} GB free — something else is using the GPU; "
                "waiting for it to free before loading calibration")
        time.sleep(120)
        waited += 120
        if waited > 43200:  # 12 h
            log("ABORT: GPU still busy after 12 h; not starting calibration")
            return 1
    if waited:
        log(f"GPU free again ({free_gb():.0f} GB) after {waited // 60} min wait")

    # sanity: fp32 router patch present (D0 — calibration must not run without it)
    import importlib.util
    spec = importlib.util.find_spec("mlx_lm.models.hy_v3")
    src = Path(spec.origin).read_text()
    if "astype(mx.float32)" not in src:
        log("ABORT: fp32 router patch missing from installed fork (see RESTORE.md)")
        return 1

    log("stage A: REAP calibration (true criterion, this is the long pole)")
    rc = run([
        PY, "scripts/04_stream_calibrate_hy3_mlx.py",
        "--model", "models/hy3-mlx-base-ar",
        "--prompts", "data/hy3_reap_calibration/prompts.jsonl",
        "--out", "dist/hy3-reap-saliency-v1.json",
    ])
    if rc != 0:
        log(f"ABORT: calibration failed rc={rc}")
        return rc

    sal = json.loads((REPO / "dist/hy3-reap-saliency-v1.json").read_text())
    n_layers = len(sal.get("layers", {}))
    facets = set()
    for layer in sal["layers"].values():
        facets.update(layer.get("facets", {}))
    log(f"calibration done: {n_layers} MoE layers, facets: {sorted(facets)}")

    log("stage B: dry-run prune plan (NO --write)")
    rc = run([
        PY, "scripts/05_apply_reap_prune_hy3_mlx.py",
        "--model", "models/hy3-mlx-base-ar",
        "--saliency", "dist/hy3-reap-saliency-v1.json",
        "--out", "dist/hy3-reap25-pruned",
        "--ratio", "0.25",
    ])
    if rc != 0:
        log(f"ABORT: plan build failed rc={rc}")
        return rc

    log("stage C: analyzer verdict")
    rc = run([
        PY, "scripts/25_analyze_reap_plan.py",
        "dist/hy3-reap25-pruned/reap_plan.json",
        "--saliency", "dist/hy3-reap-saliency-v1.json",
        "--out", "eval/receipts/hy3_reap25_plan_report.json",
    ])
    log(f"analyzer exit {rc} (0=ACCEPT/REVIEW, 1=REJECT) — report committed either way")

    # commit the morning-coffee bundle (plan + report; saliency stays local, it's big)
    run(["git", "add", "eval/receipts/hy3_reap25_plan_report.json"])
    subprocess.run(
        ["git", "add", "-f", "dist/hy3-reap25-pruned/reap_plan.json"], cwd=REPO
    )
    run(["git", "commit", "-m",
         "Overnight: REAP calibration + dry-run 25% plan + analyzer report\n\n"
         "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n"
         "Claude-Session: https://claude.ai/code/session_01LJenYJzwFH5NTvNc76tTgY"])
    run(["git", "push", "origin", "main"])
    log("OVERNIGHT COMPLETE — plan report awaits PJB review; prune --write is the next human gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
