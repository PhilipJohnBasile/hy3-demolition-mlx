#!/usr/bin/env python3
"""Build and evaluate reap25 end-to-end (PJB: "you test it", plan was ACCEPT).

Sequence (all committing receipts; NO promotion/HF upload — that stays PJB's
explicit call): prune --write -> smoke -> is_training train-view -> heal LoRA
-> streamed fuse -> stop-smoke -> serve -> full eval (suite+hard+brutal) ->
compare vs the fresh lite-v1 baseline. Every REAP-path step here was
de-risked on tiny models tonight.

Waits for the fresh baseline (the D3 comparison reference) to finish, then
preempts the follower's lower-priority benchmark/base-baseline stages —
testing reap25 is the priority. Prune plan is already committed + ACCEPT.
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
PRUNED = "dist/hy3-reap25-pruned"
PRUNED_TRAIN = "dist/hy3-reap25-pruned-train"
ADAPTER = "dist/adapters-hy3-reap25-heal-v1"
FUSED = "dist/hy3-demolition-mlx-reap25-v1-fused"


def log(m: str) -> None:
    print(f"[reap25 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def _competing_alive() -> bool:
    pats = ("08_serve_mlx", "14_serve_mlx", "09_eval_agent", "30_benchmark",
            "04_stream_calibrate", "32_fresh_baseline")
    return any(subprocess.run(["pgrep", "-f", p], capture_output=True).returncode == 0
               for p in pats)


def free_gb() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    free = inactive = 0
    for line in out.splitlines():
        if "Pages free" in line:
            free = int(line.split()[-1].rstrip("."))
        elif "Pages inactive" in line:
            inactive = int(line.split()[-1].rstrip("."))
    return (free + inactive) * 16384 / 1e9


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


def serve_and_eval(model_abs, out, backend):
    logf = (REPO / "dist" / "serve_reap25.log").open("w")
    srv = subprocess.Popen([PY, "scripts/08_serve_mlx.py", "--model", model_abs,
                            "--port", "8080", "--prompt-cache-size", "8"],
                           cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 3600
        ready = False
        while time.time() < deadline:
            try:
                if b"reap25" in urllib.request.urlopen(
                        "http://127.0.0.1:8080/v1/models", timeout=3).read():
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(5)
        if not ready:
            log("server never came up; aborting eval")
            return 1
        return sh([PY, "scripts/09_eval_agent_toolkit.py",
                   "--base-url", "http://127.0.0.1:8080/v1", "--model", model_abs,
                   "--backend", backend, "--timeout", "2400",
                   "--cases", "eval/coding/prompts.jsonl", "eval/tool_calls/prompts.jsonl",
                   "eval/agent_repair/prompts.jsonl", "eval/json_schema/prompts.jsonl",
                   "eval/planning/prompts.jsonl", "eval/souls/prompts.jsonl",
                   "eval/hard/prompts.jsonl", "eval/brutal/prompts.jsonl",
                   "--out", out], env=TOOLKIT)
    finally:
        srv.terminate()


def main() -> int:
    # wait for the fresh baseline (the reap25 comparison reference)
    log("waiting for fresh lite-v1 baseline (the reap25 comparison reference)")
    fb = REPO / "dist" / "freshbase.log"
    while True:
        txt = fb.read_text() if fb.exists() else ""
        if "FRESH BASELINE READY" in txt:
            break
        time.sleep(60)
    log("fresh baseline done; preempting follower's benchmark/base stages")
    subprocess.run(["pkill", "-f", "32_fresh_baseline_follower"])
    settle()

    # sanity: plan exists and was ACCEPT
    plan_report = REPO / "eval/receipts/hy3_reap25_plan_report.json"
    if json.loads(plan_report.read_text())["overall"] == "REJECT":
        log("ABORT: plan report is REJECT")
        return 1

    # 1. prune --write (streams shard-by-shard; low memory)
    log("stage 1: prune --write (192 -> 144 experts)")
    if sh([PY, "scripts/05_apply_reap_prune_hy3_mlx.py",
           "--model", "models/hy3-mlx-base-ar",
           "--saliency", "dist/hy3-reap-saliency-v1.json",
           "--out", PRUNED, "--ratio", "0.25", "--write"]) != 0:
        log("ABORT: prune failed")
        return 1

    # 2. direct smoke on the pruned model
    log("stage 2: pruned-model smoke")
    settle()
    sh([PY, "scripts/13_smoke_generate_mlx_ar.py", "--model", PRUNED,
        "--prompt", "Write a one-line Python function that doubles a number.",
        "--max-tokens", "48", "--receipt", "eval/receipts/hy3_reap25_pruned_smoke.json"])

    # 3. is_training train-view of the PRUNED dir (D2 landmine)
    log("stage 3: is_training train-view of the pruned dir")
    if sh([PY, "scripts/17_prepare_hy3_train_view.py",
           "--source", PRUNED, "--out", PRUNED_TRAIN]) != 0:
        log("ABORT: train-view prep failed")
        return 1

    # 4. heal LoRA (8 layers, 200 iters, seq 2048) against the train view
    log("stage 4: heal LoRA on the balanced pack")
    settle()
    if sh([PY, "scripts/07_heal_lora_hy3_mlx.py",
           "--model", PRUNED_TRAIN, "--data", "data/hy3_lite_sft_combined",
           "--adapter-path", ADAPTER, "--iters", "200", "--num-layers", "8",
           "--max-seq-length", "2048", "--train"]) != 0:
        log("ABORT: heal failed (if iter-1 NaN, D2: drop --mask-prompt)")
        return 1

    # 5. streamed fuse against the stock-template pruned dir
    log("stage 5: streamed fuse")
    settle()
    if sh([PY, "scripts/18_fuse_lora_streamed.py",
           "--model", PRUNED, "--adapter-path", ADAPTER, "--save-path", FUSED,
           "--card", "cards/hy3-demolition-mlx-lite-v1.md"]) != 0:
        log("ABORT: fuse failed")
        return 1

    # 6. stop-behavior smoke on the fused reap25
    log("stage 6: fused reap25 stop-behavior smoke")
    settle()
    sh([PY, "scripts/13_smoke_generate_mlx_ar.py", "--model", FUSED,
        "--prompt", "Write a one-line Python function that doubles a number.",
        "--max-tokens", "64", "--receipt", "eval/receipts/hy3_reap25_fused_smoke.json"])

    git("add", "-f", f"{PRUNED}/reap_plan.json", f"{PRUNED}/config.json")
    git("add", "eval/receipts/hy3_reap25_pruned_smoke.json",
        "eval/receipts/hy3_reap25_fused_smoke.json")
    commit("reap25 built: prune -> heal -> fuse (smokes committed)")

    # 7. full eval + compare vs the fresh lite-v1 baseline
    log("stage 7: full eval (suite+hard+brutal) on reap25")
    settle()
    serve_and_eval(str(REPO / FUSED),
                   "eval/receipts/hy3_reap25_eval.jsonl", "mlx_lm_server_reap25")

    log("stage 8: compare vs fresh lite-v1 baseline")
    rc = sh([PY, "scripts/20_compare_receipts.py",
             "eval/receipts/hy3_lite_v1_fresh_baseline.jsonl",
             "eval/receipts/hy3_reap25_eval.jsonl",
             "--out", "eval/receipts/hy3_reap25_vs_lite_compare.json"])
    git("add", "eval/receipts/hy3_reap25_eval.jsonl",
        "eval/receipts/hy3_reap25_vs_lite_compare.json")
    verdict = "?"
    cmp_path = REPO / "eval/receipts/hy3_reap25_vs_lite_compare.json"
    if cmp_path.exists():
        verdict = json.loads(cmp_path.read_text()).get("verdict", "?")
    commit(f"reap25 eval + compare vs lite-v1: {verdict}")
    log(f"REAP25 TESTED — compare verdict: {verdict} (rc={rc})")
    log("NOT promoted — HF upload / tag is PJB's explicit call (see compare receipt)")

    # ---- deferred bonus stages (evals are fast; PJB wanted #15 done) ----
    log("stage 9: AR vs MTP path benchmark")
    settle()
    sh([PY, "scripts/30_benchmark_paths.py", "--ar-model", "models/hy3-mlx-base-ar",
        "--mtp-model", "models/hy3-mlx-base-mtp", "--max-tokens", "48",
        "--out", "eval/receipts/hy3_path_benchmark.json"])
    git("add", "eval/receipts/hy3_path_benchmark.json")
    commit("AR vs MTP path benchmark")

    log("stage 10: clean base-model baseline (#15)")
    settle()
    AR = str(REPO / "models" / "hy3-mlx-base-ar")
    logf = (REPO / "dist" / "serve_base_clean.log").open("w")
    srv = subprocess.Popen([PY, "scripts/08_serve_mlx.py", "--model", AR,
                            "--port", "8080", "--prompt-cache-size", "8"],
                           cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 3600
        while time.time() < deadline:
            try:
                if b"hy3-mlx-base-ar" in urllib.request.urlopen(
                        "http://127.0.0.1:8080/v1/models", timeout=3).read():
                    break
            except Exception:
                pass
            time.sleep(5)
        sh([PY, "scripts/09_eval_agent_toolkit.py", "--base-url",
            "http://127.0.0.1:8080/v1", "--model", AR, "--backend", "mlx_lm_server_base",
            "--timeout", "2400", "--cases", "eval/coding/prompts.jsonl",
            "eval/tool_calls/prompts.jsonl", "eval/agent_repair/prompts.jsonl",
            "eval/json_schema/prompts.jsonl", "eval/planning/prompts.jsonl",
            "eval/souls/prompts.jsonl", "eval/hard/prompts.jsonl", "eval/brutal/prompts.jsonl",
            "--out", "eval/receipts/hy3_base_clean_baseline.jsonl"], env=TOOLKIT)
    finally:
        srv.terminate()
    base_rows = read_jsonl(REPO / "eval/receipts/hy3_base_clean_baseline.jsonl")
    git("add", "-f", "eval/receipts/hy3_base_clean_baseline.jsonl")
    if len(base_rows) >= 38:
        sh([PY, "scripts/20_compare_receipts.py", "eval/receipts/hy3_base_clean_baseline.jsonl",
            "eval/receipts/hy3_lite_v1_fresh_baseline.jsonl",
            "--out", "eval/receipts/hy3_base_vs_lite_clean.json"])
        git("add", "eval/receipts/hy3_base_vs_lite_clean.json")
    commit(f"Clean base-model baseline #15 ({sum(r['passed'] for r in base_rows)}/{len(base_rows)}) + base-vs-lite")
    log(f"base baseline #15: {sum(r['passed'] for r in base_rows)}/{len(base_rows)}")
    log("OVERNIGHT COMPLETE — reap25 verdict, benchmark, base baseline all in git")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
