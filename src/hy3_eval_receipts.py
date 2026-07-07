"""OpenAI-compatible eval calls plus verifier receipts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    id: str
    domain: str
    prompt: str
    harness: str = ""
    expected_format: str = "text"
    facet: str = "coding"


@dataclass
class EvalReceipt:
    id: str
    domain: str
    backend: str
    model: str
    passed: bool
    stage: str
    diag: str
    output: str
    created_at: float


def load_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                cases.append(EvalCase(**json.loads(line)))
    return cases


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout: int = 600,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    return body["choices"][0]["message"].get("content") or ""


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```", re.DOTALL)


def extract_verifiable_text(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _load_verify_domain():
    toolkit = os.environ.get("AGENT_TOOLKIT_PATH", "/Users/pjb/git/agent-toolkit")
    verify_path = str(Path(toolkit) / "verify")
    if verify_path not in sys.path:
        sys.path.insert(0, verify_path)
    from verifiers import verify_domain  # type: ignore

    return verify_domain


def verify_output(case: EvalCase, output: str) -> tuple[bool, str, str]:
    verify_domain = _load_verify_domain()
    result = verify_domain(
        case.domain,
        extract_verifiable_text(output),
        harness=case.harness,
    )
    return bool(result.passed), str(result.stage), str(result.diag)


def run_cases(
    *,
    cases: list[EvalCase],
    base_url: str,
    model: str,
    backend: str,
    out: str | Path,
    system_prompt: str,
    max_tokens: int,
) -> list[EvalReceipt]:
    receipts: list[EvalReceipt] = []
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for case in cases:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": case.prompt},
            ]
            output = chat_completion(
                base_url,
                model,
                messages,
                max_tokens=max_tokens,
            )
            passed, stage, diag = verify_output(case, output)
            receipt = EvalReceipt(
                id=case.id,
                domain=case.domain,
                backend=backend,
                model=model,
                passed=passed,
                stage=stage,
                diag=diag,
                output=output,
                created_at=time.time(),
            )
            f.write(json.dumps(asdict(receipt), ensure_ascii=False) + "\n")
            f.flush()
            receipts.append(receipt)
    return receipts


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="hy3")
    parser.add_argument("--backend", default="mlx_lm")
    parser.add_argument("--max-tokens", type=int, default=512)

