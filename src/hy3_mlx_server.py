"""Serve a Hy3 MLX model with mlx_lm."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServeConfig:
    model: str
    adapter_path: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    allowed_origins: str = "*"
    max_tokens: int = 1024
    temp: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    num_draft_tokens: int = 0
    decode_concurrency: int | None = None
    prompt_concurrency: int | None = None
    prefill_step_size: int | None = None
    prompt_cache_size: int | None = None
    prompt_cache_bytes: int | None = None
    chat_template_args: str = '{"reasoning_effort":"no_think"}'
    log_level: str = "INFO"
    no_wired_limit: bool = True
    dry_run: bool = False


def python_executable() -> str:
    if os.environ.get("HY3_PYTHON"):
        return os.environ["HY3_PYTHON"]
    repo_python = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    if repo_python.exists():
        return str(repo_python)
    return sys.executable


def mlx_lm_command(cfg: ServeConfig) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    server_entry = (
        [str(repo_root / "scripts" / "14_serve_mlx_ar_nowire.py")]
        if cfg.no_wired_limit
        else ["-m", "mlx_lm", "server"]
    )
    cmd = [
        python_executable(),
        *server_entry,
        "--model",
        cfg.model,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--allowed-origins",
        cfg.allowed_origins,
        "--max-tokens",
        str(cfg.max_tokens),
        "--temp",
        str(cfg.temp),
        "--top-p",
        str(cfg.top_p),
        "--top-k",
        str(cfg.top_k),
        "--min-p",
        str(cfg.min_p),
        "--num-draft-tokens",
        str(cfg.num_draft_tokens),
        "--log-level",
        cfg.log_level,
        "--chat-template-args",
        cfg.chat_template_args,
    ]
    optional = (
        ("--adapter-path", cfg.adapter_path),
        ("--decode-concurrency", cfg.decode_concurrency),
        ("--prompt-concurrency", cfg.prompt_concurrency),
        ("--prefill-step-size", cfg.prefill_step_size),
        ("--prompt-cache-size", cfg.prompt_cache_size),
        ("--prompt-cache-bytes", cfg.prompt_cache_bytes),
    )
    for flag, value in optional:
        if value is not None:
            cmd.extend([flag, str(value)])
    return cmd


def run_server(cfg: ServeConfig) -> int:
    cmd = mlx_lm_command(cfg)
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if cfg.dry_run:
        return 0
    return subprocess.call(cmd)
