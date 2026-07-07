#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hy3_mlx_server import ServeConfig, run_server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/hy3-mlx-base")
    parser.add_argument("--adapter-path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--allowed-origins", default="*")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--num-draft-tokens", type=int, default=0)
    parser.add_argument("--decode-concurrency", type=int)
    parser.add_argument("--prompt-concurrency", type=int)
    parser.add_argument("--prefill-step-size", type=int)
    parser.add_argument("--prompt-cache-size", type=int)
    parser.add_argument("--prompt-cache-bytes", type=int)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--wired-limit",
        action="store_true",
        help="Use mlx_lm.server's default wired-limit pinning.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run_server(
            ServeConfig(
                model=args.model,
                adapter_path=args.adapter_path,
                host=args.host,
                port=args.port,
                allowed_origins=args.allowed_origins,
                max_tokens=args.max_tokens,
                temp=args.temp,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                num_draft_tokens=args.num_draft_tokens,
                decode_concurrency=args.decode_concurrency,
                prompt_concurrency=args.prompt_concurrency,
                prefill_step_size=args.prefill_step_size,
                prompt_cache_size=args.prompt_cache_size,
                prompt_cache_bytes=args.prompt_cache_bytes,
                log_level=args.log_level,
                no_wired_limit=not args.wired_limit,
                dry_run=args.dry_run,
            )
        )
    )


if __name__ == "__main__":
    main()
