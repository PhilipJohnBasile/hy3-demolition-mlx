#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def python_executable() -> str:
    if os.environ.get("HY3_PYTHON"):
        return os.environ["HY3_PYTHON"]
    repo_python = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    if repo_python.exists():
        return str(repo_python)
    return sys.executable


def run(cmd: list[str], dry_run: bool = False) -> None:
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/hy3-mlx-base")
    parser.add_argument("--data", default="data/hy3_lite_sft")
    parser.add_argument("--adapter-path", default="dist/adapters-hy3-lite")
    parser.add_argument("--save-path", default="dist/hy3-demolition-mlx-lite-fused")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    Path(args.adapter_path).mkdir(parents=True, exist_ok=True)
    py = python_executable()
    if args.train:
        run(
            [
                py,
                "-m",
                "mlx_lm",
                "lora",
                "--model",
                args.model,
                "--data",
                args.data,
                "--adapter-path",
                args.adapter_path,
                "--train",
                "--fine-tune-type",
                "lora",
                "--optimizer",
                "adamw",
                "--iters",
                str(args.iters),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--num-layers",
                str(args.num_layers),
                "--max-seq-length",
                str(args.max_seq_length),
                "--grad-checkpoint",
                "--mask-prompt",
            ],
            dry_run=args.dry_run,
        )

    if args.test:
        run(
            [
                py,
                "-m",
                "mlx_lm",
                "lora",
                "--model",
                args.model,
                "--data",
                args.data,
                "--adapter-path",
                args.adapter_path,
                "--test",
            ],
            dry_run=args.dry_run,
        )

    if args.fuse:
        run(
            [
                py,
                "-m",
                "mlx_lm",
                "fuse",
                "--model",
                args.model,
                "--adapter-path",
                args.adapter_path,
                "--save-path",
                args.save_path,
            ],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
