#!/usr/bin/env python3
"""Create a training-only Hy3 view whose chat template defaults is_training=true.

mlx_lm's ChatDataset applies the chat template with no extra kwargs, and the
stock Hy3 template only appends eos_token to the final assistant turn when
is_training is set. Training against the plain AR view therefore teaches the
model that assistant turns end without EOS, which destroys stop behavior
(observed as token-0 "!" spam after otherwise correct answers).

This view symlinks everything from the AR view and prepends a default() so the
template appends EOS during training while remaining override-able. Do not
serve from this view; use it only as the --model for mlx_lm lora. Fusion should
still target the AR view so the released model keeps the stock template.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def rel_symlink(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(os.path.relpath(source, start=target.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="models/hy3-mlx-base-ar")
    parser.add_argument("--out", default="models/hy3-mlx-base-ar-train")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    template_path = source / "chat_template.jinja"
    if not template_path.exists():
        raise FileNotFoundError(template_path)

    for item in source.iterdir():
        if item.name == "chat_template.jinja":
            continue
        if item.is_file() or item.is_symlink():
            rel_symlink(item.resolve(), out / item.name)

    template = template_path.read_text(encoding="utf-8")
    header = "{%- set is_training = is_training | default(true) %}\n"
    if not template.startswith(header):
        template = header + template
    (out / "chat_template.jinja").write_text(template, encoding="utf-8")

    print(f"created {out}")
    print("chat_template.jinja: is_training defaults to true (EOS appended to final assistant turn)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
