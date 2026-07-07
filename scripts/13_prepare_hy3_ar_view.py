#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def rel_symlink(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(os.path.relpath(source, start=target.parent))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an AR-only Hy3 MLX view that omits the MTP sidecar."
    )
    parser.add_argument("--source", default="models/hy3-mlx-base")
    parser.add_argument("--out", default="models/hy3-mlx-base-ar")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not (source / "config.json").exists():
        raise FileNotFoundError(source / "config.json")

    for item in source.iterdir():
        if item.name in {
            "config.json",
            "model-mtp.safetensors",
            "tokenizer_config.json",
        }:
            continue
        if item.is_file():
            rel_symlink(item, out / item.name)

    with (source / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["num_nextn_predict_layers"] = 0
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")

    tokenizer_config_path = source / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config_out = out / "tokenizer_config.json"
        if tokenizer_config_out.exists() or tokenizer_config_out.is_symlink():
            tokenizer_config_out.unlink()
        with tokenizer_config_path.open("r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        tokenizer_config["fix_mistral_regex"] = True
        with tokenizer_config_out.open("w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, sort_keys=True)
            f.write("\n")

    print(f"created {out}")
    print("num_nextn_predict_layers=0")
    print("model-mtp.safetensors omitted")
    print("fix_mistral_regex=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
