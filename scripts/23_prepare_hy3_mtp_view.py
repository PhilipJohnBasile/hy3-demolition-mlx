#!/usr/bin/env python3
"""Create an MTP-enabled Hy3 MLX view (the counterpart of the AR-only view).

Symlinks everything from the base checkpoint INCLUDING model-mtp.safetensors,
keeps num_nextn_predict_layers as shipped (1), and writes the tokenizer config
with fix_mistral_regex. The hy_v3-mtp mlx-lm fork auto-enables self-speculative
decoding (mtp_generate_step) whenever num_nextn_predict_layers > 0, so pointing
generate/serve at this view is all it takes to exercise the MTP path.
"""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="models/hy3-mlx-base")
    parser.add_argument("--out", default="models/hy3-mlx-base-mtp")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    config = json.loads((source / "config.json").read_text())
    if not config.get("num_nextn_predict_layers"):
        raise ValueError(f"{source}/config.json has no MTP layers; nothing to enable")
    if not (source / "model-mtp.safetensors").exists():
        raise FileNotFoundError(source / "model-mtp.safetensors")

    for item in source.iterdir():
        if item.name == "tokenizer_config.json":
            continue
        if item.is_file():
            rel_symlink(item, out / item.name)

    tokenizer_config = json.loads((source / "tokenizer_config.json").read_text())
    tokenizer_config["fix_mistral_regex"] = True
    (out / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n"
    )

    print(f"created {out}")
    print(f"num_nextn_predict_layers={config['num_nextn_predict_layers']} (MTP enabled)")
    print("model-mtp.safetensors included")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
