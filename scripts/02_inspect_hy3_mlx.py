#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hy3_weight_store import inspect_checkpoint, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/hy3-mlx-base")
    parser.add_argument("--out", default="")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    info = inspect_checkpoint(args.model)
    payload = info.__dict__
    if args.summary:
        quant = payload.get("quantization_summary") or {}
        missing = payload.get("missing_safetensor_files") or []
        bits = quant.get("bits") if isinstance(quant, dict) else None
        print(
            json.dumps(
                {
                    "path": payload["path"],
                    "model_type": payload["model_type"],
                    "layers": payload["num_hidden_layers"],
                    "experts": payload["num_experts"],
                    "experts_per_token": payload["num_experts_per_tok"],
                    "safetensors": payload["safetensor_count"],
                    "expected_safetensors": payload["expected_safetensor_count"],
                    "missing_count": len(missing),
                    "has_index": payload["has_index"],
                    "total_size": payload["total_size"],
                    "quant_bits": bits,
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(payload, indent=2))
    if args.out:
        write_json(Path(args.out), payload)


if __name__ == "__main__":
    main()
