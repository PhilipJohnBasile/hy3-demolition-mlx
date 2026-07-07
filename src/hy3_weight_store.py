"""MLX safetensors checkpoint inspection and streaming helpers."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class CheckpointInfo:
    path: str
    model_type: str | None
    num_hidden_layers: int | None
    num_experts: int | None
    num_experts_per_tok: int | None
    quantization_summary: object | None
    safetensor_files: list[str]
    safetensor_count: int
    expected_safetensor_count: int | None
    missing_safetensor_files: list[str]
    has_index: bool
    total_size: int | None


def config_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "config.json"


def load_config(model_dir: str | Path) -> dict:
    with config_path(model_dir).open() as f:
        return json.load(f)


def index_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "model.safetensors.index.json"


def safetensor_files(model_dir: str | Path) -> list[Path]:
    root = Path(model_dir)
    idx = index_path(root)
    if idx.exists():
        with idx.open() as f:
            data = json.load(f)
        names = sorted(set(data.get("weight_map", {}).values()))
        return [root / name for name in names]
    return sorted(root.glob("*.safetensors"))


def summarize_quantization(quantization: object | None) -> object | None:
    if not isinstance(quantization, dict):
        return quantization

    scalar_items = {
        key: value
        for key, value in quantization.items()
        if not isinstance(value, dict)
    }
    module_items = {
        key: value
        for key, value in quantization.items()
        if isinstance(value, dict)
    }
    bit_counts: dict[str, int] = {}
    group_size_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    for params in module_items.values():
        for source, target in (
            ("bits", bit_counts),
            ("group_size", group_size_counts),
            ("mode", mode_counts),
        ):
            value = params.get(source)
            if value is not None:
                target[str(value)] = target.get(str(value), 0) + 1

    examples = {
        key: module_items[key]
        for key in sorted(module_items)[:12]
    }
    return {
        "kind": "dict",
        "entry_count": len(quantization),
        "global": scalar_items,
        "module_entry_count": len(module_items),
        "bits": dict(sorted(bit_counts.items())),
        "group_sizes": dict(sorted(group_size_counts.items())),
        "modes": dict(sorted(mode_counts.items())),
        "examples": examples,
    }


def inspect_checkpoint(model_dir: str | Path) -> CheckpointInfo:
    root = Path(model_dir)
    cfg = load_config(root)
    idx = index_path(root)
    index_data = None
    if idx.exists():
        with idx.open() as f:
            index_data = json.load(f)
    files = safetensor_files(root)
    quantization = cfg.get("quantization") or cfg.get("quantization_config")
    expected_count, missing = missing_safetensors(files)
    return CheckpointInfo(
        path=str(root),
        model_type=cfg.get("model_type"),
        num_hidden_layers=cfg.get("num_hidden_layers"),
        num_experts=cfg.get("num_experts"),
        num_experts_per_tok=cfg.get("num_experts_per_tok"),
        quantization_summary=summarize_quantization(quantization),
        safetensor_files=[p.name for p in files],
        safetensor_count=len(files),
        expected_safetensor_count=expected_count,
        missing_safetensor_files=missing,
        has_index=idx.exists(),
        total_size=(index_data or {}).get("metadata", {}).get("total_size"),
    )


_SHARD_RE = re.compile(r"model-(\d+)-of-(\d+)\.safetensors$")


def missing_safetensors(files: list[Path]) -> tuple[int | None, list[str]]:
    seen: set[int] = set()
    expected: int | None = None
    widths: dict[int, int] = {}
    for path in files:
        match = _SHARD_RE.match(path.name)
        if not match:
            continue
        seen.add(int(match.group(1)))
        expected = int(match.group(2))
        width = len(match.group(1))
        widths[width] = widths.get(width, 0) + 1
    if expected is None:
        return None, []
    common_width = max(widths, key=lambda width: (widths[width], -width))
    final_width = max(widths) if (expected - 1) in seen else common_width

    def shard_name(idx: int) -> str:
        width = final_width if idx == expected - 1 else common_width
        return f"model-{idx:0{width}d}-of-{expected:05d}.safetensors"

    missing = [
        shard_name(idx)
        for idx in range(expected)
        if idx not in seen
    ]
    return expected, missing


def copy_non_weight_files(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)
    for path in src_path.iterdir():
        if path.name.endswith(".safetensors"):
            continue
        if path.name == "model.safetensors.index.json":
            continue
        target = dst_path / path.name
        if path.is_dir():
            if target.exists():
                continue
            shutil.copytree(path, target)
        elif path.is_file():
            shutil.copy2(path, target)


def iter_weight_shards(model_dir: str | Path) -> Iterator[Path]:
    yield from safetensor_files(model_dir)


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "__dataclass_fields__"):
        payload = asdict(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n")
