"""Hy3 REAP planning and tensor transforms.

The conservative path is plan first, then apply only with an explicit write flag.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LAYER_RE = re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.")


@dataclass
class LayerPrunePlan:
    layer: int
    keep: list[int]
    drop: list[int]


@dataclass
class ReapPlan:
    source: str
    ratio: float
    old_num_experts: int
    new_num_experts: int
    layers: list[LayerPrunePlan]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def load_saliency(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def build_plan(
    saliency: dict,
    *,
    source: str,
    ratio: float,
    num_experts: int,
) -> ReapPlan:
    if not 0 <= ratio < 0.5:
        raise ValueError("Hy3 REAP ratio must be conservative: 0 <= ratio < 0.5")
    keep_count = max(1, round(num_experts * (1.0 - ratio)))
    layers: list[LayerPrunePlan] = []
    layer_scores = saliency.get("layers", {})
    for layer_key in sorted(layer_scores, key=lambda x: int(x)):
        scores = layer_scores[layer_key].get("score_sum") or layer_scores[layer_key].get("counts")
        if scores is None:
            raise ValueError(f"missing scores for layer {layer_key}")
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        keep = sorted(ranked[:keep_count])
        drop = [i for i in range(num_experts) if i not in set(keep)]
        layers.append(LayerPrunePlan(layer=int(layer_key), keep=keep, drop=drop))
    return ReapPlan(
        source=source,
        ratio=ratio,
        old_num_experts=num_experts,
        new_num_experts=keep_count,
        layers=layers,
    )


def plan_for_layer(plan: ReapPlan, layer: int) -> LayerPrunePlan | None:
    for item in plan.layers:
        if item.layer == layer:
            return item
    return None


def layer_from_key(key: str) -> int | None:
    match = LAYER_RE.search(key)
    return int(match.group(1)) if match else None


def is_expert_axis_tensor(key: str, shape: tuple[int, ...], old_num_experts: int) -> bool:
    if not shape or shape[0] != old_num_experts:
        return False
    markers = (
        ".mlp.switch_mlp.",
        ".mlp.experts.",
        ".mlp.router.gate.",
        ".mlp.router.expert_bias",
        ".mlp.expert_bias",
    )
    return any(marker in key for marker in markers)

