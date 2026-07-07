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
    protected: dict[str, list[int]]


@dataclass
class ReapPlan:
    source: str
    ratio: float
    old_num_experts: int
    new_num_experts: int
    protected_facets: list[str]
    min_keep_per_protected_facet: int
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
    protected_facets: list[str] | None = None,
    min_keep_per_protected_facet: int = 8,
    require_protected: bool = True,
) -> ReapPlan:
    if not 0 <= ratio < 0.5:
        raise ValueError("Hy3 REAP ratio must be conservative: 0 <= ratio < 0.5")
    protected_facets = protected_facets or []
    if min_keep_per_protected_facet < 0:
        raise ValueError("min_keep_per_protected_facet must be >= 0")
    keep_count = max(1, round(num_experts * (1.0 - ratio)))
    layers: list[LayerPrunePlan] = []
    layer_scores = saliency.get("layers", {})
    if require_protected and protected_facets:
        missing = []
        for facet in protected_facets:
            if not any(facet in layer.get("facets", {}) for layer in layer_scores.values()):
                missing.append(facet)
        if missing:
            raise ValueError(
                "missing protected soul saliency for: "
                + ", ".join(sorted(missing))
                + "; run calibration with eval/souls/protected_prompts.jsonl or pass an explicit override"
            )
    def ranking_scores(data: dict) -> list[float] | None:
        """REAP mean (gate x ||f||, averaged over routed tokens) when the
        calibration recorded it; gate-sum otherwise (legacy saliency files)."""
        reap = data.get("reap_sum")
        counts = data.get("counts")
        if reap and counts:
            return [r / c if c else 0.0 for r, c in zip(reap, counts)]
        return data.get("score_sum") or counts

    for layer_key in sorted(layer_scores, key=lambda x: int(x)):
        layer_data = layer_scores[layer_key]
        scores = ranking_scores(layer_data)
        if scores is None:
            raise ValueError(f"missing scores for layer {layer_key}")
        ranked = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        protected: dict[str, list[int]] = {}
        protected_set: set[int] = set()
        facets = layer_data.get("facets", {})
        for facet in protected_facets:
            facet_scores = facets.get(facet)
            if not facet_scores:
                continue
            values = ranking_scores(facet_scores)
            if values is None:
                continue
            # Only protect experts this facet ACTUALLY routed to. Padding the
            # keep set with never-routed (count 0 -> reap-mean 0.0) experts,
            # tie-broken by ascending index, silently "protects" experts
            # 0,1,2,... that the soul never used — the opposite of soul
            # preservation. Cap protection at the number of experts with
            # positive routing for this facet in this layer.
            counts = facet_scores.get("counts") or [0] * len(values)
            routed = sum(1 for c in counts if c > 0)
            take = min(min_keep_per_protected_facet, routed)
            facet_ranked = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
            chosen = sorted(facet_ranked[:take])
            protected[facet] = chosen
            protected_set.update(chosen)
        if len(protected_set) > keep_count:
            raise ValueError(
                f"protected soul experts exceed keep budget in layer {layer_key}: "
                f"{len(protected_set)} protected > {keep_count} keep"
            )
        keep_set = set(protected_set)
        for idx in ranked:
            if len(keep_set) >= keep_count:
                break
            keep_set.add(idx)
        keep = sorted(keep_set)
        drop = [i for i in range(num_experts) if i not in set(keep)]
        layers.append(
            LayerPrunePlan(
                layer=int(layer_key),
                keep=keep,
                drop=drop,
                protected=protected,
            )
        )
    return ReapPlan(
        source=source,
        ratio=ratio,
        old_num_experts=num_experts,
        new_num_experts=keep_count,
        protected_facets=protected_facets,
        min_keep_per_protected_facet=min_keep_per_protected_facet,
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
