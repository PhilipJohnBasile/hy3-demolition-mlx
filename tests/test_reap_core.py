"""Regression tests for REAP planning core (src/hy3_reap.py).

Each test pins a real bug found and fixed on 2026-07-07 (see BUILD_NOTES).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from hy3_reap import build_plan, is_expert_axis_tensor


def _sal(counts_by_layer, facet=None):
    layers = {}
    for L, counts in counts_by_layer.items():
        reap = [c * 1.5 for c in counts]
        entry = {"counts": counts, "score_sum": [c * 0.5 for c in counts], "reap_sum": reap}
        if facet:
            entry["facets"] = {facet: {"counts": counts, "reap_sum": reap}}
        layers[str(L)] = entry
    return {"layers": layers}


def test_ratio_must_be_conservative():
    with pytest.raises(ValueError):
        build_plan(_sal({1: [1] * 8}), source="x", ratio=0.5, num_experts=8,
                   protected_facets=[], require_protected=False)


def test_keep_count_and_drop_partition():
    plan = build_plan(_sal({1: [10, 9, 8, 7, 6, 5, 4, 3]}), source="x", ratio=0.25,
                      num_experts=8, protected_facets=[], require_protected=False)
    lp = plan.layers[0]
    assert len(lp.keep) == 6 and len(lp.drop) == 2
    assert set(lp.keep) & set(lp.drop) == set()
    assert set(lp.keep) | set(lp.drop) == set(range(8))
    # highest-saliency experts kept
    assert set(lp.keep) == {0, 1, 2, 3, 4, 5}


def test_protected_facet_not_padded_with_unrouted_experts():
    # music routes to only 2 experts; min_keep=8 must NOT pad with count-0 experts
    counts = [10, 5, 0, 0, 0, 0, 0, 0]
    sal = _sal({1: counts}, facet="music")
    plan = build_plan(sal, source="x", ratio=0.25, num_experts=8,
                      protected_facets=["music"], min_keep_per_protected_facet=8)
    assert plan.layers[0].protected["music"] == [0, 1]


def test_missing_protected_saliency_raises():
    with pytest.raises(ValueError):
        build_plan(_sal({1: [1] * 8}), source="x", ratio=0.25, num_experts=8,
                   protected_facets=["music"], require_protected=True)


def test_ranking_uses_reap_mean_over_gate_sum():
    # expert 7: few tokens but huge per-token impact -> high MEAN, should be kept
    counts = [100, 100, 100, 100, 100, 100, 100, 2]
    reap = [10.0] * 7 + [50.0]  # expert 7 mean = 25 >> others' 0.1
    sal = {"layers": {"1": {"counts": counts, "reap_sum": reap,
                            "score_sum": [1.0] * 8}}}
    plan = build_plan(sal, source="x", ratio=0.25, num_experts=8,
                      protected_facets=[], require_protected=False)
    assert 7 in plan.layers[0].keep  # rare-but-strong expert survives


def test_expert_axis_detection_both_namings():
    assert is_expert_axis_tensor("model.layers.1.mlp.experts.gate_proj.weight", (192, 64, 8), 192)
    assert is_expert_axis_tensor("model.layers.1.mlp.switch_mlp.gate_proj.weight", (192, 64, 8), 192)
    assert is_expert_axis_tensor("model.layers.1.mlp.router.gate.weight", (192, 8), 192)
    # non-expert tensor of the wrong first-dim is not caught
    assert not is_expert_axis_tensor("model.layers.1.self_attn.q_proj.weight", (8192, 64), 192)
    assert not is_expert_axis_tensor("model.embed_tokens.weight", (192, 64), 192)
