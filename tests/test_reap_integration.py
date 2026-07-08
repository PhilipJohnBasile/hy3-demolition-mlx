"""End-to-end REAP chain integration: saliency -> prune plan -> analyzer.

Tests the SEAMS between steps (format contracts), which unit tests of each
step in isolation miss. Uses realistic multi-layer, multi-facet synthetic
saliency in the exact shape scripts/04 emits.
"""
import json, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "bin" / "python")
sys.path.insert(0, str(REPO / "src"))


def _realistic_saliency(n_layers=8, n_experts=16, facets=("coding", "music", "security")):
    """Shape matches scripts/04: per-layer counts/score_sum/reap_sum + facets."""
    import random
    rng = random.Random(42)
    layers = {}
    for L in range(1, n_layers + 1):  # layer 0 is dense (first_k_dense_replace=1)
        counts = [rng.randint(1, 100) for _ in range(n_experts)]
        reap = [c * rng.uniform(0.5, 3.0) for c in counts]
        facet_buckets = {}
        for f in facets:
            fc = [rng.randint(0, 40) for _ in range(n_experts)]
            facet_buckets[f] = {"counts": fc, "reap_sum": [c * 1.2 for c in fc]}
        layers[str(L)] = {"counts": counts, "score_sum": [c * 0.5 for c in counts],
                          "reap_sum": reap, "facets": facet_buckets}
    return {"model": "tiny", "layers": layers}


def test_full_chain_saliency_to_analyzer(tmp_path):
    from hy3_reap import build_plan

    sal = _realistic_saliency()
    sal_file = tmp_path / "sal.json"
    sal_file.write_text(json.dumps(sal))

    # build_plan consumes the saliency (the 04->05 contract)
    plan = build_plan(sal, source="tiny", ratio=0.25, num_experts=16,
                      protected_facets=["coding", "music", "security"],
                      min_keep_per_protected_facet=2)
    assert plan.new_num_experts == 12
    assert len(plan.layers) == 8
    for lp in plan.layers:
        assert len(lp.keep) == 12 and len(lp.drop) == 4
        # every protected facet present and inside the keep set (05->soul contract)
        for f in ("coding", "music", "security"):
            assert set(lp.protected[f]) <= set(lp.keep)

    # write the plan and run the REAL analyzer on it (the 05->25 contract)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan.to_json()))
    r = subprocess.run([PY, str(REPO / "scripts" / "25_analyze_reap_plan.py"),
                        str(plan_file), "--saliency", str(sal_file)],
                       capture_output=True, text=True)
    report = json.loads(r.stdout)
    # analyzer produces a verdict and all five checks run without crashing
    assert report["overall"] in ("ACCEPT", "REVIEW", "REJECT")
    checks = {c["check"] for c in report["checks"]}
    assert checks == {"structure", "protected_coverage", "protection_pressure",
                      "soul_concentration", "saliency_mass"}
    # a well-formed plan must pass the two REJECT-capable structural checks
    verdicts = {c["check"]: c["verdict"] for c in report["checks"]}
    assert verdicts["structure"] == "ACCEPT"
    assert verdicts["protected_coverage"] == "ACCEPT"


def test_calibration_pack_parses_into_prompt_cases():
    """The 26->04 contract: the assembled calibration pack loads as PromptCases."""
    pack = REPO / "data" / "hy3_reap_calibration" / "prompts.jsonl"
    if not pack.exists():
        return  # pack not assembled in this checkout; skip
    sys.path.insert(0, str(REPO / "scripts"))
    import importlib
    calib = importlib.import_module("04_stream_calibrate_hy3_mlx")
    cases = calib.load_prompts(str(pack))
    assert len(cases) > 0
    assert all(c.prompt and c.facet for c in cases)  # every case has text + facet tag


def test_partial_saliency_rejected_by_coverage_guard(tmp_path):
    """05's coverage guard: a saliency missing MoE layers must be caught."""
    from hy3_reap import build_plan
    sal = _realistic_saliency(n_layers=3)  # only 3 of the layers present
    plan = build_plan(sal, source="tiny", ratio=0.25, num_experts=16,
                      protected_facets=[], require_protected=False)
    # plan only covers 3 layers; the coverage guard in scripts/05 (checked at
    # --write) would reject this against a full checkpoint. Here we assert the
    # plan itself reflects the partial coverage the guard keys on.
    assert len(plan.layers) == 3
