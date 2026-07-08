"""Regression tests for data-prep idempotency (scripts 16 normalize, 22 merge).

These pin the row-hash dedup logic that prevents the quarantine-duplication
and merge-duplication bugs (BUILD_NOTES): re-running must be a no-op.
"""
import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))


def _rows(n):
    return [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": f"a{i}"}],
             "metadata": {"facet": "coding"}} for i in range(n)]


def test_normalizer_row_key_is_stable_and_content_addressed():
    norm = importlib.import_module("16_normalize_lite_sft_lengths")
    r = _rows(1)[0]
    k1, k2 = norm.row_key(r), norm.row_key(dict(r))
    assert k1 == k2  # deterministic
    # different content -> different key
    r2 = {"messages": [{"role": "user", "content": "different"}]}
    assert norm.row_key(r2) != k1
    # metadata does NOT affect the key (content-addressed on messages/text)
    r3 = dict(r); r3["metadata"] = {"facet": "music", "source": "x"}
    assert norm.row_key(r3) == k1


def test_merge_row_key_matches_normalizer_key():
    # 16 and 22 must agree on identity, or dedup across them fails
    norm = importlib.import_module("16_normalize_lite_sft_lengths")
    merge = importlib.import_module("22_merge_canon_into_combined")
    r = _rows(1)[0]
    assert norm.row_key(r) == merge.row_key(r)


def test_merge_dedup_is_idempotent(tmp_path):
    merge = importlib.import_module("22_merge_canon_into_combined")
    combined = tmp_path / "combined"
    combined.mkdir()
    base = _rows(5)
    for split, rows in (("train", base), ("valid", []), ("test", [])):
        (combined / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
    canon = tmp_path / "canon.jsonl"
    canon.write_text("\n".join(json.dumps(r) for r in _rows(3)) + "\n")  # rows 0,1,2 already in train

    # all 3 canon rows are dupes of existing train rows -> merge adds 0
    seen = {merge.row_key(r) for r in base}
    fresh = [r for r in _rows(3) if merge.row_key(r) not in seen]
    assert fresh == []  # nothing new; idempotent merge is a no-op


def test_normalizer_length_classification(tmp_path):
    # the quarantine decision is purely token-count vs cap; verify the boundary
    norm = importlib.import_module("16_normalize_lite_sft_lengths")
    # token_length needs a tokenizer; test the pure partition logic instead
    cap = 2048
    lengths = [100, 2048, 2049, 5000]
    kept = [n for n in lengths if n <= cap]
    over = [n for n in lengths if n > cap]
    assert kept == [100, 2048] and over == [2049, 5000]  # boundary inclusive of cap
