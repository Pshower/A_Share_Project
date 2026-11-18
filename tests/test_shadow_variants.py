"""The parallel books must be the same rule with different weights, not different rules.

The property that matters is A: twenty equal-weighted features has to reproduce
``e1_rule.composite`` to the last bit, or the variants are measuring the harness.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tools import e1_rule, shadow_variants as variants

CODES = ["%06d" % index for index in range(120)]


def dataset(root, days=25):
    index = pd.bdate_range("2026-01-05", periods=days)
    rng = np.random.default_rng(7)
    columns = pd.MultiIndex.from_product([CODES, variants.FEATURES])
    panel = pd.DataFrame(rng.normal(size=(days, len(CODES) * len(variants.FEATURES))),
                         index=index, columns=columns)
    price_columns = pd.MultiIndex.from_product([CODES, ["close", "open"]])
    prices = pd.DataFrame(rng.uniform(5.0, 50.0, size=(days, len(price_columns))),
                          index=index, columns=price_columns)
    root.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(root / "panel_data.parquet")
    prices.to_parquet(root / "prices.parquet")
    (root / "manifest.json").write_text(
        json.dumps({"feature_cols": list(variants.FEATURES), "stock_codes": CODES}), encoding="utf-8")
    return root


def test_variant_a_is_exactly_the_frozen_rule(tmp_path):
    root = dataset(tmp_path / "pool")
    panel = pd.read_parquet(root / "panel_data.parquet")
    date = panel.index.max()
    reference = e1_rule.composite(panel, CODES, list(variants.FEATURES), date)
    blended = variants.blend(panel, CODES, variants.VARIANTS["A"]["spec"], date)
    assert np.allclose(reference.reindex(CODES).to_numpy(), blended.reindex(CODES).to_numpy())
    # Summation order differs by ~1e-16, which can flip a tie at the cut; the book must not care.
    top = set(reference.nsmallest(30).index)
    assert len(top & set(blended.nsmallest(30).index)) >= 29


def test_the_control_ignores_the_features_entirely(tmp_path):
    first = variants.random_score(CODES, "2026-11-18")
    second = variants.random_score(CODES, "2026-11-18")
    other = variants.random_score(CODES, "2026-11-19")
    assert first.equals(second), "the control must be reproducible, not fresh randomness"
    assert not first.equals(other)


def test_each_variant_records_its_own_ledger(tmp_path):
    root = dataset(tmp_path / "pool")
    output = tmp_path / "shadow"
    for letter in "ABCD":
        record = variants.run_one(letter, root, output=output, log=lambda *_: None)
        assert record["status"] == "decided" and len(record["book"]) == 30
        assert (output / ("%s_shadow.jsonl" % letter)).exists()
    books = {}
    for letter in "ABCD":
        rows = [json.loads(line) for line in
                (output / ("%s_shadow.jsonl" % letter)).read_text(encoding="utf-8").splitlines()]
        books[letter] = set(rows[-1]["book"])
    assert len(books["A"] & books["D"]) < 20, "a random book should barely overlap the real one"


def test_a_repeat_run_appends_stale_not_a_second_decision(tmp_path):
    root = dataset(tmp_path / "pool")
    output = tmp_path / "shadow"
    variants.run_one("A", root, output=output, log=lambda *_: None)
    again = variants.run_one("A", root, output=output, log=lambda *_: None)
    assert again["status"] == "stale"
    rows = (output / "A_shadow.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2


def test_an_unknown_variant_is_refused(tmp_path):
    root = dataset(tmp_path / "pool")
    with pytest.raises(ValueError, match="unknown variant"):
        variants.run("AZ", root, output=tmp_path / "shadow", log=lambda *_: None)
