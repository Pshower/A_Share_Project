"""The parallel books must be the same rule with different weights, not different rules.

The property that matters is A: twenty equal-weighted features has to reproduce
``e1_rule.composite`` to the last bit, or the variants are measuring the harness.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tools import e1_rule, shadow_run, shadow_variants as variants, trading_calendar

CODES = ["%06d" % index for index in range(120)]


@pytest.fixture(autouse=True)
def always_trading(monkeypatch):
    """The recorder asks the exchange calendar before it books anything.

    Without this, every test in the file would pass Monday to Friday and fail at the weekend --
    the answer would be about the day the suite happens to run, not about the recorder.
    """
    monkeypatch.setattr(trading_calendar, "is_trading_day", lambda *a, **k: True)


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


def seed_state(path, panel, holdings, held_since=24):
    """Both engines start from the same book, so the test is about booking, not about picking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "running",
                                "last_panel_date": str(panel.index[held_since].date()),
                                "holdings": list(holdings),
                                "entry_date": str(panel.index[held_since].date()),
                                "runs": 1}), encoding="utf-8")


def test_variant_a_books_its_intervals_exactly_like_the_main_ledger(tmp_path, monkeypatch):
    """A is the control for the rule; this pins that it is the control for the accounting too.

    Both engines are handed the same book and the same gap to the next panel date, so the only
    thing left to disagree about is the booking: the interval return, the mark, the benchmark and
    the sessions held have to come out bit-identical.  Holding *order* is presentation -- scores tie
    exactly at the 1/2400 grid the percentiles live on -- so the book is compared as a set.
    """
    pool = dataset(tmp_path / "pool", days=30)
    panel = pd.read_parquet(pool / "panel_data.parquet")
    held = CODES[:30]
    monkeypatch.setattr(shadow_run, "STATE", tmp_path / "e1_state.json")
    monkeypatch.setattr(shadow_run, "LOG", tmp_path / "e1_shadow.jsonl")
    monkeypatch.setattr(trading_calendar, "is_trading_day", lambda *a, **k: True)

    output = tmp_path / "shadow"
    seed_state(tmp_path / "e1_state.json", panel, held)
    seed_state(variants.paths("A", output)[1], panel, held)
    main = shadow_run.run(dataset=pool, log=lambda *_: None)
    variant = variants.run_one("A", pool, output=output, log=lambda *_: None)

    assert main["decision"] == variant["decision"] == "hold"
    assert sorted(variant["book"]) == sorted(main["book"])
    for field in ("status", "panel_date", "nav", "held_from", "decision", "rebalance_due",
                  "sessions_held", "interval", "benchmark"):
        assert variant[field] == main[field], field


def test_the_first_recorded_row_books_an_explicit_null_interval(tmp_path, monkeypatch):
    """Readers index ``record["interval"]``; a missing key and a null are not the same thing."""
    pool = dataset(tmp_path / "pool")
    monkeypatch.setattr(shadow_run, "STATE", tmp_path / "e1_state.json")
    monkeypatch.setattr(shadow_run, "LOG", tmp_path / "e1_shadow.jsonl")
    monkeypatch.setattr(trading_calendar, "is_trading_day", lambda *a, **k: True)
    output = tmp_path / "shadow"
    main = shadow_run.run(dataset=pool, log=lambda *_: None)
    variant = variants.run_one("A", pool, output=output, log=lambda *_: None)
    assert main["status"] == variant["status"] == "decided"
    assert main["interval"] is None and variant["interval"] is None


def test_a_holiday_appends_nothing_in_either_ledger(tmp_path, monkeypatch):
    """A holiday is not news.  The main ledger knows that; a variant that books one has drifted."""
    pool = dataset(tmp_path / "pool")
    monkeypatch.setattr(shadow_run, "STATE", tmp_path / "e1_state.json")
    monkeypatch.setattr(shadow_run, "LOG", tmp_path / "e1_shadow.jsonl")
    monkeypatch.setattr(trading_calendar, "is_trading_day", lambda *a, **k: False)

    main = shadow_run.run(dataset=pool, log=lambda *_: None)
    output = tmp_path / "shadow"
    variant = variants.run_one("A", pool, output=output, log=lambda *_: None)

    assert main["status"] == variant["status"] == "nontrading"
    assert not (tmp_path / "e1_shadow.jsonl").exists()
    assert not (output / "A_shadow.jsonl").exists()
