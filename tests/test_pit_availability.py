"""Unit checks for the derived PIT availability rule, without network or snapshot IO."""

import pandas as pd
import pytest

from tools.derive_pit_availability import availability, lag_profile


def financials_frame():
    return pd.DataFrame(dict(period=["20200331", "20200630", "20201231"],
                             code=["000001", "000001", "000001"],
                             announcement_date=["2021-04-25", "2021-04-25", "2021-04-25"]))


def test_financials_ignore_the_vendor_latest_revision_and_use_the_statutory_deadline():
    frame = financials_frame()
    usable, basis = availability(frame, "financials")
    assert usable.tolist() == ["2020-04-30", "2020-08-31", "2021-04-30"]
    assert set(basis) == {"statutory_deadline"}


def test_observed_announcement_dates_are_used_unchanged():
    frame = pd.DataFrame(dict(period=["20200331"], code=["000001"], announcement_date=["2020-01-15"]))
    usable, basis = availability(frame, "forecasts")
    assert usable.tolist() == ["2020-01-15"]
    assert basis.tolist() == ["vendor_announcement"]
    usable, basis = availability(frame, "shareholders")
    assert usable.tolist() == ["2020-01-15"]


def test_availability_never_precedes_the_period_end_for_financials():
    usable, _ = availability(financials_frame(), "financials")
    periods = pd.to_datetime(pd.Series(["20200331", "20200630", "20201231"]))
    assert (pd.to_datetime(usable) >= periods).all()


def test_unknown_table_is_rejected():
    with pytest.raises(ValueError, match="Unknown PIT table"):
        availability(financials_frame(), "analyst_ratings")


def test_lag_profile_separates_the_vendor_defect_from_the_usable_rule():
    frame = financials_frame()
    frame["available_from"], frame["availability_basis"] = availability(frame, "financials")
    profile = lag_profile(frame)
    assert profile["announcement_date"]["median_days"] == 299
    assert profile["announcement_date"]["over_365_days"] == 1
    assert profile["available_from"]["over_365_days"] == 0
    assert profile["available_from"]["median_days"] < profile["announcement_date"]["median_days"]


def test_lag_profile_counts_forecasts_issued_before_the_period_end():
    frame = pd.DataFrame(dict(period=["20200331", "20200630"], code=["000001", "600000"],
                              announcement_date=["2020-01-15", "2020-06-01"]))
    frame["available_from"], frame["availability_basis"] = availability(frame, "forecasts")
    profile = lag_profile(frame)
    assert profile["available_from"]["before_period_end"] == 2
    assert profile["available_from"]["median_days"] < 0
