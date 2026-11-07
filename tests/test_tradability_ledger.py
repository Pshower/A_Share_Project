"""Unit checks for the derived tradability ledger, without any snapshot or file IO."""

import numpy as np
import pandas as pd

from tools.tradability_ledger import (corporate_actions, exchange_round, limit_prices, limit_rate,
                                      stock_ledger, summarize)

DAYS = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"])


def bars():
    """One limit-up day, one suspension, then a one-word limit-up day on resumption."""
    return pd.DataFrame(dict(
        date=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-07", "2020-01-08"]),
        open=[10.00, 10.50, 12.10, 12.20], high=[10.10, 11.00, 12.10, 12.60],
        low=[9.90, 10.20, 12.10, 12.00], close=[10.00, 11.00, 12.10, 12.50]))


def test_board_limit_rates_cover_main_chinext_star_and_beijing():
    assert limit_rate("000001") == 0.10
    assert limit_rate("600000") == 0.10
    assert limit_rate("300750") == 0.20
    assert limit_rate("688981") == 0.20
    assert limit_rate("830799") == 0.30
    assert limit_rate("430047") == 0.30


def test_limit_prices_round_half_up_to_two_decimals():
    np.testing.assert_allclose(exchange_round([20.251, 10.894, 12.1]), [20.25, 10.89, 12.10])
    up, down = limit_prices([18.41, 10.89], 0.10)
    np.testing.assert_allclose(up, [20.25, 11.98])
    np.testing.assert_allclose(down, [16.57, 9.80])
    # Half-up matters: 16.565 must not become 16.56 the way banker's rounding would.
    assert exchange_round(16.565) == 16.57


def test_ledger_marks_suspension_and_spans_it_with_the_last_traded_close():
    ledger = stock_ledger("000001", bars(), DAYS)
    assert len(ledger) == len(DAYS)
    assert ledger.suspended.tolist() == [False, False, True, False, False]
    resumed = ledger.set_index("date").loc[pd.Timestamp("2020-01-07")]
    assert resumed["previous_close"] == 11.00
    assert bool(resumed["at_limit_up"])


def test_one_word_limit_up_needs_an_open_equal_to_the_limit():
    ledger = stock_ledger("000001", bars(), DAYS).set_index("date")
    assert bool(ledger.loc[pd.Timestamp("2020-01-07"), "one_word_limit_up"])
    # 2020-01-03 closed at the limit but opened at 10.50, so it was tradable.
    assert not bool(ledger.loc[pd.Timestamp("2020-01-03"), "one_word_limit_up"])


def test_blocked_buy_uses_the_next_traded_open_against_the_next_limit():
    ledger = stock_ledger("000001", bars(), DAYS).set_index("date")
    # 10.50 opens below the 11.00 limit derived from the 10.00 close.
    assert not bool(ledger.loc[pd.Timestamp("2020-01-02"), "blocked_buy"])
    # The 11.00 close implies a 12.10 limit, and the resumption open is exactly 12.10.
    assert bool(ledger.loc[pd.Timestamp("2020-01-03"), "blocked_buy"])
    # A suspended row never carries an execution verdict, and the last bar has no next open.
    assert not bool(ledger.loc[pd.Timestamp("2020-01-06"), "blocked_buy"])
    assert not bool(ledger.loc[pd.Timestamp("2020-01-08"), "has_next"])


def test_next_traded_date_is_the_resumption_day_not_an_integer_position():
    ledger = stock_ledger("000001", bars(), DAYS).set_index("date")
    assert ledger.loc[pd.Timestamp("2020-01-03"), "next_traded_date"] == pd.Timestamp("2020-01-07")
    assert ledger.loc[pd.Timestamp("2020-01-06"), "next_traded_date"] == pd.Timestamp("2020-01-07")
    assert pd.isna(ledger.loc[pd.Timestamp("2020-01-08"), "next_traded_date"])
    assert ledger["next_traded_date"].min() >= DAYS.min()


def test_corporate_actions_are_factor_jumps_not_price_moves():
    unadjusted = pd.DataFrame(dict(date=DAYS[:4], close=[10.0, 11.0, 12.1, 12.5]))
    hfq = pd.DataFrame(dict(date=DAYS[:4], close=[100.0, 110.0, 121.0, 125.0]))
    actions = corporate_actions("000001", hfq, unadjusted)
    assert len(actions) == 0
    hfq.loc[3, "close"] = 125.0 / 0.97          # a three percent factor step on the last day
    actions = corporate_actions("000001", hfq, unadjusted)
    assert len(actions) == 1
    assert actions["code"].tolist() == ["000001"]
    assert actions["date"].iloc[0] == DAYS[3]
    assert actions["ratio"].iloc[0] > 0.02


def test_summary_reports_shares_against_executable_rows_only():
    ledger = stock_ledger("000001", bars(), DAYS)
    summary = summarize(ledger, pd.DataFrame(columns=["code", "date", "factor_before", "factor_after", "ratio"]))
    assert summary["rows"] == len(DAYS) and summary["codes"] == 1
    assert summary["suspended_rows"] == 1
    assert summary["traded_rows"] == len(DAYS) - 1
    assert summary["limit_up_rows"] == 2
    assert summary["blocked_buy_rows"] == 1
    assert summary["corporate_action_events"] == 0
    # Three traded rows carry a next traded day, so the share uses three as its denominator.
    assert summary["blocked_buy_share_of_executable"] == 0.333333
