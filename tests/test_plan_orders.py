"""The final plan's arithmetic, which is the part that must not be wrong.

Every case here is a place where a silent change would alter what someone buys: the floor,
the multiplier cap, the no-trade band, lot rounding and the split inside the risky sleeve.
"""
from __future__ import annotations

import pytest

from tools import plan_orders


def test_risky_weight_is_the_cppi_formula():
    # NAV 1000, peak 1000, floor 0.90 -> buffer 100 -> 6 * 0.1 = 0.6
    assert plan_orders.risky_weight(1000.0, 1000.0) == pytest.approx(0.6)


def test_risky_weight_is_capped_at_one_and_floored_at_zero():
    assert plan_orders.risky_weight(2000.0, 1000.0) == pytest.approx(1.0)
    assert plan_orders.risky_weight(890.0, 1000.0) == pytest.approx(0.0)      # below the floor
    assert plan_orders.risky_weight(500.0, 1000.0) == pytest.approx(0.0)


def test_the_no_trade_band_keeps_the_current_exposure():
    # target 0.60, held 0.63 -> inside the 5% band, so nothing moves
    targets = plan_orders.target_weights(1000.0, 1000.0, held_risky=0.63)
    assert targets["traded"] is False
    assert targets["applied"] == pytest.approx(0.63)
    assert targets["target"] == pytest.approx(0.60)


def test_outside_the_band_it_trades_back_to_target():
    targets = plan_orders.target_weights(1000.0, 1000.0, held_risky=0.50)
    assert targets["traded"] is True and targets["applied"] == pytest.approx(0.60)


def test_the_risky_sleeve_splits_fifty_fifty_and_the_safe_leg_absorbs_the_rest():
    weights = plan_orders.target_weights(1000.0, 1000.0)["weights"]
    assert weights["510300"] == pytest.approx(0.15)
    assert weights["510500"] == pytest.approx(0.15)
    assert weights["513500"] == pytest.approx(0.30)
    assert weights[plan_orders.SAFE] == pytest.approx(0.40)


def test_orders_round_down_to_whole_lots_and_net_to_cash():
    prices = {"510300": 4.0, "510500": 6.0, "513500": 2.0, "511010": 100.0}
    plan = plan_orders.orders({"510300": 0.15, "510500": 0.15, "513500": 0.30, "511010": 0.40},
                              nav=1_000_000.0, prices=prices, held={})
    by_code = {row["code"]: row for row in plan["orders"]}
    assert by_code["510300"]["target_shares"] % 100 == 0
    assert by_code["510300"]["target_shares"] == 37_500        # 150k / 4.0
    assert by_code["511010"]["target_shares"] == 4_000         # 400k / 100
    assert plan["cash"] == pytest.approx(1_000_000.0 - plan["invested"])


def test_orders_report_the_delta_against_what_is_held():
    prices = {"510300": 4.0, "510500": 6.0, "513500": 2.0, "511010": 100.0}
    plan = plan_orders.orders({"510300": 0.15, "510500": 0.15, "513500": 0.30, "511010": 0.40},
                              nav=1_000_000.0, prices=prices, held={"510300": 40_000})
    row = [item for item in plan["orders"] if item["code"] == "510300"][0]
    assert row["action"] == "sell" and row["delta"] == -2_500


def test_a_missing_price_is_an_error_not_a_free_position():
    with pytest.raises(ValueError, match="No price"):
        plan_orders.orders({"510300": 0.5}, nav=1000.0, prices={}, held={})
