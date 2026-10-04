"""Deterministic, sell-first execution with partial fills and audit records."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BrokerConfig:
    commission: float = 0.0003
    min_commission: float = 0.0
    sell_tax: float = 0.0
    slippage: float = 0.0
    lot_size: int = 100
    max_weight: float = 1.0
    volume_fraction: float | None = None

    def __post_init__(self):
        for name in ("commission", "sell_tax", "slippage"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"Invalid {name}")
        if not np.isfinite(self.min_commission) or self.min_commission < 0:
            raise ValueError("Invalid min_commission")
        if not isinstance(self.lot_size, int) or self.lot_size < 1:
            raise ValueError("lot_size must be a positive integer")
        if not np.isfinite(self.max_weight) or not 0 < self.max_weight <= 1:
            raise ValueError("Invalid max_weight")
        if self.volume_fraction is not None and (
                not np.isfinite(self.volume_fraction) or not 0 < self.volume_fraction <= 1):
            raise ValueError("Invalid volume_fraction")


class BrokerSimulator:
    def __init__(self, config=None):
        self.config = config or BrokerConfig()
        self.reset()

    def reset(self):
        self.orders = []
        self._date = None
        self._used_volume = None

    def validate_weights(self, weights, n_stocks):
        weights = np.asarray(weights, dtype=float)
        if (weights.shape != (n_stocks,) or not np.isfinite(weights).all()
                or np.any(weights < 0) or np.any(weights > self.config.max_weight + 1e-7)
                or weights.sum() > 1 + 1e-7):
            raise ValueError("Weights must be finite, long-only and within position limits")
        return weights.copy()

    def validate_deltas(self, delta_weights, n_stocks):
        delta_weights = np.asarray(delta_weights, dtype=float)
        if delta_weights.shape != (n_stocks,) or not np.isfinite(delta_weights).all():
            raise ValueError("Delta weights must be a finite vector matching the portfolio")
        return delta_weights.copy()

    def _validate_delta_options(self, delta_weights, freeze_mask, top_n, exit_mask):
        n = len(delta_weights)

        def mask(value):
            result = np.zeros(n, dtype=bool) if value is None else np.asarray(value)
            if result.shape != (n,) or not np.isin(result, [0, 1]).all():
                raise ValueError("Delta execution masks must be boolean vectors matching the portfolio")
            return result.astype(bool, copy=True)

        if top_n is not None and (type(top_n) is not int or top_n < 0):
            raise ValueError("top_n must be a nonnegative integer")
        frozen, exits = mask(freeze_mask), mask(exit_mask)
        if np.any(exits & (delta_weights >= 0)):
            raise ValueError("An exit request requires a negative delta weight")
        return frozen, exits

    def fee(self, amount, selling=False):
        if amount <= 0:
            return 0.0
        return max(self.config.min_commission, amount * self.config.commission) + (
            amount * self.config.sell_tax if selling else 0.0)

    def rebalance(self, portfolio, weights, prices, valuation_prices, date, *,
                  buy_mask=None, sell_mask=None, volume=None, reasons=None,
                  limit_up=None, limit_down=None):
        n = portfolio.n_stocks
        weights = self.validate_weights(weights, n)
        execution = self._prepare_execution(portfolio, prices, valuation_prices, date,
                                            buy_mask, sell_mask, volume, reasons, limit_up, limit_down)
        prices, valid, equity = execution["prices"], execution["valid"], execution["equity"]

        # Targets use opening marks, while each fill uses its side-specific slippage.
        targets = portfolio.positions.copy()
        targets[valid] = np.floor(weights[valid] * equity / prices[valid])
        targets[weights == 0] = 0
        delta = targets - portfolio.positions
        pending = list(np.flatnonzero(delta < 0)) + list(np.flatnonzero(delta > 0))
        pending += [i for i in np.flatnonzero(~valid & (weights > 0)) if i not in pending]
        return self._execute_requests(portfolio, delta, weights, execution, pending)

    def trade_deltas(self, portfolio, delta_weights, prices, valuation_prices, date, *,
                     buy_mask=None, sell_mask=None, volume=None, reasons=None,
                     limit_up=None, limit_down=None, freeze_mask=None, top_n=None, exit_mask=None):
        delta_weights = self.validate_deltas(delta_weights, portfolio.n_stocks)
        frozen, exits = self._validate_delta_options(delta_weights, freeze_mask, top_n, exit_mask)
        execution = self._prepare_execution(portfolio, prices, valuation_prices, date,
                                            buy_mask, sell_mask, volume, reasons, limit_up, limit_down)
        equity, valid = execution["equity"], execution["valid"]
        if not np.isfinite(equity) or equity <= 0:
            raise ValueError("Delta execution requires positive finite opening equity")
        frozen |= (portfolio.positions > 0) & ~execution["sell"]
        marks = np.where(valid, execution["prices"], execution["valuation_prices"])
        priced = np.isfinite(marks) & (marks > 0)
        requested = np.zeros(portfolio.n_stocks)
        requested[priced] = np.sign(delta_weights[priced]) * np.floor(
            np.abs(delta_weights[priced]) * equity / marks[priced])
        requested[exits] = -portfolio.positions[exits]
        pending = list(np.flatnonzero((delta_weights < 0) & ((requested < 0) | ~valid)))
        pending += list(np.flatnonzero((delta_weights > 0) & ((requested > 0) | ~valid)))
        weights = np.maximum(0, portfolio.positions + requested) * np.where(priced, marks, 0) / equity
        return self._execute_requests(portfolio, requested, weights, execution, pending,
                                      freeze_mask=frozen, delta_weights=delta_weights,
                                      top_n=top_n, exit_mask=exits)

    def _prepare_execution(self, portfolio, prices, valuation_prices, date,
                           buy_mask, sell_mask, volume, reasons, limit_up, limit_down):
        n = portfolio.n_stocks

        def vector(value, default):
            array = np.full(n, default, dtype=float) if value is None else np.asarray(value, dtype=float)
            if array.shape != (n,):
                raise ValueError("Execution arrays must match the portfolio")
            return array.copy()

        prices = vector(prices, np.nan)
        valuation_prices = vector(valuation_prices, np.nan)
        buy = vector(buy_mask, 1)
        sell = vector(sell_mask, 1)
        if not np.isin(buy, [0, 1]).all() or not np.isin(sell, [0, 1]).all():
            raise ValueError("Execution masks must be boolean")
        buy, sell = buy.astype(bool), sell.astype(bool)
        volumes = vector(volume, np.nan)
        upper, lower = vector(limit_up, np.nan), vector(limit_down, np.nan)
        if np.any(np.isfinite(upper) & (upper <= 0)) or np.any(np.isfinite(lower) & (lower <= 0)):
            raise ValueError("Price limits must be positive")
        if reasons is not None and len(reasons) != n:
            raise ValueError("Reason lists must match the portfolio")
        equity = portfolio.value(valuation_prices)
        date = pd.Timestamp(date).normalize()
        if self._date is not None and date < self._date:
            raise ValueError("Execution dates cannot go backwards")
        portfolio.start_day(date)
        if self._date != date:
            self._date = date
            self._used_volume = np.zeros(n)
        elif self._used_volume.shape != (n,):
            raise ValueError("A broker instance must use a fixed asset set")
        return dict(prices=prices, valuation_prices=valuation_prices, buy=buy, sell=sell,
                    volumes=volumes, upper=upper, lower=lower, reasons=reasons, equity=equity,
                    date=date, valid=np.isfinite(prices) & (prices > 0))

    def _execute_requests(self, portfolio, delta, weights, execution, pending, *,
                          freeze_mask=None, delta_weights=None, top_n=None, exit_mask=None):
        prices, valid, date = execution["prices"], execution["valid"], execution["date"]
        buy, sell, volumes = execution["buy"], execution["sell"], execution["volumes"]
        upper, lower, reasons = execution["upper"], execution["lower"], execution["reasons"]
        records = []
        for i in pending:
            requested = float(delta[i])
            selling = requested < 0 if delta_weights is None else delta_weights[i] < 0
            why = []
            qty = abs(requested)
            price = float(prices[i])
            filled = 0.0
            fee = 0.0
            if not valid[i]:
                why.append("missing_open")
                qty = 0
            elif freeze_mask is not None and freeze_mask[i]:
                why.append("frozen_holdings")
                if reasons and reasons[i]:
                    why.extend(reasons[i])
                qty = 0
            elif not (sell[i] if selling else buy[i]):
                why.extend(reasons[i] if reasons and reasons[i] else ["sell_mask" if selling else "buy_mask"])
                qty = 0
            else:
                if delta_weights is not None and not selling:
                    room = max(0, np.floor(self.config.max_weight * execution["equity"] / prices[i])
                               - portfolio.positions[i])
                    if qty > room:
                        qty = room
                        why.append("position_cap")
                    if (top_n is not None and portfolio.positions[i] == 0
                            and np.count_nonzero(portfolio.positions > 0) >= top_n):
                        qty = 0
                        why.append("top_n_limit")
                if selling and np.isfinite(lower[i]) and price <= lower[i]:
                    why.append("limit_down")
                    qty = 0
                if not selling and np.isfinite(upper[i]) and price >= upper[i]:
                    why.append("limit_up")
                    qty = 0
                price *= 1 - self.config.slippage if selling else 1 + self.config.slippage
                if selling and np.isfinite(lower[i]):
                    price = max(price, lower[i])
                if not selling and np.isfinite(upper[i]):
                    price = min(price, upper[i])
                if selling and qty > portfolio.available_positions[i]:
                    qty = portfolio.available_positions[i]
                    why.append("t_plus_one")
                if self.config.volume_fraction is not None:
                    cap = (max(0, np.floor(volumes[i] * self.config.volume_fraction) - self._used_volume[i])
                           if np.isfinite(volumes[i]) and volumes[i] >= 0 else 0)
                    if qty > cap:
                        qty = cap
                        why.append("volume_limit")
                if not selling:
                    rounded = np.floor(qty / self.config.lot_size) * self.config.lot_size
                    if rounded < qty:
                        why.append("buy_lot")
                    qty = rounded
                    # Binary search includes minimum commission and cannot overdraw cash.
                    lo, hi = 0, int(qty // self.config.lot_size)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        amount = mid * self.config.lot_size * price
                        if amount + self.fee(amount) <= portfolio.cash + 1e-8:
                            lo = mid
                        else:
                            hi = mid - 1
                    affordable = lo * self.config.lot_size
                    if affordable < qty:
                        qty = affordable
                        why.append("insufficient_cash")
                if qty > 0:
                    fee = self.fee(qty * price, selling)
                    if selling and portfolio.cash + qty * price < fee:
                        why.append("insufficient_cash_for_fee")
                        qty, fee = 0, 0.0
                    else:
                        filled = -qty if selling else qty
                        portfolio.fill(i, filled, price, fee)
                        self._used_volume[i] += qty
            record = dict(date=date, code=portfolio.stock_codes[i], side="sell" if selling else "buy",
                          target_weight=float(weights[i]), requested_shares=requested,
                          filled_shares=float(filled), price=price, fee=float(fee),
                          slippage_cost=float(filled * (price - prices[i])) if filled else 0.0,
                          status="rejected" if filled == 0 else "filled" if filled == requested else "partial",
                          reasons=why, cash_after=portfolio.cash,
                          position_after=float(portfolio.positions[i]))
            if delta_weights is not None:
                record.update(requested_delta_weight=float(delta_weights[i]),
                              requested_trade_value=float(abs(delta_weights[i]) * execution["equity"]),
                              opening_equity=float(execution["equity"]), exit_requested=bool(exit_mask[i]))
            records.append(record)
        self.orders.extend(records)
        return records
