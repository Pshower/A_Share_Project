"""Long-only cash and share accounting, including T+1 settlement."""

import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self, initial_capital=1e6, stock_codes=None):
        if not np.isfinite(initial_capital) or initial_capital <= 0:
            raise ValueError("initial_capital must be finite and positive")
        self.initial_capital = float(initial_capital)
        self.stock_codes = list(stock_codes or [])
        if len(set(self.stock_codes)) != len(self.stock_codes):
            raise ValueError("stock_codes must be unique")
        self.n_stocks = len(self.stock_codes)
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.positions = np.zeros(self.n_stocks, dtype=np.float64)
        self.available_positions = np.zeros(self.n_stocks, dtype=np.float64)
        self.cost_basis = np.zeros(self.n_stocks, dtype=np.float64)
        self.current_date = None
        self.total_fees = 0.0
        self.receivables = []
        self.share_locks = []
        self.event_cash = 0.0
        self.event_shares = np.zeros(self.n_stocks)

    def start_day(self, date):
        date = pd.Timestamp(date).normalize()
        if pd.isna(date) or (self.current_date is not None and date < self.current_date):
            raise ValueError("Trading dates must be valid and nondecreasing")
        if self.current_date is None or date > self.current_date:
            self.event_cash = sum(amount for due, amount in self.receivables if due <= date)
            self.cash += self.event_cash
            self.receivables = [(due, amount) for due, amount in self.receivables if due > date]
            self.share_locks = [(due, asset, qty) for due, asset, qty in self.share_locks if due > date]
            self.available_positions = self.positions.copy()
            for _, asset, qty in self.share_locks:
                self.available_positions[asset] -= qty
            self.event_shares = np.zeros(self.n_stocks)
            self.current_date = date

    @property
    def receivable_value(self):
        return sum(amount for _, amount in self.receivables)

    def apply_corporate_action(self, asset, cash_per_share, share_multiplier, payment_date, share_available_date):
        """Apply ex-date entitlements to pre-event holdings; bonus shares can remain locked."""
        if self.current_date is None or not 0 <= asset < self.n_stocks:
            raise ValueError("Corporate action requires a trading day and valid asset")
        payment_date, share_available_date = pd.Timestamp(payment_date), pd.Timestamp(share_available_date)
        if (not np.isfinite([cash_per_share, share_multiplier]).all() or cash_per_share < 0
                or share_multiplier < 1 or pd.isna(payment_date) or pd.isna(share_available_date)
                or payment_date < self.current_date or share_available_date < self.current_date):
            raise ValueError("Invalid corporate action dates or ratios")
        old = self.positions[asset]
        bonus = old * (share_multiplier - 1)
        if not np.isclose(bonus, round(bonus), rtol=0, atol=1e-8):
            raise ValueError("Fractional bonus shares require an explicit cash-in-lieu policy")
        bonus = float(round(bonus))
        dividend = old * cash_per_share
        if payment_date == self.current_date:
            self.cash += dividend
            self.event_cash += dividend
        elif dividend:
            self.receivables.append((payment_date, dividend))
        if bonus:
            self.positions[asset] += bonus
            self.event_shares[asset] += bonus
            self.cost_basis[asset] /= share_multiplier
            if share_available_date <= self.current_date:
                self.available_positions[asset] += bonus
            else:
                self.share_locks.append((share_available_date, asset, bonus))

    def _vector(self, values, name):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.n_stocks,):
            raise ValueError(f"{name} must have shape ({self.n_stocks},)")
        return values

    def value(self, prices):
        prices = self._vector(prices, "prices")
        held = self.positions > 0
        if np.any(~np.isfinite(prices[held]) | (prices[held] <= 0)):
            raise ValueError("Held assets require finite positive valuation prices")
        return float(self.cash + self.receivable_value + np.dot(self.positions[held], prices[held]))

    def fill(self, asset, shares, price, fee):
        if self.current_date is None:
            raise ValueError("Call start_day before trading")
        if not 0 <= asset < self.n_stocks:
            raise ValueError("Invalid asset index")
        if (not np.isfinite([shares, price, fee]).all() or price <= 0 or fee < 0
                or shares == 0 or shares != np.floor(shares)):
            raise ValueError("Fill requires integer shares, positive price and nonnegative fee")
        if shares < 0 and -shares > self.available_positions[asset]:
            raise ValueError("Sell exceeds T+1 available shares")
        new_cash = self.cash - shares * price - fee
        if new_cash < -1e-8:
            raise ValueError("Insufficient cash including fees")
        old_shares = self.positions[asset]
        if shares > 0:
            self.cost_basis[asset] = (
                old_shares * self.cost_basis[asset] + shares * price + fee
            ) / (old_shares + shares)
        else:
            self.available_positions[asset] += shares
        self.positions[asset] += shares
        if self.positions[asset] == 0:
            self.cost_basis[asset] = 0
        self.cash = max(0.0, float(new_cash))
        self.total_fees += fee

    def update_positions(self, new_positions, prices, transaction_cost=0.0):
        """Apply an exact integer target atomically; use the broker for partial fills."""
        target = self._vector(new_positions, "new_positions")
        prices = self._vector(prices, "prices")
        if (not np.isfinite(target).all() or np.any(target < 0)
                or np.any(target != np.floor(target))):
            raise ValueError("Target shares must be finite nonnegative integers")
        if not np.isfinite(transaction_cost) or not 0 <= transaction_cost < 1:
            raise ValueError("Invalid transaction_cost")
        delta = target - self.positions
        active = delta != 0
        if np.any(~np.isfinite(prices[active]) | (prices[active] <= 0)):
            raise ValueError("Trades require finite positive prices")
        if self.current_date is None or np.any(-delta > self.available_positions):
            raise ValueError("Call start_day and respect T+1 available shares")
        net = float(np.dot(delta[active], prices[active]))
        fees = float(np.dot(np.abs(delta[active]), prices[active]) * transaction_cost)
        if self.cash - net - fees < -1e-8:
            raise ValueError("Insufficient cash including fees")
        for indices in (np.flatnonzero(delta < 0), np.flatnonzero(delta > 0)):
            for i in indices:
                self.fill(i, delta[i], prices[i], abs(delta[i]) * prices[i] * transaction_cost)
        return fees, net

    def apply_weights(self, weights, prices, transaction_cost=0.0):
        """Compatibility helper for same-price, unconstrained research execution."""
        from .broker import BrokerConfig, BrokerSimulator

        if self.current_date is None:
            raise ValueError("Call start_day before trading")
        broker = BrokerSimulator(BrokerConfig(commission=transaction_cost))
        orders = broker.rebalance(self, weights, prices, prices, self.current_date)
        return (sum(o["fee"] for o in orders),
                sum(o["filled_shares"] * o["price"] for o in orders if o["filled_shares"]))

    def get_weights(self, prices):
        total = self.value(prices)
        weights = np.zeros(self.n_stocks)
        held = self.positions > 0
        if total > 0:
            weights[held] = self.positions[held] * np.asarray(prices)[held] / total
        return weights

    def get_cash_ratio(self, prices):
        total = self.value(prices)
        return self.cash / total if total > 0 else 1.0
