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

    def start_day(self, date):
        date = pd.Timestamp(date).normalize()
        if pd.isna(date) or (self.current_date is not None and date < self.current_date):
            raise ValueError("Trading dates must be valid and nondecreasing")
        if self.current_date is None or date > self.current_date:
            self.available_positions = self.positions.copy()
            self.current_date = date

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
        return float(self.cash + np.dot(self.positions[held], prices[held]))

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
