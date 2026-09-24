"""Aligned feature and execution data. No network access or price unscaling."""

from pathlib import Path

import numpy as np
import pandas as pd


class MarketDataProvider:
    def __init__(self, panel_path, feature_cols=None, *, price_path,
                 price_basis, start_date=None, end_date=None, lookback=1):
        if price_basis not in {"unadjusted", "hfq_research"}:
            raise ValueError("Explicit price_basis must be unadjusted or hfq_research")
        if not isinstance(lookback, int) or lookback < 1:
            raise ValueError("lookback must be a positive integer")
        self.panel = self._read(panel_path)
        price_panel = self._read(price_path)
        self.stock_codes = self.panel.columns.get_level_values(0).unique().tolist()
        self.n_stocks = len(self.stock_codes)
        if not self.n_stocks or not all(isinstance(c, str) for c in self.stock_codes):
            raise ValueError("Stock codes must be nonempty string identifiers")
        if set(price_panel.columns.get_level_values(0)) != set(self.stock_codes):
            raise ValueError("Feature and price stock sets must match")
        self.feature_cols = list(feature_cols) if feature_cols is not None else [
            f for f in self.panel.columns.get_level_values(1).unique()
            if f not in {"code", "股票代码"}
        ]
        if not self.feature_cols or len(set(self.feature_cols)) != len(self.feature_cols):
            raise ValueError("Feature names must be nonempty and unique")
        self.lookback = lookback
        self.price_basis = price_basis
        all_dates = self.panel.index
        self._features = np.stack([
            self.panel.loc[:, [(code, f) for f in self.feature_cols]].to_numpy(dtype=float)
            for code in self.stock_codes
        ], axis=1)
        price_panel = price_panel.reindex(all_dates)
        self._fields = {}
        for field in ["open", "close", "volume", "suspended", "limit_up", "limit_down"]:
            present = [(code, field) in price_panel.columns for code in self.stock_codes]
            if any(present) and not all(present):
                raise ValueError(f"Field {field} must cover every stock")
            if all(present):
                values = price_panel.loc[:, [(c, field) for c in self.stock_codes]].to_numpy(dtype=float)
                if field in {"open", "close", "limit_up", "limit_down"}:
                    if np.any(np.isinf(values) | (np.isfinite(values) & (values <= 0))):
                        raise ValueError(f"{field} must be positive or missing, never standardized")
                if field == "volume" and np.any(np.isinf(values) | (values < 0)):
                    raise ValueError("Volume must be nonnegative shares or missing")
                if field == "suspended" and np.any(np.isfinite(values) & ~np.isin(values, [0, 1])):
                    raise ValueError("suspended must be 0, 1 or missing")
                self._fields[field] = values
            elif field in {"open", "close"}:
                raise ValueError(f"Separate price input requires {field}")
        if {"limit_up", "limit_down"} <= self._fields.keys():
            if np.any(self._fields["limit_up"] < self._fields["limit_down"]):
                raise ValueError("limit_up must not be below limit_down")
        self._valuation = pd.DataFrame(self._fields["close"]).ffill().to_numpy()
        selected = np.ones(len(all_dates), dtype=bool)
        if start_date is not None:
            selected &= all_dates >= pd.Timestamp(start_date)
        if end_date is not None:
            selected &= all_dates <= pd.Timestamp(end_date)
        self._indices = np.flatnonzero(selected)
        self.dates = all_dates[selected]
        self.n_dates = len(self.dates)
        if self.n_dates < 2:
            raise ValueError("At least two dates are needed for next-day execution")
        self.prices = pd.DataFrame(self._valuation[selected], index=self.dates, columns=self.stock_codes)
        self.assumptions = [
            "No corporate-action cash/share processing; provide an event-free interval or use explicit HFQ research mode.",
            "Missing closing quotes carry forward only for valuation; no delisting recovery model.",
        ]
        if price_basis == "hfq_research":
            self.assumptions.append("HFQ research prices are not real execution quotes or real share quantities.")
        for field in ["suspended", "limit_up", "limit_down"]:
            if field not in self._fields:
                self.assumptions.append(f"No {field} data: this constraint is not simulated.")
        self.current_index = 0

    @staticmethod
    def _read(source):
        frame = pd.read_parquet(Path(source)) if not isinstance(source, pd.DataFrame) else source.copy()
        if (not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2
                or frame.columns.has_duplicates):
            raise ValueError("Panel columns must be unique (code, field) pairs")
        frame.index = pd.to_datetime(frame.index)
        if (frame.index.has_duplicates or frame.index.isna().any()
                or not frame.index.is_monotonic_increasing):
            raise ValueError("Dates must be unique, valid and increasing")
        return frame

    def observation(self, idx=None):
        idx = self.current_index if idx is None else idx
        j = self._indices[idx]
        window = np.full((self.lookback, self.n_stocks, len(self.feature_cols)), np.nan)
        history = self._features[max(0, j - self.lookback + 1):j + 1]
        window[-len(history):] = history
        valid = np.isfinite(window).all(axis=(0, 2)) & np.isfinite(self._valuation[j])
        values = np.where(np.isfinite(window), window, 0).transpose(1, 0, 2)
        with np.errstate(over="ignore"):
            values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Features overflow float32")
        quote = self.execution(idx)
        return values, valid, quote["buy_mask"] & valid, quote["sell_mask"]

    def get_state(self, idx=None):
        idx = self.current_index if idx is None else idx
        return self.observation(idx)[0][:, -1, :], self.prices.iloc[idx].to_numpy(copy=True)

    def execution(self, idx):
        j = self._indices[idx]
        opens = self._fields["open"][j].copy()
        buy = np.isfinite(opens) & (opens > 0)
        sell = buy.copy()
        reasons = [[] for _ in self.stock_codes]
        for i in np.flatnonzero(~buy):
            reasons[i].append("missing_open")
        if "suspended" in self._fields:
            blocked = self._fields["suspended"][j] != 0
            buy[blocked] = sell[blocked] = False
            for i in np.flatnonzero(blocked):
                reasons[i].append("suspended_or_unknown")
        limits = {}
        for field, mask, comparison in [("limit_up", buy, np.greater_equal),
                                         ("limit_down", sell, np.less_equal)]:
            values = self._fields.get(field)
            limits[field] = np.full(self.n_stocks, np.nan) if values is None else values[j].copy()
            if values is not None:
                blocked = ~np.isfinite(values[j]) | comparison(opens, values[j])
                mask[blocked] = False
                for i in np.flatnonzero(blocked):
                    reasons[i].append(field + "_or_unknown")
        # Opening execution cannot use the same day's total volume or closing price.
        reference = (self._fields["volume"][j - 1].copy()
                     if "volume" in self._fields and j > 0 else np.full(self.n_stocks, np.nan))
        previous = self._valuation[j - 1] if j > 0 else np.full(self.n_stocks, np.nan)
        valuation = np.where(np.isfinite(opens), opens, previous)
        return dict(prices=opens, valuation_prices=valuation, buy_mask=buy, sell_mask=sell,
                    volume=reference, reasons=reasons, **limits)

    def reset(self):
        self.current_index = 0
        return self.get_state()

    def get_next_prices(self):
        idx = self.current_index + 1
        return None if idx >= self.n_dates else self.prices.iloc[idx].to_numpy(copy=True)

    def step(self):
        if self.current_index < self.n_dates:
            self.current_index += 1
        return self.current_index < self.n_dates

    @property
    def current_date(self):
        return self.dates[self.current_index] if self.current_index < self.n_dates else None
