"""Synthetic inherited holdings, valued only at the episode's initial close."""

import numpy as np

MODES = ("cash", "equal", "concentrated", "random")


def initialize_account(portfolio, prices, eligible, config, mode, rng):
    if mode not in MODES:
        raise ValueError("Unknown initial portfolio mode")
    prices = np.asarray(prices, dtype=float)
    eligible = np.asarray(eligible, dtype=bool) & np.isfinite(prices) & (prices > 0)
    weights = np.zeros(portfolio.n_stocks)
    indices = np.flatnonzero(eligible)
    if mode != "cash" and len(indices):
        if mode == "concentrated":
            indices = rng.choice(indices, size=min(5, len(indices)), replace=False)
        fractions = rng.dirichlet(np.ones(len(indices))) if mode == "random" else np.full(len(indices), 1 / len(indices))
        exposure = rng.uniform(0.2, 0.9) if mode == "random" else 0.8
        weights[indices] = np.minimum(fractions * exposure, config.max_weight)
        quantities = np.floor(weights[indices] * portfolio.initial_capital / prices[indices] / config.lot_size) * config.lot_size
        portfolio.positions[indices] = quantities
        portfolio.available_positions[indices] = quantities
        portfolio.cost_basis[indices] = prices[indices]
        portfolio.cash = portfolio.initial_capital - float(np.dot(quantities, prices[indices]))
    if portfolio.cash < -1e-7:
        raise AssertionError("Initial portfolio overdraws capital")
    return dict(mode=mode, holding_count=int((portfolio.positions > 0).sum()),
                cash_ratio=portfolio.cash / portfolio.initial_capital,
                basis="synthetic_settled_holdings_at_initial_close_no_entry_fee")
