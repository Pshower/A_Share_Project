"""Metrics use daily net equity; turnover is gross traded notional / prior equity."""

import numpy as np


def performance(daily, periods_per_year=252):
    equity = daily["equity"].to_numpy(dtype=float)
    if (len(equity) < 2 or not np.isfinite(equity).all()
            or np.any(equity[:-1] <= 0) or equity[-1] < 0):
        raise ValueError("Equity must be finite and positive before the terminal observation")
    returns = equity[1:] / equity[:-1] - 1
    cumulative = equity[-1] / equity[0] - 1
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    drawdown = equity / np.maximum.accumulate(equity) - 1
    return dict(cumulative_return=float(cumulative),
                annual_return=float((1 + cumulative) ** (periods_per_year / len(returns)) - 1),
                annual_volatility=volatility * np.sqrt(periods_per_year),
                sharpe=float(returns.mean() / volatility * np.sqrt(periods_per_year)) if volatility > 1e-15 else None,
                max_drawdown=float(drawdown.min()),
                total_fees=float(daily.fee.sum()), total_slippage_cost=float(daily.slippage_cost.sum()),
                gross_turnover=float(daily.turnover.sum()),
                mean_daily_turnover=float(daily.turnover.iloc[1:].mean()),
                mean_cash_ratio=float((daily.cash / daily.equity).iloc[1:].mean()) if "cash" in daily else None,
                final_equity=float(equity[-1]), trading_steps=len(returns))
