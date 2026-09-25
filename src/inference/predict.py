"""Construct an as-of observation without stepping into the following day."""

import numpy as np
import pandas as pd

from src.envs.market import MarketDataProvider
from src.envs.portfolio import Portfolio


def predict_action(agent, data, as_of, account=None, display_codes=None):
    if agent.contract != data.contract:
        raise ValueError("Model/data contract mismatch")
    date = pd.Timestamp(as_of).normalize()
    if date not in data.features.index:
        raise ValueError("Select a trading date present in this dataset")
    display_codes = data.codes if display_codes is None else display_codes
    if not display_codes or not set(display_codes) <= set(data.codes):
        raise ValueError("Displayed stocks must be a nonempty subset of the model pool")
    features = data.features.loc[:date, pd.IndexSlice[data.codes, :]]
    prices = data.prices.loc[:date, pd.IndexSlice[data.codes, :]]
    market = MarketDataProvider(features, data.manifest["feature_cols"], price_path=prices,
                                price_basis=data.contract["price_basis"], lookback=data.config["lookback"])
    market.current_index = market.n_dates - 1
    portfolio = Portfolio(data.settings["initial_capital"], data.codes)
    portfolio.start_day(date)
    account = account or dict(cash=data.settings["initial_capital"], positions=[], receivables=[])
    if not np.isfinite(account["cash"]) or account["cash"] < 0:
        raise ValueError("Cash must be finite and nonnegative")
    portfolio.cash = float(account["cash"])
    seen = set()
    for position in account.get("positions", []):
        code = position["code"]
        qty = position["shares"]
        available = position.get("available_shares", qty)
        if (code in seen or code not in data.codes or not np.isfinite([qty, available]).all()
                or qty < 0 or available < 0 or available > qty or qty != int(qty) or available != int(available)):
            raise ValueError("Invalid, duplicated or out-of-pool account position")
        seen.add(code)
        index = data.codes.index(code)
        portfolio.positions[index], portfolio.available_positions[index] = qty, available
    for item in account.get("receivables", []):
        due, amount = pd.Timestamp(item["date"]), item["amount"]
        if due <= date or not np.isfinite(amount) or amount < 0:
            raise ValueError("Receivables must be nonnegative and payable after the as-of date")
        portfolio.receivables.append((due, float(amount)))
    closes = market.prices.iloc[-1].to_numpy()
    equity = portfolio.value(closes)
    if equity <= 0:
        raise ValueError("Account equity must be positive")
    features, valid, buy, sell = market.observation()
    obs = dict(stock_features=features, valid_mask=valid.astype(np.int8), buy_mask=buy.astype(np.int8),
               sell_mask=sell.astype(np.int8), portfolio=np.append(portfolio.get_weights(closes),
               portfolio.cash / equity).astype(np.float32),
               receivable_ratio=np.array([portfolio.receivable_value / equity], dtype=np.float32))
    mapped = agent.inspect_action(obs)
    rows = []
    for i, code in enumerate(data.codes):
        current, target = float(obs["portfolio"][i]), float(mapped["weights"][i])
        change = target - current
        intent = "unchanged" if abs(change) <= 1e-6 else "buy" if change > 0 else "sell"
        if portfolio.positions[i] > 0 and target == 0:
            intent = "exit"
        rows.append(dict(code=code, current_weight=current, target_weight=target, difference=change,
                         intent=intent, reference_amount=change * equity, valid=bool(valid[i]),
                         buy_allowed=bool(buy[i]), sell_allowed=bool(sell[i]),
                         shares=int(portfolio.positions[i]), available_shares=int(portfolio.available_positions[i]),
                         close_reference=float(closes[i]) if np.isfinite(closes[i]) else None,
                         latent=float(mapped["latent"][i]), eligible=bool(mapped["eligible"][i])))
    return dict(as_of=str(date.date()), price_basis=data.contract["price_basis"], equity=equity,
                current_cash_ratio=portfolio.cash / equity, target_cash_ratio=mapped["cash_weight"],
                pool_size=len(data.codes), display_codes=list(display_codes), rows=rows,
                account=account, contract=data.contract, deterministic=True,
                note="Historical target weights and close-reference intentions, not next-day fills or price forecasts.")
