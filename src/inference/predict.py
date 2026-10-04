"""Construct an as-of observation without stepping into the following day."""

import numpy as np
import pandas as pd

from src.envs.market import MarketDataProvider
from src.envs.portfolio import Portfolio


def predict_action(agent, data, as_of, account=None, display_codes=None, top_n=None):
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
        if data.contract.get("action_mode") == "stock_trade" and qty > 0 and "available_shares" not in position:
            raise ValueError("Stock trade held positions require explicit available_shares; unknown settlement cannot imply fully sellable shares")
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
    if data.contract.get("action_mode") == "stock_delta":
        from src.agents.delta_mapping import enrich_delta_observation
        from src.envs.broker import BrokerConfig
        requested_top = data.config.get("top_n", 20) if top_n is None else top_n
        if requested_top not in set(data.config.get("top_n_choices", []) + [data.config.get("top_n", 20)]):
            raise ValueError("Requested top_n was not registered for this delta model")
        # External accounts have no dated release ledger; do not guess that locks expire overnight.
        obs = enrich_delta_observation(obs, portfolio, closes, BrokerConfig(**data.settings["broker"]),
                                       requested_top)
    elif data.contract.get("action_mode") == "stock_trade":
        from src.agents.trade_mapping import enrich_trade_observation
        from src.envs.broker import BrokerConfig
        requested_top = data.config.get("top_n", 20) if top_n is None else top_n
        if requested_top not in set(data.config.get("top_n_choices", []) + [data.config.get("top_n", 20)]):
            raise ValueError("Requested top_n was not registered for this trade model")
        obs = enrich_trade_observation(obs, portfolio, closes, BrokerConfig(**data.settings["broker"]), requested_top,
                                      data.config.get("max_trade_weight", .05), data.config.get("max_commission_ratio", .001),
                                      execution_date=None)
    elif top_n is not None:
        raise ValueError("Legacy models do not support conditional top_n")
    mapped = agent.inspect_action(obs)
    if data.contract.get("action_mode") == "stock_trade":
        from src.agents.trade_mapping import DIRECTION_NAMES
        rows = []
        for i, code in enumerate(data.codes):
            reference = float(closes[i]) if np.isfinite(closes[i]) else None
            direction = int(mapped["directions"][i])
            shares = int(mapped["shares"][i])
            rows.append(dict(code=code, direction=direction, direction_name=DIRECTION_NAMES[direction],
                direction_probabilities=mapped["direction_probabilities"][i].tolist(),
                size_latent=float(mapped["size_latent"][i]), size_mean=float(mapped["size_mean"][i]),
                size_active=bool(mapped["size_active"][i]), selected_lots=int(mapped["selected_lots"][i]),
                proposed_shares=int(mapped["proposed_shares"][i]), projected_request_shares=shares,
                requested_lots=float(mapped["requested_lots"][i]), lot_bounds=mapped["lot_bounds"][i].tolist(),
                legal_directions=mapped["legal_directions"][i].tolist(), priority=float(mapped["priority"][i]),
                frozen=bool(mapped["freeze_mask"][i]), exit_requested=bool(mapped["exit_mask"][i]),
                top_n_filtered=bool(mapped["top_n_filtered"][i]), cash_filtered=bool(mapped["cash_filtered"][i]),
                shares=int(portfolio.positions[i]), available_shares=int(portfolio.available_positions[i]),
                current_weight=float(obs["portfolio"][i]), close_reference=reference,
                projected_reference_amount=shares * reference if reference is not None else None,
                valid=bool(valid[i]), buy_allowed=bool(buy[i]), sell_allowed=bool(sell[i])))
        return dict(as_of=str(date.date()), action_mode="stock_trade", price_basis=data.contract["price_basis"],
            equity=equity, current_cash_ratio=portfolio.cash / equity,
            projected_close_reference_cash_ratio=mapped["cash_weight"], pool_size=len(data.codes),
            display_codes=list(display_codes), rows=rows, account=account, contract=data.contract,
            top_n=mapped["top_n"], frozen_weight=mapped["frozen_weight"], size_sigma=mapped["size_sigma"],
            deterministic=True, note="Historical direction and share-count requests using known close marks and current availability; not executed fills or future quotes.")
    rows = []
    for i, code in enumerate(data.codes):
        current, target = float(obs["portfolio"][i]), float(mapped["weights"][i])
        change = target - current
        intent = "unchanged" if abs(change) <= 1e-6 else "buy" if change > 0 else "sell"
        if portfolio.positions[i] > 0 and target == 0:
            intent = "exit"
        row = dict(code=code, current_weight=current, target_weight=target, difference=change,
                         intent=intent, reference_amount=change * equity, valid=bool(valid[i]),
                         buy_allowed=bool(buy[i]), sell_allowed=bool(sell[i]),
                         shares=int(portfolio.positions[i]), available_shares=int(portfolio.available_positions[i]),
                         close_reference=float(closes[i]) if np.isfinite(closes[i]) else None,
                         latent=float(mapped["latent"][i]), eligible=bool(mapped["eligible"][i]))
        if data.contract.get("action_mode") == "stock_delta":
            row.update(requested_delta_weight=float(mapped["delta_weights"][i]),
                       frozen=bool(mapped["freeze_mask"][i]), exit_requested=bool(mapped["exit_mask"][i]))
        rows.append(row)
    result = dict(as_of=str(date.date()), price_basis=data.contract["price_basis"], equity=equity,
                current_cash_ratio=portfolio.cash / equity, target_cash_ratio=mapped["cash_weight"],
                pool_size=len(data.codes), display_codes=list(display_codes), rows=rows,
                account=account, contract=data.contract, deterministic=True,
                note=("Historical trade increments and close-reference intentions, not next-day fills or price forecasts."
                      if data.contract.get("action_mode") == "stock_delta" else
                      "Historical target weights and close-reference intentions, not next-day fills or price forecasts."))
    if data.contract.get("action_mode") == "stock_delta":
        result.update(action_mode="stock_delta", top_n=mapped["top_n"], frozen_weight=mapped["frozen_weight"])
    return result
