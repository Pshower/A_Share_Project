"""Pre-open daily advisory: trend analysis plus per-stock buy and sell actions.

The program answers one question: with the data already on disk and an engine that has
cleared the backtest gate, what should the portfolio hold for the next open, and what does
each name's trend look like right now?

Engines
    momentum20    top-k by 20-day momentum among tradeable names, equal weight; this is the
                  same ``Baseline("momentum")`` rule the backtest evaluates
    equal_weight  every tradeable name, equal weight
    cash          stay in cash
    ppo           a trained PPO archive evaluated under the dataset's own contract

Backends
    panel   a research dataset (features plus prices); any date inside the stored panel and
            the only backend that can drive the PPO engine
    online  an online HFQ history snapshot (bars only); always the last complete bar

Reports
    ``--mode validate --curves DIR`` additionally writes, per fold, the report directory the
    local workbench already renders: ``metrics.json`` keyed by strategy plus one folder per
    strategy holding ``daily.csv``, ``orders.csv`` and ``positions.parquet``. The numbers come
    from the same run that produces the gate verdict, so a curve can never disagree with it.

Hurdle
    A whole 100-share lot is a real trading requirement, so ``equal_weight`` over 291 names at
    retail capital is not a portfolio anybody can hold: each target is 0.34% of the account and
    the broker rejects nearly every order. The gate therefore compares against ``random20`` -
    ``--draws`` random portfolios holding the same number of names on the same schedule under
    the same lot rules. Only the selection differs, so only the selection is being tested.

Lots
    ``--capital`` turns each target weight into whole lots at the unadjusted quote. HFQ prices
    are corporate-action adjusted (``000001`` is 1,792.94 HFQ against 11.53 traded), so sizing
    from them would be meaningless; a target smaller than one lot is reported as
    ``below_one_lot`` rather than quietly rounded to zero.

Every output is research material. No order is sent, no account is connected, and the
advisory only calls itself validated when a saved backtest artifact covers that engine on
the registered folds.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.metrics import performance
from src.backtest.run import run_policy, run_strategy
from src.data.preprocess import ROOT, read_config, write_json
from src.data.snapshots import Snapshots
from src.envs.broker import BrokerConfig

MOMENTUM_WINDOW = 20
TRADING_DAYS = 252
LOT_SIZE = 100
RULE_ENGINES = ("momentum20", "equal_weight", "cash")
ENGINES = RULE_ENGINES + ("ppo",)
CALENDAR_PREFERENCE = ("c_4e276f88d8ff4018",)


def momentum_scores(closes, window=MOMENTUM_WINDOW):
    """Trailing return over ``window`` trading days, computed from the bar panel itself."""
    return closes / closes.shift(window) - 1.0


def trend_frame(closes, as_of, volumes=None):
    """Per-stock trend metrics at ``as_of``; every value uses bars up to and including it."""
    window = closes.loc[:as_of]
    if window.empty:
        raise ValueError("No bars at or before the advisory date")
    close = window.iloc[-1]
    result = pd.DataFrame({"close": close})
    for days in (1, 5, 20, 60, 120):
        result[f"r{days}"] = close / window.shift(days).iloc[-1] - 1.0
    ma20 = window.rolling(20).mean()
    ma60 = window.rolling(60).mean()
    result["ma20"] = ma20.iloc[-1]
    result["ma60"] = ma60.iloc[-1]
    result["ma20_dev"] = close / ma20.iloc[-1] - 1.0
    result["ma60_dev"] = close / ma60.iloc[-1] - 1.0
    result["ma20_slope"] = ma20.iloc[-1] / ma20.iloc[-21] - 1.0 if len(ma20) > 21 else np.nan
    result["dd60"] = close / window.iloc[-60:].max() - 1.0
    result["dd250"] = close / window.iloc[-250:].max() - 1.0
    returns = window.pct_change()
    result["vol20"] = returns.iloc[-20:].std() * np.sqrt(TRADING_DAYS)
    result["up_days20"] = (returns.iloc[-20:] > 0).sum()
    if volumes is not None:
        volume = volumes.loc[:as_of]
        if not volume.empty:
            result["volume_ratio"] = volume.iloc[-5:].mean() / volume.iloc[-60:].mean()
    return result


def select_weights(engine, closes, as_of, *, top_k=20, max_weight=0.05, codes=None, tradeable=None):
    """Target weights for one engine, restricted to names that can actually be traded."""
    if engine not in RULE_ENGINES:
        raise ValueError(f"{engine} is not a rule engine")
    codes = list(closes.columns) if codes is None else list(codes)
    window = closes.loc[:as_of, codes]
    scores = momentum_scores(window).iloc[-1]
    if tradeable is None:
        tradeable = window.iloc[-1].notna()
    else:
        tradeable = tradeable.reindex(codes).fillna(False).astype(bool)
    weights = pd.Series(0.0, index=codes)
    if engine == "cash":
        return weights, scores, tradeable
    eligible = [code for code in codes if tradeable[code] and np.isfinite(scores.get(code, np.nan))]
    if engine == "momentum20":
        # Stable ordering: score first, then code, so a tie never depends on dict order.
        eligible = sorted(eligible, key=lambda code: (-float(scores[code]), code))[:top_k]
    if eligible:
        weights[eligible] = min(1.0 / len(eligible), max_weight)
    return weights, scores, tradeable


def classify(target, current, *, tolerance=1e-6):
    """Turn a weight change into an action label."""
    change = float(target) - float(current)
    if abs(change) <= tolerance:
        return "HOLD"
    if current <= tolerance and target > tolerance:
        return "BUY"
    if target <= tolerance:
        return "SELL"
    return "ADD" if change > 0 else "TRIM"


def reasons(row, rank, median_vol, holding):
    """Short, checkable phrases behind one recommendation."""
    notes = []
    if np.isfinite(row.get("r20", np.nan)) and np.isfinite(row.get("r60", np.nan)):
        notes.append("20日 %.1f%% / 60日 %.1f%%" % (row["r20"] * 100, row["r60"] * 100))
    if np.isfinite(row.get("ma20_dev", np.nan)):
        notes.append("MA20 %+.1f%%" % (row["ma20_dev"] * 100))
    if np.isfinite(row.get("ma20_slope", np.nan)):
        notes.append("MA20 %s" % ("上行" if row["ma20_slope"] > 0 else "下行"))
    if np.isfinite(row.get("dd60", np.nan)):
        notes.append("60日回撤 %.1f%%" % (row["dd60"] * 100))
    if np.isfinite(median_vol) and np.isfinite(row.get("vol20", np.nan)):
        notes.append("波动 %s中位" % ("高于" if row["vol20"] > median_vol else "低于"))
    if rank is not None:
        notes.append("动量排名 #%d" % rank)
    if holding:
        notes.append("当前已持有")
    return "；".join(notes)


def _names(root, codes):
    """Best-effort Chinese names from the newest saved market breadth snapshot."""
    directory = Path(root) / "data/online/market"
    if not directory.exists():
        return {}
    snapshots = sorted((path for path in directory.iterdir() if not path.name.startswith(".")),
                       key=lambda path: path.stat().st_mtime, reverse=True)
    for snapshot in snapshots:
        path = snapshot / "breadth.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["code", "name"])
        return {str(code): str(name) for code, name in zip(frame["code"], frame["name"])}
    return {}


def _calendar_dates(root):
    store = Snapshots(root)
    for identifier in CALENDAR_PREFERENCE:
        try:
            _, directory = store.get("calendar", identifier)
        except (FileNotFoundError, ValueError):
            continue
        frame = pd.read_parquet(directory / "calendar.parquet")
        return pd.to_datetime(frame["trade_date"]).sort_values().reset_index(drop=True)
    return None


def next_open(as_of, calendar=None):
    """The first trading day strictly after the advisory date."""
    if calendar is not None and len(calendar):
        later = calendar[calendar > pd.Timestamp(as_of)]
        if len(later):
            return str(later.iloc[0].date())
    return None


def load_panel(config):
    """Research dataset view; the only backend that can drive a PPO archive."""
    from src.training.env_factory import ResearchData
    data = ResearchData(config)
    closes = data.prices.xs("close", axis=1, level=1) if "close" in data.prices.columns.get_level_values(1) \
        else data.prices
    volumes = data.prices.xs("volume", axis=1, level=1) if "volume" in data.prices.columns.get_level_values(1) \
        else None
    # The backtest only trades names whose whole feature window is finite, so mirror that rule.
    valid = data.features.notna().groupby(level=0, axis=1).all().reindex(columns=closes.columns, fill_value=False)
    return dict(source="panel", label=Path(data.research["dataset"]["clean_dir"]).name,
                codes=list(data.codes), closes=closes, volumes=volumes, valid=valid, data=data,
                price_basis=data.manifest["price_basis"])


def _unadjusted_close(directory, code):
    """Real quote for lot sizing.

    HFQ research prices are adjusted for corporate actions and are several orders of
    magnitude away from the traded quote (``000001`` is 1,792.94 HFQ against 11.53 real), so a
    100-share lot can only be sized from the unadjusted series.
    """
    path = Path(directory) / "unadjusted" / f"{code}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["date", "close"]).set_index("date").sort_index()
    frame["close"] = frame["close"].where(frame["close"] > 0)
    return frame["close"]


def load_online(history_id, root=ROOT):
    """Online HFQ bars; rule engines only, because no scaler or model contract is bound."""
    from src.data.preprocess import DataPreprocessor
    store = Snapshots(root)
    saved, directory = store.get("history", history_id)
    files = sorted((directory / "hfq").glob("*.parquet"))
    if not files:
        raise ValueError("History snapshot has no HFQ bars")
    closes, volumes, valid, real = {}, {}, {}, {}
    for path in files:
        frame = pd.read_parquet(path).set_index("date").sort_index()
        frame["valid_price"] = frame["close"].notna() & (frame["close"] > 0)
        features = DataPreprocessor.add_features(frame)
        closes[path.stem] = frame["close"]
        volumes[path.stem] = frame["volume"]
        # Same mask the environment uses: every registered feature must be finite.
        valid[path.stem] = features.notna().all(axis=1)
        quote = _unadjusted_close(directory, path.stem)
        if quote is not None:
            real[path.stem] = quote
    return dict(source="online", label=history_id, codes=list(closes),
                closes=pd.DataFrame(closes).sort_index(), volumes=pd.DataFrame(volumes).sort_index(),
                valid=pd.DataFrame(valid).sort_index(), data=None, price_basis="hfq",
                real_closes=pd.DataFrame(real).sort_index() if real else None)


def lot_plan(weights, prices, capital, *, lot=LOT_SIZE):
    """Turn target weights into whole lots at the real quote; a lot is the executable unit.

    Returns one entry per code: the target value, how many lots that buys, what those lots are
    actually worth and why a target went unfilled. ``below_one_lot`` is not a rounding nicety -
    it means the position cannot be opened at all at this capital.
    """
    if capital is None or not np.isfinite(capital) or capital <= 0:
        raise ValueError("Capital must be a positive number to size lots")
    plan = {}
    for code, weight in weights.items():
        price = float(prices.get(code, np.nan))
        target_value = capital * float(weight)
        if float(weight) <= 0:
            # Nothing is bought here, so a missing quote is not a problem worth reporting.
            plan[code] = dict(price=price if np.isfinite(price) else None, lots=0,
                              target_value=0.0, executable_value=0.0, executable_weight=0.0,
                              status="idle")
            continue
        if not np.isfinite(price) or price <= 0:
            plan[code] = dict(price=None, lots=0, target_value=float(target_value),
                              executable_value=0.0, executable_weight=0.0, status="no_price")
            continue
        unit = price * lot
        lots = int(target_value // unit)
        value = lots * unit
        status = "below_one_lot" if lots == 0 else "ok"
        plan[code] = dict(price=price, lots=lots, target_value=float(target_value),
                          executable_value=float(value), executable_weight=float(value / capital),
                          status=status)
    return plan


def tradeable_at(stage, as_of):
    """A name can be traded when its stored close exists and its feature window is complete."""
    closes = stage["closes"].loc[as_of]
    valid = stage.get("valid")
    if valid is None:
        return closes.notna()
    return closes.notna() & valid.loc[as_of].reindex(closes.index).fillna(False).astype(bool)


def build_advice(stage, as_of, engine, *, top_k=20, max_weight=0.05, holdings=None,
                 model=None, evidence=None, capital=None, root=ROOT):
    as_of = pd.Timestamp(as_of).normalize()
    if as_of not in stage["closes"].index:
        raise ValueError(f"{as_of.date()} is not a stored trading date for this backend")
    holdings = holdings or {}
    if not set(holdings) <= set(stage["codes"]):
        raise ValueError("Holdings contain codes outside the advisory pool")
    tradeable = tradeable_at(stage, as_of)
    if engine == "ppo":
        weights = _ppo_weights(stage, as_of, model, max_weight)
        scores = momentum_scores(stage["closes"].loc[:as_of, stage["codes"]]).iloc[-1]
        tradeable = tradeable.reindex(stage["codes"]).fillna(False).astype(bool)
    else:
        weights, scores, tradeable = select_weights(engine, stage["closes"], as_of, top_k=top_k,
                                                    max_weight=max_weight, codes=stage["codes"],
                                                    tradeable=tradeable)
    if engine == "ppo":
        weights = _ppo_weights(stage, as_of, model, max_weight)
    trend = trend_frame(stage["closes"], as_of, stage["volumes"])
    ranked = [code for code in scores.sort_values(ascending=False).index
              if np.isfinite(scores.get(code, np.nan))]
    rank = {code: position + 1 for position, code in enumerate(ranked)}
    median_vol = float(trend["vol20"].median()) if trend["vol20"].notna().any() else np.nan
    names = _names(root, stage["codes"])
    plan, lot_note = {}, None
    if capital is not None:
        quotes = stage.get("real_closes")
        if quotes is None:
            lot_note = "缺少未复权行情：HFQ 价格不能换算整手，因此只给目标权重。"
        else:
            plan = lot_plan(pd.Series(weights), quotes.loc[:as_of].iloc[-1], capital)
    rows = []
    for code in stage["codes"]:
        target, current = float(weights.get(code, 0.0)), float(holdings.get(code, 0.0))
        tradeable_now = bool(tradeable.get(code, False))
        action = classify(target, current) if tradeable_now else "UNAVAILABLE"
        row = trend.loc[code] if code in trend.index else pd.Series(dtype=float)
        lots = plan.get(code)
        rows.append(dict(
            code=code, name=names.get(code, ""), action=action, target_weight=target,
            current_weight=current, weight_change=target - current,
            close=float(row.get("close", np.nan)) if np.isfinite(row.get("close", np.nan)) else None,
            lot_price=None if lots is None else lots["price"],
            lots=None if lots is None else lots["lots"],
            lot_status=None if lots is None else lots["status"],
            target_value=None if lots is None else round(lots["target_value"], 2),
            executable_value=None if lots is None else round(lots["executable_value"], 2),
            executable_weight=None if lots is None else lots["executable_weight"],
            momentum_20=float(scores[code]) if np.isfinite(scores.get(code, np.nan)) else None,
            momentum_rank=rank.get(code),
            **{key: (float(row[key]) if key in row.index and np.isfinite(row[key]) else None)
               for key in ("r1", "r5", "r20", "r60", "r120", "ma20_dev", "ma60_dev", "ma20_slope",
                           "dd60", "dd250", "vol20", "up_days20", "volume_ratio")},
            tradeable=tradeable_now,
            reason=reasons(row, rank.get(code), median_vol, bool(current > 0)),
        ))
    rows.sort(key=lambda item: (-(item["momentum_20"] if item["momentum_20"] is not None else -9),
                                item["code"]))
    above_ma20 = trend["ma20_dev"].dropna()
    blocked = [row["code"] for row in rows
               if row["lot_status"] == "below_one_lot" and row["target_weight"] > 0]
    summary = dict(
        as_of=str(as_of.date()), next_open=next_open(as_of, _calendar_dates(root)), engine=engine,
        source=stage["source"], source_label=stage["label"], price_basis=stage["price_basis"],
        pool=len(stage["codes"]), tradeable=int(sum(tradeable.values)),
        invested_weight=float(weights.sum()), cash_weight=float(1.0 - weights.sum()),
        positions=int((weights > 0).sum()),
        above_ma20_share=float((above_ma20 > 0).mean()) if len(above_ma20) else None,
        median_r20=float(trend["r20"].median()) if trend["r20"].notna().any() else None,
        median_r60=float(trend["r60"].median()) if trend["r60"].notna().any() else None,
        counts={action: sum(1 for row in rows if row["action"] == action)
                for action in ("BUY", "ADD", "HOLD", "TRIM", "SELL", "UNAVAILABLE")},
        top_k=int(top_k), max_weight=float(max_weight),
        capital=None if capital is None else float(capital),
        lot_size=LOT_SIZE,
        executable_weight=(None if not plan else float(sum(item["executable_value"] for item in plan.values()) / capital)),
        blocked_targets=len(blocked),
        blocked_codes=blocked,
        lot_note=lot_note,
        evidence=evidence, validated=bool(evidence and evidence.get("gate", {}).get("pass")),
        actionable=False,
        limitation=("基于已保存数据的研究建议。没有绑定订单、账户或成交模型；权重是目标而非保证成交。"
                    "决策参考是已保存的收盘价，运行时次日开盘价仍未知。"),
    )
    return dict(summary=summary, rows=rows)


def _model_directory(model):
    """PPOAgent.load reads a directory holding model.zip and metadata.json."""
    path = Path(model)
    return path.parent if path.suffix == ".zip" else path


def _ppo_weights(stage, as_of, model, max_weight):
    if stage["data"] is None:
        raise ValueError("The PPO engine needs the panel backend and its feature contract")
    if model is None:
        raise ValueError("The PPO engine needs --model pointing at a trained archive")
    from src.agents.ppo_agent import PPOAgent
    agent = PPOAgent.load(_model_directory(model), stage["data"].contract, "cpu")
    from src.inference.predict import predict_action
    action = predict_action(agent, stage["data"], as_of)
    weights = pd.Series(0.0, index=stage["codes"])
    for row in action["rows"]:
        weights[row["code"]] = min(float(row["target_weight"]), max_weight)
    return weights


def _table(rows, limit=None):
    head = "| 代码 | 名称 | 目标权重 | 手数 | 整手金额 | 可执行权重 | 现权重 | 收盘(HFQ) | 5日 | 20日 | 60日 | MA20偏离 | 60日回撤 | 20日波动 |\n"
    head += "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    def cell(value, scale=100, suffix="%"):
        return "—" if value is None else f"{value * scale:+.2f}{suffix}"
    def lots(row):
        if row.get("lots") is None:
            return "—"
        return str(row["lots"]) if row["lot_status"] != "below_one_lot" else "**0**"
    def unit(row):
        price = row.get("lot_price")
        return "—" if not price else f"{price * LOT_SIZE:,.0f}"
    body = []
    for row in (rows if limit is None else rows[:limit]):
        executable = row.get("executable_weight")
        body.append(("| {code} | {name} | {target:.2%} | {lots} | {unit} | {exec} | {current:.2%} | "
                     "{close} | {r5} | {r20} | {r60} | {ma20} | {dd60} | {vol} |").format(
            code=row["code"], name=row["name"] or "—",
            target=row["target_weight"], current=row["current_weight"], lots=lots(row),
            unit=unit(row), exec="—" if executable is None else f"{executable:.2%}",
            close="—" if row["close"] is None else f"{row['close']:.2f}",
            r5=cell(row["r5"]), r20=cell(row["r20"]), r60=cell(row["r60"]),
            ma20=cell(row["ma20_dev"]), dd60=cell(row["dd60"]),
            vol=cell(row["vol20"], suffix="%")))
    return head + "\n".join(body) + "\n"


def render_markdown(advice, *, top=20):
    summary, rows = advice["summary"], advice["rows"]
    buys = [row for row in rows if row["action"] in ("BUY", "ADD")]
    sells = [row for row in rows if row["action"] in ("SELL", "TRIM")]
    evidence = summary["evidence"]
    lines = [
        f"# 开盘前交易建议 {summary['next_open'] or summary['as_of'] + ' 之后'}",
        "",
        f"- 决策日：{summary['as_of']}（{summary['source_label']}，价格口径 {summary['price_basis']}）",
        f"- 引擎：`{summary['engine']}`　目标持仓 {summary['positions']} 只，"
        f"股票仓位 {summary['invested_weight']:.1%}，现金 {summary['cash_weight']:.1%}",
        (f"- 按 {summary['capital']:,.0f} 元与 100 股整手换算，实际可建仓 {summary['executable_weight']:.1%}，"
         f"目标金额不足一手的标的 {summary['blocked_targets']} 只" if summary.get("capital") else
         "- 未给资金规模，只输出目标权重（整手换算需要 `--capital`）"),
        (f"- {summary['lot_note']}" if summary.get("lot_note") else None),
        f"- 回测证据：{'已通过' if summary['validated'] else '**未通过或缺失**'}",
        f"- 动作分布：买入 {summary['counts']['BUY']}，加仓 {summary['counts']['ADD']}，"
        f"持有 {summary['counts']['HOLD']}，减仓 {summary['counts']['TRIM']}，"
        f"卖出 {summary['counts']['SELL']}，不可交易 {summary['counts']['UNAVAILABLE']}",
        "",
        "## 趋势总览",
        "",
        f"- 池内站上 MA20 比例：{summary['above_ma20_share']:.1%}" if summary["above_ma20_share"] is not None else "- 池内站上 MA20 比例：—",
        f"- 20 日动量中位：{summary['median_r20']:+.2%}" if summary["median_r20"] is not None else "- 20 日动量中位：—",
        f"- 60 日动量中位：{summary['median_r60']:+.2%}" if summary["median_r60"] is not None else "- 60 日动量中位：—",
        f"- 可交易标的：{summary['tradeable']} / {summary['pool']}",
        "",
        "## 买入与加仓",
        "",
    ]
    lines = [line for line in lines if line is not None]
    lines.append(_table(buys, top) if buys else "无可执行买入。\n")
    if summary.get("capital"):
        lines += ["", "手数按未复权报价换算（`收盘(HFQ)` 列是复权研究价，只用于算收益，不能用来算整手）；"
                      "`**0**` 表示目标金额连一手都买不起，该标的无法建仓。"]
    lines += ["", "## 卖出与减仓", ""]
    lines.append(_table(sells, top) if sells else "无可执行卖出。\n")
    lines += ["", "## 证据与口径", ""]
    if evidence:
        for fold, metrics in sorted(evidence.get("folds", {}).items()):
            lines.append(f"- {fold} 折：累计收益 {metrics['cumulative_return']:+.2%}，"
                         f"最大回撤 {metrics['max_drawdown']:.2%}，夏普 "
                         f"{metrics['sharpe'] if metrics['sharpe'] is None else round(metrics['sharpe'], 3)}")
        gate = evidence.get("gate", {})
        lines.append(f"- 门槛：{gate.get('rule', '未登记')} → {'通过' if gate.get('pass') else '未通过'}")
        lines.append(f"- 证据文件：`{evidence.get('path', '')}`")
    else:
        lines.append("- 未找到该引擎的回测证据，本报告只是信号演示，不构成已验证策略。")
    lines += [
        f"- {summary['limitation']}",
        "- 目标权重来自收盘数据，次日开盘价未知；不含真实账户、真实撮合与交易所认证交易状态。",
        "",
    ]
    return "\n".join(lines)


def load_evidence(path):
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload.setdefault("path", str(path))
    return payload


def _fold_models(model, folds):
    """Accept one archive for every fold, or ``fold=path`` pairs for per-fold archives."""
    if model is None:
        return {}
    text = str(model)
    if "=" not in text:
        return {str(fold): text for fold in folds}
    pairs = dict(item.split("=", 1) for item in text.split(",") if item.strip())
    missing = [str(fold) for fold in folds if str(fold) not in pairs]
    if missing:
        raise ValueError(f"No archive supplied for folds {missing}")
    return pairs


def write_report(directory, arms, metrics, *, title, notes=(), metadata=None):
    """Write one report directory in the layout the local workbench already reads.

    ``arms`` maps a strategy name to its ``(daily, holdings, orders)`` backtest output; the
    directory is overwritten in place so a repeated run of the same engine stays idempotent.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, (daily, holdings, orders) in arms.items():
        folder = directory / name
        folder.mkdir(exist_ok=True)
        daily.to_csv(folder / "daily.csv", index=False)
        holdings.to_parquet(folder / "positions.parquet")
        orders.to_csv(folder / "orders.csv", index=False)
    write_json(directory / "metrics.json", metrics)
    pd.DataFrame(metrics).T.to_csv(directory / "metrics.csv")
    if metadata is not None:
        write_json(directory / "experiment.json", metadata)
    lines = [f"# {title}", "",
             "| 策略 | 累计收益 | 年化波动 | Sharpe | 最大回撤 | Sortino | 费用 | 现金比例 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, value in metrics.items():
        sharpe, sortino = value.get("sharpe"), value.get("sortino")
        lines.append("| {} | {:+.4%} | {:.4%} | {} | {:.4%} | {} | {:.2f} | {:.2%} |".format(
            name, value["cumulative_return"], value["annual_volatility"],
            "—" if sharpe is None else f"{sharpe:.3f}", value["max_drawdown"],
            "—" if sortino is None else f"{sortino:.3f}", value["total_fees"],
            value["mean_cash_ratio"]))
    lines += ["", *notes, ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return directory


class RandomTopK:
    """A null portfolio: same universe, masks, schedule and lot rules, no information.

    ``equal_weight`` over the whole pool is not a usable hurdle at retail capital: 291 targets
    of 0.34% each are smaller than one 100-share lot, so the broker correctly rejects almost
    every order and the arm sits in cash. Drawing the same number of names as the signal arm
    keeps the lot drag, the turnover and the exposure identical, so only the selection differs.
    """

    def __init__(self, seed, *, top_k=20, max_weight=0.05, interval=20):
        self.seed, self.top_k, self.max_weight, self.interval = seed, top_k, max_weight, interval

    def act(self, observation, step):
        universe = len(observation["valid_mask"])
        if step % self.interval:
            return None
        eligible = np.flatnonzero(observation["valid_mask"] & observation["buy_mask"])
        weights = np.zeros(universe)
        if not len(eligible):
            return weights
        rng = np.random.default_rng(self.seed * 1_000_003 + step)
        chosen = rng.choice(eligible, size=min(self.top_k, len(eligible)), replace=False)
        weights[chosen] = min(1.0 / len(chosen), self.max_weight)
        return weights


def validate_engine(config, engine, output, *, folds=(2023, 2024), model=None, top_k=20,
                    max_weight=0.05, curves=None, draws=20):
    """Backtest one engine against a same-shape random null on the registered folds.

    The hurdle is the median of ``draws`` random portfolios that hold the same number of names
    on the same schedule under the same lot rules. Whole lots are never relaxed here: they are
    a real trading requirement, which is exactly why an all-name equal weight cannot be the
    benchmark.
    """
    folds_root = Path(config["research_config"]).parent.parent
    archives = _fold_models(model, folds)
    arm = engine.replace("20", "")
    results, reference, artifacts, controls = {}, {}, {}, {}
    for fold in folds:
        fold_config = Path(folds_root) / str(fold) / "research.json"
        if not fold_config.exists():
            raise ValueError(f"Missing fold config {fold_config}")
        from src.training.env_factory import ResearchData
        local = deepcopy(config)
        local["research_config"] = str(fold_config)
        data = ResearchData(local)
        settings = deepcopy(data.settings)
        settings["rebalance_every"] = 20
        settings["momentum_top_k"] = top_k
        if engine == "ppo":
            if str(fold) not in archives:
                raise ValueError("PPO validation requires a trained archive per fold")
            from src.agents.ppo_agent import PPOAgent
            agent = PPOAgent.load(_model_directory(archives[str(fold)]), data.contract, "cpu")
            daily, holdings, orders, _ = run_policy(data.market("val"), agent, settings)
        else:
            daily, holdings, orders, _ = run_strategy(data.market("val"), arm, settings)
        results[str(fold)] = performance(daily)
        baseline, base_holdings, base_orders, _ = run_strategy(data.market("val"), "equal_weight", settings)
        reference[str(fold)] = performance(baseline)
        if curves is not None:
            artifacts[str(fold)] = {arm: (daily, holdings, orders)}
            # ``--engine equal_weight`` compares the rule with itself; keep one copy.
            artifacts[str(fold)].setdefault("equal_weight", (baseline, base_holdings, base_orders))
        if draws > 0:
            control_weight = BrokerConfig(**settings["broker"]).max_weight
            runs = []
            for seed in range(draws):
                random_daily, random_holdings, random_orders, _ = run_policy(
                    data.market("val"),
                    RandomTopK(seed, top_k=top_k, max_weight=control_weight, interval=20),
                    settings)
                runs.append((performance(random_daily)["cumulative_return"],
                             random_daily, random_holdings, random_orders))
            values = [item[0] for item in runs]
            controls[str(fold)] = dict(draws=draws, median=float(np.median(values)),
                                       p05=float(np.percentile(values, 5)),
                                       p95=float(np.percentile(values, 95)),
                                       values=[float(value) for value in values])
            if curves is not None:
                # One line for the chart: the draw closest to the median, chosen deterministically.
                middle = min(runs, key=lambda item: abs(item[0] - controls[str(fold)]["median"]))
                artifacts[str(fold)]["random20"] = middle[1:]
    returns = [metrics["cumulative_return"] for metrics in results.values()]
    gaps = [results[fold]["cumulative_return"] - reference[fold]["cumulative_return"] for fold in results]
    random_gaps = {fold: results[fold]["cumulative_return"] - controls[fold]["median"]
                   for fold in controls}
    beats_null = all(value >= 0 for value in random_gaps.values())
    rule = ("两折累计收益均为正，且两折都不低于同 20 只、同调仓节奏、同整手约束的随机组合中位数"
            if draws > 0 else "两折累计收益均为正")
    passed = all(value > 0 for value in returns) and (beats_null if draws > 0 else True)
    payload = dict(engine=engine, folds=results, equal_weight=reference,
                   gaps={fold: gap for fold, gap in zip(results, gaps)},
                   controls=controls, controls_gap=random_gaps,
                   gate=dict(rule=rule, folds=len(results), draws=int(draws),
                             **{"pass": bool(passed)},
                             equal_weight_is_reference_only=True),
                   generated_at=datetime.now(timezone.utc).isoformat(),
                   settings=dict(top_k=top_k, max_weight=max_weight, rebalance_every=20))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    if curves is not None:
        settings = payload["settings"]
        for fold in folds:
            key = str(fold)
            window = artifacts[key][arm][0]["date"].astype(str).str[:10]
            fold_metrics = {arm: results[key]}
            fold_metrics.setdefault("equal_weight", reference[key])
            control = controls.get(key)
            if control is not None and "random20" in artifacts[key]:
                fold_metrics["random20"] = performance(artifacts[key]["random20"][0])
            gate = "通过" if payload["gate"]["pass"] else "未通过"
            cash = "，".join(f"{name} {value['mean_cash_ratio']:.1%}"
                             for name, value in fold_metrics.items())
            notes = [
                f"- 引擎 `{engine}` 与各对照使用同一股票池、同一执行约束、同一区间。",
                f"- 共 {len(window) - 1} 个交易日，每 {settings['rebalance_every']} 个交易日调仓，"
                f"动量取前 {settings['top_k']} 只。",
            ]
            if control is not None:
                notes += [
                    f"- 门槛对照 `random20` 是 {control['draws']} 个随机组合的中位数（每 20 个交易日在同一"
                    f"可交易池里重抽 {settings['top_k']} 只，同样整手、同样节奏）；5%~95% 区间 "
                    f"{control['p05']:+.2%} ~ {control['p95']:+.2%}，中位 {control['median']:+.2%}。"
                    f"图中只画最接近中位的那一条。",
                    f"- 门槛：{payload['gate']['rule']} → {gate}"
                    f"（`{engine}` 对随机中位 {payload['controls_gap'][key]:+.4%}）。",
                    "- 等权整池对照仅作参考：100 万资金下每只目标不足一手，现实中建不起来，不能当门槛。",
                ]
            else:
                notes.append(f"- 门槛：{payload['gate']['rule']} → {gate}（对等权 {payload['gaps'][key]:+.4%}）。")
            notes += [
                f"- 平均现金占比：{cash}。整手（100 股）约束下，标的越多、单只目标金额越小，"
                "越容易买不进；仓位不同时收益差不能全部归因于选股。",
                "- HFQ 研究价格，不是真实成交价；费用与滑点已计入净值。",
                "- 这是两段约三个月的验证窗口之一，只说明策略在该窗口有效，不是资金部署证据。",
            ]
            write_report(
                Path(curves) / engine / f"fold_{key}", artifacts[key], fold_metrics,
                title=f"{engine} 与可行对照 {key} 折（{window.iloc[0]} 至 {window.iloc[-1]}）",
                notes=notes,
                metadata=dict(split="val", generator="tools/daily_advice.py --mode validate",
                              engine=engine, fold=key, window=[window.iloc[0], window.iloc[-1]],
                              settings=settings, gate=payload["gate"]["pass"],
                              controls=control),
            )
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["advise", "validate"], default="advise")
    parser.add_argument("--engine", choices=ENGINES, default="momentum20")
    parser.add_argument("--config", default="configs/ppo.json", help="PPP/research config for the panel backend")
    parser.add_argument("--research-config", help="override the dataset, for example a fold research.json")
    parser.add_argument("--history", help="online history snapshot id; selects the online backend")
    parser.add_argument("--model", help="trained PPO archive; use fold=path pairs for per-fold archives")
    parser.add_argument("--folds", default="2023,2024", help="comma separated fold years for validate mode")
    parser.add_argument("--as-of", help="advisory date; defaults to the last stored date")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--holdings", help="JSON mapping of current weights, for example {\"000001\": 0.05}")
    parser.add_argument("--evidence", help="saved validate payload for this engine")
    parser.add_argument("--curves", help="validate mode: directory for per-fold workbench reports")
    parser.add_argument("--draws", type=int, default=20,
                        help="validate mode: random portfolios behind the gate hurdle; 0 disables")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="advise mode: account size used to convert weights into whole lots")
    parser.add_argument("--output", required=True, help="directory for advice.md and advice.json")
    arguments = parser.parse_args(argv)
    config = read_config(arguments.config) if Path(arguments.config).exists() else None
    if config is not None and arguments.research_config:
        config["research_config"] = arguments.research_config
    if arguments.mode == "validate":
        if config is None:
            parser.error("validate mode needs --config")
        payload = validate_engine(config, arguments.engine, Path(arguments.output) / "evidence.json",
                                  folds=tuple(int(item) for item in arguments.folds.split(",") if item.strip()),
                                  model=arguments.model, top_k=arguments.top_k,
                                  max_weight=arguments.max_weight, curves=arguments.curves,
                                  draws=arguments.draws)
        for fold, metrics in sorted(payload["folds"].items()):
            control = payload["controls"].get(fold)
            extra = ("" if control is None else
                     f" random20 median={control['median']:+.4f} "
                     f"[{control['p05']:+.4f},{control['p95']:+.4f}]")
            print(f"{fold}: return={metrics['cumulative_return']:+.4f} "
                  f"sharpe={metrics['sharpe']} maxdd={metrics['max_drawdown']:.4f}{extra}")
        print(f"gate: {'pass' if payload['gate']['pass'] else 'fail'}")
        return 0
    if arguments.history:
        stage = load_online(arguments.history)
    elif config is not None:
        stage = load_panel(config)
    else:
        parser.error("advise mode needs --history or --config")
    as_of = pd.Timestamp(arguments.as_of) if arguments.as_of else stage["closes"].index[-1]
    holdings = json.loads(arguments.holdings) if arguments.holdings else {}
    advice = build_advice(stage, as_of, arguments.engine, top_k=arguments.top_k,
                          max_weight=arguments.max_weight, holdings=holdings,
                          model=arguments.model, evidence=load_evidence(arguments.evidence),
                          capital=arguments.capital)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "advice.json", advice)
    (output / "advice.md").write_text(render_markdown(advice), encoding="utf-8")
    summary = advice["summary"]
    print(f"as_of={summary['as_of']} next_open={summary['next_open']} engine={summary['engine']} "
          f"positions={summary['positions']} validated={summary['validated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
