"""Bounded, causal feature recipes. No user code or expressions are evaluated."""

import re
import numpy as np
import pandas as pd

DEFINITION_VERSION = 1
RESERVED_COLUMNS = {"date", "code", "open", "close", "high", "low", "volume", "amount", "valid_price"}
KINDS = {
    "momentum": ("窗口收益", "C / C.shift(n) - 1"),
    "ma_gap": ("均线偏离", "C / rolling_mean(C, n) - 1"),
    "log_volatility": ("对数收益波动率", "rolling_std(log(C / C.shift(1)), n, ddof=0)"),
    "drawdown": ("窗口回撤", "C / rolling_max(C, n) - 1"),
    "atr_ratio": ("ATR 比例（简单均值）", "rolling_mean(TR, n) / C"),
    "relative_volume": ("相对成交量", "V / rolling_mean(V.shift(1), n)"),
    "relative_amount": ("相对成交额", "A / rolling_mean(A.shift(1), n)"),
    "bb_width": ("布林带宽", "4 * rolling_std(C, n, ddof=0) / rolling_mean(C, n)"),
}


def validate_definitions(definitions, reserved=()):
    if not isinstance(definitions, list) or len(definitions) > 12:
        raise ValueError("At most 12 extra features are supported")
    names = set(reserved) | RESERVED_COLUMNS
    result = []
    for definition in definitions:
        if not isinstance(definition, dict) or set(definition) != {"name", "kind", "window"}:
            raise ValueError("Feature requires exactly name, kind and window")
        name, kind, window = (definition[k] for k in ["name", "kind", "window"])
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{2,39}", name) or name in names:
            raise ValueError("Feature names must be unique ASCII identifiers, not existing columns")
        if not isinstance(kind, str) or kind not in KINDS or type(window) is not int or not 2 <= window <= 250:
            raise ValueError("Unknown feature type or window outside 2..250")
        names.add(name)
        result.append(dict(name=name, kind=kind, window=window))
    return result


def compute_extra_features(frame, definitions):
    close = frame.close.where(frame.valid_price.eq(True))
    result = pd.DataFrame(index=frame.index)
    for definition in definitions:
        n, kind = definition["window"], definition["kind"]
        if kind == "momentum":
            value = close.pct_change(n, fill_method=None)
        elif kind == "ma_gap":
            value = close / close.rolling(n).mean().replace(0, np.nan) - 1
        elif kind == "log_volatility":
            value = np.log(close / close.shift(1)).rolling(n).std(ddof=0)
        elif kind == "drawdown":
            value = close / close.rolling(n).max().replace(0, np.nan) - 1
        elif kind == "atr_ratio":
            tr = pd.concat([frame.high - frame.low, (frame.high - close.shift(1)).abs(),
                            (frame.low - close.shift(1)).abs()], axis=1).max(axis=1)
            tr = tr.where(frame.valid_price.eq(True) & close.shift(1).notna())
            value = tr.rolling(n).mean() / close
        elif kind in {"relative_volume", "relative_amount"}:
            raw = frame["volume" if kind == "relative_volume" else "amount"].where(frame.valid_price.eq(True))
            value = raw / raw.shift(1).rolling(n).mean().replace(0, np.nan)
        elif kind == "bb_width":
            value = 4 * close.rolling(n).std(ddof=0) / close.rolling(n).mean().replace(0, np.nan)
        else:
            raise ValueError("Unknown feature type")
        result[definition["name"]] = value.where(frame.valid_price.eq(True))
    return result.replace([np.inf, -np.inf], np.nan)
