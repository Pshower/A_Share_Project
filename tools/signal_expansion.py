"""Screen the unmodelled data blocks for cross-sectional predictive information.

The 20 stored price features all sit at |IC| <= 0.035 on the long sample.  Three data
families were collected but never entered a model: share counts and market capitalisation,
earnings preannouncements, and shareholder-count changes.  This tool builds one frozen
signal per family, following ``doc/signal_expansion_protocol_20261105.md``, and reports the
same daily rank IC table the price features were judged by.

Nothing is fitted, no parameter is searched, and the decisions are published in the IC
table itself: a signal earns a further test only if the protocol's criteria hold.  The
labels are gross (``close[t+h] / open[t+1] - 1``), so a positive IC says nothing about
whether the effect survives lots and slippage.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import file_hash, read_config, write_json
from tools.signal_leaderboard import summarize, windows

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
REPORT_ROOT = PROJECT_ROOT / "reports" / "runs"

MIN_NAMES = 30
STALE_DAYS = 400
EARNINGS_DECAY = 60
HOLDER_DECAY = 90
FLOAT_SMOOTH = 21

SCORE_MAP = {
    "预增": 1.0, "扭亏": 1.0,
    "略增": 0.5, "续盈": 0.5,
    "不确定": 0.0,
    "略减": -0.5, "减亏": -0.5,
    "预减": -1.0, "首亏": -1.0, "增亏": -1.0, "续亏": -1.0,
}


def next_trading_day(index: pd.DatetimeIndex, stamps: pd.Series) -> pd.Series:
    """First stored trading row strictly after the announcement date."""
    values = pd.to_datetime(stamps)
    grid = index.to_numpy()
    taken = np.full(len(values), np.datetime64("NaT"), dtype="datetime64[ns]")
    known = values.notna().to_numpy()
    if known.any():
        position = np.searchsorted(grid, values[known].to_numpy(), side="right")
        inside = position < len(grid)
        located = np.full(len(position), np.datetime64("NaT"), dtype="datetime64[ns]")
        located[inside] = grid[position[inside]]
        taken[known] = located
    return pd.Series(taken, index=stamps.index)


def latest_by_code(index, codes, events, *, tolerance_days=STALE_DAYS):
    """Value of the most recent event at or before each date, per code.

    ``merge_asof`` refuses a per-group as-of across codes here, so the lookup is an
    explicit ``searchsorted`` on each code's row positions; duplicates resolve to the
    last event, which is what "most recent announcement" means.
    """
    value = np.full((len(index), len(codes)), np.nan, dtype=float)
    positions = pd.Series(np.arange(len(index)), index=index)
    grid = np.arange(len(index))
    tolerance = np.timedelta64(tolerance_days, "D")
    stamps = index.to_numpy()
    for column, code in enumerate(codes):
        part = events[events["code"] == code].dropna(subset=["available", "value"])
        if part.empty:
            continue
        part = part.sort_values("available")
        rows = positions.reindex(part["available"]).to_numpy()
        keep = ~pd.isna(rows)
        if not keep.any():
            continue
        rows = rows[keep].astype(int)
        scores = part["value"].to_numpy(dtype=float)[keep]
        last = np.searchsorted(rows, grid, side="right") - 1
        usable = last >= 0
        if not usable.any():
            continue
        ages = stamps[grid[usable]] - stamps[rows[last[usable]]]
        fresh = ages <= tolerance
        target = grid[usable][fresh]
        value[target, column] = scores[last[usable][fresh]]
    return pd.DataFrame(value, index=index, columns=codes)


def decayed_by_code(index, codes, events, decay):
    """Linear decay to zero over ``decay`` trading rows after each event."""
    value = np.full((len(index), len(codes)), np.nan, dtype=float)
    positions = pd.Series(np.arange(len(index)), index=index)
    for position, code in enumerate(codes):
        part = events[events["code"] == code].dropna(subset=["available", "value"])
        if part.empty:
            continue
        rows = positions.reindex(part["available"]).to_numpy()
        keep = ~pd.isna(rows)
        if not keep.any():
            continue
        order = np.argsort(rows[keep], kind="mergesort")
        rows = rows[keep].astype(int)[order]
        scores = part["value"].to_numpy()[order]
        grid = np.arange(len(index))
        last = np.searchsorted(rows, grid, side="right") - 1
        usable = last >= 0
        age = grid[usable] - rows[last[usable]]
        value[usable, position] = scores[last[usable]] * np.clip(1.0 - age / decay, 0.0, None)
    return pd.DataFrame(value, index=index, columns=codes)


def load_unadjusted(history_root: Path, index, codes):
    """Unadjusted close per code, preferring the file that covers the most rows."""
    matrix = pd.DataFrame(np.nan, index=index, columns=codes, dtype=float)
    for code in codes:
        frames = []
        for path in sorted(history_root.glob(f"*/unadjusted/{code}.parquet")):
            frame = pd.read_parquet(path, columns=["date", "close"]).dropna()
            if len(frame):
                frames.append(frame)
        if not frames:
            continue
        best = max(frames, key=len).drop_duplicates("date").set_index("date")["close"]
        matrix[code] = best.reindex(index)
    return matrix


def float_shares(history_root: Path, index, codes):
    """Implied free-float shares from volume / turnover, median-smoothed."""
    matrix = pd.DataFrame(np.nan, index=index, columns=codes, dtype=float)
    for code in codes:
        frames = []
        for path in sorted(history_root.glob(f"*/hfq/{code}.parquet")):
            frame = pd.read_parquet(path, columns=["date", "volume", "turnover"]).dropna()
            if len(frame):
                frames.append(frame)
        if not frames:
            continue
        best = max(frames, key=len).drop_duplicates("date").sort_values("date").set_index("date")
        usable = best["turnover"] > 0
        implied = (best.loc[usable, "volume"] / (best.loc[usable, "turnover"] / 100.0))
        implied = implied.rolling(FLOAT_SMOOTH, min_periods=max(5, FLOAT_SMOOTH // 4)).median()
        matrix[code] = implied.reindex(index)
    return matrix


def earnings_events(forecasts_path: Path, index):
    frame = pd.read_parquet(forecasts_path,
                            columns=["code", "period", "预测指标", "预告类型", "公告日期"])
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["value"] = frame["预告类型"].map(SCORE_MAP)
    frame = frame.dropna(subset=["value"])
    frame["priority"] = (frame["预测指标"] == "归属于上市公司股东的净利润").astype(int)
    frame = frame.sort_values(["code", "period", "priority", "公告日期"])
    frame = frame.groupby(["code", "period"]).tail(1)
    frame["available"] = next_trading_day(index, frame["公告日期"])
    return frame


def holder_events(shareholders_path: Path, index):
    frame = pd.read_parquet(shareholders_path,
                            columns=["code", "period", "announcement_date", "股东户数-增减比例"])
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["value"] = -frame["股东户数-增减比例"].astype(float) / 100.0
    frame = frame.dropna(subset=["value"]).drop_duplicates(["code", "period"])
    frame["available"] = next_trading_day(index, frame["announcement_date"])
    return frame


def share_events(shareholders_path: Path, index):
    frame = pd.read_parquet(shareholders_path,
                            columns=["code", "period", "announcement_date", "总股本"])
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["value"] = frame["总股本"].astype(float)
    frame = frame.dropna(subset=["value"]).drop_duplicates(["code", "period"])
    frame["available"] = next_trading_day(index, frame["announcement_date"])
    return frame


def build_signals(index, codes, *, history_root=DATA_ROOT / "online" / "history",
                  fundamentals_dir=None):
    """The four frozen signals plus their raw inputs, all on the fold grid."""
    if fundamentals_dir is None:
        raise ValueError("fundamentals_dir is required")
    fundamentals_dir = Path(fundamentals_dir)
    close = load_unadjusted(history_root, index, codes)
    shares = latest_by_code(index, codes, share_events(fundamentals_dir / "shareholders.parquet", index))
    floats = float_shares(history_root, index, codes)

    with np.errstate(divide="ignore", invalid="ignore"):
        mcap = np.where((close > 0) & (shares > 0), close * shares, np.nan)
        float_mcap = np.where((close > 0) & (floats > 0), close * floats, np.nan)

    signals = {
        "log_mcap": pd.DataFrame(np.log(mcap), index=index, columns=codes),
        "log_float_mcap": pd.DataFrame(np.log(float_mcap), index=index, columns=codes),
        "earnings_score": decayed_by_code(index, codes,
                                         earnings_events(fundamentals_dir / "forecasts.parquet", index),
                                         EARNINGS_DECAY),
        "holder_change": decayed_by_code(index, codes,
                                        holder_events(fundamentals_dir / "shareholders.parquet", index),
                                        HOLDER_DECAY),
    }
    return signals


def daily_rank_ic(values, target, dates, *, min_names=MIN_NAMES):
    """Spearman correlation per row, ignoring rows with too few usable names."""
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    out = np.full(len(values), np.nan)
    for row in range(len(values)):
        usable = np.isfinite(values[row]) & np.isfinite(target[row])
        if usable.sum() < min_names:
            continue
        left = pd.Series(values[row][usable]).rank().to_numpy()
        right = pd.Series(target[row][usable]).rank().to_numpy()
        left = left - left.mean()
        right = right - right.mean()
        scale = np.sqrt((left ** 2).sum() * (right ** 2).sum())
        if scale > 1e-12:
            out[row] = float(left @ right / scale)
    return pd.Series(out, index=dates)


def evaluate(fold_dir, horizon, *, data_root=DATA_ROOT):
    """Per-window IC for every frozen signal plus the price baseline."""
    fold_dir = Path(fold_dir)
    manifest = read_config(fold_dir / "manifest.json")
    panel = pd.read_parquet(fold_dir / "panel_data.parquet")
    prices = pd.read_parquet(fold_dir / "prices.parquet")
    codes = list(panel.columns.get_level_values(0).unique())
    index = panel.index

    from src.training.exploration_signal import forward_labels
    labels, endpoint = forward_labels(prices, codes, horizon)
    labels, endpoint = labels.reindex(index), endpoint.reindex(index)

    fundamentals_dir = sorted((data_root / "online" / "fundamentals").glob("*/financials.parquet"),
                              key=lambda path: (pd.read_parquet(path, columns=["code"]).code.nunique(),
                                                path.stat().st_size))[-1].parent
    signals = build_signals(index, codes, history_root=data_root / "online" / "history",
                            fundamentals_dir=fundamentals_dir)
    signals["momentum_20"] = panel.xs("momentum_20", axis=1, level=1).reindex(columns=codes)

    target = labels.to_numpy(dtype=float)
    settled = endpoint.fillna(pd.Timestamp.max).to_numpy() <= np.datetime64(pd.Timestamp.max)
    results = {}
    for name, (start, end) in windows(manifest).items():
        settled = endpoint.fillna(pd.Timestamp.max).to_numpy() <= np.datetime64(end)
        mask = np.asarray((index >= start) & (index <= end), dtype=bool) & settled
        block = {}
        for label, matrix in signals.items():
            series = daily_rank_ic(matrix.loc[mask].to_numpy(), target[mask], index[mask])
            table = summarize(series.to_frame(label)).reset_index(names="feature")
            block[label] = _finite(table.iloc[0].to_dict()) if len(table) else None
        results[name] = dict(start=str(pd.Timestamp(start).date()), end=str(pd.Timestamp(end).date()),
                             rows=int(mask.sum()), table=block)

    coverage = {name: float(matrix.notna().to_numpy().mean()) for name, matrix in signals.items()}
    nonzero = {name: float((matrix.fillna(0).to_numpy() != 0).mean())
               for name, matrix in signals.items() if name in ("earnings_score", "holder_change")}
    pair = signals["earnings_score"].to_numpy().ravel()
    other = signals["holder_change"].to_numpy().ravel()
    both = np.isfinite(pair) & np.isfinite(other)
    # Spearman without scipy: Pearson on within-series ranks.
    overlap = (float(pd.Series(pair[both]).rank().corr(pd.Series(other[both]).rank()))
               if both.sum() > 10 else None)
    return dict(dataset=str(fold_dir), version=manifest["version"], horizon=horizon,
                codes=len(codes), rows=len(index), fundamentals=str(fundamentals_dir.relative_to(PROJECT_ROOT))
                if fundamentals_dir.is_relative_to(PROJECT_ROOT) else str(fundamentals_dir),
                coverage=coverage, nonzero=nonzero, earnings_holder_spearman=overlap,
                windows=results)


def _finite(record):
    """NaN/None are not valid JSON numbers and print badly in a table."""
    out = {}
    for key, value in record.items():
        if isinstance(value, float) and not np.isfinite(value):
            out[key] = None
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def _fmt(value, spec):
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return "—"


def _markdown(payload):
    lines = [f"# 新数据块截面 IC（前 {payload['horizon']} 交易日，{payload['version']}）", "",
             f"- 代码 {payload['codes']} 只，面板 {payload['rows']} 行",
             f"- 基本面来源 `{payload['fundamentals']}`",
             f"- 覆盖率：" + "、".join(f"{name} {value:.1%}" for name, value in payload["coverage"].items()),
             f"- 非零占比：" + "、".join(f"{name} {value:.1%}" for name, value in payload["nonzero"].items()),
             f"- `earnings_score` 与 `holder_change` 的 Spearman 相关：{payload['earnings_holder_spearman']}",
             ""]
    signals = ["log_mcap", "log_float_mcap", "earnings_score", "holder_change", "momentum_20"]
    for name, block in payload["windows"].items():
        lines += [f"## {name} {block['start']} ~ {block['end']}（{block['rows']} 个决策日）", "",
                  "| 信号 | 均值 IC | IC 波动 | 稳定性 | 正比例 | 决策日 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for signal in signals:
            row = block["table"].get(signal)
            if not row:
                lines.append(f"| {signal} | — | — | — | — | 0 |")
                continue
            lines.append("| {} | {} | {} | {} | {} | {} |".format(
                signal, _fmt(row["mean_ic"], "+.4f"), _fmt(row["ic_std"], ".4f"),
                _fmt(row["stability"], "+.2f"), _fmt(row["hit_rate"], ".1%"), row["dates"]))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, help="fold dataset directory")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    payload = evaluate(arguments.fold, arguments.horizon)
    payload["inputs"] = {
        "signal_expansion.py": file_hash(Path(__file__)),
        "protocol": file_hash(PROJECT_ROOT / "doc" / "signal_expansion_protocol_20261105.md"),
    }
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "expansion.json", payload)
    (output / "expansion.md").write_text(_markdown(payload), encoding="utf-8")
    for name, block in payload["windows"].items():
        ranked = [(label, row) for label, row in block["table"].items()
                  if row and row.get("mean_ic") is not None]
        ranked.sort(key=lambda item: -abs(item[1]["mean_ic"]))
        if ranked:
            best_label, best = ranked[0]
            print(f"{name}: best={best_label} ic={best['mean_ic']:+.4f} stability={best['stability']:+.2f}")
        else:
            print(f"{name}: no settled labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
