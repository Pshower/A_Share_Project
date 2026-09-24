"""Offline, versioned feature building with training-only scaling."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RENAME = {"日期": "date", "股票代码": "code", "开盘": "open", "收盘": "close",
          "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
          "换手率": "turnover"}
FEATURES = ["return", "log_return", "range_ratio", "body_ratio", "volume_ratio", "turnover",
            "rsi_14", "macd_ratio", "macd_signal_ratio", "bb_position"] + [
    name for window in [5, 10, 20, 60]
    for name in [f"momentum_{window}", f"close_ma_ratio_{window}"]
] + ["volatility_20", "volatility_60"]


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return config


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class DataPreprocessor:
    def __init__(self, raw_dir, clean_dir, stock_list_path, start_date, end_date,
                 train_end, val_end, train_codes=None, volume_multiplier=100,
                 raw_pattern="{code}_data_*_hfq.csv", version="research_v2"):
        self.raw_dir = Path(raw_dir)
        self.clean_dir = Path(clean_dir)
        self.stock_list_path = Path(stock_list_path)
        self.start_date, self.train_end, self.val_end, self.end_date = map(
            pd.Timestamp, [start_date, train_end, val_end, end_date])
        if not self.start_date < self.train_end < self.val_end < self.end_date:
            raise ValueError("Expected start_date < train_end < val_end < end_date")
        codes = [c.strip() for c in self.stock_list_path.read_text(encoding="utf-8-sig").splitlines() if c.strip()]
        if not codes or len(codes) != len(set(codes)) or any(len(c) != 6 or not c.isdigit() for c in codes):
            raise ValueError("Stock list must contain unique six-digit string codes")
        self.stock_list = codes
        self.train_codes = codes if train_codes is None else list(train_codes)
        if not self.train_codes or not set(self.train_codes) <= set(codes):
            raise ValueError("train_codes must be a nonempty subset of the stock list")
        if not np.isfinite(volume_multiplier) or volume_multiplier <= 0:
            raise ValueError("volume_multiplier must be positive")
        self.volume_multiplier = volume_multiplier
        self.raw_pattern = raw_pattern
        self.version = version

    def _read_stock_data(self, code):
        candidates = sorted(self.raw_dir.glob(self.raw_pattern.format(code=code)))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one HFQ source for {code}, found {len(candidates)}")
        path = candidates[0]
        data = pd.read_csv(path, dtype={"股票代码": str, "code": str})
        return data, path

    def clean_single_stock(self, df):
        df = df.rename(columns=RENAME).copy()
        required = ["date", "code", "open", "close", "high", "low", "volume", "amount", "turnover"]
        if not set(required) <= set(df.columns):
            raise ValueError("Source does not satisfy the OHLCV schema")
        df = df[required]
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        if df["date"].isna().any() or df.duplicated(["date", "code"]).any():
            raise ValueError("Missing dates or duplicate stock/date rows")
        df = df.sort_values("date").reset_index(drop=True)
        for col in required[2:]:
            df[col] = pd.to_numeric(df[col], errors="raise").replace([np.inf, -np.inf], np.nan)
        prices = ["open", "close", "high", "low"]
        good = (df[prices].notna().all(axis=1) & (df[prices] > 0).all(axis=1)
                & (df["low"] <= df[["open", "close"]].min(axis=1))
                & (df["high"] >= df[["open", "close"]].max(axis=1)))
        df["valid_price"] = good
        # A missing closing/high/low quote must not suppress a valid opening fill.
        for col in prices:
            df.loc[df[col] <= 0, col] = np.nan
        for col in ["volume", "amount", "turnover"]:
            df.loc[df[col] < 0, col] = np.nan
        df["volume"] *= self.volume_multiplier
        return df

    @staticmethod
    def add_features(df):
        result = pd.DataFrame(index=df.index)
        close = df["close"].where(df["valid_price"].eq(True))
        result["return"] = close.pct_change(fill_method=None)
        result["log_return"] = np.log(close / close.shift(1))
        result["range_ratio"] = (df["high"] - df["low"]) / close.shift(1)
        result["body_ratio"] = (close - df["open"]) / df["open"]
        result["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
        result["turnover"] = df["turnover"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        result["rsi_14"] = gain / (gain + loss)
        result.loc[(gain + loss) == 0, "rsi_14"] = 0.5
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        result["macd_ratio"] = macd / close
        result["macd_signal_ratio"] = macd.ewm(span=9, adjust=False).mean() / close
        middle, std = close.rolling(20).mean(), close.rolling(20).std(ddof=0)
        result["bb_position"] = (close - middle + 2 * std) / (4 * std)
        result.loc[std == 0, "bb_position"] = 0.5
        for window in [5, 10, 20, 60]:
            result[f"momentum_{window}"] = close.pct_change(window, fill_method=None)
            result[f"close_ma_ratio_{window}"] = close / close.rolling(window).mean() - 1
        for window in [20, 60]:
            result[f"volatility_{window}"] = result["return"].rolling(window).std(ddof=0)
        result.loc[~df["valid_price"].eq(True), :] = np.nan
        return result[FEATURES].replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def normalize_features(df, fit=True, stats=None):
        features = df[FEATURES].copy()
        if fit:
            usable = features.dropna()
            if usable.empty:
                raise ValueError("No complete training feature rows")
            stats = {}
            for col in FEATURES:
                scale = float(usable[col].std(ddof=0))
                stats[col] = {"mean": float(usable[col].mean()), "std": scale if scale > 1e-12 else 1.0}
        if stats is None or set(stats) != set(FEATURES):
            raise ValueError("Scaler features do not match the feature schema")
        for col in FEATURES:
            features[col] = (features[col] - stats[col]["mean"]) / stats[col]["std"]
        return features, stats

    @staticmethod
    def align_to_panel(combined_df, feature_cols=None, fill_method=None, output_path=None):
        if fill_method is not None:
            raise ValueError("Feature panels must retain missing values for explicit masks")
        columns = FEATURES if feature_cols is None else list(feature_cols)
        panel = combined_df.pivot(index="date", columns="code", values=columns)
        panel = panel.swaplevel(0, 1, axis=1).sort_index(axis=1)
        panel.columns.names = ["code", "feature"]
        if output_path is not None:
            panel.to_parquet(output_path)
        return panel

    def split_time_series(self, df):
        return (df[df.date <= self.train_end],
                df[(df.date > self.train_end) & (df.date <= self.val_end)],
                df[df.date > self.val_end])

    def process_all_stocks_and_save_scaler(self):
        if self.clean_dir.exists():
            raise FileExistsError(f"Use a new version directory; refusing to overwrite {self.clean_dir}")
        sources, rows, audit = [], {}, []
        for code in self.stock_list:
            raw, path = self._read_stock_data(code)
            clean = self.clean_single_stock(raw)
            if not clean.code.eq(code).all():
                raise ValueError(f"Stock code mismatch in {path}")
            in_range = clean.date.between(self.start_date, self.end_date)
            record = dict(code=code, source=path.name, sha256=file_hash(path), rows=len(clean),
                          source_start=str(clean.date.min().date()), source_end=str(clean.date.max().date()),
                          rows_in_range=int(in_range.sum()), invalid_price_rows=int((~clean.loc[in_range, "valid_price"]).sum()))
            record["status"] = "included" if in_range.any() else "no_rows_in_requested_dates"
            audit.append(record)
            sources.append({"file": path.name, "sha256": record["sha256"]})
            if in_range.any():
                rows[code] = clean[clean.date <= self.end_date].set_index("date")
        if not rows:
            raise ValueError("No local data within requested dates")
        calendar = pd.DatetimeIndex(sorted(set().union(*(set(df.index) for df in rows.values()))))
        dates = calendar[calendar >= self.start_date]
        features, prices, masks, source_masks = {}, {}, {}, {}
        for code, raw in rows.items():
            aligned = raw.reindex(calendar)
            feature = self.add_features(aligned).reindex(dates)
            features[code] = feature
            prices[code] = aligned[["open", "close", "volume"]].reindex(dates)
            masks[code] = feature.notna().all(axis=1) & prices[code]["close"].notna()
            source_masks[code] = pd.Series(dates.isin(raw.index), index=dates)
        long = pd.concat(features, names=["code", "date"]).reset_index()
        train = long[(long.date <= self.train_end) & long.code.isin(self.train_codes)]
        _, stats = self.normalize_features(train)
        scaled, _ = self.normalize_features(long, fit=False, stats=stats)
        combined = pd.concat([long[["date", "code"]], scaled], axis=1)
        panel = self.align_to_panel(combined)
        price_panel = pd.concat(prices, axis=1).sort_index(axis=1)
        price_panel.columns.names = ["code", "feature"]
        valid = pd.DataFrame(masks)[panel.columns.get_level_values(0).unique()]
        splits = self.split_time_series(combined)
        if any(part.empty for part in splits):
            raise ValueError("Each chronological split must contain rows")
        quality = []
        for code, feature in features.items():
            quality.append(dict(code=code, valid_rows=int(valid[code].sum()),
                                missing_feature_fraction=float(feature.isna().mean().mean()),
                                missing_source_dates=int((~source_masks[code]).sum())))
        self.clean_dir.mkdir(parents=True)
        combined.to_parquet(self.clean_dir / "all_stocks_features.parquet", index=False)
        panel.to_parquet(self.clean_dir / "panel_data.parquet")
        price_panel.to_parquet(self.clean_dir / "prices.parquet")
        valid.to_parquet(self.clean_dir / "valid_mask.parquet")
        pd.DataFrame(source_masks).to_parquet(self.clean_dir / "source_mask.parquet")
        for name, part in zip(["train", "val", "test"], splits):
            part.to_parquet(self.clean_dir / f"{name}.parquet", index=False)
        for code, group in combined.groupby("code", sort=False):
            group.to_parquet(self.clean_dir / f"{code}_features.parquet", index=False)
        pd.DataFrame(audit).to_csv(self.clean_dir / "source_audit.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(quality).to_csv(self.clean_dir / "quality.csv", index=False)
        fit_rows = train.dropna(subset=FEATURES)
        scaler = dict(schema_version=2, feature_cols=FEATURES, stats=stats,
                      train_start=str(self.start_date.date()), train_end=str(self.train_end.date()),
                      train_codes=self.train_codes, fit_rows=len(fit_rows), ddof=0)
        write_json(self.clean_dir / "scaler_params.json", scaler)
        fitted, _ = self.normalize_features(fit_rows, fit=False, stats=stats)
        fitted.agg(["mean", "std"]).T.to_csv(self.clean_dir / "training_feature_distribution.csv")
        manifest = dict(version=self.version, schema_version=2, price_basis="hfq_research",
                        feature_cols=FEATURES, stock_codes=list(features), n_dates=len(dates),
                        rows=len(combined), source_rows=sum(a["rows_in_range"] for a in audit),
                        start_date=str(dates.min().date()), end_date=str(dates.max().date()),
                        train_end=str(self.train_end.date()), val_end=str(self.val_end.date()),
                        sources=sources, volume_unit="shares", volume_multiplier=self.volume_multiplier,
                        stock_list_sha256=file_hash(self.stock_list_path),
                        limitations=["Fixed 2026 stock-list snapshot: historical selection/survivorship bias remains.",
                                     "Calendar is the union of local quote dates, not an independent exchange calendar.",
                                     "HFQ-only research execution; no real unadjusted execution quotes or corporate-action ledger.",
                                     "Missing observations are not classified as suspension without authoritative status data.",
                                     "No historical ST, price-limit or delisting metadata."])
        manifest["build_config"] = dict(version=self.version, start_date=str(self.start_date.date()),
                                      end_date=str(self.end_date.date()), train_end=str(self.train_end.date()),
                                      val_end=str(self.val_end.date()), train_codes=self.train_codes,
                                      raw_pattern=self.raw_pattern, volume_multiplier=self.volume_multiplier)
        manifest["builder_sha256"] = file_hash(__file__)
        manifest["artifacts"] = {p.name: file_hash(p) for p in self.clean_dir.glob("*.parquet")}
        manifest["artifacts"]["scaler_params.json"] = file_hash(self.clean_dir / "scaler_params.json")
        write_json(self.clean_dir / "manifest.json", manifest)
        return combined

    def transform_new_data(self, raw_df, scaler_path, code=None):
        scaler = json.loads(Path(scaler_path).read_text(encoding="utf-8"))
        if scaler.get("schema_version") != 2 or scaler["feature_cols"] != FEATURES:
            raise ValueError("Incompatible scaler version")
        clean = self.clean_single_stock(raw_df)
        if code is not None and not clean.code.eq(code).all():
            raise ValueError("Stock code mismatch")
        if clean.code.nunique() != 1:
            raise ValueError("Transform one stock at a time with sufficient historical warmup")
        # Caller supplies the complete trading-date history, including missing-date rows.
        feature = self.add_features(clean.set_index("date"))
        scaled, _ = self.normalize_features(feature, fit=False, stats=scaler["stats"])
        scaled["code"] = clean.code.iloc[0]
        return scaled.reset_index()


def build_from_config(config):
    params = dict(config["dataset"])
    for key in ["raw_dir", "clean_dir", "stock_list_path"]:
        params[key] = resolve_path(params[key])
    return DataPreprocessor(**params).process_all_stocks_and_save_scaler()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research.json")
    args = parser.parse_args()
    result = build_from_config(read_config(args.config))
    print(f"Built {len(result)} stock/date rows from local files")
