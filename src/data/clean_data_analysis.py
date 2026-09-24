"""Inspect the versioned dataset without imposing per-stock unit variance."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


class CleanDataAnalyzer:
    def __init__(self, clean_dir):
        self.clean_dir = Path(clean_dir)

    def generate_report(self, save_path=None):
        meta = json.loads((self.clean_dir / "manifest.json").read_text(encoding="utf-8"))
        df = pd.read_parquet(self.clean_dir / "all_stocks_features.parquet")
        features = meta["feature_cols"]
        values = df[features].to_numpy()
        if np.isinf(values).any():
            raise ValueError("Infinite features detected")
        records = []
        for code, group in df.groupby("code"):
            records.append(dict(code=code, rows=len(group), valid_rows=int(group[features].notna().all(axis=1).sum()),
                                missing_ratio=float(group[features].isna().mean().mean())))
        report = pd.DataFrame(records)
        if save_path is not None:
            report.to_csv(save_path, index=False)
        scaler = json.loads((self.clean_dir / "scaler_params.json").read_text(encoding="utf-8"))
        training = df[(df.date <= pd.Timestamp(meta["train_end"])) & df.code.isin(scaler["train_codes"])]
        distribution = training.dropna(subset=features)[features].agg(["mean", "std"]).T
        print(f"stocks={len(report)} rows={len(df)} valid_rows={report.valid_rows.sum()}")
        print("Training-feature distribution only; per-stock and out-of-sample unit variance is not required:")
        print(distribution.to_string())
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2] / "data/clean/research_v3")
    args = parser.parse_args()
    CleanDataAnalyzer(args.data).generate_report()
