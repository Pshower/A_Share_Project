"""Derive past-only research folds from immutable, unclipped local features."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import file_hash, read_config, write_json


FOLDS = {
    str(year): dict(train_end=f"{year}-03-31", val_start=f"{year}-04-01", val_end=f"{year}-06-30")
    for year in (2023, 2024, 2025)
}
PROTECTED_START = pd.Timestamp("2026-07-01")
DERIVATION_VERSION = "past_only_affine_fold_v1"


def _read_verified(directory):
    manifest = read_config(directory / "manifest.json")
    if manifest.get("schema_version") != 2:
        raise ValueError("Only schema-2 local feature panels are supported")
    names = ("panel_data.parquet", "prices.parquet", "scaler_params.json")
    hashes = {name: file_hash(directory / name) for name in names}
    if any(manifest["artifacts"].get(name) != value for name, value in hashes.items()):
        raise ValueError("Original dataset artifact hash mismatch")
    scaler = read_config(directory / "scaler_params.json")
    if scaler.get("feature_cols") != manifest["feature_cols"] or scaler.get("schema_version") != 2:
        raise ValueError("Original scaler schema mismatch")
    if any(scaler.get(name) is not None for name in ("clip", "clip_range", "clip_value")):
        raise ValueError("A clipped scaler cannot be inverted without original features")
    panel = pd.read_parquet(directory / "panel_data.parquet")
    prices = pd.read_parquet(directory / "prices.parquet")
    codes, features = manifest["stock_codes"], manifest["feature_cols"]
    expected = pd.MultiIndex.from_product([codes, features], names=["code", "feature"])
    if (panel.columns.has_duplicates or set(panel.columns) != set(expected)
            or not panel.index.equals(prices.index) or panel.index.has_duplicates
            or not panel.index.is_monotonic_increasing):
        raise ValueError("Original panel dates or columns do not match the manifest")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError("Panel must have datetime trading-date index")
    panel = panel.reindex(columns=expected)
    prices = prices.reindex(columns=pd.MultiIndex.from_product(
        [codes, prices.columns.get_level_values(1).unique()], names=["code", "feature"]))
    for feature in features:
        stat = scaler["stats"][feature]
        if not np.isfinite(stat["mean"]) or not np.isfinite(stat["std"]) or stat["std"] <= 0:
            raise ValueError("Original scaler statistics must be finite with positive scale")
    return manifest, scaler, panel, prices, hashes


def inverse_affine_features(panel, scaler, feature_cols):
    """Recover raw features; no future row contributes to a subsequent fold fit."""
    raw = panel.copy()
    for feature in feature_cols:
        columns = raw.columns.get_level_values(1) == feature
        stat = scaler["stats"][feature]
        raw.loc[:, columns] = raw.loc[:, columns] * stat["std"] + stat["mean"]
    return raw


def fit_fold_scaler(raw, prices, feature_cols, training_codes, train_end):
    training = raw.loc[raw.index <= pd.Timestamp(train_end)]
    rows = []
    for code in training_codes:
        values = training[code].reindex(columns=feature_cols).to_numpy(dtype=np.float64)
        close = prices.loc[training.index, (code, "close")].to_numpy(dtype=np.float64)
        valid = np.isfinite(values).all(axis=1) & np.isfinite(close) & (close > 0)
        rows.append(values[valid])
    usable = np.concatenate(rows, axis=0)
    if not len(usable):
        raise ValueError("No complete, valid training feature rows in this fold")
    mean, std = usable.mean(axis=0), usable.std(axis=0, ddof=0)
    std = np.where(std > 1e-12, std, 1.0)
    return {name: dict(mean=float(mean[i]), std=float(std[i])) for i, name in enumerate(feature_cols)}, len(usable)


def derive_fold(original_dir, original_research_config, output_dir, fold_id, training_codes=None):
    """Write a new ResearchData-compatible directory, never altering original data.

    ``training_codes`` limits scaler fitting only. The panel retains all stocks so
    held-out-stock environments can share the exact same past-only transformation.
    """
    fold_id = str(fold_id)
    if fold_id not in FOLDS:
        raise ValueError("Registered folds are 2023, 2024 and 2025 Q2")
    original_dir, output_dir = Path(original_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("Use a new fold directory; refusing to overwrite artifacts")
    manifest, old_scaler, panel, prices, original_hashes = _read_verified(original_dir)
    research = (deepcopy(original_research_config) if isinstance(original_research_config, dict)
                else read_config(original_research_config))
    codes, features = manifest["stock_codes"], manifest["feature_cols"]
    train_codes = list(codes if training_codes is None else training_codes)
    if not train_codes or len(set(train_codes)) != len(train_codes) or not set(train_codes) <= set(codes):
        raise ValueError("training_codes must be a nonempty unique subset of manifest stocks")
    fold = FOLDS[fold_id]
    # Physically exclude the protected period, before inverse transform or fitting.
    panel = panel.loc[panel.index < PROTECTED_START]
    prices = prices.loc[panel.index]
    if panel.empty or panel.index.min() >= pd.Timestamp(fold["train_end"]):
        raise ValueError("Insufficient dates for the registered fold")
    if not (panel.index > pd.Timestamp(fold["val_end"])).any():
        raise ValueError("ResearchData schema requires a nonempty post-validation split")
    raw = inverse_affine_features(panel, old_scaler, features)
    stats, fit_rows = fit_fold_scaler(raw, prices, features, train_codes, fold["train_end"])
    scaled = raw.copy()
    for feature, stat in stats.items():
        columns = scaled.columns.get_level_values(1) == feature
        scaled.loc[:, columns] = (raw.loc[:, columns] - stat["mean"]) / stat["std"]
    identity = dict(fold_id=fold_id, directory=str(output_dir), training_codes=train_codes,
                    source_manifest_sha256=file_hash(original_dir / "manifest.json"))
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    version = f"exploration_{fold_id}_{identity_hash}"
    scaler = dict(schema_version=2, feature_cols=features, stats=stats, train_codes=train_codes,
                  train_start=manifest["build_config"]["start_date"], train_end=fold["train_end"],
                  fit_rows=fit_rows, ddof=0)
    for key in ("extra_features", "feature_definition_version"):
        if key in old_scaler:
            scaler[key] = deepcopy(old_scaler[key])
    build = deepcopy(manifest["build_config"])
    build.update(version=version, train_end=fold["train_end"], val_end=fold["val_end"],
                 end_date="2026-06-30", train_codes=train_codes)
    derived = deepcopy(manifest)
    derived.update(version=version, start_date=str(panel.index.min().date()),
                   end_date=str(panel.index.max().date()), train_end=fold["train_end"], val_end=fold["val_end"],
                   n_dates=len(panel), rows=len(panel) * len(codes), build_config=build,
                   builder_sha256=file_hash(__file__), derivation_version=DERIVATION_VERSION,
                   protected_period_excluded=True, source_identity=dict(
                       directory=str(original_dir), manifest_sha256=file_hash(original_dir / "manifest.json"),
                       artifacts=original_hashes, original_version=manifest["version"],
                       raw_feature_recovery="inverse unclipped original affine scaler; fold refit uses past rows only"))
    derived.pop("source_rows", None)
    derived["limitations"] = list(manifest.get("limitations", [])) + [
        "2023-2025 Q2 are historical research folds, not independent prospective deployment evidence.",
        "Original 2026 scaler is inverted, not used for fold model inputs; floating-point inverse error remains.",
        "Post-validation rows through 2026-06-30 are retained only for loader compatibility, not evaluated."]
    research["dataset"].update(build)
    research["dataset"]["clean_dir"] = str(output_dir)
    output_dir.mkdir(parents=True)
    scaled.to_parquet(output_dir / "panel_data.parquet")
    prices.to_parquet(output_dir / "prices.parquet")
    valid = pd.DataFrame({code: np.isfinite(scaled[code].to_numpy()).all(axis=1)
                          & np.isfinite(prices[(code, "close")]) & (prices[(code, "close")] > 0)
                          for code in codes}, index=panel.index)
    valid.to_parquet(output_dir / "valid_mask.parquet")
    write_json(output_dir / "scaler_params.json", scaler)
    derived["artifacts"] = {name: file_hash(output_dir / name) for name in
                            ("panel_data.parquet", "prices.parquet", "valid_mask.parquet", "scaler_params.json")}
    write_json(output_dir / "manifest.json", derived)
    write_json(output_dir / "research.json", research)
    return dict(fold_id=fold_id, directory=str(output_dir), research_config=str(output_dir / "research.json"),
                manifest=str(output_dir / "manifest.json"), scaler=str(output_dir / "scaler_params.json"),
                panel=str(output_dir / "panel_data.parquet"), prices=str(output_dir / "prices.parquet"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--research-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", choices=tuple(FOLDS), required=True)
    args = parser.parse_args()
    derive_fold(args.original_dir, args.research_config, args.output, args.fold)


if __name__ == "__main__":
    main()
