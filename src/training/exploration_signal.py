"""Fixed ridge signal diagnostics: past-only fit, purged forward labels, no trades."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import file_hash, read_config, write_json
from src.training.exploration_data import FOLDS, PROTECTED_START


HORIZONS = (1, 5, 20)
RIDGE_ALPHA = 1.0
LABEL_DEFINITION = "close[t+h] / open[t+1] - 1; h union-calendar trading rows; complete positive open/close path"


def forward_labels(prices, codes, horizon):
    """Require all intervening open/close rows; never jump over missing quotes."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    dates = prices.index
    labels = pd.DataFrame(np.nan, index=dates, columns=codes, dtype=float)
    endpoint = pd.Series(pd.NaT, index=dates, dtype="datetime64[ns]")
    if len(dates) <= horizon:
        return labels, endpoint
    endpoint.iloc[:-horizon] = dates[horizon:].to_numpy()
    for code in codes:
        opening = prices[(code, "open")].to_numpy(dtype=float)
        closing = prices[(code, "close")].to_numpy(dtype=float)
        valid = np.isfinite(opening) & np.isfinite(closing) & (opening > 0) & (closing > 0)
        path = np.ones(len(dates) - horizon, dtype=bool)
        for offset in range(1, horizon + 1):
            path &= valid[offset:offset + len(path)]
        values = np.full(len(path), np.nan)
        values[path] = closing[horizon:][path] / opening[1:1 + len(path)][path] - 1
        labels.iloc[:-horizon, labels.columns.get_loc(code)] = values
    return labels, endpoint


def split_examples(panel, labels, endpoint, codes, features, start, end):
    """Purge labels ending after each split boundary, then apply feature masks."""
    end = pd.Timestamp(end)
    decisions = (panel.index <= end) & endpoint.notna().to_numpy() & (endpoint <= end).to_numpy()
    if start is not None:
        decisions &= panel.index >= pd.Timestamp(start)
    dates = panel.index[decisions]
    rows, targets, keys = [], [], []
    for code in codes:
        values = panel.loc[dates, code].reindex(columns=features).to_numpy(dtype=float)
        target = labels.loc[dates, code].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1) & np.isfinite(target)
        rows.append(values[valid])
        targets.append(target[valid])
        keys.extend((date, code, endpoint.loc[date]) for date in dates[valid])
    x = np.concatenate(rows, axis=0)
    y = np.concatenate(targets, axis=0)
    if not len(x):
        raise ValueError("No valid purged examples in split")
    frame = pd.DataFrame(keys, columns=["date", "code", "label_endpoint"])
    return x, y, frame, dict(candidate_dates=len(dates), retained_rows=len(x),
                             masked_rows=len(dates) * len(codes) - len(x),
                             max_decision_date=str(frame.date.max().date()),
                             max_label_endpoint=str(frame.label_endpoint.max().date()),
                             label_end_purge_dates=int(((panel.index <= end) & (endpoint > end).to_numpy()).sum()))


def fit_ridge(x, y, alpha=RIDGE_ALPHA):
    """Center labels and refit input scaling using only label-eligible train rows."""
    if (x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(x)
            or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError("Ridge inputs and positive alpha must be finite")
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    constant = std <= 1e-12
    std = np.where(constant, 1.0, std)
    z, center = (x - mean) / std, float(y.mean())
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - center))
    return dict(alpha=float(alpha), feature_mean=mean.tolist(), feature_std=std.tolist(),
                coef=coef.tolist(), intercept=center, zero_variance_features=constant.tolist())


def predict_ridge(model, x):
    return ((x - np.asarray(model["feature_mean"])) / np.asarray(model["feature_std"])) @ np.asarray(model["coef"]) + model["intercept"]


def _correlation(a, b):
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def per_date_metrics(predictions, method, top_n=20):
    records = []
    for date, group in predictions.groupby("date", sort=True):
        score, label = group[method].to_numpy(), group.label.to_numpy()
        finite = np.isfinite(score) & np.isfinite(label)
        group, score, label = group.loc[finite], score[finite], label[finite]
        varied = len(score) >= 3 and np.std(score) > 1e-12
        pearson = _correlation(score, label)
        spearman = _correlation(pd.Series(score).rank(method="average").to_numpy(),
                                pd.Series(label).rank(method="average").to_numpy())
        spread = top_return = bottom_return = None
        if varied and len(score) >= 2 * top_n:
            # Code tie-break is stable, recorded, and irrelevant to the zero baseline.
            ordered = group.sort_values([method, "code"], ascending=[False, True])
            top_return, bottom_return = float(ordered.label.iloc[:top_n].mean()), float(ordered.label.iloc[-top_n:].mean())
            spread = top_return - bottom_return
        records.append(dict(date=str(date.date()), method=method, coverage=len(group),
                            zero_variance_score=not varied, pearson_ic=pearson, spearman_ic=spearman,
                            top20_gross_return=top_return, bottom20_gross_return=bottom_return,
                            top20_bottom20_gross_spread=spread))
    return records


def _aggregate(rows):
    result = dict(dates=len(rows), covered_stock_dates=sum(row["coverage"] for row in rows),
                  zero_variance_score_dates=sum(row["zero_variance_score"] for row in rows))
    for metric in ("pearson_ic", "spearman_ic", "top20_gross_return", "bottom20_gross_return", "top20_bottom20_gross_spread"):
        values = [row[metric] for row in rows if row[metric] is not None]
        result[metric] = dict(mean=float(np.mean(values)) if values else None,
                              std=float(np.std(values, ddof=0)) if values else None, valid_dates=len(values))
    return result


def run_signal_diagnostics(fold_paths, output_dir):
    """Fit fixed alpha=1 models on each fold and save every validation prediction."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite diagnostic output")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    all_metrics, all_predictions, cases = [], [], []
    for item in fold_paths:
        directory = Path(item["directory"] if isinstance(item, dict) else item).resolve()
        manifest, scaler = read_config(directory / "manifest.json"), read_config(directory / "scaler_params.json")
        fold_id = str(manifest["train_end"][:4])
        if fold_id not in FOLDS or manifest["train_end"] != FOLDS[fold_id]["train_end"] or manifest["val_end"] != FOLDS[fold_id]["val_end"]:
            raise ValueError("Signal folds must match registered historical boundaries")
        for name in ("panel_data.parquet", "prices.parquet", "scaler_params.json"):
            if file_hash(directory / name) != manifest["artifacts"][name]:
                raise ValueError("Fold artifact hash mismatch")
        panel, prices = pd.read_parquet(directory / "panel_data.parquet"), pd.read_parquet(directory / "prices.parquet")
        if not panel.index.equals(prices.index) or panel.index.max() >= PROTECTED_START:
            raise ValueError("Fold dates mismatch or protected-period rows present")
        codes, features = manifest["stock_codes"], manifest["feature_cols"]
        fit_codes = scaler["train_codes"]
        expected = pd.MultiIndex.from_product([codes, features])
        if (scaler.get("schema_version") != 2 or scaler.get("feature_cols") != features
                or scaler.get("train_end") != manifest["train_end"] or not fit_codes
                or len(set(fit_codes)) != len(fit_codes) or not set(fit_codes) <= set(codes)
                or manifest["build_config"]["train_codes"] != fit_codes
                or panel.columns.has_duplicates or set(panel.columns) != set(expected)
                or panel.index.has_duplicates or not panel.index.is_monotonic_increasing):
            raise ValueError("Signal fold scaler, stock or feature schema mismatch")
        for horizon in HORIZONS:
            labels, endpoint = forward_labels(prices, codes, horizon)
            train_x, train_y, _, train_audit = split_examples(panel, labels, endpoint, fit_codes, features, None, manifest["train_end"])
            val_x, val_y, frame, val_audit = split_examples(panel, labels, endpoint, codes, features, FOLDS[fold_id]["val_start"], manifest["val_end"])
            model = fit_ridge(train_x, train_y)
            model.update(fold_id=fold_id, horizon=horizon, feature_cols=features, training_codes=fit_codes,
                         label_definition=LABEL_DEFINITION, manifest_sha256=file_hash(directory / "manifest.json"),
                         train_audit=train_audit)
            model_path = output_dir / "models" / f"ridge_{fold_id}_h{horizon}.json"
            write_json(model_path, model)
            frame["label"], frame["ridge"] = val_y, predict_ridge(model, val_x)
            frame["momentum_5"] = val_x[:, features.index("momentum_5")]
            frame["momentum_20"] = val_x[:, features.index("momentum_20")]
            frame["zero"], frame["fold_id"], frame["horizon"] = 0.0, fold_id, horizon
            methods = {}
            for method in ("ridge", "momentum_5", "momentum_20", "zero"):
                rows = per_date_metrics(frame, method)
                methods[method] = _aggregate(rows)
                all_metrics.extend(dict(row, fold_id=fold_id, horizon=horizon) for row in rows)
            cases.append(dict(fold_id=fold_id, horizon=horizon, train=train_audit, validation=val_audit,
                              methods=methods, ridge_model=str(model_path), ridge_model_sha256=file_hash(model_path),
                              training_pool=list(fit_codes), validation_pool=list(codes)))
            all_predictions.append(frame)
    pd.DataFrame(all_metrics).to_csv(output_dir / "per_date.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_parquet(output_dir / "predictions.parquet", index=False)
    summary = dict(schema_version=1, label_definition=LABEL_DEFINITION, horizons=list(HORIZONS), ridge_alpha=RIDGE_ALPHA,
                   cases=cases, test_evaluated=False, protected_period_excluded=True,
                   limitations=["Diagnostic labels use HFQ research prices and the union quote calendar.",
                                "IC and overlapping top-bottom gross spreads are not net, executable strategy returns.",
                                "Fixed 2026 stock membership creates historical survivorship/selection bias.",
                                "Historical fold results are research evidence, not unseen prospective testing."],
                   artifacts={name: file_hash(output_dir / name) for name in ("per_date.csv", "predictions.parquet")})
    write_json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_signal_diagnostics(args.fold, args.output)


if __name__ == "__main__":
    main()
