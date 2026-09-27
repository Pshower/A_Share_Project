"""Read-only model/source preflight for explicit close-date research requests."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.preprocess import read_config, file_hash
from src.data.snapshots import Snapshots
from src.inference.eod import require_close_capture


def capabilities():
    return dict(default_history_source="sina", sources=[
        dict(id="sina", label="新浪", history_bases=["hfq", "unadjusted"], quotes=False, supplements=["dividend"]),
        dict(id="eastmoney", label="东方财富", history_bases=["hfq", "unadjusted"], quotes=True,
             supplements=["profile", "fund_flow", "minute", "index_daily"]),
    ], quote_source="eastmoney", market_ui_version="model-comparison-v1")


def model_requirements(catalog, model_id):
    from src.inference.live import feature_history_origin
    cfg, record, _ = catalog.model_config(model_id)
    path = Path(cfg["research_config"])
    research = read_config(path if path.is_absolute() else catalog.root / path)
    origin, _ = feature_history_origin(research)
    contract = record["metadata"]["contract"]
    return dict(model_id=model_id, model_sha256=record["metadata"]["model_sha256"],
                codes=contract["stock_codes"], history_origin=str(origin.date()),
                price_basis=record["price_basis"], trained=record["trained"],
                source=contract.get("data_source"), train_end=research["dataset"]["train_end"],
                val_end=research["dataset"]["val_end"],
                policy_type="momentum_residual" if contract.get("prior") else "sparse_ppo" if contract.get("action_top_k") else "dense_ppo",
                scaler_sha256=contract["artifacts"]["scaler_params.json"])


def comparison_preflight(catalog, body, now=None):
    now = now or datetime.now(timezone.utc)
    snapshot, directory = Snapshots(catalog.root).get("history", body.history_id)
    entries = {e["code"]: e for e in snapshot["entries"] if e["basis"] == "hfq" and e["status"] == "ready"}
    dates = {}
    rows = []
    for model_id in body.model_ids:
        errors = []
        try:
            requirement = model_requirements(catalog, model_id)
            if not requirement["trained"]:
                errors.append("模型尚未训练")
            if not requirement["source"]:
                errors.append("模型来源未认证")
            elif requirement["source"] != snapshot["request"]["provider"]:
                errors.append("模型训练来源与日线来源不一致")
            if requirement["price_basis"] != "hfq_research":
                errors.append("当前行情计划仅支持HFQ研究模型")
            if snapshot["status"] != "ready":
                errors.append("日线批次未完整保存")
            if body.display_code not in requirement["codes"]:
                errors.append("图表股票不在该模型股票池中")
            if not set(body.research_weights) <= set(requirement["codes"]):
                errors.append("研究持仓含模型池外股票，不能静默丢弃")
            if snapshot["request"]["start_date"] > requirement["history_origin"]:
                errors.append("历史数据起点晚于模型特征锚点")
            missing = [c for c in requirement["codes"] if c not in entries]
            if missing:
                errors.append("缺少HFQ股票：" + ", ".join(missing[:10]))
            if not errors:
                for code in requirement["codes"]:
                    require_close_capture(body.close_as_of, entries[code]["received_at"], now)
                    if code not in dates:
                        dates[code] = set(pd.to_datetime(pd.read_parquet(directory / "hfq" / f"{code}.parquet", columns=["date"]).date).dt.date)
                    if body.close_as_of not in dates[code]:
                        errors.append(f"{code} 缺少指定收盘日 {body.close_as_of}")
            rows.append(dict(**requirement, ready=not errors, errors=errors,
                             historical_label="训练期内回溯" if str(body.close_as_of) <= requirement["train_end"] else
                                              "开发验证期回溯" if str(body.close_as_of) <= requirement["val_end"] else "训练后单日研究推理"))
        except (ValueError, KeyError, OSError) as exc:
            rows.append(dict(model_id=model_id, ready=False, errors=[str(exc)]))
    return dict(history_id=body.history_id, source=snapshot["request"]["provider"],
                history_manifest_sha256=file_hash(directory / "manifest.json"),
                close_as_of=str(body.close_as_of), models=rows,
                note="预检不计算特征或预测；最终任务仍校验特征窗口、历史修订与模型契约")
