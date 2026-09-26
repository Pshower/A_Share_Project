"""Local research HTTP API. Start with python -m src.web.serve."""

import asyncio
from contextlib import asynccontextmanager
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import secrets
import sys
from typing import Literal

from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

from src.data.preprocess import ROOT, read_config
from .catalog import Catalog, inside
from .jobs import JobManager, TERMINAL, read_events
from .store import Store
from .schemas import Training, Build, Evaluation, ModelRequest, Prediction, Transfer, Label
from .schemas import OnlineHistory, OnlineQuotes, DailyPlan, OnlineBuild, MonitorRequest
from src.data.snapshots import Snapshots


def create_app(root=ROOT, start_workers=True):
    root = Path(root).resolve()
    catalog, store = Catalog(root), Store(root / ".workbench")
    manager = JobManager(store, root)
    market_managers = [JobManager(store, root, lane="history"), JobManager(store, root, lane="quotes")]
    snapshots = Snapshots(root)
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        for record in catalog.reports():
            if record["split"] == "test":
                contract = record["experiment"].get("contract", {})
                try:
                    path = root / record["relative_path"] / "ppo/daily.csv"
                    dates = pd.read_csv(path, usecols=["date"])["date"]
                    store.expose("test_performance", contract.get("dataset_version", "unknown"), str(dates.iloc[1])[:10], str(dates.iloc[-1])[:10], record["id"])
                except (OSError, ValueError, IndexError):
                    pass
        if start_workers:
            manager.start()
            for runner in market_managers:
                runner.start()
        yield
        manager.close()
        for runner in market_managers:
            runner.close()

    app = FastAPI(title="A-share Research Workbench", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.store, app.state.manager, app.state.catalog = store, manager, catalog

    @app.middleware("http")
    async def local_security(request: Request, call_next):
        if request.url.hostname not in {"127.0.0.1", "localhost", "testserver"}:
            return JSONResponse({"error": "Untrusted Host"}, status_code=403)
        origin = request.headers.get("origin")
        expected = str(request.base_url).rstrip("/")
        if origin and origin != expected:
            return JSONResponse({"error": "Cross-origin requests are disabled"}, status_code=403)
        if request.method not in {"GET", "HEAD"} and not secrets.compare_digest(request.headers.get("x-csrf-token", ""), token):
            return JSONResponse({"error": "Invalid local session token"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api") else "no-cache"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"error": str(error)}, status_code=409)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, error):
        return JSONResponse({"error": "Required local artifact is missing"}, status_code=404)

    @app.get("/api/session")
    def session():
        return dict(token=token)

    @app.get("/api/system")
    def system():
        from src.runtime.device import device_status
        packages = {}
        for name in ["torch", "stable-baselines3", "fastapi", "gymnasium"]:
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        return dict(interpreter=sys.executable, environment=Path(sys.prefix).name,
                    packages=packages, **device_status(), heavy_slots=1,
                    price_basis="hfq_research", workspace=str(root), exposures=store.exposures())

    @app.get("/api/defaults")
    def defaults():
        return dict(ppo=read_config(root / "configs/ppo.json"),
                    light_codes=read_config(root / "configs/ppo_lightweight.json")["stock_codes"])

    @app.get("/api/market/status")
    def market_status():
        from datetime import datetime
        from src.data.online import CHINA
        return dict(provider="eastmoney", history_basis=["hfq", "unadjusted"], quote_basis="unadjusted",
                    today=str(datetime.now(CHINA).date()), min_poll_seconds=60, max_quote_stocks=50,
                    real_account_compatible=False, supports_orders=False,
                    timestamp_semantics="unverified f86 Unix-seconds field; not exchange-certified trade time")

    @app.get("/api/market/model-requirements/{model_id}")
    def model_requirements(model_id: str):
        cfg, record, _ = catalog.model_config(model_id)
        research_path = Path(cfg["research_config"])
        research = read_config(research_path if research_path.is_absolute() else root / research_path)
        from src.inference.live import feature_history_origin
        origin, _ = feature_history_origin(research)
        return dict(codes=record["metadata"]["contract"]["stock_codes"],
                    history_origin=str(origin.date()),
                    price_basis=record["price_basis"], trained=record["trained"])

    @app.get("/api/market/snapshots/{kind}")
    def online_snapshots(kind: str):
        if kind not in Snapshots.KINDS:
            raise ValueError("Unknown snapshot kind")
        return snapshots.list(kind)

    @app.get("/api/market/snapshots/{kind}/{snapshot_id}")
    def online_snapshot(kind: str, snapshot_id: str, plan_id: str | None = None):
        if kind not in Snapshots.KINDS:
            raise ValueError("Unknown snapshot kind")
        record, _ = snapshots.get(kind, snapshot_id)
        if kind == "quotes":
            from src.inference.live import judge_quotes
            plan = snapshots.get("plans", plan_id)[0] if plan_id else None
            return dict(snapshot=record, judgment=judge_quotes(record, plan))
        return record

    @app.get("/api/market/history/{snapshot_id}/bars")
    def history_bars(snapshot_id: str, code: str, basis: Literal["hfq", "unadjusted"] = "unadjusted"):
        record, directory = snapshots.get("history", snapshot_id)
        entry = next((e for e in record["entries"] if e["code"] == code and e["basis"] == basis), None)
        if not entry or entry["status"] != "ready":
            raise ValueError("No saved bars for this stock and price basis")
        name = f"{basis}/{code}.parquet"
        if name not in record["files"]:
            raise ValueError("Bars are not registered in the snapshot")
        frame = pd.read_parquet(inside(directory / name, directory))
        frame = frame.sort_values("date")
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        columns = ["date", "open", "close", "low", "high", "volume", "amount"]
        return dict(snapshot_id=snapshot_id, code=code, basis=basis, volume_unit="shares",
                    received_at=entry.get("received_at"),
                    rows=json.loads(frame[columns].to_json(orient="records")))

    @app.post("/api/market/downloads")
    def online_download(body: OnlineHistory, idempotency_key: str | None = Header(default=None)):
        if body.resume_id:
            old, _ = snapshots.get("history", body.resume_id)
            expected = dict(provider="eastmoney", codes=body.codes, start_date=str(body.start_date), end_date=str(body.end_date), bases=body.bases)
            if old["request"] != expected:
                raise ValueError("Retry request differs from original capture")
        return submit("online_history", body.model_dump(mode="json"), idempotency_key)

    def quote_pool(body):
        if body.plan_id:
            plan, _ = snapshots.get("plans", body.plan_id)
            if body.codes != plan["stock_codes"]:
                raise ValueError("Quote the entire ordered plan pool, not just displayed stocks")

    @app.post("/api/market/quotes")
    def online_quote(body: OnlineQuotes, idempotency_key: str | None = Header(default=None)):
        quote_pool(body)
        return submit("online_quotes", body.model_dump(mode="json"), idempotency_key)

    @app.post("/api/market/plans")
    def daily_plan(body: DailyPlan, idempotency_key: str | None = Header(default=None)):
        record, _ = catalog.model(body.model_id)
        history, _ = snapshots.get("history", body.history_id)
        if not record["trained"] or history["status"] != "ready":
            raise ValueError("A trained model and complete download are required")
        return submit("daily_plan", body.model_dump(mode="json"), idempotency_key)

    @app.post("/api/market/build")
    def online_build(body: OnlineBuild, idempotency_key: str | None = Header(default=None)):
        catalog.dataset(body.dataset_id)
        history, _ = snapshots.get("history", body.history_id)
        if history["status"] != "ready" or "hfq" not in history["request"]["bases"]:
            raise ValueError("A complete HFQ snapshot is required")
        return submit("online_build", body.model_dump(mode="json"), idempotency_key)

    @app.post("/api/market/monitors")
    def monitor_start(body: MonitorRequest):
        quote_pool(body)
        return store.create_monitor(body.model_dump(mode="json"))

    @app.get("/api/market/monitors")
    def monitors():
        with store.connect() as conn:
            ids = [row[0] for row in conn.execute("SELECT id FROM monitors ORDER BY created DESC LIMIT 20")]
        return [store.monitor(identifier) for identifier in ids]

    @app.get("/api/market/monitors/{monitor_id}")
    def monitor_detail(monitor_id: str):
        from src.inference.live import judge_quotes
        record = store.monitor(monitor_id)
        record["last_job_record"] = store.get(record["last_job"]) if record["last_job"] else None
        last = record["last_job_record"]
        if last and last["status"] == "succeeded" and last["result"].get("snapshot_id"):
            quote = snapshots.get("quotes", last["result"]["snapshot_id"])[0]
            plan = snapshots.get("plans", record["payload"]["plan_id"])[0] if record["payload"]["plan_id"] else None
            record["judgment"] = judge_quotes(quote, plan)
        return record

    @app.post("/api/market/monitors/{monitor_id}/heartbeat")
    def monitor_renew(monitor_id: str):
        return store.renew_monitor(monitor_id)

    @app.post("/api/market/monitors/{monitor_id}/stop")
    def monitor_stop(monitor_id: str):
        record = store.stop_monitor(monitor_id)
        if record["last_job"]:
            manager.request_stop(record["last_job"])
        return store.monitor(monitor_id)

    @app.get("/api/datasets")
    def datasets():
        return [{k: v for k, v in row.items() if k != "manifest"} for row in catalog.datasets()]

    @app.get("/api/datasets/{dataset_id}")
    def dataset(dataset_id: str):
        record, path = catalog.dataset(dataset_id)
        return dict(**record, quality=catalog.stocks(dataset_id))

    def validate_pool(dataset_id, codes):
        record, _ = catalog.dataset(dataset_id)
        if len(set(codes)) != len(codes) or not set(codes) <= set(record["manifest"]["stock_codes"]):
            raise ValueError("Selected stocks are duplicated or outside the dataset")

    def submit(kind, payload, key):
        job = store.enqueue(kind, payload, key)
        return JSONResponse(job, status_code=202)

    @app.post("/api/training/jobs")
    def train(body: Training, idempotency_key: str | None = Header(default=None)):
        validate_pool(body.dataset_id, body.codes)
        return submit("training", body.model_dump(mode="json"), idempotency_key)

    @app.post("/api/training/validate")
    def check(body: Training, idempotency_key: str | None = Header(default=None)):
        validate_pool(body.dataset_id, body.codes)
        return submit("check", body.model_dump(mode="json"), idempotency_key)

    @app.post("/api/datasets/build")
    def build(body: Build, idempotency_key: str | None = Header(default=None)):
        validate_pool(body.dataset_id, body.codes)
        if (root / "data/clean" / body.version).exists():
            raise ValueError("Dataset version already exists; successful versions are immutable")
        return submit("build", body.model_dump(mode="json"), idempotency_key)

    @app.get("/api/jobs")
    def jobs():
        return store.jobs()

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str):
        job = store.get(job_id)
        directory = manager.directory(job_id)
        event_list = read_events(directory)
        log = directory / "worker.log"
        job["events"] = event_list[-2000:]
        job["metrics"] = [e for e in event_list if e["type"] in {"metrics", "validation"}]
        job["log"] = log.read_text(encoding="utf-8", errors="replace")[-20000:] if log.exists() else ""
        if job["result"] and job["result"].get("prediction_path"):
            job["prediction"] = read_config(inside(root / job["result"]["prediction_path"], catalog.runs))
        if job["result"] and job["result"].get("output"):
            output = inside(root / job["result"]["output"], root)
            if (output / "training_audit.json").exists():
                job["audit"] = read_config(output / "training_audit.json")
            if (output / "check.json").exists():
                job["check"] = read_config(output / "check.json")
        return job

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str):
        path = manager.directory(job_id) / "worker.log"
        if not path.exists():
            raise ValueError("Log is not available yet")
        return FileResponse(path, filename=job_id + ".log")

    @app.post("/api/jobs/{job_id}/stop")
    def stop(job_id: str):
        return manager.request_stop(job_id)

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        store.get(job_id)
        try:
            after = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            after = 0

        async def stream():
            cursor = after
            while not await request.is_disconnected():
                batch = await asyncio.to_thread(read_events, manager.directory(job_id), cursor)
                for event in batch:
                    cursor = event["sequence"]
                    yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                if store.get(job_id)["status"] in TERMINAL:
                    yield "event: complete\ndata: {}\n\n"
                    return
                yield ": heartbeat\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.get("/api/models")
    def models():
        labels = store.labels()
        return [dict(**row, label=labels.get(row["id"], row["name"])) for row in catalog.models()]

    @app.patch("/api/models/{model_id}/labels")
    def label(model_id: str, body: Label):
        catalog.model(model_id)
        store.label(model_id, body.label)
        return dict(saved=True)

    @app.post("/api/models/{model_id}/transfer")
    def transfer(model_id: str, body: Transfer, idempotency_key: str | None = Header(default=None)):
        if body.model_id != model_id:
            raise ValueError("Model ID mismatch")
        _, record, _ = catalog.model_config(model_id)
        if not record["trained"]:
            raise ValueError("Select a trained model")
        return submit("transfer", body.model_dump(mode="json"), idempotency_key)

    @app.post("/api/freezes")
    def freeze(body: ModelRequest, idempotency_key: str | None = Header(default=None)):
        catalog.model(body.model_id)
        return submit("freeze", body.model_dump(mode="json"), idempotency_key)

    @app.post("/api/evaluations")
    def evaluate(body: Evaluation, idempotency_key: str | None = Header(default=None)):
        record, _ = catalog.model(body.model_id)
        if not record["trained"]:
            raise ValueError("Select a trained model")
        if body.split == "test":
            freeze_job = store.get(body.frozen_job_id)
            if freeze_job["kind"] != "freeze" or freeze_job["status"] != "succeeded" or freeze_job["payload"]["model_id"] != body.model_id:
                raise ValueError("A successful freeze for this model is required")
            store.expose("test_requested", record["dataset"], "configured test start", "configured test end", body.model_id)
        return submit("evaluation", body.model_dump(mode="json"), idempotency_key)

    @app.post("/api/predictions")
    def predict(body: Prediction, idempotency_key: str | None = Header(default=None)):
        record, _ = catalog.model(body.model_id)
        if not record["trained"]:
            raise ValueError("Select a trained or explicitly transferred model")
        if not set(body.display_codes) <= set(record["metadata"]["contract"]["stock_codes"]):
            raise ValueError("Display filter does not change the model pool; transfer explicitly for a new pool")
        return submit("prediction", body.model_dump(mode="json"), idempotency_key)

    @app.get("/api/reports")
    def reports():
        return catalog.reports()

    @app.get("/api/reports/{report_id}")
    def report_detail(report_id: str):
        record, directory = catalog.report(report_id)
        series = {}
        for name in record["metrics"]:
            path = inside(directory / name / "daily.csv", directory)
            if path.exists():
                frame = pd.read_csv(path)
                series[name] = json.loads(frame.to_json(orient="records"))
        return dict(**record, series=series)

    @app.get("/api/reports/{report_id}/orders")
    def orders(report_id: str, strategy: str = "ppo", page: int = 0, date: str | None = None):
        record, directory = catalog.report(report_id)
        if strategy not in record["metrics"] or page < 0:
            raise ValueError("Invalid strategy or page")
        path = inside(directory / strategy / "orders.csv", directory)
        frame = pd.read_csv(path, dtype={"code": str})
        if date:
            frame = frame[frame.date.str[:10] == date]
        return dict(total=len(frame), rows=json.loads(frame.iloc[page * 50:(page + 1) * 50].to_json(orient="records")))

    @app.get("/api/reports/{report_id}/positions")
    def positions(report_id: str, strategy: str = "ppo", page: int = 0, date: str | None = None):
        record, directory = catalog.report(report_id)
        if strategy not in record["metrics"] or page < 0:
            raise ValueError("Invalid strategy or page")
        frame = pd.read_parquet(inside(directory / strategy / "positions.parquet", directory))
        dates = frame.index.strftime("%Y-%m-%d")
        selected = date or dates[-1]
        if selected not in dates:
            raise ValueError("No holdings snapshot on the requested date")
        row = frame.iloc[list(dates).index(selected)]
        held = row[row > 0].sort_values(ascending=False)
        records = [dict(date=selected, code=str(code), shares=float(qty)) for code, qty in held.items()]
        return dict(total=len(records), date=selected, rows=records[page * 50:(page + 1) * 50])

    @app.get("/api/artifacts/{report_id}/{name:path}")
    def artifact(report_id: str, name: str):
        path = catalog.artifact(report_id, name)
        return FileResponse(path, filename=path.name)

    @app.get("/api/predictions/{job_id}/download")
    def prediction_download(job_id: str):
        job = store.get(job_id)
        if job["status"] != "succeeded" or job["kind"] != "prediction":
            raise ValueError("Prediction is not ready")
        path = inside(root / job["result"]["prediction_path"], catalog.runs)
        return FileResponse(path, filename="prediction.json")

    dist = root / "web/dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app
