"""Offline API, accounting and task state tests; browser execution is separate."""

from copy import deepcopy
from pathlib import Path
import shutil

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from src.agents.ppo_agent import PPOAgent, make_model
from src.data.preprocess import read_config, write_json, ROOT
from src.inference.predict import predict_action
from src.runtime import events
from src.web.app import create_app
from src.web.catalog import Catalog, inside, identifier
from src.web.jobs import JobManager
from src.web.store import Store
from tests.test_ppo_framework import local_data, no_parameter_updates

pytestmark = pytest.mark.usefixtures("no_parameter_updates")


@pytest.fixture
def web_root(local_data, tmp_path):
    root = tmp_path
    destination = root / "data/clean/fixture"
    shutil.copytree(local_data.research["dataset"]["clean_dir"], destination)
    research = deepcopy(local_data.research)
    research["dataset"]["clean_dir"] = str(destination)
    write_json(destination / "research_config.json", research)
    (root / "configs").mkdir()
    write_json(root / "configs/research.json", research)
    write_json(root / "configs/ppo.json", local_data.config)
    write_json(root / "configs/ppo_lightweight.json", dict(stock_codes=local_data.codes))
    agent = PPOAgent(make_model(local_data.vector_env(), local_data.config), local_data.contract)
    agent.model.num_timesteps = 8
    agent.save(root / "reports/runs/fixture/best_8", local_data.config)
    agent.model.get_env().close()
    return root


@pytest.mark.local_ipc
def test_api_security_idempotency_and_readonly_catalog(web_root):
    app = create_app(web_root, start_workers=False)
    with TestClient(app) as client:
        token = client.get("/api/session").json()["token"]
        datasets = client.get("/api/datasets").json()
        models = client.get("/api/models").json()
        assert len(datasets) == 1 and models[0]["trained"]
        assert client.get("/api/system").json()["heavy_slots"] == 1
        assert client.get("/api/defaults").status_code == 200
        data = dict(dataset_id=datasets[0]["id"], codes=["000001"], total_timesteps=16,
                    n_steps=8, batch_size=4)
        assert client.post("/api/training/jobs", json=data).status_code == 403
        assert client.get("/api/models", headers={"Origin": "https://outside.invalid"}).status_code == 403
        headers = {"X-CSRF-Token": token, "Idempotency-Key": "same-request"}
        a = client.post("/api/training/jobs", json=data, headers=headers)
        b = client.post("/api/training/jobs", json=data, headers=headers)
        assert a.status_code == 202 and a.json()["id"] == b.json()["id"]
        assert len(client.get("/api/jobs").json()) == 1
        c = client.post("/api/training/jobs", json={**data, "total_timesteps": 32}, headers=headers)
        assert c.status_code == 409
        result = client.post(f"/api/jobs/{a.json()['id']}/stop", json={}, headers={"X-CSRF-Token": token})
        assert result.json()["status"] == "cancelled"
        assert client.post("/api/training/jobs", json={**data, "batch_size": 3}, headers={"X-CSRF-Token": token}).status_code == 422
        assert client.post("/api/evaluations", json=dict(model_id=models[0]["id"], split="test"), headers={"X-CSRF-Token": token}).status_code == 422
        assert client.post("/api/predictions", json=dict(model_id=models[0]["id"], as_of="2020-01-01", display_codes=["999999"]), headers={"X-CSRF-Token": token}).status_code == 409
        assert client.patch(f"/api/models/{models[0]['id']}/labels", json=dict(label="research checkpoint"), headers={"X-CSRF-Token": token}).status_code == 200
        assert client.get("/api/models").json()[0]["label"] == "research checkpoint"


def test_real_path_containment_rejects_parent_and_absolute_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        inside(tmp_path / "../outside", tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        inside(Path(tmp_path.anchor), tmp_path)


def test_causal_prediction_last_date_and_display_filter(local_data):
    data = local_data
    env = data.vector_env()
    agent = PPOAgent(make_model(env, data.config), data.contract)
    date = data.features.index[100]
    account = dict(cash=90000, positions=[dict(code=data.codes[0], shares=100, available_shares=0)],
                   receivables=[dict(date="2030-01-01", amount=100)])
    before = predict_action(agent, data, date, account, [data.codes[0]])
    original = predict_action(agent, data, date, account, data.codes)
    assert before["rows"] == original["rows"]
    data.features.loc[data.features.index > date] = 99999
    data.prices.loc[data.prices.index > date] *= 10
    after = predict_action(agent, data, date, account, [data.codes[0]])
    assert before == after
    last = predict_action(agent, data, data.features.index[-1], None, data.codes)
    assert last["as_of"] == str(data.features.index[-1].date())
    assert last["pool_size"] == 2
    assert before["rows"][0]["available_shares"] == 0
    assert before["equity"] > 90000
    with pytest.raises(ValueError, match="account position"):
        predict_action(agent, data, date, dict(cash=10, positions=[dict(code="999999", shares=1)]))
    with pytest.raises(ValueError, match="account position"):
        predict_action(agent, data, date, dict(cash=10, positions=[dict(code=data.codes[0], shares=1, available_shares=2)]))
    with pytest.raises(ValueError, match="trading date"):
        predict_action(agent, data, "2030-01-01")
    env.close()


def test_ledger_and_stale_job_recovery(tmp_path):
    store = Store(tmp_path / "state")
    job = store.enqueue("prediction", dict(model_id="fixture"), "key")
    manager = JobManager(store, tmp_path)
    store.update(job["id"], "running")
    with store.connect() as conn:
        conn.execute("UPDATE jobs SET updated=? WHERE id=?", ("2000-01-01T00:00:00+00:00", job["id"]))
    assert not manager.reconcile(store.get(job["id"]))
    assert store.get(job["id"])["status"] == "interrupted"
    store.expose("test_observation", "fixture", "2020-01-01", "2020-01-01", "source")
    store.expose("test_observation", "fixture", "2020-01-01", "2020-01-01", "source")
    assert len(Store(tmp_path / "state").exposures()) == 1


def test_events_stop_without_final_success(tmp_path):
    events.configure(tmp_path)
    try:
        events.emit("phase", phase="training")
        (tmp_path / "stop").touch()
        with pytest.raises(events.Cancelled):
            events.check_cancel()
        assert '"sequence": 1' in (tmp_path / "events.jsonl").read_text()
    finally:
        events.configure(None)


def test_recent_worker_is_not_duplicated_during_import_startup(tmp_path):
    store = Store(tmp_path / "state")
    job = store.enqueue("check", {})
    store.update(job["id"], "running")
    manager = JobManager(store, tmp_path)
    assert manager.reconcile(store.get(job["id"]))
    assert store.get(job["id"])["status"] == "running"


def test_multiple_managers_share_one_atomic_heavy_slot(tmp_path):
    first, second = Store(tmp_path / "state"), Store(tmp_path / "state")
    a, b = first.enqueue("check", {}), first.enqueue("check", {})
    assert first.claim_next()["id"] == a["id"]
    assert second.claim_next() is None
    first.update(a["id"], "succeeded")
    assert second.claim_next()["id"] == b["id"]
    assert first.claim_next() is None


@pytest.mark.local_ipc
def test_holdings_endpoint_and_external_network_still_blocked(web_root):
    import socket
    directory = web_root / "reports/runs/comparison"
    (directory / "ppo").mkdir(parents=True)
    write_json(directory / "metrics.json", {"ppo": {"cumulative_return": 0}})
    pd.DataFrame({"000001": [0, 100]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"])).to_parquet(directory / "ppo/positions.parquet")
    app = create_app(web_root, start_workers=False)
    with TestClient(app) as client:
        report_id = client.get("/api/reports").json()[0]["id"]
        last = client.get(f"/api/reports/{report_id}/positions").json()
        assert last["date"] == "2024-01-02" and last["rows"][0]["shares"] == 100
        assert client.get(f"/api/reports/{report_id}/positions?date=2024-01-01").json()["rows"] == []
        assert client.get(f"/api/reports/{report_id}/positions?date=2024-01-03").status_code == 409
        with socket.socket() as sock, pytest.raises(AssertionError, match="forbidden"):
            sock.connect(("198.51.100.1", 80))
