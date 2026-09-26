"""Offline network simulation and strict live-research boundaries."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.agents.ppo_agent import PPOAgent, make_model
from src.data.online import download_history, capture_quotes, normalize_quote, normalize_history
from src.data.snapshots import Snapshots
from src.inference.live import build_daily_plan, judge_quotes
from src.runtime import events
from src.web.app import create_app
from src.web.store import Store
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_web_workbench import web_root

pytestmark = pytest.mark.usefixtures("no_parameter_updates")


class FakeProvider:
    def __init__(self, data=None, fail=False, timestamp=None):
        self.data, self.fail = data, fail
        self.calls = []
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def history(self, code, start, end, basis):
        self.calls.append((code, basis))
        if self.fail:
            raise ConnectionError("fixture unavailable")
        raw = pd.read_csv(self.data.research["dataset"]["raw_dir"] + f"/{code}_data_fixture_hfq.csv", dtype={"code": str})
        raw["volume"] /= 100
        if basis == "unadjusted":
            raw[["open", "close", "high", "low"]] /= 2
        return raw

    def quote(self, code):
        self.calls.append(code)
        if self.fail:
            raise ConnectionError("fixture unavailable")
        return dict(f57=code, f58="fixture", f43=10.0, f60=9.8, f51=11., f52=9., f86=int(self.timestamp.timestamp()))


def history_fixture(data, root):
    request = data.research["dataset"]
    return download_history(data.codes, request["start_date"], request["end_date"], ["hfq", "unadjusted"],
                            allow_network=True, root=root, provider=FakeProvider(data))


def test_explicit_network_consent_and_separate_immutable_bases(local_data, tmp_path):
    provider = FakeProvider(local_data)
    with pytest.raises(ValueError, match="consent"):
        download_history(local_data.codes, "2020-01-01", "2020-12-31", ["hfq"], root=tmp_path, provider=provider)
    with pytest.raises(ValueError, match="consent"):
        capture_quotes(local_data.codes, root=tmp_path, provider=provider)
    assert not provider.calls
    batch = history_fixture(local_data, tmp_path)
    saved, path = Snapshots(tmp_path).get("history", batch["id"])
    assert saved["status"] == "ready"
    high = pd.read_parquet(path / "hfq/000001.parquet")
    raw = pd.read_parquet(path / "unadjusted/000001.parquet")
    np.testing.assert_allclose(high.close, raw.close * 2)
    assert high.volume.iloc[0] == 100000
    second = history_fixture(local_data, tmp_path)
    assert second["id"] != batch["id"]
    assert Snapshots(tmp_path).get("history", batch["id"])[0] == saved


def test_failure_preserves_good_capture_and_explicit_retry(local_data, tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.online.time.sleep", lambda _: None)
    provider = FakeProvider(local_data)
    original = provider.history
    provider.history = lambda code, start, end, basis: original(code, start, end, basis) if basis == "hfq" else (_ for _ in ()).throw(ConnectionError("unavailable"))
    dates = local_data.research["dataset"]
    batch = download_history(["000001"], dates["start_date"], dates["end_date"], ["hfq", "unadjusted"], allow_network=True, root=tmp_path, provider=provider)
    assert batch["status"] == "partial"
    retry_provider = FakeProvider(local_data)
    retried = download_history(["000001"], dates["start_date"], dates["end_date"], ["hfq", "unadjusted"], allow_network=True, root=tmp_path, provider=retry_provider, resume_id=batch["id"])
    assert retried["status"] == "ready" and retry_provider.calls == [("000001", "unadjusted")]
    assert retried["entries"][0]["reused_from"] == batch["id"]
    assert Snapshots(tmp_path).get("history", batch["id"])[0]["status"] == "partial"


def test_bad_schema_duplicate_codes_and_unknown_quote_times(local_data):
    raw = FakeProvider(local_data).history("000001", "", "", "hfq")
    with pytest.raises(ValueError, match="identity"):
        normalize_history(pd.concat([raw, raw.iloc[:1]]), "000001", "2020-01-01", "2020-12-31")
    raw.loc[0, "low"] = 9999
    with pytest.raises(ValueError, match="OHLC"):
        normalize_history(raw, "000001", "2020-01-01", "2020-12-31")
    now = datetime.now(timezone.utc)
    quote = normalize_quote(dict(f57="000001", f43=10, f86=0), "000001", now.isoformat())
    assert quote["provider_timestamp"] is None
    quote = normalize_quote(dict(f57="000001", f43=10, f86=(now + timedelta(days=1)).timestamp()), "000001", now.isoformat())
    assert quote["provider_timestamp"] is None


def test_stale_missing_and_fresh_quotes_never_become_orders(tmp_path):
    now = datetime.now(timezone.utc)
    snapshot = capture_quotes(["000001"], allow_network=True, root=tmp_path, provider=FakeProvider(timestamp=now))
    result = judge_quotes(snapshot, now=now + timedelta(seconds=1))
    assert result["rows"][0]["status"] == "quotes_only" and not result["actionable"]
    old = judge_quotes(snapshot, now=now + timedelta(minutes=5))
    assert old["rows"][0]["status"] == "waiting"
    assert "stale_quote" in old["rows"][0]["reasons"]
    partial = deepcopy(snapshot)
    partial["rows"] = []
    assert judge_quotes(partial)["rows"][0]["status"] == "waiting"


def test_daily_binding_reuses_scaler_and_keeps_original_contract(local_data, tmp_path):
    data = local_data
    history = history_fixture(data, tmp_path)
    agent = PPOAgent(make_model(data.vector_env(), data.config), data.contract)
    before = deepcopy(agent.contract)
    first = build_daily_plan(agent, data, history["id"], model_id="fixture", research_weights={"000001": 0.8}, root=tmp_path)
    second = build_daily_plan(agent, data, history["id"], model_id="fixture", research_weights={"000001": 0.8}, root=tmp_path)
    assert agent.contract == before
    assert first["rows"] == second["rows"]
    assert not first["real_account_compatible"] and first["compatibility_status"] == "research_only"
    assert first["scaler_sha256"] == data.contract["artifacts"]["scaler_params.json"]
    assert first["rows"][0]["research_intent"] == "sell"
    assert all(r["max_abs_scaled_difference"] < 1e-6 for r in first["revision"].values())
    agent.model.get_env().close()


def test_revision_requires_explicit_research_acceptance(local_data, tmp_path):
    data = local_data
    provider = FakeProvider(data)
    original = provider.history
    def revised(*args):
        frame = original(*args)
        frame.loc[70, ["open", "close", "high", "low"]] *= 1.1
        return frame
    provider.history = revised
    config = data.research["dataset"]
    h = download_history(data.codes, config["start_date"], config["end_date"], ["hfq"], allow_network=True, root=tmp_path, provider=provider)
    agent = PPOAgent(make_model(data.vector_env(), data.config), data.contract)
    with pytest.raises(ValueError, match="revisions"):
        build_daily_plan(agent, data, h["id"], model_id="fixture", root=tmp_path)
    plan = build_daily_plan(agent, data, h["id"], model_id="fixture", root=tmp_path, accept_revisions=True)
    assert plan["revisions_accepted"] and not plan["real_account_compatible"]
    agent.model.get_env().close()


def test_feature_anchor_precedes_study_start_when_raw_history_is_longer(local_data, tmp_path):
    from src.data.preprocess import DataPreprocessor, write_json
    from src.training.env_factory import ResearchData
    from src.inference.live import feature_history_origin
    research = deepcopy(local_data.research)
    research["dataset"].update(start_date="2020-02-01", clean_dir=str(tmp_path / "later_study"))
    DataPreprocessor(**research["dataset"]).process_all_stocks_and_save_scaler()
    config = deepcopy(local_data.config)
    config["research_config"] = str(tmp_path / "later_research.json")
    write_json(config["research_config"], research)
    data = ResearchData(config)
    origin, _ = feature_history_origin(research)
    assert origin == pd.Timestamp("2020-01-01")
    h = download_history(data.codes, "2020-01-01", research["dataset"]["end_date"], ["hfq"],
                         allow_network=True, root=tmp_path, provider=FakeProvider(data))
    agent = PPOAgent(make_model(data.vector_env(), config), data.contract)
    plan = build_daily_plan(agent, data, h["id"], model_id="fixture", root=tmp_path)
    assert plan["history_origin"] == "2020-01-01"
    assert not plan["revisions_accepted"]
    agent.model.get_env().close()


def test_capture_day_bar_does_not_become_final_just_by_waiting(local_data, tmp_path):
    h = history_fixture(local_data, tmp_path)
    metadata, path = Snapshots(tmp_path).get("history", h["id"])
    # Simulate a capture during the fixture's last day; advancing the wall clock cannot finalize that bar.
    final = local_data.features.index[-1]
    for entry in metadata["entries"]:
        if entry["basis"] == "hfq":
            entry["received_at"] = final.tz_localize("Asia/Shanghai").isoformat()
    from src.data.preprocess import write_json
    write_json(path / "manifest.json", metadata)
    agent = PPOAgent(make_model(local_data.vector_env(), local_data.config), local_data.contract)
    plan = build_daily_plan(agent, local_data, h["id"], model_id="fixture", root=tmp_path)
    assert pd.Timestamp(plan["feature_as_of"]) < final
    agent.model.get_env().close()


def test_revision_acknowledgement_cannot_bypass_missing_history_anchor(local_data, tmp_path):
    provider = FakeProvider(local_data)
    original = provider.history
    provider.history = lambda *args: original(*args).iloc[20:].copy()
    config = local_data.research["dataset"]
    batch = download_history(local_data.codes, config["start_date"], config["end_date"], ["hfq"],
                             allow_network=True, root=tmp_path, provider=provider)
    agent = PPOAgent(make_model(local_data.vector_env(), local_data.config), local_data.contract)
    with pytest.raises(ValueError, match="truncated"):
        build_daily_plan(agent, local_data, batch["id"], model_id="fixture", root=tmp_path, accept_revisions=True)
    agent.model.get_env().close()


def test_quote_permission_is_checked_between_network_calls(tmp_path):
    calls = []
    def stopped():
        calls.append(1)
        raise events.Cancelled("expired")
    provider = FakeProvider()
    with pytest.raises(events.Cancelled):
        capture_quotes(["000001"], allow_network=True, root=tmp_path, provider=provider, permission_check=stopped)
    assert not provider.calls and calls


def test_lane_isolation_monitor_lease_and_no_duplicate_poll(tmp_path, monkeypatch):
    store = Store(tmp_path / "state")
    for kind in ["training", "online_history", "online_quotes"]:
        store.enqueue(kind, {})
    assert store.claim_next("research")["kind"] == "training"
    assert store.claim_next("history")["kind"] == "online_history"
    assert store.claim_next("quotes")["kind"] == "online_quotes"
    assert store.claim_next("quotes") is None
    payload = dict(codes=["000001"], allow_network=True, plan_id=None, interval_seconds=60, duration_seconds=300)
    monitor = store.create_monitor(payload)
    store.tick_monitors()
    job_id = store.monitor(monitor["id"])["last_job"]
    store.tick_monitors()
    assert store.monitor(monitor["id"])["last_job"] == job_id
    with store.connect() as conn:
        conn.execute("UPDATE monitors SET lease=0 WHERE id=?", (monitor["id"],))
    store.tick_monitors()
    assert store.monitor(monitor["id"])["status"] == "expired"
    assert store.get(job_id)["status"] == "cancelled"
    with pytest.raises(ValueError, match="expired"):
        store.renew_monitor(monitor["id"])


@pytest.mark.local_ipc
def test_online_api_requires_explicit_consent_and_never_fetches_on_get(web_root):
    app = create_app(web_root, start_workers=False)
    with TestClient(app) as client:
        token = client.get("/api/session").json()["token"]
        headers = {"X-CSRF-Token": token}
        assert client.get("/api/market/status").json()["supports_orders"] is False
        assert client.get("/api/market/snapshots/history").json() == []
        assert client.get("/api/jobs").json() == []
        body = dict(codes=["000001"], start_date="2020-01-01", end_date="2020-02-01", bases=["hfq"])
        assert client.post("/api/market/downloads", json=body, headers=headers).status_code == 422
        assert client.post("/api/market/downloads", json={**body, "allow_network": True}, headers=headers).status_code == 202
        assert client.post("/api/market/quotes", json=dict(codes=["000001"], allow_network=False), headers=headers).status_code == 422
        mon = client.post("/api/market/monitors", json=dict(codes=["000001"], allow_network=True), headers=headers)
        assert mon.status_code == 200
        assert client.post(f"/api/market/monitors/{mon.json()['id']}/stop", json={}, headers=headers).json()["status"] == "stopped"
