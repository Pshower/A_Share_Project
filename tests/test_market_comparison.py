"""Offline source defaults and per-model comparison boundaries."""

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from src.data.online import download_history
from src.web.app import create_app
from src.web.schemas import ModelComparison, OnlineHistory
from src.web import market_research
from src.web.catalog import Catalog
from tests.test_bulk_download import Provider


def request(**changes):
    return ModelComparison(**dict(dict(model_ids=["good"], history_id="fixture", close_as_of="2020-01-02",
                                      display_code="000001", account_mode="cash", acknowledge_research=True), **changes))


@pytest.mark.parametrize("change", [dict(model_ids=["a", "a"]), dict(model_ids=list("abcd")),
                                    dict(account_mode="weights"), dict(research_weights={"000001": .1}),
                                    dict(account_mode="weights", research_weights={"000001": 1.1})])
def test_comparison_schema_requires_explicit_consistent_account(change):
    with pytest.raises(ValidationError):
        request(**change)


def test_preflight_is_per_model_and_requires_exact_date(tmp_path, monkeypatch):
    batch = download_history(["000001"], "2020-01-01", "2020-01-03", ["hfq"],
                             allow_network=True, root=tmp_path, provider=Provider())
    assert batch["request"]["provider"] == "sina"
    def requirements(catalog, model_id):
        return dict(model_id=model_id, model_sha256=model_id, source="sina" if model_id == "good" else "eastmoney",
                    trained=True, codes=["000001"], price_basis="hfq_research", history_origin="2020-01-01",
                    train_end="2019-01-01", val_end="2020-12-31")
    monkeypatch.setattr(market_research, "model_requirements", requirements)
    catalog = SimpleNamespace(root=tmp_path)
    result = market_research.comparison_preflight(catalog, request(history_id=batch["id"], model_ids=["good", "bad"]))
    assert result["models"][0]["ready"]
    assert not result["models"][1]["ready"]
    assert result["models"][0]["historical_label"] == "开发验证期回溯"
    absent = market_research.comparison_preflight(catalog, request(history_id=batch["id"], close_as_of="2020-01-01"))
    assert not absent["models"][0]["ready"]
    assert "缺少指定收盘日" in absent["models"][0]["errors"][0]
    outside = market_research.comparison_preflight(catalog, request(history_id=batch["id"], display_code="999999"))
    assert not outside["models"][0]["ready"]


@pytest.mark.local_ipc
def test_default_source_and_legacy_retry_keep_original_provider(tmp_path):
    with TestClient(create_app(tmp_path, start_workers=False)) as client:
        headers = {"X-CSRF-Token": client.get("/api/session").json()["token"]}
        status = client.get("/api/market/status").json()
        assert status["default_history_source"] == "sina" and status["quote_source"] == "eastmoney"
        body = dict(codes=["000001"], start_date="2020-01-01", end_date="2020-01-03", bases=["hfq"], allow_network=True)
        assert OnlineHistory(**body).source == "sina"
        response = client.post("/api/market/downloads", json=body, headers=headers)
        assert response.status_code == 202 and response.json()["payload"]["source"] == "sina"
        for source in ["sina", "eastmoney"]:
            saved = download_history(**dict(codes=body["codes"], start=body["start_date"], end=body["end_date"],
                                           bases=body["bases"], allow_network=True, root=tmp_path, provider=Provider(), source=source))
            retry = client.post("/api/market/downloads", json={**body, "resume_id": saved["id"]}, headers=headers)
            assert retry.status_code == 202 and retry.json()["payload"]["source"] == source
            other = "eastmoney" if source == "sina" else "sina"
            assert client.post("/api/market/downloads", json={**body, "resume_id": saved["id"], "source": other}, headers=headers).status_code == 409


@pytest.mark.local_ipc
def test_comparison_partial_enqueue_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(market_research, "comparison_preflight", lambda *_: dict(history_manifest_sha256="hash", models=[
        dict(model_id="good", model_sha256="sha", ready=True, historical_label="开发验证期回溯", errors=[]),
        dict(model_id="bad", ready=False, errors=["模型来源未认证"])]))
    with TestClient(create_app(tmp_path, start_workers=False)) as client:
        headers = {"X-CSRF-Token": client.get("/api/session").json()["token"], "Idempotency-Key": "compare"}
        body = request(model_ids=["good", "bad"]).model_dump(mode="json")
        assert client.post("/api/market/model-comparison/preflight", json=body, headers=headers).status_code == 200
        assert client.get("/api/jobs").json() == []
        a = client.post("/api/market/model-comparison", json=body, headers=headers)
        b = client.post("/api/market/model-comparison", json=body, headers=headers)
        assert a.status_code == 202 and a.json() == b.json()
        assert a.json()["results"][1]["status"] == "blocked"
        jobs = client.get("/api/jobs").json()
        assert len(jobs) == 1 and jobs[0]["payload"]["strict_source"]


@pytest.mark.local_ipc
def test_saved_performance_excludes_test_and_other_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(Catalog, "model", lambda *_: ({"metadata": {"model_sha256": "exact"}}, None))
    monkeypatch.setattr(Catalog, "reports", lambda *_: [
        dict(id="matching", split="val", experiment={"provenance": {"loaded_model_sha256": "exact"}}),
        dict(id="protected", split="test", relative_path="reports/runs/protected", experiment={"provenance": {"loaded_model_sha256": "exact"}}),
        dict(id="other", split="val", experiment={"provenance": {"loaded_model_sha256": "other"}}),
    ])
    with TestClient(create_app(tmp_path, start_workers=False)) as client:
        assert [r["id"] for r in client.get("/api/market/model-performance/fixture").json()] == ["matching"]
