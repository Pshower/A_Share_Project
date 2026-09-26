import pandas as pd
import pytest

from src.data.online import download_history, normalize_history
from src.data.sina import Sina
from src.data.source_bundle import inspect_bundle
from src.data.snapshots import Snapshots
from src.data.preprocess import write_json
from tests.test_bulk_download import Provider


def test_sina_units_are_canonical_once(tmp_path, monkeypatch):
    import akshare
    raw = pd.DataFrame(dict(date=["2020-01-02"], open=[10.], close=[11.], high=[12.], low=[9.],
                            volume=[123400.], amount=[1234500.], turnover=[0.012]))
    calls = []
    def fake(**kwargs):
        calls.append(kwargs)
        return raw
    monkeypatch.setattr(akshare, "stock_zh_a_daily", fake)
    provider = Sina(tmp_path)
    parsed = normalize_history(provider.history("000001", "2020-01-01", "2020-01-03", "hfq"),
                               "000001", "2020-01-01", "2020-01-03")
    assert parsed.volume.iloc[0] == 123400
    assert parsed.turnover.iloc[0] == pytest.approx(1.2)
    assert calls[0]["symbol"] == "sz000001" and calls[0]["adjust"] == "hfq"
    assert raw.turnover.iloc[0] == 0.012


def test_source_declared_and_resume_cannot_switch_provider(tmp_path):
    original = download_history(["000001"], "2020-01-01", "2020-01-03", ["hfq"],
                                allow_network=True, root=tmp_path, provider=Provider(), source="sina")
    assert original["request"]["provider"] == "sina"
    with pytest.raises(ValueError, match="original request"):
        download_history(["000001"], "2020-01-01", "2020-01-03", ["hfq"],
                         allow_network=True, root=tmp_path, provider=Provider(), source="eastmoney", resume_id=original["id"])
    second = download_history(["000001"], "2020-01-01", "2020-01-03", ["hfq"],
                              allow_network=True, root=tmp_path, provider=Provider())
    with pytest.raises(ValueError, match="mix"):
        inspect_bundle([original["id"], second["id"]], ["000001"], "2020-01-01", "2020-01-03", tmp_path)
