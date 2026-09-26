import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.online import download_history, history_time_budget
from src.data.snapshots import Snapshots
from src.data.stock_codes import parse_stock_codes
from src.runtime import events
from src.web.app import create_app
from src.web.schemas import OnlineHistory


class Provider:
    def __init__(self, cancel=False):
        self.calls = []
        self.cancel = cancel

    def history(self, code, start, end, basis):
        self.calls.append((code, basis))
        if self.cancel and len(self.calls) == 2:
            raise events.Cancelled("fixture stop")
        return pd.DataFrame(dict(date=["2020-01-02", "2020-01-03"], code=[code, code],
                                 open=[10., 11.], close=[11., 12.], high=[12., 13.], low=[9., 10.],
                                 volume=[100., 200.], amount=[100000., 200000.], turnover=[1., 2.]))


def test_stock_list_parser():
    result = parse_stock_codes('\ufeffcode,name\n1,"name, quoted"\n000001,duplicate\n600030,other', 'csv')
    assert result == dict(codes=["000001", "600030"], duplicates=1, normalized=1)
    assert parse_stock_codes('000001；600030\n000001')["duplicates"] == 1
    for text, kind in [('code,name\nABC,bad', 'csv'), ('a,b\n1,2', 'csv'), ('000001 bad', 'txt'), ('', 'txt')]:
        with pytest.raises(ValueError):
            parse_stock_codes(text, kind)


def test_batch_limit_and_budget():
    codes = [f'{i:06d}' for i in range(1, 501)]
    request = dict(codes=codes, start_date='2020-01-01', end_date='2020-01-03', allow_network=True)
    assert len(OnlineHistory(**request).codes) == 500
    with pytest.raises(ValueError):
        OnlineHistory(**{**request, 'codes': codes + ['600000']})
    assert history_time_budget(codes, ['hfq', 'unadjusted']) == 3600
    assert history_time_budget(codes[:1], ['hfq']) == 600


def test_large_batch_progress_is_complete(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(events, 'emit', lambda kind, **kw: recorded.append(dict(type=kind, **kw)))
    provider = Provider()
    codes = [f'{i:06d}' for i in range(1, 122)]
    batch = download_history(codes, '2020-01-01', '2020-01-03', ['hfq'],
                             allow_network=True, root=tmp_path, provider=provider)
    assert batch['status'] == 'ready' and len(batch['entries']) == len(provider.calls) == 121
    last = [r for r in recorded if r['type'] == 'download_progress'][-1]
    assert last['completed'] == last['saved'] == last['total'] == 121
    assert last['failed'] == 0


def test_cancel_publishes_partial_and_retry_reuses_completed_files(tmp_path):
    codes = ['000001', '000002', '600030']
    with pytest.raises(events.Cancelled, match='snapshot'):
        download_history(codes, '2020-01-01', '2020-01-03', ['hfq'],
                         allow_network=True, root=tmp_path, provider=Provider(cancel=True))
    batch = Snapshots(tmp_path).list('history')[0]
    assert batch['status'] == 'partial' and batch['cancelled']
    assert [e['status'] for e in batch['entries']] == ['ready', 'failed', 'failed']
    provider = Provider()
    retry = download_history(codes, '2020-01-01', '2020-01-03', ['hfq'], allow_network=True,
                             root=tmp_path, provider=provider, resume_id=batch['id'])
    assert retry['status'] == 'ready' and provider.calls == [('000002', 'hfq'), ('600030', 'hfq')]
    assert retry['entries'][0]['received_at'] == batch['entries'][0]['received_at']


@pytest.mark.local_ipc
def test_code_import_api_is_local_and_does_not_enqueue(tmp_path):
    with TestClient(create_app(tmp_path, start_workers=False)) as client:
        token = client.get('/api/session').json()['token']
        body = dict(text='代码,名称\n1,fixture\n600030,fixture', format='csv')
        assert client.post('/api/market/parse-codes', json=body).status_code == 403
        result = client.post('/api/market/parse-codes', json=body, headers={'X-CSRF-Token': token})
        assert result.json()['codes'] == ['000001', '600030']
        assert client.get('/api/jobs').json() == []
