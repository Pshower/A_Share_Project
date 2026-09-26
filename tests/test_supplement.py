from types import SimpleNamespace
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.supplement import capture_supplement, validate_request, validate_table, api_arguments, TYPES, SupplementProvider
from src.data.snapshots import Snapshots
from src.runtime import events
from src.web.app import create_app
from src.web.store import lane_for, Store


def request(kind='profile', codes=None):
    return dict(data_type=kind, codes=codes or ['000001'], market='sz', start_date='2020-01-01', end_date='2020-01-07', period='1')


class Provider:
    def __init__(self, fail=None):
        self.fail, self.calls = fail, []

    def fetch(self, kind, code, settings):
        self.calls.append(code)
        if code == self.fail:
            raise ValueError('fixture unavailable')
        if kind == 'profile':
            return pd.DataFrame({'item': ['股票代码', '股票简称'], 'value': [code, '=FORMULA_FIXTURE']})
        if kind == 'fund_flow':
            return pd.DataFrame({'日期': ['2020-01-02'], '主力净流入-净额': [123.0]})
        if kind == 'minute':
            return pd.DataFrame({'时间': ['2020-01-02 09:31:00'], '开盘': [0.0], '收盘': [10.0], '成交量': [100.0]})
        if kind == 'index_daily':
            return pd.DataFrame({'日期': ['2020-01-02'], '开盘': [100.0], '收盘': [101.0], '最高': [102.0], '最低': [99.0]})
        return pd.DataFrame({'公告日期': ['2020-01-02'], '派息': [2.0]})


def test_explicit_consent_and_parameter_whitelist(tmp_path):
    provider = Provider()
    with pytest.raises(ValueError, match='consent'):
        capture_supplement(request(), provider=provider, root=tmp_path)
    assert provider.calls == [] and lane_for('online_supplement') == 'history'
    assert api_arguments('minute', '000001', request('minute'))['adjust'] == ''
    assert api_arguments('fund_flow', '000001', request('fund_flow'))['market'] == 'sz'
    for bad in [{**request(), 'data_type': 'arbitrary'}, {**request(), 'codes': ['../secret']},
                {**request('minute'), 'end_date': '2020-03-01'}, {**request(), 'codes': [f'{i:06d}' for i in range(21)]}]:
        with pytest.raises(ValueError):
            validate_request(bad)


@pytest.mark.parametrize('kind', list(TYPES))
def test_capture_each_type_keeps_units_and_is_not_model_ready(tmp_path, kind):
    result = capture_supplement(request(kind), allow_network=True, provider=Provider(), root=tmp_path)
    assert result['status'] == 'ready' and not result['model_ready'] and not result['history_complete']
    record, path = Snapshots(tmp_path).get('supplements', result['id'])
    table = json.loads((path / '000001.json').read_text(encoding='utf-8'))
    if kind == 'profile':
        assert table['data'][1][1] == '=FORMULA_FIXTURE'
        assert "'=FORMULA_FIXTURE" in (path / '000001.csv').read_text(encoding='utf-8-sig')
    if kind == 'minute':
        assert table['data'][0][1] == 0 and table['data'][0][-1] == 100
    assert record['entries'][0]['received_at']


def test_partial_retry_reuses_files_and_cancel_publishes(tmp_path):
    settings = request(codes=['000001', '600030'])
    first = capture_supplement(settings, allow_network=True, provider=Provider(fail='600030'), root=tmp_path)
    assert first['status'] == 'partial'
    provider = Provider()
    second = capture_supplement(settings, allow_network=True, provider=provider, root=tmp_path, resume_id=first['id'])
    assert provider.calls == ['600030'] and second['entries'][0]['received_at'] == first['entries'][0]['received_at']
    with pytest.raises(ValueError, match='original'):
        capture_supplement(request('minute'), allow_network=True, provider=provider, root=tmp_path, resume_id=first['id'])
    class Stopped(Provider):
        def fetch(self, *args):
            raise events.Cancelled('fixture')
    with pytest.raises(events.Cancelled):
        capture_supplement(settings, allow_network=True, provider=Stopped(), root=tmp_path)
    assert Snapshots(tmp_path).list('supplements')[0]['cancelled']


def test_invalid_identity_and_duplicate_timestamps():
    with pytest.raises(ValueError, match='identity'):
        validate_table(Provider().fetch('profile', '600030', request()), 'profile', '000001', request())
    frame = Provider().fetch('minute', '000001', request())
    with pytest.raises(ValueError, match='duplicate'):
        validate_table(pd.concat([frame, frame]), 'minute', '000001', request('minute'))


def test_supplement_uses_history_slot_not_training_slot(tmp_path):
    store = Store(tmp_path / 'state')
    job = store.enqueue('online_supplement', request())
    assert store.claim_next('research') is None
    assert store.claim_next('history')['id'] == job['id']
    store.enqueue('online_history', {})
    assert store.claim_next('history') is None


def test_sdk_timeout_stops_owned_process(tmp_path, monkeypatch):
    class Process:
        alive, terminated, closed = True, False, False
        def start(self): pass
        def is_alive(self): return self.alive
        def join(self, timeout=None): pass
        def terminate(self): self.alive, self.terminated = False, True
        def close(self): self.closed = True
    process = Process()
    monkeypatch.setattr('src.data.supplement.mp.get_context', lambda _: SimpleNamespace(Process=lambda **kw: process))
    clock = iter([0, 100])
    monkeypatch.setattr('src.data.supplement.time', SimpleNamespace(monotonic=lambda: next(clock)))
    provider = SupplementProvider(tmp_path)
    monkeypatch.setattr(provider.pacer, 'pace', lambda: None)
    with pytest.raises(TimeoutError):
        provider.fetch('profile', '000001', request())
    assert process.terminated and process.closed


@pytest.mark.local_ipc
def test_readonly_preview_export_and_consent_api(tmp_path):
    batch = capture_supplement(request(), allow_network=True, provider=Provider(), root=tmp_path)
    with TestClient(create_app(tmp_path, start_workers=False)) as client:
        headers = {'X-CSRF-Token': client.get('/api/session').json()['token']}
        assert len(client.get('/api/market/supplement-types').json()) == 5
        assert client.post('/api/market/supplements', json=request(), headers=headers).status_code == 422
        url = f"/api/market/supplements/{batch['id']}"
        assert client.get(url + '/rows?code=000001').json()['total'] == 2
        assert client.get(url + '/files/000001/csv').status_code == 200
        assert client.get(url + '/rows?code=600000').status_code == 409
        assert client.get('/api/jobs').json() == []
