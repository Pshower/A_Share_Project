from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.coverage import training_coverage
from src.training.env_factory import ResearchData
from src.web.app import create_app
from src.web.catalog import Catalog
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_web_workbench import web_root

pytestmark = pytest.mark.usefixtures('no_parameter_updates')


def test_common_dates_use_all_stocks_lookback_and_no_gap_joining():
    dates = pd.bdate_range('2020-01-01', periods=10)
    features = pd.DataFrame(1., index=dates, columns=pd.MultiIndex.from_product([['000001', '000002'], ['x']]))
    prices = pd.DataFrame(10., index=dates, columns=pd.MultiIndex.from_product([['000001', '000002'], ['open', 'close', 'volume']]))
    features.loc[dates[:3], ('000002', 'x')] = np.nan
    one = training_coverage(features, prices, ['000001'], 2, str(dates[-1].date()))
    both = training_coverage(features, prices, ['000001', '000002'], 2, str(dates[-1].date()))
    assert one['train_start'] == str(dates[1].date())
    assert both['train_start'] == str(dates[4].date())
    features.loc[dates[6], ('000002', 'x')] = np.nan
    gap = training_coverage(features, prices, ['000001', '000002'], 2, str(dates[-1].date()))
    assert gap['train_start'] == str(dates[8].date()) and gap['train_rows'] == 2
    short = training_coverage(features, prices, ['000001', '000002'], 3, str(dates[-1].date()))
    assert short['train_rows'] == 1 and not short['eligible']
    prices.loc[dates[-1], ('000002', 'close')] = np.nan
    assert not training_coverage(features, prices, ['000001', '000002'], 2, str(dates[-1].date()))['eligible']


def test_actual_training_provider_uses_same_intersection(local_data):
    cfg = deepcopy(local_data.config)
    cfg['training_date_policy'] = 'intersection'
    data = ResearchData(cfg)
    market = data.market('train')
    assert str(market.dates[0].date()) == data.training_coverage['train_start']
    assert str(market.dates[-1].date()) == data.training_coverage['train_end']
    assert data.contract['training_date_policy'] == 'intersection'
    assert 'training_date_policy' not in local_data.contract
    assert data.market('val').dates[0] == local_data.market('val').dates[0]


@pytest.mark.local_ipc
def test_coverage_and_original_bars_are_local_and_checked(web_root):
    catalog = Catalog(web_root)
    dataset = catalog.datasets()[0]['id']
    with TestClient(create_app(web_root, start_workers=False)) as client:
        headers = {'X-CSRF-Token': client.get('/api/session').json()['token']}
        url = f'/api/datasets/{dataset}/coverage'
        response = client.post(url, json={'codes': ['000001', '600000'], 'lookback': 1}, headers=headers)
        assert response.status_code == 200
        result = response.json()
        assert result['eligible'] and result['train_start'] <= result['train_end']
        assert len(result['stocks']) == 2 and result['history_matches'] == []
        assert client.post(url, json={'codes': ['999999']}, headers=headers).status_code == 409
        bars = client.get(f'/api/market/datasets/{dataset}/bars?code=000001')
        assert bars.status_code == 200 and bars.json()['basis'] == 'hfq'
        assert bars.json()['rows'] and bars.json()['volume_unit'] == 'shares'
        assert client.get('/api/jobs').json() == []
        source = web_root / 'raw/000001_data_fixture_hfq.csv'
        source.write_bytes(source.read_bytes() + b'\n')
        assert client.get(f'/api/market/datasets/{dataset}/bars?code=000001').status_code == 409
