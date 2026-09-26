from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.feature_definitions import validate_definitions, compute_extra_features
from src.data.preprocess import DataPreprocessor, FEATURES, read_config, write_json, file_hash
from src.training.env_factory import ResearchData
from src.agents.ppo_agent import PPOAgent, make_model
from src.web.app import create_app
from src.web.catalog import Catalog
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_web_workbench import web_root
from tests.test_research_pipeline import fixture_config
from tests.test_online_market import history_fixture
from src.inference.live import build_daily_plan

EXTRA = [dict(name='custom_atr', kind='atr_ratio', window=14),
         dict(name='custom_dd', kind='drawdown', window=20)]


def test_definitions_reject_invalid_or_colliding_names():
    for definitions in [[dict(name='return', kind='momentum', window=2)],
                        [dict(name='code', kind='momentum', window=2)],
                        [dict(name='close', kind='momentum', window=2)],
                        [EXTRA[0], EXTRA[0]], [dict(name='bad', kind='eval', window=2)],
                        [dict(name='bad', kind='momentum', window=-1)],
                        [dict(name='bad', kind='momentum', window=True)], {}, EXTRA * 7]:
        with pytest.raises(ValueError):
            validate_definitions(definitions, FEATURES)


def test_extra_formulas_and_prefix_causality():
    index = pd.bdate_range('2020-01-01', periods=8)
    frame = pd.DataFrame(dict(close=[10., 12., 11., 13., 14., 12., 16., 17.],
                              high=[11., 13., 12., 14., 15., 13., 17., 18.],
                              low=[9., 11., 10., 12., 13., 11., 15., 16.],
                              volume=[10., 20., 30., 40., 50., 60., 70., 80.],
                              amount=[100., 200., 300., 400., 500., 600., 700., 800.], valid_price=True), index=index)
    definitions = [dict(name='test_' + kind, kind=kind, window=2) for kind in
                   ['momentum', 'ma_gap', 'log_volatility', 'drawdown', 'atr_ratio', 'relative_volume', 'relative_amount', 'bb_width']]
    result = compute_extra_features(frame, definitions)
    assert result.iloc[2].test_momentum == pytest.approx(0.1)
    assert result.iloc[2].test_drawdown == pytest.approx(11 / 12 - 1)
    assert result.iloc[2].test_atr_ratio == pytest.approx(2.5 / 11)
    assert result.iloc[2].test_relative_volume == 2
    changed = frame.copy()
    changed.loc[index[5]:, ['close', 'high', 'low', 'volume', 'amount']] *= 100
    pd.testing.assert_frame_equal(result.iloc[:5], compute_extra_features(changed, definitions).iloc[:5])
    frame['volume'] = 0
    assert compute_extra_features(frame, definitions).test_relative_volume.isna().all()


def test_feature_version_scaler_and_training_contract(local_data, tmp_path):
    research = deepcopy(local_data.research)
    original_hash = file_hash(local_data.research['dataset']['clean_dir'] + '/panel_data.parquet')
    research['dataset'].update(clean_dir=str(tmp_path / 'custom'), version='custom', extra_features=EXTRA)
    processor = DataPreprocessor(**research['dataset'])
    processor.process_all_stocks_and_save_scaler()
    scaler = read_config(tmp_path / 'custom/scaler_params.json')
    assert scaler['feature_cols'] == FEATURES + ['custom_atr', 'custom_dd']
    assert scaler['extra_features'] == EXTRA
    raw = pd.read_csv(research['dataset']['raw_dir'] + '/000001_data_fixture_hfq.csv', dtype={'code': str})
    transformed = processor.transform_new_data(raw, tmp_path / 'custom/scaler_params.json', code='000001')
    panel = pd.read_parquet(tmp_path / 'custom/panel_data.parquet')
    np.testing.assert_allclose(transformed.custom_atr, panel[('000001', 'custom_atr')], equal_nan=True)
    research_path = tmp_path / 'custom_research.json'
    write_json(research_path, research)
    cfg = deepcopy(local_data.config)
    cfg['research_config'] = str(research_path)
    custom = ResearchData(cfg)
    env = custom.vector_env()
    old_env = local_data.vector_env()
    try:
        assert env.observation_space['stock_features'].shape[-1] == 22
        agent = PPOAgent(make_model(env, cfg), custom.contract)
        observation = {key: value[0] for key, value in env.reset().items()}
        assert np.isfinite(agent.act(observation)).all()
        old = PPOAgent(make_model(old_env, local_data.config), local_data.contract)
        with pytest.raises(ValueError, match='feature_cols'):
            old.transfer_to(env, cfg, custom.contract)
        batch = history_fixture(custom, tmp_path)
        plan = build_daily_plan(agent, custom, batch['id'], model_id='fixture', root=tmp_path)
        assert plan['status'] == 'ready'
    finally:
        env.close()
        old_env.close()
    assert file_hash(local_data.research['dataset']['clean_dir'] + '/panel_data.parquet') == original_hash


def test_custom_scaler_ignores_validation_changes(tmp_path):
    research = fixture_config(tmp_path)
    research['dataset']['extra_features'] = EXTRA
    DataPreprocessor(**research['dataset']).process_all_stocks_and_save_scaler()
    first = read_config(tmp_path / 'clean/scaler_params.json')
    path = tmp_path / 'raw/000001_data_fixture_hfq.csv'
    raw = pd.read_csv(path, dtype={'code': str})
    raw.loc[90:, ['open', 'close', 'high', 'low']] *= 100
    raw.to_csv(path, index=False)
    research['dataset']['clean_dir'] = str(tmp_path / 'changed')
    DataPreprocessor(**research['dataset']).process_all_stocks_and_save_scaler()
    assert first['stats'] == read_config(tmp_path / 'changed/scaler_params.json')['stats']


@pytest.mark.local_ipc
def test_preview_and_schema_guard(web_root):
    dataset = Catalog(web_root).datasets()[0]['id']
    with TestClient(create_app(web_root, start_workers=False)) as client:
        headers = {'X-CSRF-Token': client.get('/api/session').json()['token']}
        response = client.post(f'/api/datasets/{dataset}/features/preview', headers=headers,
                               json=dict(code='000001', extra_features=EXTRA))
        assert response.status_code == 200, response.text
        assert response.json()['scaled'] is False and response.json()['stats'][0]['valid'] > 0
        assert client.get('/api/jobs').json() == []
        bad = client.post(f'/api/datasets/{dataset}/features/preview', headers=headers,
                          json=dict(code='000001', extra_features=[dict(name='return', kind='momentum', window=5)]))
        assert bad.status_code == 409
