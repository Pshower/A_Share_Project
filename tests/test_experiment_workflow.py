from copy import deepcopy
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.web.schemas import Training, ExperimentRecipe, ExperimentApproval
from src.web.store import Store
from src.web.catalog import Catalog
from src.web.app import create_app
from src.web.experiments import preflight, stage_jobs
from src.data.source_bundle import compose_bundle, inspect_bundle
from src.data.online import download_history
from src.data.snapshots import Snapshots
from src.agents.ppo_agent import make_model
from src.training.evaluate import selection_assessment, selection_score
from src.backtest.metrics import performance
from tests.test_ppo_framework import local_data, no_parameter_updates
from tests.test_web_workbench import web_root
from tests.test_online_market import FakeProvider

pytestmark = pytest.mark.usefixtures('no_parameter_updates')


def recipe(root):
    dataset = Catalog(root).datasets()[0]['id']
    return ExperimentRecipe(name='fixture', training=Training(dataset_id=dataset, codes=['000001'],
                            total_timesteps=16, n_steps=8, batch_size=4, device='cpu'), seeds=[42, 123]).model_dump(mode='json')


def test_recipe_stages_require_approval_and_current_checks(web_root):
    store = Store(web_root / '.workbench')
    original = recipe(web_root)
    exp = store.create_experiment(original)
    assert preflight(store, exp['id'], web_root)['can_check']
    with pytest.raises(ValueError, match='approval'):
        stage_jobs(store, exp['id'], 'train', ExperimentApproval(), web_root)
    with pytest.raises(ValueError, match='fresh'):
        stage_jobs(store, exp['id'], 'train', ExperimentApproval(allow_training=True), web_root)
    checks = stage_jobs(store, exp['id'], 'check', ExperimentApproval(), web_root)
    assert len(checks) == 2 and all(j['kind'] == 'check' for j in checks)
    assert [j['id'] for j in checks] == [j['id'] for j in stage_jobs(store, exp['id'], 'check', ExperimentApproval(), web_root)]
    for job in checks:
        store.update(job['id'], 'succeeded', {})
    trains = stage_jobs(store, exp['id'], 'train', ExperimentApproval(allow_training=True), web_root)
    assert len(trains) == 2 and all(j['kind'] == 'training' for j in trains)
    assert store.experiment(exp['id'])['recipe'] == original
    copy = store.create_experiment({**original, 'name': 'revision', 'parent_id': exp['id']})
    assert copy['id'] != exp['id']


def test_partial_download_is_retryable_with_explicit_new_action(tmp_path):
    store = Store(tmp_path)
    exp = store.create_experiment({'name': 'fixture'})
    first = store.enqueue_experiment(exp['id'], 'download', 'online_history', {}, 'same')
    store.update(first['id'], 'succeeded', {'capture_status': 'partial'})
    second = store.enqueue_experiment(exp['id'], 'download', 'online_history', {}, 'same')
    assert first['id'] != second['id']


def test_source_bundle_missing_conflicts_and_provenance(local_data, tmp_path):
    settings = local_data.research['dataset']
    start, end = settings['start_date'], settings['end_date']
    a = download_history(['000001'], start, end, ['hfq'], allow_network=True, provider=FakeProvider(local_data), root=tmp_path)
    b = download_history(['600000'], start, end, ['hfq'], allow_network=True, provider=FakeProvider(local_data), root=tmp_path)
    assert inspect_bundle([a['id']], ['000001', '600000'], start, end, tmp_path)['missing'] == ['600000']
    joined = compose_bundle([a['id'], b['id']], ['000001', '600000'], start, end, tmp_path)
    assert joined['status'] == 'ready' and joined['entries'][0]['received_at'] == a['entries'][0]['received_at']
    provider = FakeProvider(local_data)
    original = provider.history
    def revised(*args):
        frame = original(*args)
        frame[['open', 'high', 'low', 'close']] *= 2
        return frame
    provider.history = revised
    c = download_history(['000001'], start, end, ['hfq'], allow_network=True, provider=provider, root=tmp_path)
    assert inspect_bundle([a['id'], c['id']], ['000001'], start, end, tmp_path)['conflicts']
    with pytest.raises(ValueError, match='conflicting'):
        compose_bundle([a['id'], c['id']], ['000001'], start, end, tmp_path)
    assert Snapshots(tmp_path).get('history', joined['id'])[0]['source_history_ids'] == [a['id'], b['id']]


def test_schedule_and_risk_selection(local_data):
    cfg = deepcopy(local_data.config)
    cfg.update(learning_rate_schedule='linear', learning_rate_end_factor=0.1)
    env = local_data.vector_env()
    model = make_model(env, cfg)
    assert model.lr_schedule(1) == pytest.approx(cfg['ppo']['learning_rate'])
    assert model.lr_schedule(-1) == pytest.approx(cfg['ppo']['learning_rate'] * 0.1)
    env.close()
    metrics = dict(sharpe=1.2, cumulative_return=0.1, max_drawdown=-0.2, mean_daily_turnover=0.1, trading_steps=30)
    assert selection_score(metrics) == (1.2, 0.1)
    assert not selection_assessment(metrics, {'max_drawdown': 0.1})['eligible']
    assert selection_score(metrics, {'min_steps': 60}) is None
    assert selection_score({**metrics, 'sharpe': None}) is None


def test_added_risk_metrics_keep_undefined_values_explicit():
    frame = pd.DataFrame(dict(equity=[100., 110., 99., 120.], fee=0., slippage_cost=0., turnover=0., cash=10.))
    m = performance(frame)
    assert m['sortino'] is not None and m['calmar'] is not None and m['longest_underwater_steps'] == 1
    frame['equity'] = 100.
    flat = performance(frame)
    assert flat['sharpe'] is None and flat['calmar'] is None and flat['sortino'] is None


@pytest.mark.local_ipc
def test_experiment_api_is_immutable_and_does_not_implicitly_train(web_root):
    with TestClient(create_app(web_root, start_workers=False)) as client:
        headers = {'X-CSRF-Token': client.get('/api/session').json()['token']}
        response = client.post('/api/experiments', json=recipe(web_root), headers=headers)
        assert response.status_code == 200
        exp = response.json()
        assert exp['recipe']['template_manifest_sha256']
        assert client.get('/api/jobs').json() == []
        assert client.post(f"/api/experiments/{exp['id']}/train", json={}, headers=headers).status_code == 409
        assert client.post(f"/api/experiments/{exp['id']}/check", json={}, headers=headers).status_code == 200
        assert all(j['kind'] == 'check' for j in client.get('/api/jobs').json())
