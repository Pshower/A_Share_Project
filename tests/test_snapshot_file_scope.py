import pytest
from src.data.online import download_history
from src.data.snapshots import Snapshots
from tests.test_bulk_download import Provider


def test_selective_read_checks_only_requested_file_but_full_read_still_protects_all(tmp_path, monkeypatch):
    import src.data.snapshots as module
    batch = download_history(['000001', '600000'], '2020-01-01', '2020-01-03', ['hfq'],
                             allow_network=True, root=tmp_path, provider=Provider())
    snapshots = Snapshots(tmp_path)
    _, directory = snapshots.get('history', batch['id'])
    original = module.file_hash
    calls = []
    def tracked(path):
        calls.append(path.relative_to(directory).as_posix())
        return original(path)
    monkeypatch.setattr(module, 'file_hash', tracked)
    snapshots.get('history', batch['id'], files=['hfq/000001.parquet'])
    assert calls == ['hfq/000001.parquet']
    # Simulate corruption outside the displayed stock, never in production data.
    (directory / 'hfq/600000.csv').write_text('corrupt', encoding='utf-8')
    snapshots.get('history', batch['id'], files=['hfq/000001.parquet'])
    with pytest.raises(ValueError, match='mismatch'):
        snapshots.get('history', batch['id'])
    (directory / 'hfq/000001.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='mismatch'):
        snapshots.get('history', batch['id'], files=['hfq/000001.parquet'])
    with pytest.raises(ValueError, match='registered'):
        snapshots.get('history', batch['id'], files=['../escape'])
