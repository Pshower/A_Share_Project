from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from src.web.store import Store
from src.web.jobs import JobManager


def test_cancel_claim_race_never_releases_claimed_slot(tmp_path):
    for n in range(20):
        store = Store(tmp_path / str(n))
        job = store.enqueue("prediction", {})
        gate = Barrier(2)
        def claim():
            gate.wait()
            return store.claim_next()
        def cancel():
            gate.wait()
            return JobManager(store, tmp_path).request_stop(job["id"])
        with ThreadPoolExecutor(2) as executor:
            a, b = executor.submit(claim), executor.submit(cancel)
            claimed, stopped = a.result(), b.result()
        assert stopped["status"] == ("stop_requested" if claimed else "cancelled")
        second = store.enqueue("prediction", {})
        following = store.claim_next()
        assert following is None if claimed else following["id"] == second["id"]


def test_terminal_result_cannot_be_overwritten_or_cancelled(tmp_path):
    store = Store(tmp_path)
    job = store.enqueue("prediction", {})
    store.update(job["id"], "succeeded", {"preserve": True})
    store.update(job["id"], "failed", error="stale manager")
    assert store.request_stop(job["id"])["result"] == {"preserve": True}
    assert store.get(job["id"])["status"] == "succeeded"
