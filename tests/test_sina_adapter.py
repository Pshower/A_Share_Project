"""The Sina adapter is required by ``src/data/online.py`` and by the daily forward panel.

A branch cut once deleted ``src/data/sina.py`` while ``online.py`` still imported it at three call
sites; nothing failed until the 20:00 S2 job tried to download and died with ``ModuleNotFoundError``.
These two tests fail the moment that happens again, without touching the network.
"""
from __future__ import annotations


def test_the_sina_adapter_exists_and_is_wired():
    from src.data.online import Eastmoney
    from src.data.sina import Sina

    assert Sina.name == "sina"
    assert issubclass(Sina, Eastmoney)


def test_the_online_module_still_routes_history_to_sina():
    import inspect

    from src.data import online

    source = inspect.getsource(online.download_history)
    assert "from src.data.sina import Sina" in source, (
        "download_history no longer imports the Sina adapter; if this was deliberate, "
        "remove the source='sina' branch and update tools/forward_panel.py too"
    )


def test_the_forward_panel_imports():
    import tools.forward_panel as panel

    assert hasattr(panel, "capture")
