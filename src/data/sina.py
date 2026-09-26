"""Sina AKShare adapter with pacing at each HTTP request, not each SDK call."""

from unittest.mock import patch

import requests

from src.data.online import Eastmoney
from src.data.rate_limit import AdaptivePacer


class Sina(Eastmoney):
    name = "sina"

    def __init__(self, root):
        super().__init__(root)
        self.limiter = AdaptivePacer(self.root / "data/online/providers/sina")

    def history(self, code, start, end, basis):
        import akshare as ak
        if basis not in {"hfq", "unadjusted"}:
            raise ValueError("Explicit Sina history basis required")
        original = requests.sessions.Session.request
        def request(session, *args, **kwargs):
            self.pace()
            kwargs["timeout"] = (5, 12)
            response = original(session, *args, **kwargs)
            response.raise_for_status()
            return response
        # Workers are serial; do not use this scoped SDK hook in concurrent threads.
        with patch.object(requests.sessions.Session, "request", request):
            frame = ak.stock_zh_a_daily(symbol=("sh" if code.startswith("6") else "sz") + code,
                start_date=start.replace("-", ""), end_date=end.replace("-", ""),
                adjust="hfq" if basis == "hfq" else "")
        frame = frame.copy()
        frame["code"] = code
        # Canonical history normalizer expects Eastmoney-style lots and percent.
        frame["volume"] /= 100
        frame["turnover"] *= 100
        return frame

    def quote(self, code, permission_check=None):
        raise ValueError("Sina daily models do not silently use Eastmoney live quotes")
