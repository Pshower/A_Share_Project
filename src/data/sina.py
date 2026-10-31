"""Sina AKShare adapter with pacing at each HTTP request, not each SDK call."""

import re
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
        if permission_check:
            permission_check()
        self.pace()
        symbol = ("sh" if code.startswith("6") else "sz") + code
        response = requests.get("https://hq.sinajs.cn/list=" + symbol, headers={
            "Referer": "https://finance.sina.com.cn/",
            "User-Agent": "Mozilla/5.0",
        }, timeout=(5, 12))
        response.raise_for_status()
        text = response.content.decode("gb18030", errors="replace")
        match = re.search(r'hq_str_' + re.escape(symbol) + r'="([^"]*)"', text)
        if not match:
            raise ValueError("Sina quote response schema changed")
        fields = match.group(1).split(",")
        if len(fields) < 32 or not fields[0] or fields[3] in {"", "0", "0.000"}:
            raise ValueError("Sina returned no tradable quote")
        return dict(code=code, name=fields[0], open=fields[1], previous_close=fields[2], last=fields[3],
                    high=fields[4], low=fields[5], bid=fields[6], ask=fields[7], volume=fields[8],
                    amount=fields[9], date=fields[30], time=fields[31],
                    status=fields[32] if len(fields) > 32 else "")
