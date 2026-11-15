"""The frozen E1 rule, in one place.

``shadow_run`` needs the score and the buffer rule to record a book; ``e1_topk`` needs the same two
to backtest it.  Keeping two copies would let the backtest drift away from what the recorder
writes -- the exact failure the ledger was built to prevent -- so the rule lives here and both
tools import it.

Nothing in this module may be tuned: the twenty features are negated (lower score is the buy), the
book keeps whatever still ranks inside the buffer and refills from the top.  Changing either is a
new pre-registration, not a parameter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOP_K = 30
BUFFER = 60


def composite(panel, codes, features, date):
    """Negated rank-equal-weighted score on ``date``; lower is better."""
    block = panel.loc[date]
    ranks, usable = np.zeros(len(codes)), 0
    for name in features:
        try:
            values = block.xs(name, level=1).reindex(codes).to_numpy(dtype=float)
        except KeyError:
            continue
        if not np.isfinite(values).any():
            continue
        ordered = pd.Series(values).rank(method="average", pct=True).to_numpy()
        ranks += np.nan_to_num(ordered, nan=0.5)
        usable += 1
    if not usable:
        return None
    score = ranks / usable          # mean percentile; the smallest are the reversal buy
    return pd.Series(score, index=codes)


def composite_frame(panel, codes, features):
    """The same score for every date at once, for backtests.

    Vectorised because a six-year daily sweep cannot afford the per-date loop above; the arithmetic
    is identical (cross-sectional percentile per feature, averaged, lower is the buy).
    """
    blocks = []
    for name in features:
        try:
            block = panel.xs(name, axis=1, level=1)
        except KeyError:
            continue
        blocks.append(block.reindex(columns=codes).rank(axis=1, pct=True))
    if not blocks:
        return pd.DataFrame(index=panel.index, columns=codes, dtype=float)
    stacked = np.stack([frame.to_numpy(dtype=float) for frame in blocks], axis=0)
    usable = np.isfinite(stacked).sum(axis=0)
    total = np.nansum(np.where(np.isfinite(stacked), stacked, 0.0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(usable > 0, total / np.maximum(usable, 1), np.nan)
    return pd.DataFrame(mean, index=panel.index, columns=codes)


def decide(score, held, top_k=TOP_K, buffer=None):
    """Keep whatever is still inside the buffer, then fill from the top of the ranking.

    ``buffer=None`` means the frozen relationship ``2 * top_k`` (30 names, exit below rank 60).
    Callers that need a different band pass it explicitly -- and then it is a different rule, not a
    parameter of this one.
    """
    buffer = 2 * top_k if buffer is None else int(buffer)
    order = score.dropna().sort_values(kind="stable")
    rank_of = {code: position for position, code in enumerate(order.index)}
    keep = [code for code in held if rank_of.get(code, 10 ** 9) < buffer]
    keep = sorted(keep, key=lambda code: rank_of.get(code, 10 ** 9))[:top_k]
    for code in order.index:
        if len(keep) >= top_k:
            break
        if code not in keep:
            keep.append(code)
    # Sorted by today's rank: the caller renders this as "the top K of the day", and a list that puts
    # buffer survivors (ranked up to 59) ahead of new entrants (ranked 1-30) makes that slice wrong.
    return sorted(keep, key=lambda code: rank_of.get(code, 10 ** 9))
