"""Baselines consume only the current observation."""

import numpy as np


class Baseline:
    def __init__(self, name, feature_cols, max_weight=1.0, rebalance_every=20, top_k=20):
        if name not in {"cash", "buy_hold", "equal_weight", "momentum"}:
            raise ValueError("Unknown baseline")
        if rebalance_every < 1 or top_k < 1:
            raise ValueError("Strategy intervals and top_k must be positive")
        self.name = name
        self.interval = rebalance_every
        self.top_k = top_k
        self.max_weight = max_weight
        self.momentum_index = feature_cols.index("momentum_20") if name == "momentum" else None
        self.invested = False

    def act(self, observation, step):
        n = len(observation["valid_mask"])
        if self.name == "cash":
            return np.zeros(n)
        if self.name == "buy_hold" and self.invested:
            return None
        if self.name in {"equal_weight", "momentum"} and step % self.interval:
            return None
        selected = np.flatnonzero(observation["valid_mask"] & observation["buy_mask"])
        weights = np.zeros(n)
        if self.name == "momentum":
            scores = observation["stock_features"][selected, -1, self.momentum_index]
            selected = selected[np.argsort(-scores, kind="stable")[:self.top_k]]
        if len(selected):
            weights[selected] = min(1.0 / len(selected), self.max_weight)
            self.invested = True
        return weights
