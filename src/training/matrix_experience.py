"""Audit sampled training accounts and dates without changing environment behavior."""

from collections import Counter
import hashlib
import json

import gymnasium as gym


def _sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EpisodeAudit(gym.Wrapper):
    """Record the actual accounts and decisions experienced by a trading policy."""

    def __init__(self, env):
        super().__init__(env)
        self.episodes = []
        self._active_episode = None
        self._decision_date_counts = Counter()
        self._mode_steps = Counter()

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        base = self.unwrapped
        positions = {
            code: float(quantity)
            for code, quantity in zip(base.market.stock_codes, base.portfolio.positions)
            if quantity != 0
        }
        cash = float(base.portfolio.cash)
        date = base.market.current_date.date().isoformat()
        account = dict(positions=positions, cash=cash)
        record = dict(
            episode=len(self.episodes) + 1,
            start_date=date,
            mode=info["initial_account"]["mode"],
            initial_cash=cash,
            holding_count=len(positions),
            initial_positions=positions,
            account_sha256=_sha256(account),
            initial_state_sha256=_sha256(dict(account, date=date)),
            steps=0,
            decision_dates=[],
            rewards=[],
            reward_sum=0.0,
            completed=False,
            terminated=False,
            truncated=False,
            end_date=None,
        )
        if "top_n" in observation:
            record["top_n"] = int(observation["top_n"][0])
            record["conditioned_state_sha256"] = _sha256(dict(account, date=date, top_n=record["top_n"]))
        self.episodes.append(record)
        self._active_episode = record
        return observation, info

    def step(self, action):
        decision_date = self.unwrapped.market.current_date.date().isoformat()
        result = self.env.step(action)
        _, reward, terminated, truncated, _ = result
        episode = self._active_episode
        episode["steps"] += 1
        episode["decision_dates"].append(decision_date)
        episode["rewards"].append(float(reward))
        episode["reward_sum"] += float(reward)
        episode["completed"] = bool(terminated or truncated)
        episode["terminated"] = bool(terminated)
        episode["truncated"] = bool(truncated)
        episode["end_date"] = self.unwrapped.market.current_date.date().isoformat()
        self._decision_date_counts.update([decision_date])
        self._mode_steps.update([episode["mode"]])
        return result

    def stats(self):
        # VecEnv resets after a terminal step, even when training stops immediately.
        experienced = [episode for episode in self.episodes if episode["steps"]]
        result = dict(
            total_steps=sum(self._decision_date_counts.values()),
            started_episodes=len(experienced),
            completed_episodes=sum(episode["completed"] for episode in experienced),
            unique_initial_accounts=len({episode["account_sha256"] for episode in experienced}),
            unique_initial_states=len({episode["initial_state_sha256"] for episode in experienced}),
            unique_decision_dates=len(self._decision_date_counts),
            decision_date_counts=dict(sorted(self._decision_date_counts.items())),
            mode_steps=dict(sorted(self._mode_steps.items())),
            mode_episodes=dict(sorted(Counter(episode["mode"] for episode in experienced).items())),
        )
        if experienced and "top_n" in experienced[0]:
            result.update(top_n_episodes=dict(sorted(Counter(str(e["top_n"]) for e in experienced).items())),
                          top_n_steps=dict(sorted(Counter({str(k): sum(e["steps"] for e in experienced if e["top_n"] == k)
                                                          for k in {e["top_n"] for e in experienced}}).items())),
                          unique_conditioned_states=len({e["conditioned_state_sha256"] for e in experienced}))
        return result
