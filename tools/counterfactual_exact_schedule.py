"""Exact phase/account/date coverage for the isolated paired teacher source."""

from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd

from src.envs.initial_account import MODES


DATE_STRATA = ("early", "middle", "latest")
SCHEDULE_SEED_SALT = 0xE541
DATE_SEED_SALT = 0xE54D
SCHEDULE_VERSION = "exact_phase_account_date_240_v1"


class ExactPhaseSampler:
    """One cycle covers all 20 phases, four accounts and three date thirds."""

    def __init__(self, dates, episode_steps=80):
        self.dates = pd.DatetimeIndex(dates)
        if (type(episode_steps) is not int or episode_steps < 1
                or self.dates.hasnans or self.dates.has_duplicates
                or not self.dates.is_monotonic_increasing or self.dates.tz is not None):
            raise ValueError("Exact schedule requires positive steps and unique ordered naive dates")
        self.episode_steps = episode_steps
        self.maximum_start = len(self.dates) - episode_steps - 1
        if self.maximum_start < 2:
            raise ValueError("Three date thirds need at least three feasible episode starts")
        chunks = np.array_split(np.arange(self.maximum_start + 1, dtype=np.int64), 3)
        self.starts = {name: chunk for name, chunk in zip(DATE_STRATA, chunks)}
        self.seed(0)

    def seed(self, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("Exact schedule seed must be a nonnegative integer")
        self.seed_value = seed
        self.schedule_rng = np.random.default_rng(seed ^ SCHEDULE_SEED_SALT)
        self.date_rng = np.random.default_rng(seed ^ DATE_SEED_SALT)
        self.remaining = []
        self.episode_count = 0
        self.history = []

    def next(self):
        if not self.remaining:
            combinations = list(product(MODES, DATE_STRATA, range(20)))
            self.remaining = [combinations[int(index)]
                              for index in self.schedule_rng.permutation(len(combinations))]
        mode, stratum, phase = self.remaining.pop()
        choices = self.starts[stratum]
        index = int(choices[int(self.date_rng.integers(len(choices)))])
        result = dict(episode=self.episode_count, initial_mode=str(mode),
                      date_stratum=str(stratum), phase_bin=int(phase // 5),
                      phase=int(phase), start_index=index,
                      start_date=str(self.dates[index].date()))
        self.episode_count += 1
        self.history.append(deepcopy(result))
        return result

    def audit(self):
        return dict(version=SCHEDULE_VERSION, seed=self.seed_value,
                    schedule_seed_salt=SCHEDULE_SEED_SALT, date_seed_salt=DATE_SEED_SALT,
                    episode_steps=self.episode_steps, maximum_start=self.maximum_start,
                    date_strata={key: value.tolist() for key, value in self.starts.items()},
                    cycle_episodes=240, episode_count=self.episode_count,
                    history=deepcopy(self.history),
                    rng_states={"schedule": deepcopy(self.schedule_rng.bit_generator.state),
                                "date": deepcopy(self.date_rng.bit_generator.state)})
