"""Two-action categorical controller with trainable calendar initialization."""

from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor


PHASE_PERIOD = 20
HOLD, REBALANCE_M20 = 0, 1
TIMING_ACTION_NAMES = ("hold", "rebalance_m20")
TIMING_NET_ARCH = {"pi": [], "vf": [64, 64]}


class CalendarTimingPolicy(ActorCriticPolicy):
    """Linear Actor and ordinary Critic; no persistent rule prior in forward.

    The first twenty inputs are a phase one-hot. Initialization selects a
    rebalance at phase zero and HOLD otherwise. Those calendar coefficients,
    every other Actor coefficient, and the Critic are all ordinary trainable
    parameters. PPO receives only the sampled macro action, not stock orders.
    """

    def __init__(self, observation_space, action_space, lr_schedule, *,
                 calendar_margin=2., net_arch=None, **kwargs):
        if (isinstance(calendar_margin, bool)
                or not isinstance(calendar_margin, (int, float, np.integer, np.floating))
                or not np.isfinite(calendar_margin)
                or calendar_margin <= 0):
            raise ValueError("calendar_margin must be finite and positive")
        if (not isinstance(observation_space, gym.spaces.Box)
                or len(observation_space.shape) != 1
                or observation_space.shape[0] < PHASE_PERIOD
                or observation_space.dtype != np.dtype("float32")):
            raise ValueError("Timing policy requires a float32 vector Box with twenty phase coordinates")
        if (not isinstance(action_space, gym.spaces.Discrete)
                or action_space.n != 2 or action_space.start != 0):
            raise ValueError("Timing policy requires Discrete(2) starting at zero")
        if net_arch is not None and net_arch != TIMING_NET_ARCH:
            raise ValueError("Timing policy requires linear pi and vf=[64,64]")
        if kwargs.get("features_extractor_class", FlattenExtractor) is not FlattenExtractor:
            raise ValueError("Timing policy requires the unmodified vector features")
        if kwargs.get("activation_fn", nn.Tanh) is not nn.Tanh or kwargs.get("use_sde", False):
            raise ValueError("Timing policy requires Tanh Critic and categorical actions without SDE")
        self.calendar_margin = float(calendar_margin)
        super().__init__(observation_space, action_space, lr_schedule,
                         net_arch=deepcopy(TIMING_NET_ARCH), **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if not isinstance(self.action_dist, CategoricalDistribution):
            raise ValueError("Timing policy requires SB3's categorical distribution")
        # This runs once at construction, never during forward or PPO updates.
        with torch.no_grad():
            self.action_net.weight.zero_()
            self.action_net.bias.zero_()
            self.action_net.weight[HOLD, 1:PHASE_PERIOD] = self.calendar_margin
            self.action_net.weight[REBALANCE_M20, 0] = self.calendar_margin

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_vector(self, observation):
        if (observation.ndim != 2
                or observation.shape[0] < 1
                or observation.shape[1] != self.observation_space.shape[0]
                or not torch.isfinite(observation).all()):
            raise ValueError("Timing observation must be a finite [B,F] vector")
        phase = observation[:, :PHASE_PERIOD]
        if (not ((phase == 0) | (phase == 1)).all()
                or not (phase.sum(1) == 1).all()):
            raise ValueError("Timing phase must be an exact twenty-way one-hot")

    def forward(self, obs, deterministic=False):
        self._validate_vector(obs)
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._validate_vector(obs)
        if (actions.numel() != obs.shape[0] or not torch.isfinite(actions).all()
                or not ((actions == HOLD) | (actions == REBALANCE_M20)).all()):
            raise ValueError("Timing actions must be one integer code 0 or 1 per observation")
        return super().evaluate_actions(obs, actions.reshape(-1))

    def get_distribution(self, obs):
        self._validate_vector(obs)
        return super().get_distribution(obs)

    def predict_values(self, obs):
        self._validate_vector(obs)
        return super().predict_values(obs)

    def _get_constructor_parameters(self):
        result = super()._get_constructor_parameters()
        result.update(calendar_margin=self.calendar_margin)
        return result
