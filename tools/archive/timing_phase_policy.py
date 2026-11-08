"""Tools-only due-phase context interaction for the unchanged timing actions."""

from collections import OrderedDict

import gymnasium as gym
import torch
from torch import nn

from src.agents.timing_policy import CalendarTimingPolicy, PHASE_PERIOD

OBSERVATION_DIMENSION = 129
CONTEXT_DIMENSION = OBSERVATION_DIMENSION - PHASE_PERIOD
PHASE_ARCHITECTURES = ("linear", "due_phase_residual")


class DuePhaseActionHead(nn.Module):
    """Add context sensitivity to REBALANCE only when phase zero is due."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = OBSERVATION_DIMENSION, 2
        devices = [] if base.weight.device.type == "cpu" else [base.weight.device]
        with torch.random.fork_rng(devices=devices):
            self.residual = nn.Linear(CONTEXT_DIMENSION, 1, bias=False,
                                      device=base.weight.device, dtype=base.weight.dtype)
            with torch.no_grad():
                self.residual.weight.zero_()

    def forward(self, features):
        correction = features[:, :1] * self.residual(features[:, PHASE_PERIOD:])
        return self.base(features) + torch.cat((torch.zeros_like(correction), correction), dim=1)


class DuePhaseTimingPolicy(CalendarTimingPolicy):
    """The baseline Actor/Critic plus one zero-initialized 109D due vector.

    SB3's forward, evaluate_actions and get_distribution all use action_net.
    This changes no observation, macro code, Critic or economic order decoder.
    """

    def __init__(self, observation_space, action_space, lr_schedule, *, calendar_margin=2., **kwargs):
        if (not isinstance(observation_space, gym.spaces.Box)
                or observation_space.shape != (OBSERVATION_DIMENSION,)):
            raise ValueError("Due-phase policy requires the registered 129D observation")
        if isinstance(calendar_margin, bool) or calendar_margin != 2.:
            raise ValueError("Due-phase architecture fixes calendar_margin=2")
        super().__init__(observation_space, action_space, lr_schedule,
                         calendar_margin=calendar_margin, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = DuePhaseActionHead(self.action_net)
        # The optimizer built by SB3 did not know the newly registered vector.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def mapped_baseline_state_dict(policy):
    """Expose the shared tensors under their original baseline names, read-only."""
    if type(policy) not in (CalendarTimingPolicy, DuePhaseTimingPolicy):
        raise ValueError("Only the two registered timing policy classes can be mapped")
    result = OrderedDict()
    for name, value in policy.state_dict().items():
        if name == "action_net.residual.weight":
            continue
        result[name.replace("action_net.base.", "action_net.")] = value
    return result
