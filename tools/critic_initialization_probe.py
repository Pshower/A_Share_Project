"""Future-study engineering helpers; never imported by the frozen live study.

The initializer consumes no random values. The scoped clipping observer calls the
installed PyTorch clipping implementation once, without extra forward/backward.
It owns a process-global hook and must not be used by concurrent training threads.
"""

from functools import wraps
import hashlib
import json

import torch
from stable_baselines3 import PPO

from src.agents.timing_policy import CalendarTimingPolicy


VERSION = "fresh_critic_head_gain_and_transparent_clip_v1"
GAINS = (1., .01, 0.)
_ACTIVE_PROBE = None
_INITIALIZATION_MARKER = "_critic_head_gain_initialization_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_sha256(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameters(model):
    _require(type(model) is PPO and type(model.policy) is CalendarTimingPolicy
             and model.policy.parameter_count == 12805, "A registered CalendarTimingPolicy PPO is required")
    parameters = dict(model.policy.named_parameters())
    _require(len(parameters) == 8 and set(parameters) == set(model.policy.state_dict())
             and all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in parameters.values()),
             "The complete eight finite policy parameter tensors are required")
    return parameters


def initialize_critic_head(model, gain=1.):
    """Scale only the final value head, once, before any PPO experience/update.

    Freshness is established by the native clocks, empty buffer/Adam, absent
    gradients and this helper's one-shot marker. No seed is reset or sampled.
    """
    _require(type(gain) in (float, int) and gain in GAINS, "Critic gain must be exactly 1, .01 or 0")
    parameters = _parameters(model)
    _require(not hasattr(model, _INITIALIZATION_MARKER), "Critic initialization was already applied")
    _require(model.num_timesteps == 0 and model._n_updates == 0
             and model.rollout_buffer.pos == 0 and not model.rollout_buffer.full
             and not model.policy.optimizer.state and all(p.grad is None for p in parameters.values()),
             "Critic initialization requires a fresh PPO, empty buffer/Adam and no gradients")
    head = model.policy.value_net
    _require(type(head) is torch.nn.Linear and head.in_features == 64 and head.out_features == 1,
             "The default final Linear(64,1) value head is required")
    before = {name: tensor_sha256(p) for name, p in parameters.items()}
    with torch.no_grad():
        head.weight.mul_(float(gain))
        head.bias.mul_(float(gain))
    after = {name: tensor_sha256(p) for name, p in parameters.items()}
    head_names = {"value_net.weight", "value_net.bias"}
    unchanged = all(before[name] == after[name] for name in parameters if name not in head_names)
    _require(unchanged and not model.policy.optimizer.state and model.num_timesteps == 0
             and model._n_updates == 0, "Critic initialization altered non-head or PPO state")
    record = dict(version=VERSION, gain=float(gain), before_tensor_sha256=before,
        after_tensor_sha256=after, only_value_head_may_change=unchanged,
        changed_tensors=[name for name in parameters if before[name] != after[name]],
        actual_rl_steps=0, optimizer_state_entries=0, new_forward_calls=0,
        new_backward_calls=0, rng_values_consumed=0)
    setattr(model, _INITIALIZATION_MARKER, dict(version=VERSION, gain=float(gain)))
    return record


def _gradients(parameters):
    return {name: None if p.grad is None else p.grad.detach().clone() for name, p in parameters.items()}


def _norms(parameters):
    sums = dict(actor=0., critic=0.)
    for name, parameter in parameters.items():
        if parameter.grad is not None:
            square = float(parameter.grad.detach().to(dtype=torch.float64).square().sum().item())
            sums["actor" if name.startswith("action_net.") else "critic"] += square
    return dict(actor=sums["actor"] ** .5, critic=sums["critic"] ** .5,
                global_norm=(sums["actor"] + sums["critic"]) ** .5)


class ClipGradientProbe:
    """Scoped, transparent observer of the original ``clip_grad_norm_`` call.

    ``records`` hold scalar L2 measurements. ``gradient_snapshots`` retain all
    pre/post gradients for the first actual clipping call at each distinct
    ``model.num_timesteps`` rollout boundary. This is not a train-loop rewrite.
    No additional loss evaluation, backward pass or random draw is performed.
    """

    def __init__(self, model):
        self.model = model
        self.parameters = _parameters(model)
        self.records = []
        self.gradient_snapshots = []
        self.foreign_clip_calls = 0
        self._seen_rollouts = set()
        self._installed = False
        self._original = self._wrapper = None

    def __enter__(self):
        global _ACTIVE_PROBE
        _require(not self._installed and _ACTIVE_PROBE is None, "Only one scoped clipping probe may be active")
        _require(tuple(id(p) for p in self.model.policy.parameters()) == tuple(id(p) for p in self.parameters.values()),
                 "Policy parameters changed after creating the probe")
        self._original = torch.nn.utils.clip_grad_norm_
        expected = tuple(id(p) for p in self.parameters.values())

        @wraps(self._original)
        def observe(parameters, *args, **kwargs):
            ordered = [parameters] if torch.is_tensor(parameters) else list(parameters)
            if tuple(id(p) for p in ordered) != expected:
                self.foreign_clip_calls += 1
                return self._original(ordered, *args, **kwargs)
            clock = int(self.model.num_timesteps)
            first = clock not in self._seen_rollouts
            before = _norms(self.parameters)
            gradients = _gradients(self.parameters) if first else None
            record = dict(call_index=len(self.records) + 1, rollout_clock=clock,
                minibatch_clip_index=1 + sum(row["rollout_clock"] == clock for row in self.records),
                first_minibatch_of_observed_rollout=first, before=before,
                norm_measurement="L2, float64 accumulation; returned norm remains native PyTorch dtype",
                norm_type=kwargs.get("norm_type", args[1] if len(args) > 1 else 2.),
                foreach=kwargs.get("foreach", args[3] if len(args) > 3 else None),
                error_if_nonfinite=kwargs.get("error_if_nonfinite", args[2] if len(args) > 2 else False))
            try:
                returned = self._original(ordered, *args, **kwargs)
            except Exception as error:
                record.update(status="raised", error_type=type(error).__name__, after=_norms(self.parameters))
                self.records.append(record)
                raise
            max_norm = kwargs.get("max_norm", args[0] if args else None)
            coefficient = torch.clamp(float(max_norm) / (returned.detach() + 1e-6), max=1.)
            record.update(status="completed", after=_norms(self.parameters),
                returned_norm=float(returned.detach().item()), returned_norm_dtype=str(returned.dtype),
                max_norm=float(max_norm), clip_coefficient=float(coefficient.item()))
            self.records.append(record)
            self._seen_rollouts.add(clock)
            if first:
                self.gradient_snapshots.append(dict(call_index=record["call_index"], rollout_clock=clock,
                    parameter_order=list(self.parameters), before=gradients, after=_gradients(self.parameters)))
            return returned

        self._wrapper = observe
        torch.nn.utils.clip_grad_norm_ = observe
        self._installed = True
        _ACTIVE_PROBE = self
        return self

    def __exit__(self, error_type, error, traceback):
        global _ACTIVE_PROBE
        changed = torch.nn.utils.clip_grad_norm_ is not self._wrapper
        torch.nn.utils.clip_grad_norm_ = self._original
        self._installed = False
        _ACTIVE_PROBE = None
        if changed and error is None:
            raise RuntimeError("The process-global clipping hook changed while the probe owned it")
        return False

    def summary(self):
        return dict(version=VERSION, calls=len(self.records), completed_calls=sum(r["status"] == "completed" for r in self.records),
            retained_rollout_first_minibatches=len(self.gradient_snapshots), foreign_clip_calls=self.foreign_clip_calls,
            new_forward_calls=0, new_backward_calls=0, rng_values_consumed=0,
            original_clip_implementation_called_once_per_observed_call=True,
            process_global_hook_requires_single_training_thread=True, records=self.records)
