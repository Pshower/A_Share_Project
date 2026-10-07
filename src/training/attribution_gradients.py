"""Read-only actor/critic gradient attribution on a fixed PPO rollout batch."""

import hashlib
import json
import pickle
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .research_matrix import preserve_training_rng


def _tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _state_digest(policy, gradients=False):
    digest = hashlib.sha256()
    values = policy.named_parameters() if gradients else policy.state_dict().items()
    for name, value in sorted(values):
        digest.update(name.encode("utf-8"))
        value = value.grad if gradients else value
        if value is None:
            digest.update(b"NONE")
        else:
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(list(value.shape)).encode("ascii"))
            digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _rng_digest(device):
    digest = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()), protocol=4))
    digest.update(_tensor_bytes(torch.get_rng_state()))
    if device.type == "cuda":
        digest.update(_tensor_bytes(torch.cuda.get_rng_state(device)))
    return digest.hexdigest()


def _numpy_tensor(value, device, name):
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
        raise TypeError(name + " must be a real numeric NumPy array")
    if not np.isfinite(value).all():
        raise FloatingPointError(name + " must contain only finite values")
    with torch.inference_mode(False):
        return torch.as_tensor(value.copy(), device=device).detach()


def _parameter_groups(named):
    def component(name):
        return name.split("network.", 1)[-1]

    market = [i for i, (name, _) in enumerate(named) if component(name).startswith("market_encoder.")]
    account = [i for i, (name, _) in enumerate(named) if component(name).startswith("account_encoder.")]
    legacy = [i for i, (name, _) in enumerate(named) if component(name).startswith("encoder.")]
    combined = sorted(set(market + account + legacy))
    if not combined:
        raise ValueError("No identifiable market/account or shared encoder parameters")
    return dict(market_encoder=market, account_encoder=account, combined_encoder=combined,
                stock_body=[i for i, (name, _) in enumerate(named) if component(name).startswith("stock_body.")],
                actor_heads=[i for i, (name, _) in enumerate(named) if component(name).startswith(
                    ("direction_head.", "size_head.")) or component(name) == "log_std"],
                value_head=[i for i, (name, _) in enumerate(named) if component(name).startswith("value_head.")],
                all_parameters=list(range(len(named))))


def _vector(grads, named, indices):
    if not indices:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate([
        np.zeros(parameter.numel(), dtype=np.float64) if grads[i] is None
        else grads[i].detach().cpu().to(torch.float64).reshape(-1).numpy()
        for i in indices for _, parameter in [named[i]]
    ])


def _gradient_pair(actor, critic, critic_raw):
    norm_actor, norm_critic = float(np.linalg.norm(actor)), float(np.linalg.norm(critic))
    dot = float(np.dot(actor, critic))
    cosine = (None if norm_actor == 0 or norm_critic == 0 else
              float(np.clip(dot / (norm_actor * norm_critic), -1, 1)))
    return dict(norm_actor=norm_actor, norm_critic=norm_critic,
                norm_critic_raw=float(np.linalg.norm(critic_raw)), dot=dot, cosine=cosine,
                combinednorm=float(np.linalg.norm(actor + critic)))


def gradient_diagnostics(model, batch, vf_coef=.5, clip_range=.2):
    """Attribute clipped PPO actor and weighted Critic gradients without updating.

    ``model`` may be a PPO instance or its policy. ``batch`` contains NumPy
    observations, original packed actions, old_log_prob, advantages and returns.
    Mask/lot/share observation dtypes are retained. Floating targets are detached
    and cast to the evaluated loss dtype, as PPO's float rollout targets require.

    Critic norms/dot/cosine use vf_coef * MSE; norm_critic_raw separately records
    unweighted MSE. combinednorm is ||g_actor + g_critic|| before global clipping
    or Adam, not the actual optimizer update. Entropy is recorded, not included
    in the actor surrogate (the registered study uses ent_coef=0).
    """
    if not np.isfinite(vf_coef) or vf_coef < 0 or not np.isfinite(clip_range) or clip_range < 0:
        raise ValueError("vf_coef and clip_range must be finite nonnegative scalars")
    policy = getattr(model, "policy", model)
    if getattr(model, "clip_range_vf", None) is not None:
        raise ValueError("Value clipping is unsupported without the registered old_values target")
    named = [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad]
    if not named:
        raise ValueError("Diagnostic policy must have trainable parameters")
    device = named[0][1].device
    if any(parameter.device != device for _, parameter in named):
        raise ValueError("Diagnostic policy parameters must use one device")
    if any(isinstance(module, (nn.modules.dropout._DropoutNd, nn.modules.batchnorm._BatchNorm))
           for module in policy.modules()):
        raise ValueError("Read-only attribution requires a network without Dropout or BatchNorm")
    groups = _parameter_groups(named)
    required = {"observations", "actions", "old_log_prob", "advantages", "returns"}
    if not isinstance(batch, dict) or not required <= set(batch):
        raise ValueError("Diagnostic batch is missing required PPO rollout fields")
    if not isinstance(batch["observations"], dict) or not batch["observations"]:
        raise TypeError("observations must be a nonempty NumPy-array dictionary")
    actions = _numpy_tensor(batch["actions"], device, "actions")
    if actions.ndim != 2 or not len(actions):
        raise ValueError("Original packed actions must have shape [batch,2N]")
    action_shape = getattr(getattr(policy, "action_space", None), "shape", None)
    if action_shape is not None and tuple(actions.shape[1:]) != tuple(action_shape):
        raise ValueError("Packed actions do not match the policy action space")
    size = len(actions)
    observations = {name: _numpy_tensor(value, device, "observations." + name)
                    for name, value in batch["observations"].items()}
    if any(value.ndim < 1 or len(value) != size for value in observations.values()):
        raise ValueError("Observation arrays must share the action batch dimension")
    targets = {name: _numpy_tensor(batch[name], device, name)
               for name in ("old_log_prob", "advantages", "returns")}
    if any(value.shape not in {(size,), (size, 1)} for value in targets.values()):
        raise ValueError("PPO targets must have shape [batch] or [batch,1]")
    modes = [(module, module.training) for module in policy.modules()]
    grad_references = [(parameter, parameter.grad) for parameter in policy.parameters()]
    distribution = getattr(policy, "action_dist", None)
    distribution_state = None if distribution is None else distribution.__dict__.copy()
    parameter_before, grad_before = _state_digest(policy), _state_digest(policy, gradients=True)
    rng_before = _rng_digest(device)
    try:
        with preserve_training_rng(device), torch.inference_mode(False), torch.enable_grad():
            policy.train(True)
            values, log_prob, entropy = policy.evaluate_actions(observations, actions)
            values, log_prob = values.flatten(), log_prob.flatten()
            if values.shape != (size,) or log_prob.shape != (size,):
                raise ValueError("Policy values and joint log_prob must be one scalar per sample")
            if not torch.isfinite(values).all() or not torch.isfinite(log_prob).all():
                raise FloatingPointError("Policy values and original-action joint log_prob must be finite")
            advantages = targets["advantages"].flatten().to(log_prob.dtype).detach()
            normalized = bool(getattr(model, "normalize_advantage", True) and size > 1)
            if normalized:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            old_log_prob = targets["old_log_prob"].flatten().to(log_prob.dtype).detach()
            returns = targets["returns"].flatten().to(values.dtype).detach()
            ratio = torch.exp(log_prob - old_log_prob)
            actor_loss = -torch.minimum(advantages * ratio,
                advantages * ratio.clamp(1 - clip_range, 1 + clip_range)).mean()
            critic_raw = F.mse_loss(values, returns)
            critic_scaled = vf_coef * critic_raw
            if not torch.isfinite(actor_loss) or not torch.isfinite(critic_scaled):
                raise FloatingPointError("PPO attribution losses must be finite")
            parameters = [parameter for _, parameter in named]
            actor_grads = torch.autograd.grad(actor_loss, parameters, retain_graph=True, allow_unused=True)
            critic_raw_grads = torch.autograd.grad(critic_raw, parameters, retain_graph=True, allow_unused=True)
            critic_grads = torch.autograd.grad(critic_scaled, parameters, allow_unused=True)
            attributed = {}
            for name, indices in groups.items():
                actor_vector = _vector(actor_grads, named, indices)
                critic_vector = _vector(critic_grads, named, indices)
                raw_vector = _vector(critic_raw_grads, named, indices)
                if not all(np.isfinite(vector).all() for vector in (actor_vector, critic_vector, raw_vector)):
                    raise FloatingPointError("PPO attribution gradients must be finite")
                attributed[name] = dict(_gradient_pair(actor_vector, critic_vector, raw_vector),
                    parameter_count=sum(named[i][1].numel() for i in indices),
                    parameter_names=[named[i][0] for i in indices])
            log_ratio = log_prob - old_log_prob
            result = dict(schema_version=1, batch_size=size, parameter_count=sum(p.numel() for p in parameters),
                losses=dict(actor=float(actor_loss.detach()), critic_raw=float(critic_raw.detach()),
                    critic_scaled=float(critic_scaled.detach()), vf_coef=float(vf_coef), clip_range=float(clip_range),
                    normalize_advantage=normalized, advantage_std_ddof=1,
                    entropy_mean=None if entropy is None else float(entropy.detach().mean()),
                    conditional_size_entropy_mean=(float(distribution.conditional_size_entropy().detach().mean())
                        if hasattr(distribution, "conditional_size_entropy") else None),
                    entropy_coefficient=float(getattr(model, "ent_coef", 0.0)), actor_entropy_included=False,
                    clip_fraction=float(((ratio - 1).abs() > clip_range).float().mean().detach()),
                    approximate_kl=float((torch.exp(log_ratio) - 1 - log_ratio).mean().detach())),
                groups=attributed, value_head_critic_gradnorm=attributed["value_head"]["norm_critic"],
                actor_head_actor_gradnorm=attributed["actor_heads"]["norm_actor"],
                actor_head_critic_gradnorm=attributed["actor_heads"]["norm_critic"],
                observation_dtypes={name: str(value.dtype) for name, value in observations.items()},
                action_dtype=str(actions.dtype), targets_detached=True,
                effective_target_dtypes=dict(old_log_prob=str(old_log_prob.dtype), advantages=str(advantages.dtype),
                                             returns=str(returns.dtype)),
                interpretation="Gradient directions before clipping/Adam, not optimizer updates or trading quality.")
    finally:
        for module, training in modes:
            module.training = training
        if distribution_state is not None:
            distribution.__dict__.clear()
            distribution.__dict__.update(distribution_state)
        if parameter_before != _state_digest(policy):
            raise AssertionError("Read-only gradient attribution changed policy state")
        if grad_before != _state_digest(policy, gradients=True) or any(
                parameter.grad is not before for parameter, before in grad_references):
            raise AssertionError("Read-only gradient attribution changed existing .grad tensors")
        if rng_before != _rng_digest(device):
            raise AssertionError("Read-only gradient attribution changed an RNG stream")
    result["state_preservation"] = dict(parameter_sha256=parameter_before, gradient_sha256=grad_before,
        parameters_unchanged=True, gradients_unchanged=True, rng_unchanged=True, training_modes_restored=True,
        optimizer_updates=0)
    return result
