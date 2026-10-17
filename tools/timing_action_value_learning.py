"""Train-only ridge action values mapped into the ordinary timing Actor."""

import hashlib
import json

import numpy as np


FEATURE_DIMENSION = 129
PHASE_PERIOD = 20
RIDGE_ALPHA = 1.0
TARGET_RMS_FLOOR = 1e-8
ACTOR_GAIN = 8.0
CALENDAR_DEADBAND = 0.25
FIT_VERSION = "timing_rule_action_value_ridge_v1"


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_digest(value):
    array = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _features(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise ValueError("Action-value features must be real numeric values")
    array = np.asarray(raw, dtype=np.float64)
    if (array.ndim != 2 or array.shape[0] < 1
            or array.shape[1] != FEATURE_DIMENSION or not np.isfinite(array).all()):
        raise ValueError("Action-value features must be finite [N,129]")
    phase = array[:, :PHASE_PERIOD]
    if not np.isin(phase, (0, 1)).all() or not (phase.sum(axis=1) == 1).all():
        raise ValueError("Action-value features require an exact twenty-way phase one-hot")
    return array


def _targets(value, count):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise ValueError("Action-value targets must be real numeric values")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError("Action-value targets must be one finite Delta per observation")
    return array


def _finite(value, name):
    if not np.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite action-value {name}")
    return value


def _validate_fit(fit):
    if (not isinstance(fit, dict) or fit.get("version") != FIT_VERSION
            or fit.get("feature_dimension") != FEATURE_DIMENSION
            or type(fit.get("n_train")) is not int or fit["n_train"] < 2
            or isinstance(fit.get("alpha"), bool) or fit.get("alpha") != RIDGE_ALPHA
            or fit.get("objective") != "mean_squared_error_plus_alpha_l2_unpenalized_intercept"
            or fit.get("target_units") != "H20_discounted_relative_log_reward_difference"):
        raise ValueError("Unknown action-value fit or ridge contract")
    payload = {key: value for key, value in fit.items() if key != "fit_sha256"}
    if fit.get("fit_sha256") != _digest(payload):
        raise ValueError("Action-value fit SHA mismatch")
    arrays = {}
    for name in ("x_mean", "x_scale", "coef_normalized", "coef_observation"):
        array = np.asarray(fit.get(name), dtype=np.float64)
        if array.shape != (FEATURE_DIMENSION,) or not np.isfinite(array).all():
            raise ValueError(f"Invalid action-value {name}")
        arrays[name] = array
    if (arrays["x_scale"] <= 0).any():
        raise ValueError("Action-value feature scales must be positive")
    constant = np.asarray(fit.get("constant_feature_mask"))
    if (constant.shape != (FEATURE_DIMENSION,) or constant.dtype.kind != "b"
            or not (arrays["x_scale"][constant] == 1).all()):
        raise ValueError("Invalid constant-column action-value scales")
    for name in ("target_rms", "raw_target_rms", "intercept_normalized", "intercept_observation", "train_target_mean"):
        value = fit.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Invalid action-value {name}")
    if (fit["raw_target_rms"] < 0
            or fit["target_rms"] != max(fit["raw_target_rms"], TARGET_RMS_FLOOR)):
        raise ValueError("Action-value target RMS is below the registered floor")
    coefficient = arrays["coef_normalized"] / arrays["x_scale"]
    intercept = fit["intercept_normalized"] - arrays["x_mean"] @ coefficient
    if (not np.allclose(coefficient, arrays["coef_observation"], rtol=1e-13, atol=1e-13)
            or not np.isclose(intercept, fit["intercept_observation"], rtol=1e-13, atol=1e-13)):
        raise ValueError("Action-value observation-coordinate mapping differs")
    return arrays


def fit_action_value(X_train, y_train, *, alpha=RIDGE_ALPHA):
    """Fit mean squared ridge; neither holdout inputs nor labels are accepted.

    The intercept is not penalized. ``target_rms`` converts normalized scores
    back to discounted relative-log reward differences, not portfolio ROI.
    """
    if isinstance(alpha, bool) or alpha != RIDGE_ALPHA:
        raise ValueError("The registered mean-squared ridge alpha is exactly one")
    features = _features(X_train)
    if len(features) < 2:
        raise ValueError("Action-value fitting requires at least two training rows")
    target = _targets(y_train, len(features))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean = _finite(features.mean(axis=0), "training means")
        constant = np.all(features == features[:1], axis=0)
        mean[constant] = features[0, constant]
        centered = _finite(features - mean, "centered training features")
        deviation = _finite(np.sqrt(np.mean(np.square(centered), axis=0)), "training deviations")
        if ((deviation == 0) & ~constant).any():
            raise FloatingPointError("Nonconstant action-value column has underflowed deviation")
        scale = np.where(constant, 1.0, deviation)
        standardized = _finite(centered / scale, "standardized training features")
        maximum = float(np.max(np.abs(target)))
        raw_rms = float(maximum * np.sqrt(np.mean(np.square(target / maximum)))) if maximum else 0.0
        target_rms = max(raw_rms, TARGET_RMS_FLOOR)
        normalized = _finite(target / target_rms, "normalized training targets")
        intercept = float(normalized.mean())
        matrix = standardized.T @ standardized / len(features) + alpha * np.eye(FEATURE_DIMENSION)
        rhs = standardized.T @ (normalized - intercept) / len(features)
        coefficient = _finite(np.linalg.solve(matrix, rhs), "ridge coefficients")
        residual = _finite(standardized @ coefficient + intercept - normalized, "training residuals")
        gradient = 2 * (standardized.T @ residual / len(features) + alpha * coefficient)
        intercept_gradient = float(2 * residual.mean())
        observation_coefficient = _finite(coefficient / scale, "observation coefficients")
        observation_intercept = float(intercept - mean @ observation_coefficient)
        _finite(observation_intercept, "observation intercept")
        train_mse = float(np.mean(np.square(residual)))
        penalty = float(alpha * coefficient @ coefficient)
    fit = dict(version=FIT_VERSION, feature_dimension=FEATURE_DIMENSION,
        n_train=len(features), alpha=float(alpha),
        objective="mean_squared_error_plus_alpha_l2_unpenalized_intercept",
        target_units="H20_discounted_relative_log_reward_difference",
        x_mean=mean.tolist(), x_scale=scale.tolist(), constant_feature_mask=constant.tolist(),
        target_rms=target_rms, raw_target_rms=raw_rms,
        coef_normalized=coefficient.tolist(), intercept_normalized=intercept,
        coef_observation=observation_coefficient.tolist(), intercept_observation=observation_intercept,
        train_target_mean=float(target.mean()),
        training=dict(mse_normalized=train_mse, mse_delta=train_mse * target_rms ** 2,
            ridge_penalty=penalty, objective=train_mse + penalty,
            coefficient_gradient_max_abs=float(np.max(np.abs(gradient))),
            intercept_gradient_abs=abs(intercept_gradient)),
        training_input_sha256=_array_digest(features), training_target_sha256=_array_digest(target),
        counters=dict(model_fits=1, training_label_exposures=len(features),
            rl_steps=0, neural_forwards=0, gradient_updates=0, optimizer_updates=0))
    fit["fit_sha256"] = _digest(fit)
    _validate_fit(fit)
    return fit


def predict(fit, X, *, normalized=False):
    """Predict original Delta units by default, or dimensionless normalized q."""
    if type(normalized) is not bool:
        raise ValueError("normalized must be an explicit boolean")
    arrays = _validate_fit(fit)
    features = _features(X)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        score = _finite(((features - arrays["x_mean"]) / arrays["x_scale"])
                        @ arrays["coef_normalized"] + fit["intercept_normalized"], "predictions")
        return score if normalized else _finite(score * fit["target_rms"], "Delta predictions")


def diagnostics(fit, X_hold, y_hold, phase=None):
    """Describe fixed-rule labels; no fitting, selection, or strategy ROI."""
    features = _features(X_hold)
    targets = _targets(y_hold, len(features))
    phases = features[:, :PHASE_PERIOD].argmax(axis=1)
    if phase is not None:
        supplied = np.asarray(phase)
        if (supplied.shape != phases.shape or supplied.dtype.kind not in "iu"
                or not np.array_equal(supplied, phases)):
            raise ValueError("Diagnostic phase must equal each input's phase one-hot")
    score = predict(fit, features, normalized=True)
    prediction = score * fit["target_rms"]
    calendar = phases == 0
    margin = ACTOR_GAIN * (score + CALENDAR_DEADBAND * (2 * calendar.astype(float) - 1))
    _finite(margin, "diagnostic actor margins")
    action = margin > 0
    gain = (action.astype(float) - calendar.astype(float)) * targets

    def summarize(mask):
        return dict(count=int(mask.sum()),
            mse_delta=float(np.mean(np.square(prediction[mask] - targets[mask]))),
            mse_training_mean_delta=float(np.mean(np.square(fit["train_target_mean"] - targets[mask]))),
            mean_single_decision_calendar_difference=float(gain[mask].mean()),
            extra_rebalances=int(np.count_nonzero(mask & action & ~calendar)),
            skipped_due_rebalances=int(np.count_nonzero(mask & ~action & calendar)))

    result = summarize(np.ones(len(features), dtype=bool))
    result.update(fit_sha256=fit["fit_sha256"],
        actor_mapping=dict(gain=ACTOR_GAIN, calendar_deadband=CALENDAR_DEADBAND,
            tie_action="HOLD", deadband_is_statistical_confidence=False),
        phase_counts={str(value): int(np.count_nonzero(phases == value)) for value in range(PHASE_PERIOD)},
        by_phase={str(value): summarize(phases == value) for value in range(PHASE_PERIOD)
                  if np.any(phases == value)},
        strategy_roi=False, on_policy_advantage=False,
        counters=dict(model_fits=0, rl_steps=0, neural_forwards=0, gradient_updates=0, optimizer_updates=0))
    return result


def _tensor_digest(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def apply_actor_warmup(model, fit):
    """Transfer ridge into the ordinary Actor before any PPO experience/update."""
    import torch
    from stable_baselines3 import PPO
    from src.agents.timing_policy import CalendarTimingPolicy

    arrays = _validate_fit(fit)
    policy = getattr(model, "policy", None)
    if (type(model) is not PPO or type(policy) is not CalendarTimingPolicy
            or policy.parameter_count != 12805 or policy.calendar_margin != 2.0
            or policy.observation_space.shape != (FEATURE_DIMENSION,)
            or type(policy.action_net) is not torch.nn.Linear
            or policy.action_net.in_features != FEATURE_DIMENSION or policy.action_net.out_features != 2):
        raise ValueError("Warmup requires the original 12,805-parameter CalendarTimingPolicy PPO")
    if (model.num_timesteps != 0 or model._n_updates != 0
            or model.rollout_buffer.pos != 0 or model.rollout_buffer.full):
        raise ValueError("Warmup must precede every PPO timestep, buffer entry and update")
    optimizer = policy.optimizer
    if not isinstance(optimizer, torch.optim.Adam) or optimizer.state:
        raise ValueError("Warmup requires a fresh empty PPO Adam state")
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    parameter_ids = [id(parameter) for parameter in policy.parameters()]
    if (len(optimizer_parameters) != len(parameter_ids)
            or len({id(parameter) for parameter in optimizer_parameters}) != len(parameter_ids)
            or {id(parameter) for parameter in optimizer_parameters} != set(parameter_ids)):
        raise ValueError("PPO optimizer must cover every policy parameter exactly once")
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise ValueError("Warmup requires no pre-existing neural gradients")
    expected_weight = torch.zeros_like(policy.action_net.weight)
    expected_weight[0, 1:PHASE_PERIOD] = 2.0
    expected_weight[1, 0] = 2.0
    if (not torch.equal(policy.action_net.weight, expected_weight)
            or not torch.equal(policy.action_net.bias, torch.zeros_like(policy.action_net.bias))):
        raise ValueError("Warmup must start from the registered original calendar Actor")
    before = {name: tensor.detach().clone() for name, tensor in policy.state_dict().items()}
    if any(not torch.isfinite(tensor).all() for tensor in before.values()):
        raise FloatingPointError("Warmup requires finite initial policy tensors")
    coefficient = ACTOR_GAIN * arrays["coef_observation"].copy()
    coefficient[0] += 2 * ACTOR_GAIN * CALENDAR_DEADBAND
    intercept = ACTOR_GAIN * (fit["intercept_observation"] - CALENDAR_DEADBAND)
    with np.errstate(over="ignore", invalid="ignore"):
        weights = torch.as_tensor(np.stack((-0.5 * coefficient, 0.5 * coefficient)),
            device=policy.action_net.weight.device, dtype=policy.action_net.weight.dtype)
        bias = torch.as_tensor([-0.5 * intercept, 0.5 * intercept],
            device=policy.action_net.bias.device, dtype=policy.action_net.bias.dtype)
    if not torch.isfinite(weights).all() or not torch.isfinite(bias).all():
        raise FloatingPointError("Ridge transfer is not finite in the Actor dtype")
    with torch.no_grad():
        policy.action_net.weight.copy_(weights)
        policy.action_net.bias.copy_(bias)
    after = policy.state_dict()
    actor_names = {"action_net.weight", "action_net.bias"}
    unchanged = all(torch.equal(tensor, after[name]) for name, tensor in before.items() if name not in actor_names)
    if not unchanged or optimizer.state or model.num_timesteps != 0 or model._n_updates != 0:
        with torch.no_grad():
            for name, tensor in policy.state_dict().items():
                tensor.copy_(before[name])
        raise RuntimeError("Warmup modified a Critic tensor, PPO state or clock")
    return dict(version=FIT_VERSION, fit_sha256=fit["fit_sha256"], parameter_count=policy.parameter_count,
        mapping=dict(gain=ACTOR_GAIN, calendar_deadband=CALENDAR_DEADBAND,
            formula="logit_R-logit_H=8*(q_hat+0.25*(2*phase_0-1))",
            deadband_is_statistical_confidence=False, actor_heads="minus/plus half margin"),
        before_tensor_sha256={name: _tensor_digest(tensor) for name, tensor in before.items()},
        after_tensor_sha256={name: _tensor_digest(tensor) for name, tensor in after.items()},
        critic_and_all_nonactor_tensors_bitwise_unchanged=unchanged,
        ppo_optimizer_state_entries=len(optimizer.state), ppo_num_timesteps=model.num_timesteps,
        ppo_update_count=model._n_updates,
        counters=dict(model_fits=0, rl_steps=0, neural_forwards=0, gradient_updates=0, optimizer_updates=0))
