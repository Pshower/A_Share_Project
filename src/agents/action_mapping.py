"""A single latent-action mapping shared by PPO rollouts and inference.

Two ways to turn latents into target weights:

* with cash (``full_invest=False``): N stock latents plus one cash latent, softmaxed together.
  The cash latent competes with the stocks, so the book can sit partly in cash.
* fully invested (``full_invest=True``): N stock latents only.  The softmax is normalised to
  one and the per-name cap is honoured by freezing offending names and redistributing the
  remaining budget pro rata.  Cash then appears only as an infeasible-residual: if the
  eligible names times the cap cannot reach one, the shortfall is reported rather than hidden.

The fully-invested form exists so that "return minus the equal-weight pool" is a pure selection
result.  With a cash action the same reward also pays for being invested, and that gradient
swamps the selection part.
"""

import gymnasium as gym
import numpy as np

MAPPING_VERSION = "masked_softmax_cap_cash_v1"
TOP_K_VERSION = "masked_topk_softmax_cap_cash_v2"
FULL_INVEST_VERSION = "masked_softmax_cap_full_invest_v3"
FULL_INVEST_TOP_K_VERSION = "masked_topk_softmax_cap_full_invest_v4"


def mapping_version(top_k=None, full_invest=False):
    if full_invest:
        return FULL_INVEST_TOP_K_VERSION if top_k is not None else FULL_INVEST_VERSION
    return MAPPING_VERSION if top_k is None else TOP_K_VERSION


def cap_to_one(probabilities, max_weight, *, tolerance=1e-12):
    """Scale a vector to sum to one while honouring a per-name cap.

    Names that breach the cap are frozen at it and the remaining budget is redistributed pro
    rata among the rest; that is the only allocation preserving both the ordering and the
    total.  When every eligible name is capped and the total still falls short of one, the
    shortfall is left as cash for the caller to report -- an infeasible request is a fact, not
    something to paper over.
    """
    weights = np.asarray(probabilities, dtype=np.float64).copy()
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    total = weights.sum()
    if len(weights) == 0 or total <= 0:
        return np.zeros_like(weights)
    weights = weights / total
    frozen = np.zeros(len(weights), dtype=bool)
    for _ in range(len(weights) + 1):
        breach = (weights > max_weight + tolerance) & ~frozen
        if not breach.any():
            break
        weights[breach] = max_weight
        frozen |= breach
        free = ~frozen
        budget = 1.0 - float(weights[frozen].sum())
        if budget <= 0 or not free.any():
            break
        remainder = float(weights[free].sum())
        if remainder <= 0:
            break
        weights[free] = weights[free] / remainder * budget
    return weights


def _scores(raw, bound):
    return np.clip(raw, -bound, bound)


def map_action(latent, observation, max_weight, bound=8.0, top_k=None, full_invest=False):
    n = len(observation["valid_mask"])
    raw = np.asarray(latent, dtype=np.float64)
    expected = n if full_invest else n + 1
    if raw.shape != (expected,) or not np.isfinite(raw).all():
        raise ValueError(f"Expected a finite latent vector with {expected} entries")
    if not np.isfinite(bound) or bound <= 0 or not 0 < max_weight <= 1:
        raise ValueError("Invalid action bound or max_weight")
    eligible = np.asarray(observation["valid_mask"], dtype=bool) & np.asarray(observation["buy_mask"], dtype=bool)
    scores = _scores(raw, bound)
    if full_invest:
        masked = scores.copy()
        masked[~eligible] = -np.inf
        if top_k is not None:
            if type(top_k) is not int or top_k < 1:
                raise ValueError("top_k must be a positive integer")
            ranked = np.argsort(-masked, kind="stable")
            masked[ranked[top_k:]] = -np.inf
        weights = np.zeros(n, dtype=np.float64)
        if eligible.any() and np.isfinite(masked.max()):
            exp = np.exp(masked - masked.max())
            total = float(exp.sum())
            if total > 0:
                weights = cap_to_one(exp / total, max_weight)
        weights = np.minimum(weights, np.float32(max_weight)).astype(np.float32)
        held = float(weights.astype(np.float64).sum())
        if held > 1:
            weights *= np.nextafter(np.float32(1 / held), np.float32(0))
            held = float(weights.astype(np.float64).sum())
        return dict(weights=weights, cash_weight=float(1 - held),
                    latent=raw.copy(), clipped_latent=scores, eligible=eligible,
                    full_invest=True, feasible=bool(held >= 1 - 1e-6),
                    saturation_fraction=float(np.mean(np.abs(raw) >= bound)))
    masked = scores.copy()
    masked[:-1][~eligible] = -np.inf
    if top_k is not None:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        ranked = np.argsort(-masked[:-1], kind="stable")
        masked[ranked[top_k:]] = -np.inf
    exp = np.exp(masked - masked.max())
    probabilities = exp / exp.sum()
    weights = np.minimum(probabilities[:-1], max_weight).astype(np.float32)
    # Roundoff must not create leverage or exceed the broker's float64 cap.
    weights = np.minimum(weights, np.nextafter(np.float32(max_weight), np.float32(0)))
    if weights.astype(np.float64).sum() > 1:
        weights *= np.nextafter(np.float32(1 / weights.astype(np.float64).sum()), np.float32(0))
    return dict(weights=weights, cash_weight=float(1 - weights.astype(np.float64).sum()),
                latent=raw.copy(), clipped_latent=scores, eligible=eligible,
                full_invest=False, feasible=True,
                saturation_fraction=float(np.mean(np.abs(raw) >= bound)))


class LatentActionWrapper(gym.Wrapper):
    def __init__(self, env, bound=8.0, top_k=None, full_invest=False):
        super().__init__(env)
        if not np.isfinite(bound) or bound <= 0:
            raise ValueError("bound must be finite and positive")
        self.bound = float(bound)
        self.top_k = top_k
        self.full_invest = bool(full_invest)
        size = env.market.n_stocks if self.full_invest else env.market.n_stocks + 1
        self.action_space = gym.spaces.Box(-bound, bound, (size,), np.float32)
        self._observation = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._observation = obs
        return obs, info

    def step(self, action):
        if self._observation is None:
            raise RuntimeError("Reset before submitting a latent action")
        mapped = map_action(action, self._observation, self.env.broker.config.max_weight,
                            self.bound, self.top_k, self.full_invest)
        obs, reward, terminated, truncated, info = self.env.step(mapped["weights"])
        self._observation = None if terminated or truncated else obs
        info["action_diagnostics"] = mapped
        return obs, reward, terminated, truncated, info
