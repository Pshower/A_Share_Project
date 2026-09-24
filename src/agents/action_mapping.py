"""A single latent-action mapping shared by PPO rollouts and inference."""

import gymnasium as gym
import numpy as np

MAPPING_VERSION = "masked_softmax_cap_cash_v1"


def map_action(latent, observation, max_weight, bound=8.0):
    n = len(observation["valid_mask"])
    raw = np.asarray(latent, dtype=np.float64)
    if raw.shape != (n + 1,) or not np.isfinite(raw).all():
        raise ValueError("Expected a finite latent vector with N+1 entries")
    if not np.isfinite(bound) or bound <= 0 or not 0 < max_weight <= 1:
        raise ValueError("Invalid action bound or max_weight")
    eligible = np.asarray(observation["valid_mask"], dtype=bool) & np.asarray(observation["buy_mask"], dtype=bool)
    scores = np.clip(raw, -bound, bound)
    masked = scores.copy()
    masked[:-1][~eligible] = -np.inf
    exp = np.exp(masked - masked.max())
    probabilities = exp / exp.sum()
    weights = np.minimum(probabilities[:-1], max_weight).astype(np.float32)
    # Roundoff must not create leverage or exceed the broker's float64 cap.
    weights = np.minimum(weights, np.nextafter(np.float32(max_weight), np.float32(0)))
    if weights.astype(np.float64).sum() > 1:
        weights *= np.float32(1 / weights.astype(np.float64).sum())
    return dict(weights=weights, cash_weight=float(1 - weights.astype(np.float64).sum()),
                latent=raw.copy(), clipped_latent=scores, eligible=eligible,
                saturation_fraction=float(np.mean(np.abs(raw) >= bound)))


class LatentActionWrapper(gym.Wrapper):
    def __init__(self, env, bound=8.0):
        super().__init__(env)
        if not np.isfinite(bound) or bound <= 0:
            raise ValueError("bound must be finite and positive")
        self.bound = float(bound)
        self.action_space = gym.spaces.Box(-bound, bound, (env.market.n_stocks + 1,), np.float32)
        self._observation = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._observation = obs
        return obs, info

    def step(self, action):
        if self._observation is None:
            raise RuntimeError("Reset before submitting a latent action")
        mapped = map_action(action, self._observation, self.env.broker.config.max_weight, self.bound)
        obs, reward, terminated, truncated, info = self.env.step(mapped["weights"])
        self._observation = None if terminated or truncated else obs
        info["action_diagnostics"] = mapped
        return obs, reward, terminated, truncated, info
