"""Wrapper that replaces the environment reward with a shaped composite."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np

from opengoalrl.rewards.base_reward import RewardComponent


class RewardWrapper(gym.Wrapper):
    """Compute a weighted sum of :class:`RewardComponent` instances.

    The original environment reward is **replaced** by the composite reward.
    A per-component breakdown is stored in ``info["reward_components"]``.
    """

    def __init__(
        self,
        env: gym.Env,
        components: Sequence[RewardComponent],
    ) -> None:
        super().__init__(env)
        self.components = list(components)
        self._prev_obs: Optional[np.ndarray] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_obs = obs
        return obs, info

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, _raw_reward, terminated, truncated, info = self.env.step(action)

        breakdown: dict[str, float] = {}
        total = 0.0
        for comp in self.components:
            value = comp.compute(self._prev_obs, action, obs, info)
            weighted = comp.weight * value
            breakdown[repr(comp)] = weighted
            total += weighted

        info["reward_components"] = breakdown
        self._prev_obs = obs
        return obs, total, terminated, truncated, info
