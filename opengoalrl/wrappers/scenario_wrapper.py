"""Wrapper that injects scenario metadata and tracks episode statistics."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np


class ScenarioWrapper(gym.Wrapper):
    """Adds scenario metadata to ``info`` and tracks per-episode stats.

    Attributes injected into ``info`` on every step:

    * ``scenario`` – name of the active scenario
    * ``episode_steps`` – steps taken so far in the current episode
    * ``episode_goals`` – goals scored so far in the current episode
    """

    def __init__(self, env: gym.Env, scenario_name: str = "") -> None:
        super().__init__(env)
        self.scenario_name = scenario_name or getattr(env, "scenario_name", "unknown")
        self._episode_steps = 0
        self._episode_goals = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._episode_steps = 0
        self._episode_goals = 0
        info["scenario"] = self.scenario_name
        return obs, info

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_steps += 1
        if info.get("score_reward", 0.0) > 0:
            self._episode_goals += 1
        info["scenario"] = self.scenario_name
        info["episode_steps"] = self._episode_steps
        info["episode_goals"] = self._episode_goals
        return obs, reward, terminated, truncated, info
