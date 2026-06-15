"""Base Gymnasium environment wrapping a Google Research Football scenario."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

import gfootball.env as grf_env

from opengoalrl.envs.grf_compat import patch_grf_api as _patch_grf_api


class BaseScenarioEnv(gym.Env):
    """Gymnasium-compatible wrapper around a single GRF academy scenario.

    Subclasses set ``scenario_name`` and optionally override the termination
    helpers to implement scenario-specific episode boundaries.

    Parameters
    ----------
    scenario_name:
        GRF scenario id passed to ``create_environment``.
    max_steps:
        Maximum steps before the episode is truncated.
    representation:
        GRF observation representation (default ``"simple115v2"``).
    rewards:
        GRF reward type fed to ``create_environment`` (default ``"scoring"``).
    render_mode:
        ``"human"`` to render, ``None`` for headless.
    other_config_options:
        Extra kwargs forwarded to ``gfootball.env.create_environment``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario_name: str,
        max_steps: int = 400,
        representation: str = "simple115v2",
        rewards: str = "scoring",
        render_mode: Optional[str] = None,
        **other_config_options: Any,
    ) -> None:
        super().__init__()
        self.scenario_name = scenario_name
        self.max_steps = max_steps
        self._step_count = 0

        self._grf_env = grf_env.create_environment(
            env_name=scenario_name,
            representation=representation,
            rewards=rewards,
            render=render_mode == "human",
            **other_config_options,
        )
        _patch_grf_api(self._grf_env)

        obs_shape = self._grf_env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self._grf_env.action_space.n)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        result = self._grf_env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        self._step_count = 0
        return np.asarray(obs, dtype=np.float32), {}

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step_result = self._grf_env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated_grf, truncated_grf, info = step_result
            done = terminated_grf or truncated_grf
        else:
            obs, reward, done, info = step_result
        obs = np.asarray(obs, dtype=np.float32)
        self._step_count += 1

        score_reward = reward
        info["score_reward"] = float(score_reward)

        terminated = done or self._is_goal_scored(obs, info) or self._is_ball_cleared(obs)
        truncated = self._is_timeout()

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        pass

    def close(self) -> None:
        self._grf_env.close()

    # ------------------------------------------------------------------
    # Overridable termination helpers
    # ------------------------------------------------------------------

    def _is_goal_scored(self, obs: np.ndarray, info: dict[str, Any]) -> bool:
        return info.get("score_reward", 0.0) != 0.0

    def _is_ball_cleared(self, obs: np.ndarray) -> bool:
        return False

    def _is_timeout(self) -> bool:
        return self._step_count >= self.max_steps
