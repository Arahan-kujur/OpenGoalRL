"""Multi-agent Google Research Football environment layer."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from opengoalrl.envs.grf_compat import patch_grf_api
from opengoalrl.scenarios.spec import GRF_SCENARIO_MAP


class MultiAgentFootballEnv(gym.Env):
    """Parallel multi-agent env controlling multiple left-team players.

    Uses GRF ``number_of_left_players_agent_controls`` and returns a
    dict observation keyed by agent id for PettingZoo-style consumers.
    For SB3 parameter-sharing training, use :meth:`as_single_agent`.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario: str = "three_vs_one",
        n_agents: int = 3,
        max_steps: int = 400,
        representation: str = "simple115v2",
        render_mode: Optional[str] = None,
        team_reward: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        import gfootball.env as grf_env

        self.scenario = scenario
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.team_reward = team_reward
        self._step_count = 0

        grf_scenario = GRF_SCENARIO_MAP.get(scenario, scenario)
        self._grf_env = grf_env.create_environment(
            env_name=grf_scenario,
            representation=representation,
            rewards="scoring",
            render=render_mode == "human",
            number_of_left_players_agent_controls=n_agents,
            **kwargs,
        )
        patch_grf_api(self._grf_env)

        obs_shape = self._grf_env.observation_space.shape
        self._obs_shape = obs_shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32,
        )
        self.action_space = gym.spaces.MultiDiscrete(
            [self._grf_env.action_space.n] * n_agents,
        )
        self.agent_ids = [f"agent_{i}" for i in range(n_agents)]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        result = self._grf_env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs = np.asarray(obs, dtype=np.float32)
        self._step_count = 0
        if obs.ndim == 1:
            obs = np.stack([obs] * self.n_agents)
        info = {"scenario": self.scenario, "n_agents": self.n_agents}
        return self._obs_dict(obs), info

    def step(self, actions):
        if isinstance(actions, dict):
            action_arr = [actions[aid] for aid in self.agent_ids]
        else:
            action_arr = list(actions)
        step_result = self._grf_env.step(action_arr)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
            terminated, truncated = done, False
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = np.stack([obs] * self.n_agents)
        self._step_count += 1

        if self.team_reward:
            rewards = {aid: float(reward) for aid in self.agent_ids}
        else:
            share = float(reward) / self.n_agents
            rewards = {aid: share for aid in self.agent_ids}

        info = dict(info)
        info["scenario"] = self.scenario
        info["team_reward"] = float(reward)
        info["score_reward"] = float(reward)

        truncated = self._step_count >= self.max_steps
        return self._obs_dict(obs), rewards, terminated, truncated, info

    def _obs_dict(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        if obs.ndim == 1:
            return {aid: obs for aid in self.agent_ids}
        return {aid: obs[i] for i, aid in enumerate(self.agent_ids)}

    def as_single_agent(self) -> "SingleAgentMAView":
        return SingleAgentMAView(self)

    def close(self) -> None:
        self._grf_env.close()


class SingleAgentMAView(gym.Env):
    """Flatten multi-agent obs/actions for parameter-sharing PPO training."""

    def __init__(self, ma_env: MultiAgentFootballEnv) -> None:
        self.ma_env = ma_env
        n = ma_env.n_agents
        shape = (n * ma_env._obs_shape[0],)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=shape, dtype=np.float32,
        )
        self.action_space = ma_env.action_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.ma_env.reset(seed=seed, options=options)
        return self._flatten_obs(obs), info

    def step(self, actions):
        obs, rewards, term, trunc, info = self.ma_env.step(actions)
        team_r = info.get("team_reward", sum(rewards.values()))
        return self._flatten_obs(obs), float(team_r), term, trunc, info

    @staticmethod
    def _flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([obs[k] for k in sorted(obs.keys())]).astype(np.float32)

    def close(self) -> None:
        self.ma_env.close()
