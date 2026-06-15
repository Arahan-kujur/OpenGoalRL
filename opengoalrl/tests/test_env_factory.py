"""Unit tests for shared environment construction -- no GRF required."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from opengoalrl.utils import env_factory
from opengoalrl.wrappers.action_wrapper import ActionWrapper
from opengoalrl.wrappers.observation_wrapper import ObservationWrapper
from opengoalrl.wrappers.reward_wrapper import RewardWrapper
from opengoalrl.wrappers.scenario_wrapper import ScenarioWrapper


class _MockEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 400, render_mode: str | None = None) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(115,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(19)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(115, dtype=np.float32), {"score_reward": 0.0}

    def step(self, action):
        return np.zeros(115, dtype=np.float32), 0.0, True, False, {"score_reward": 0.0}


def test_build_env_preserves_documented_wrapper_order(monkeypatch):
    monkeypatch.setitem(env_factory.ENV_REGISTRY, "empty_goal", _MockEnv)

    env = env_factory.build_env({
        "environment": {"scenario": "empty_goal", "max_steps": 123},
        "rewards": [{"type": "goal", "weight": 1.0}],
    })

    assert isinstance(env, ActionWrapper)
    assert isinstance(env.env, ObservationWrapper)
    assert isinstance(env.env.env, RewardWrapper)
    assert isinstance(env.env.env.env, ScenarioWrapper)
    assert isinstance(env.env.env.env.env, _MockEnv)
    assert env.env.env.env.env.max_steps == 123


def test_build_env_skips_reward_wrapper_when_no_rewards(monkeypatch):
    monkeypatch.setitem(env_factory.ENV_REGISTRY, "empty_goal", _MockEnv)

    env = env_factory.build_env({
        "environment": {"scenario": "empty_goal"},
        "rewards": [],
    })

    assert isinstance(env, ActionWrapper)
    assert isinstance(env.env, ObservationWrapper)
    assert isinstance(env.env.env, ScenarioWrapper)
    assert isinstance(env.env.env.env, _MockEnv)
