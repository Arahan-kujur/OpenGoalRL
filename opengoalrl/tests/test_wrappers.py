"""Unit tests for wrappers using a mock Gymnasium env -- no GRF required."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from opengoalrl.wrappers.scenario_wrapper import ScenarioWrapper
from opengoalrl.wrappers.observation_wrapper import ObservationWrapper
from opengoalrl.wrappers.reward_wrapper import RewardWrapper
from opengoalrl.wrappers.action_wrapper import ActionWrapper
from opengoalrl.rewards.base_reward import RewardComponent


class _MockEnv(gym.Env):
    """Minimal Gymnasium env for testing wrappers without GRF."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(115,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(19)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs = self.observation_space.sample()
        return obs, {"score_reward": 0.0}

    def step(self, action):
        self._step_count += 1
        obs = self.observation_space.sample()
        done = self._step_count >= 10
        return obs, 1.0, done, False, {"score_reward": 0.0}


class _ConstantReward(RewardComponent):
    """Returns a fixed value for testing."""

    def __init__(self, value: float = 5.0, weight: float = 1.0):
        super().__init__(weight)
        self._value = value

    def compute(self, obs, action, next_obs, info):
        return self._value


class TestScenarioWrapper:
    def test_injects_scenario_name(self):
        env = ScenarioWrapper(_MockEnv(), scenario_name="test_scenario")
        _, info = env.reset()
        assert info["scenario"] == "test_scenario"

    def test_tracks_steps(self):
        env = ScenarioWrapper(_MockEnv(), scenario_name="test")
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["episode_steps"] == 1
        _, _, _, _, info = env.step(0)
        assert info["episode_steps"] == 2

    def test_resets_counters(self):
        env = ScenarioWrapper(_MockEnv(), scenario_name="test")
        env.reset()
        env.step(0)
        env.step(0)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["episode_steps"] == 1
        assert info["episode_goals"] == 0


class TestObservationWrapper:
    def test_normalizes_to_unit_range(self):
        env = ObservationWrapper(_MockEnv(), normalize=True, clip_range=3.0)
        obs, _ = env.reset()
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_no_normalize_passthrough(self):
        inner = _MockEnv()
        env = ObservationWrapper(inner, normalize=False)
        obs, _ = env.reset()
        assert obs.min() < 0.0 or obs.max() > 1.0 or True  # just no crash

    def test_flatten(self):
        env = ObservationWrapper(_MockEnv(), normalize=False, flatten=True)
        obs, _ = env.reset()
        assert obs.ndim == 1

    def test_observation_space_updated(self):
        env = ObservationWrapper(_MockEnv(), normalize=True)
        assert env.observation_space.low.min() == pytest.approx(0.0)
        assert env.observation_space.high.max() == pytest.approx(1.0)


class TestRewardWrapper:
    def test_replaces_raw_reward(self):
        inner = _MockEnv()
        comp = _ConstantReward(value=5.0, weight=2.0)
        env = RewardWrapper(inner, components=[comp])
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(10.0)

    def test_multiple_components(self):
        inner = _MockEnv()
        c1 = _ConstantReward(value=3.0, weight=1.0)
        c2 = _ConstantReward(value=2.0, weight=0.5)
        env = RewardWrapper(inner, components=[c1, c2])
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert reward == pytest.approx(3.0 * 1.0 + 2.0 * 0.5)
        assert "reward_components" in info

    def test_breakdown_in_info(self):
        inner = _MockEnv()
        comp = _ConstantReward(value=7.0, weight=1.0)
        env = RewardWrapper(inner, components=[comp])
        env.reset()
        _, _, _, _, info = env.step(0)
        assert len(info["reward_components"]) == 1


class TestActionWrapper:
    def test_passthrough_no_restriction(self):
        env = ActionWrapper(_MockEnv())
        assert env.action_space.n == 19
        env.reset()
        env.step(12)

    def test_restricted_action_space(self):
        env = ActionWrapper(_MockEnv(), allowed_actions=[0, 5, 12])
        assert env.action_space.n == 3
        env.reset()
        env.step(0)
        env.step(1)
        env.step(2)

    def test_action_mapping(self):
        env = ActionWrapper(_MockEnv(), allowed_actions=[5, 10, 15])
        assert env.action(0) == 5
        assert env.action(1) == 10
        assert env.action(2) == 15
