"""Tests for shared rollout utilities."""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from opengoalrl.utils.rollout import (
    EpisodeResult,
    extract_step_stats,
    run_episode,
    summarize_results,
)


class _FakeEnv(gym.Env):
    def __init__(self, steps=3):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(115,))
        self.action_space = gym.spaces.Discrete(19)
        self._steps = steps
        self._count = 0

    def reset(self, *, seed=None, options=None):
        self._count = 0
        return np.zeros(115, dtype=np.float32), {"scenario": "test"}

    def step(self, action):
        self._count += 1
        obs = np.zeros(115, dtype=np.float32)
        obs[88] = 0.1 * self._count
        done = self._count >= self._steps
        info = {"score_reward": 1.0 if self._count == self._steps else 0.0}
        if action == 12:
            info["reward_components"] = {"ShotReward": 1.0}
        return obs, 0.5, done, False, info


class TestRollout:
    def test_run_episode(self):
        env = _FakeEnv(steps=2)
        result = run_episode(env, lambda o: 12, collect_tactical=True)
        assert isinstance(result, EpisodeResult)
        assert result.steps == 2
        assert result.tactical is not None

    def test_extract_step_stats(self):
        g, s, b = extract_step_stats({
            "score_reward": 1.0,
            "reward_components": {"ShotReward": 2.0, "BallInBoxReward": 1.0},
        })
        assert g == 1 and s == 1 and b == 1

    def test_summarize(self):
        results = [
            EpisodeResult(1, 1.0, 10, 1, 0, 0),
            EpisodeResult(2, 0.0, 20, 0, 0, 0),
        ]
        s = summarize_results(results)
        assert s["scoring_rate"] == 50.0
