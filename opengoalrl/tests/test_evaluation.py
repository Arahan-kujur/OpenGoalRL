"""Unit tests for shared evaluation rollout utilities -- no GRF required."""

from __future__ import annotations

import csv

import gymnasium as gym
import numpy as np
import pytest

from opengoalrl.utils.rollout import (
    BASIC_CSV_HEADER,
    run_episode,
    run_rollouts,
    summarize_results,
    write_episode_csv,
)


class _MetricEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        done = self._step_count >= 3
        info = {
            "score_reward": 1.0 if self._step_count == 3 else 0.0,
            "reward_components": {
                "ShotReward": 1.0 if self._step_count == 1 else 0.0,
                "BallInBoxReward": 1.0 if self._step_count in (1, 2) else 0.0,
            },
        }
        return np.zeros(4, dtype=np.float32), 2.0, done, False, info


def test_run_episode_collects_common_metrics():
    metrics = run_episode(_MetricEnv(), lambda obs: 0, episode=7)

    assert metrics.episode == 7
    assert metrics.reward == pytest.approx(6.0)
    assert metrics.steps == 3
    assert metrics.goals == 1
    assert metrics.shots == 1
    assert metrics.ball_in_box == 2


def test_run_episodes_and_summary():
    metrics = run_rollouts(_MetricEnv(), lambda obs: 0, 2)
    summary = summarize_results(metrics)

    assert len(metrics) == 2
    assert summary["mean_reward"] == pytest.approx(6.0)
    assert summary["total_goals"] == pytest.approx(2)
    assert summary["scoring_rate"] == pytest.approx(100.0)
    assert summary["shot_rate"] == pytest.approx(100.0)
    assert summary["ball_in_box_rate"] == pytest.approx(100.0)


def test_write_episode_metrics_csv_uses_existing_schema(tmp_path):
    metrics = run_rollouts(_MetricEnv(), lambda obs: 0, 1)
    out = write_episode_csv(metrics, tmp_path / "eval.csv")

    with open(out, newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == BASIC_CSV_HEADER
    assert rows[1] == ["1", "6.0", "3", "1", "1", "2"]
