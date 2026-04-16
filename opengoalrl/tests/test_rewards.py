"""Pure unit tests for reward components -- no GRF installation required."""

from __future__ import annotations

import numpy as np
import pytest

from opengoalrl.rewards.ball_position_reward import BallInBoxReward
from opengoalrl.rewards.shot_reward import ShotReward
from opengoalrl.rewards.goal_reward import GoalReward
from opengoalrl.rewards.distance_reward import DistanceToGoalReward


def _obs(ball_x: float = 0.0, ball_y: float = 0.0) -> np.ndarray:
    """Synthetic simple115v2 observation with controllable ball position."""
    obs = np.zeros(115, dtype=np.float32)
    obs[88] = ball_x
    obs[89] = ball_y
    return obs


class TestBallInBoxReward:
    def test_inside_box(self):
        comp = BallInBoxReward()
        assert comp.compute(_obs(), 0, _obs(0.9, 0.0), {}) == 1.0

    def test_outside_box_x(self):
        comp = BallInBoxReward()
        assert comp.compute(_obs(), 0, _obs(0.5, 0.0), {}) == 0.0

    def test_outside_box_y(self):
        comp = BallInBoxReward()
        assert comp.compute(_obs(), 0, _obs(0.9, 0.3), {}) == 0.0

    def test_boundary_x(self):
        comp = BallInBoxReward()
        assert comp.compute(_obs(), 0, _obs(0.8, 0.0), {}) == 1.0

    def test_boundary_y(self):
        comp = BallInBoxReward()
        assert comp.compute(_obs(), 0, _obs(0.9, 0.19), {}) == 1.0

    def test_weight(self):
        comp = BallInBoxReward(weight=3.0)
        assert comp.weight == 3.0


class TestShotReward:
    def test_shot_action(self):
        comp = ShotReward()
        assert comp.compute(_obs(), 12, _obs(), {}) == 1.0

    def test_non_shot_action(self):
        comp = ShotReward()
        for action in [0, 1, 5, 11, 13, 18]:
            assert comp.compute(_obs(), action, _obs(), {}) == 0.0


class TestGoalReward:
    def test_goal_scored(self):
        comp = GoalReward()
        assert comp.compute(_obs(), 0, _obs(), {"score_reward": 1.0}) == 1.0

    def test_goal_conceded(self):
        comp = GoalReward()
        r = comp.compute(_obs(), 0, _obs(), {"score_reward": -1.0})
        assert r == -1.0

    def test_no_goal(self):
        comp = GoalReward()
        assert comp.compute(_obs(), 0, _obs(), {"score_reward": 0.0}) == 0.0

    def test_missing_key(self):
        comp = GoalReward()
        assert comp.compute(_obs(), 0, _obs(), {}) == 0.0

    def test_custom_concede_penalty(self):
        comp = GoalReward(concede_penalty=-5.0)
        r = comp.compute(_obs(), 0, _obs(), {"score_reward": -1.0})
        assert r == -5.0


class TestDistanceToGoalReward:
    def test_at_goal(self):
        comp = DistanceToGoalReward()
        r = comp.compute(_obs(), 0, _obs(1.0, 0.0), {})
        assert r == pytest.approx(1.0)

    def test_far_from_goal(self):
        comp = DistanceToGoalReward()
        r = comp.compute(_obs(), 0, _obs(-1.0, 0.0), {})
        assert r < 0.2

    def test_closer_is_higher(self):
        comp = DistanceToGoalReward()
        r_close = comp.compute(_obs(), 0, _obs(0.9, 0.0), {})
        r_far = comp.compute(_obs(), 0, _obs(0.0, 0.0), {})
        assert r_close > r_far

    def test_returns_float(self):
        comp = DistanceToGoalReward()
        r = comp.compute(_obs(), 0, _obs(0.5, 0.3), {})
        assert isinstance(r, float)

    def test_normalized_range(self):
        comp = DistanceToGoalReward()
        r = comp.compute(_obs(), 0, _obs(0.5, 0.3), {})
        assert 0.0 <= r <= 1.0
