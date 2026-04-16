"""Integration tests for OpenGoalRL environments (require gfootball).

These tests need ``gfootball`` installed with system dependencies.
They are skipped automatically when GRF is not available, so the
rest of the test suite (test_rewards, test_wrappers, test_config) can
run in CI without GRF.

Run with::

    pytest opengoalrl/tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import gfootball  # noqa: F401
    HAS_GRF = True
except ImportError:
    HAS_GRF = False

pytestmark = pytest.mark.skipif(not HAS_GRF, reason="gfootball not installed")

from opengoalrl.envs.corner_kick import CornerKickEnv
from opengoalrl.envs.penalty import PenaltyEnv
from opengoalrl.rewards.ball_position_reward import BallInBoxReward
from opengoalrl.rewards.shot_reward import ShotReward
from opengoalrl.rewards.goal_reward import GoalReward
from opengoalrl.rewards.distance_reward import DistanceToGoalReward
from opengoalrl.wrappers.scenario_wrapper import ScenarioWrapper
from opengoalrl.wrappers.observation_wrapper import ObservationWrapper
from opengoalrl.wrappers.reward_wrapper import RewardWrapper
from opengoalrl.wrappers.action_wrapper import ActionWrapper


# ── Environment tests ────────────────────────────────────────────────

class TestCornerKickEnv:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.env = CornerKickEnv()
        yield
        self.env.close()

    def test_creation(self):
        assert self.env is not None
        assert self.env.observation_space is not None
        assert self.env.action_space is not None

    def test_reset_returns_valid_obs(self):
        obs, info = self.env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == self.env.observation_space.shape
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self):
        self.env.reset(seed=42)
        action = self.env.action_space.sample()
        result = self.env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


class TestPenaltyEnv:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.env = PenaltyEnv()
        yield
        self.env.close()

    def test_creation(self):
        assert self.env is not None

    def test_reset_and_step(self):
        obs, _ = self.env.reset(seed=0)
        assert obs.shape == self.env.observation_space.shape
        obs2, reward, term, trunc, info = self.env.step(0)
        assert obs2.shape == self.env.observation_space.shape


# ── Reward component tests ───────────────────────────────────────────

def _dummy_obs(ball_x: float = 0.0, ball_y: float = 0.0) -> np.ndarray:
    """Create a simple115v2-shaped observation with given ball position."""
    obs = np.zeros(115, dtype=np.float32)
    obs[88] = ball_x
    obs[89] = ball_y
    return obs


class TestRewardComponents:
    def test_ball_in_box_positive(self):
        comp = BallInBoxReward()
        r = comp.compute(
            _dummy_obs(), action=0, next_obs=_dummy_obs(0.9, 0.0), info={},
        )
        assert r == 1.0

    def test_ball_in_box_negative(self):
        comp = BallInBoxReward()
        r = comp.compute(
            _dummy_obs(), action=0, next_obs=_dummy_obs(0.5, 0.0), info={},
        )
        assert r == 0.0

    def test_shot_reward(self):
        comp = ShotReward()
        assert comp.compute(_dummy_obs(), 12, _dummy_obs(), info={}) == 1.0
        assert comp.compute(_dummy_obs(), 0, _dummy_obs(), info={}) == 0.0

    def test_goal_reward_scored(self):
        comp = GoalReward()
        r = comp.compute(
            _dummy_obs(), action=0, next_obs=_dummy_obs(),
            info={"score_reward": 1.0},
        )
        assert r == 1.0

    def test_goal_reward_conceded(self):
        comp = GoalReward()
        r = comp.compute(
            _dummy_obs(), action=0, next_obs=_dummy_obs(),
            info={"score_reward": -1.0},
        )
        assert r < 0

    def test_distance_to_goal(self):
        comp = DistanceToGoalReward()
        r_close = comp.compute(
            _dummy_obs(), action=0, next_obs=_dummy_obs(0.9, 0.0), info={},
        )
        r_far = comp.compute(
            _dummy_obs(), action=0, next_obs=_dummy_obs(-0.5, 0.0), info={},
        )
        assert r_close > r_far


# ── Wrapper stack tests ──────────────────────────────────────────────

class TestWrapperStack:
    @pytest.fixture(autouse=True)
    def _setup(self):
        base = CornerKickEnv()
        wrapped = ScenarioWrapper(base)
        wrapped = RewardWrapper(wrapped, components=[
            GoalReward(weight=10.0),
            BallInBoxReward(weight=1.0),
        ])
        wrapped = ObservationWrapper(wrapped)
        wrapped = ActionWrapper(wrapped)
        self.env = wrapped
        yield
        self.env.close()

    def test_stack_reset(self):
        obs, info = self.env.reset(seed=7)
        assert isinstance(obs, np.ndarray)
        assert "scenario" in info

    def test_stack_step(self):
        self.env.reset(seed=7)
        action = self.env.action_space.sample()
        obs, reward, term, trunc, info = self.env.step(action)
        assert "reward_components" in info
        assert isinstance(reward, float)
