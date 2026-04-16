"""Reward for keeping the ball inside the opponent's penalty area."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengoalrl.rewards.base_reward import RewardComponent

# In GRF's simple115v2 observation the ball position is encoded at indices
# 88 (x) and 89 (y), normalised to roughly [-1, 1].  The opponent's penalty
# area spans approximately x > 0.8, -0.2 < y < 0.2.
_BALL_X_IDX = 88
_BALL_Y_IDX = 89
_BOX_X_MIN = 0.8
_BOX_Y_ABS_MAX = 0.20


class BallInBoxReward(RewardComponent):
    """Returns +1 when the ball is inside the opponent's penalty box."""

    def compute(
        self,
        obs: np.ndarray,
        action: int,
        next_obs: np.ndarray,
        info: dict[str, Any],
    ) -> float:
        ball_x = float(next_obs[_BALL_X_IDX])
        ball_y = float(next_obs[_BALL_Y_IDX])
        in_box = ball_x >= _BOX_X_MIN and abs(ball_y) <= _BOX_Y_ABS_MAX
        return 1.0 if in_box else 0.0
