"""Corner-kick scenario environment."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from opengoalrl.envs.base_env import BaseScenarioEnv

_BALL_X_IDX = 88
_CLEAR_X_THRESHOLD = 0.8


class CornerKickEnv(BaseScenarioEnv):
    """GRF ``academy_corner`` wrapped as a Gymnasium env.

    Terminates early when a goal is scored **or** the ball is cleared out of
    the danger zone (ball x drops below the threshold).
    """

    def __init__(
        self,
        max_steps: int = 400,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            scenario_name="academy_corner",
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )

    def _is_ball_cleared(self, obs: np.ndarray) -> bool:
        return float(obs[_BALL_X_IDX]) < _CLEAR_X_THRESHOLD
