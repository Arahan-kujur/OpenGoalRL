"""Penalty-kick scenario environment."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from opengoalrl.envs.base_env import BaseScenarioEnv

_SHOT_ACTION = 12


class PenaltyEnv(BaseScenarioEnv):
    """GRF ``academy_single_goal_versus_lazy`` as a penalty-kick proxy.

    Terminates when a goal is scored, when the agent takes a shot (regardless
    of outcome), or on timeout.
    """

    def __init__(
        self,
        max_steps: int = 150,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            scenario_name="academy_single_goal_versus_lazy",
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )
        self._shot_taken = False

    def reset(self, *, seed=None, options=None):
        self._shot_taken = False
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        if action == _SHOT_ACTION:
            self._shot_taken = True
        return super().step(action)

    def _is_ball_cleared(self, obs: np.ndarray) -> bool:
        return self._shot_taken
