"""Close-range empty-goal scenario -- trivial scoring environment."""

from __future__ import annotations

from typing import Any, Optional

from opengoalrl.envs.base_env import BaseScenarioEnv


class EmptyGoalCloseEnv(BaseScenarioEnv):
    """GRF ``academy_empty_goal_close`` wrapped as a Gymnasium env.

    Ball starts very close to an unguarded goal.  Designed as the easiest
    possible scenario so the agent learns to score within minutes.
    """

    def __init__(
        self,
        max_steps: int = 200,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            scenario_name="academy_empty_goal_close",
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )
