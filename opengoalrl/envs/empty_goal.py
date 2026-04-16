"""Empty-goal scenario -- easy scoring from midfield."""

from __future__ import annotations

from typing import Any, Optional

from opengoalrl.envs.base_env import BaseScenarioEnv


class EmptyGoalEnv(BaseScenarioEnv):
    """GRF ``academy_empty_goal`` wrapped as a Gymnasium env.

    One attacker near centre with the ball facing an unguarded goal.
    Slightly harder than ``EmptyGoalCloseEnv`` because the player must
    run further before shooting.
    """

    def __init__(
        self,
        max_steps: int = 300,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            scenario_name="academy_empty_goal",
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )
