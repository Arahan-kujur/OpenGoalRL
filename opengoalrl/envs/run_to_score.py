"""Run-to-score scenario -- dribble past defenders."""

from __future__ import annotations

from typing import Any, Optional

from opengoalrl.envs.base_env import BaseScenarioEnv


class RunToScoreEnv(BaseScenarioEnv):
    """GRF ``academy_run_to_score`` wrapped as a Gymnasium env.

    One attacker must dribble past a line of defenders and score.
    Medium difficulty -- requires learning directional movement.
    """

    def __init__(
        self,
        max_steps: int = 400,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            scenario_name="academy_run_to_score",
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )
