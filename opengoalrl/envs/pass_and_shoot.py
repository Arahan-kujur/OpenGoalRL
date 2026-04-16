"""Pass-and-shoot scenario -- two attackers vs keeper."""

from __future__ import annotations

from typing import Any, Optional

from opengoalrl.envs.base_env import BaseScenarioEnv


class PassAndShootEnv(BaseScenarioEnv):
    """GRF ``academy_pass_and_shoot_with_keeper`` wrapped as a Gymnasium env.

    Two left-side attackers near the box against a keeper and one defender.
    Requires learning to pass then shoot -- a coordination challenge.
    """

    def __init__(
        self,
        max_steps: int = 400,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            scenario_name="academy_pass_and_shoot_with_keeper",
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )
