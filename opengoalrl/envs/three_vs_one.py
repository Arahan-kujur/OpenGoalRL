"""3-vs-1 with keeper scenario -- multi-attacker overload."""

from __future__ import annotations

from typing import Any, Optional

from opengoalrl.envs.base_env import BaseScenarioEnv


class ThreeVsOneEnv(BaseScenarioEnv):
    """GRF ``academy_3_vs_1_with_keeper`` wrapped as a Gymnasium env.

    Three attackers against one defender and a goalkeeper.  Requires
    decision-making about when to pass and when to shoot.
    """

    def __init__(
        self,
        max_steps: int = 400,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            scenario_name="academy_3_vs_1_with_keeper",
            max_steps=max_steps,
            render_mode=render_mode,
            **kwargs,
        )
